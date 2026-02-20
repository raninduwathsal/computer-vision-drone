import os
import select
import socket
import struct
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

import cv2
import numpy as np
from pymavlink import mavutil

try:
	from pupil_apriltags import Detector  # type: ignore[import-not-found]
except ImportError as exc:
	Detector = None  # type: ignore[assignment]
	APRILTAG_IMPORT_ERROR = exc
else:
	APRILTAG_IMPORT_ERROR = None


MAVLINK_CONNECTION = os.getenv("MAVLINK_CONNECTION", "udp:0.0.0.0:14550")
CAMERA_HOST = os.getenv("CAMERA_HOST", "localhost")
CAMERA_PORT = int(os.getenv("CAMERA_PORT", "5599"))
CAMERA_CHANNELS = int(os.getenv("CAMERA_CHANNELS", "1"))

TARGET_ALT_M = float(os.getenv("TARGET_ALT_M", "2.0"))
ALT_HARD_MAX_M = float(os.getenv("ALT_HARD_MAX_M", "2.3"))
MAIN_LOOP_HZ = float(os.getenv("MAIN_LOOP_HZ", "35.0"))
MAX_MISSION_S = float(os.getenv("MAX_MISSION_S", "600"))

WAIT_AFTER_TAKEOFF_S = float(os.getenv("WAIT_AFTER_TAKEOFF_S", "1.0"))
FORWARD_SPEED_MPS = float(os.getenv("FORWARD_SPEED_MPS", "0.20"))
YAW_TOL_DEG = float(os.getenv("YAW_TOL_DEG", "2.0"))
YAW_STABLE_FRAMES = int(os.getenv("YAW_STABLE_FRAMES", "6"))
TURN_RIGHT_DEG = float(os.getenv("TURN_RIGHT_DEG", "89.0"))
TAKEOFF_SPEED_UP_CM_S = float(os.getenv("TAKEOFF_SPEED_UP_CM_S", "45.0"))
TAKEOFF_ACCEL_Z_CMSS = float(os.getenv("TAKEOFF_ACCEL_Z_CMSS", "65.0"))
TURN_MAX_YAW_RATE_RAD_S = float(os.getenv("TURN_MAX_YAW_RATE_RAD_S", "0.32"))
LINE_PID_START_DIST_M = float(os.getenv("LINE_PID_START_DIST_M", "2.0"))
LINE_TRACK_SPEED_MPS = float(os.getenv("LINE_TRACK_SPEED_MPS", "0.18"))
LINE_FALLBACK_SPEED_MPS = float(os.getenv("LINE_FALLBACK_SPEED_MPS", "0.12"))
LINE_MAX_VY_MPS = float(os.getenv("LINE_MAX_VY_MPS", "0.16"))
LINE_MAX_YAW_RATE_RAD_S = float(os.getenv("LINE_MAX_YAW_RATE_RAD_S", "0.22"))
LINE_CENTER_DEADBAND = float(os.getenv("LINE_CENTER_DEADBAND", "0.04"))
TAG_APPROACH_DIST_M = float(os.getenv("TAG_APPROACH_DIST_M", "0.2"))

TAG_MIN_MARGIN = float(os.getenv("TAG_MIN_MARGIN", "30.0"))
TAG_OBS_MAX_AGE_S = float(os.getenv("TAG_OBS_MAX_AGE_S", "0.35"))
APRIL_NTHREADS = int(os.getenv("APRIL_NTHREADS", "4"))
APRIL_QUAD_DECIMATE = float(os.getenv("APRIL_QUAD_DECIMATE", "2.0"))

SHOW_DEBUG = os.getenv("SHOW_CV_DEBUG", "1") == "1"
DISPLAY_EVERY_N = max(1, int(os.getenv("DISPLAY_EVERY_N", "2")))


class MissionState(Enum):
	WAIT_HEARTBEAT = auto()
	SET_GUIDED = auto()
	PREPARE_ARM = auto()
	ARMING = auto()
	TAKEOFF = auto()
	WAIT_AFTER_TAKEOFF = auto()
	SEARCH_FIRST_TAG = auto()
	TURN_RIGHT = auto()
	SEARCH_FINAL_TAG = auto()
	LANDING = auto()
	DONE = auto()


def wrap_deg(angle: float) -> float:
	return (angle + 360.0) % 360.0


def shortest_yaw_err_deg(current_deg: float, target_deg: float) -> float:
	return (target_deg - current_deg + 180.0) % 360.0 - 180.0


@dataclass
class PIDController:
	kp: float
	ki: float
	kd: float
	output_limit: float
	integral_limit: float

	def __post_init__(self) -> None:
		self.integral = 0.0
		self.last_error: Optional[float] = None
		self.last_t: Optional[float] = None

	def reset(self) -> None:
		self.integral = 0.0
		self.last_error = None
		self.last_t = None

	def update(self, error: float, now: float) -> float:
		if self.last_t is None:
			dt = 0.03
		else:
			dt = max(0.005, now - self.last_t)

		self.integral += error * dt
		self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))

		if self.last_error is None:
			derivative = 0.0
		else:
			derivative = (error - self.last_error) / dt

		output = self.kp * error + self.ki * self.integral + self.kd * derivative
		output = float(np.clip(output, -self.output_limit, self.output_limit))

		self.last_error = error
		self.last_t = now
		return output


@dataclass
class TagObservation:
	tag: Optional[Any]
	tag_id: int
	margin: float
	center_px: tuple[int, int]
	debug: np.ndarray


@dataclass
class LineObservation:
	found: bool
	x_norm: float
	angle_deg: float
	center_px: tuple[int, int]
	debug: np.ndarray


class CameraStreamClient:
	def __init__(self, host: str, port: int, channels: int = 1) -> None:
		self.host = host
		self.port = port
		self.channels = channels
		self.sock: Optional[socket.socket] = None
		self.buffer = bytearray()

	def connect(self) -> None:
		self.close()
		sock = socket.create_connection((self.host, self.port), timeout=5)
		sock.setblocking(False)
		sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
		self.sock = sock
		self.buffer.clear()
		print(f"Camera connected at {self.host}:{self.port}")

	def close(self) -> None:
		if self.sock is not None:
			try:
				self.sock.close()
			except Exception:
				pass
			self.sock = None

	def _pump_socket(self) -> None:
		if self.sock is None:
			self.connect()
		assert self.sock is not None
		for _ in range(16):
			ready, _, _ = select.select([self.sock], [], [], 0)
			if not ready:
				break
			data = self.sock.recv(65536)
			if not data:
				raise ConnectionError("Camera disconnected")
			self.buffer.extend(data)

	def read_latest_gray(self) -> Optional[np.ndarray]:
		try:
			self._pump_socket()
		except Exception:
			self.close()
			return None

		latest: Optional[np.ndarray] = None
		while True:
			if len(self.buffer) < 4:
				break
			width, height = struct.unpack("<HH", self.buffer[:4])
			frame_len = width * height * self.channels
			if len(self.buffer) < (4 + frame_len):
				break

			payload = bytes(self.buffer[4 : 4 + frame_len])
			del self.buffer[: 4 + frame_len]

			if self.channels == 1:
				latest = np.frombuffer(payload, dtype=np.uint8).reshape((height, width))
			elif self.channels == 3:
				rgb = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3))
				latest = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

		if len(self.buffer) > 3_000_000:
			self.buffer.clear()

		return latest


class MavController:
	def __init__(self, connection: str) -> None:
		self.master = mavutil.mavlink_connection(connection)
		self.last_n = 0.0
		self.last_e = 0.0
		self.last_d = 0.0
		self.last_alt = 0.0
		self.last_yaw_deg = 0.0
		self.last_hb_t = 0.0

	def close(self) -> None:
		self.master.close()

	def poll(self) -> None:
		while True:
			msg = self.master.recv_match(blocking=False)
			if msg is None:
				break
			mtype = msg.get_type()
			if mtype == "HEARTBEAT":
				sys_id = int(msg.get_srcSystem())
				comp_id = int(msg.get_srcComponent())
				if sys_id > 0:
					self.master.target_system = sys_id
					self.master.target_component = comp_id
				self.last_hb_t = time.monotonic()
			elif mtype == "LOCAL_POSITION_NED":
				self.last_n = float(msg.x)
				self.last_e = float(msg.y)
				self.last_d = float(msg.z)
			elif mtype == "GLOBAL_POSITION_INT":
				self.last_alt = float(msg.relative_alt) / 1000.0
			elif mtype == "ATTITUDE":
				self.last_yaw_deg = wrap_deg(np.degrees(float(msg.yaw)))

	def heartbeat_seen(self) -> bool:
		return (time.monotonic() - self.last_hb_t) < 2.0 and self.master.target_system > 0

	def request_message_interval(self, message_id: int, hz: float) -> None:
		if hz <= 0:
			return
		interval_us = int(1_000_000 / hz)
		self.master.mav.command_long_send(
			self.master.target_system,
			self.master.target_component,
			mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
			0,
			float(message_id),
			float(interval_us),
			0,
			0,
			0,
			0,
			0,
		)

	def configure_telemetry(self) -> None:
		self.request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 30.0)
		self.request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 20.0)
		self.request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 30.0)

	def set_mode(self, mode_name: str) -> None:
		mode_map = self.master.mode_mapping()
		if mode_map is None or mode_name not in mode_map:
			raise RuntimeError(f"Mode {mode_name} unsupported")
		self.master.mav.set_mode_send(
			self.master.target_system,
			mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
			mode_map[mode_name],
		)

	def arm_force(self) -> None:
		self.master.mav.command_long_send(
			self.master.target_system,
			self.master.target_component,
			mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
			0,
			1,
			21196,
			0,
			0,
			0,
			0,
			0,
		)

	def motors_armed(self) -> bool:
		return bool(self.master.motors_armed())

	def takeoff(self, altitude_m: float) -> None:
		self.master.mav.command_long_send(
			self.master.target_system,
			self.master.target_component,
			mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			altitude_m,
		)

	def land(self) -> None:
		self.set_mode("LAND")

	def altitude_estimate_m(self) -> float:
		local_alt = -float(self.last_d)
		if local_alt > 0.05:
			return max(float(self.last_alt), local_alt)
		return float(self.last_alt)

	def set_param(self, name: str, value: float) -> None:
		self.master.mav.param_set_send(
			self.master.target_system,
			self.master.target_component,
			name.encode("utf-8"),
			float(value),
			mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
		)

	def configure_sitl_prearm(self) -> None:
		for pname, pval in [
			("ARMING_CHECK", 0.0),
			("FS_THR_ENABLE", 0.0),
			("FS_GCS_ENABLE", 0.0),
			("DISARM_DELAY", 60.0),
			("WPNAV_SPEED_UP", TAKEOFF_SPEED_UP_CM_S),
			("WPNAV_ACCEL_Z", TAKEOFF_ACCEL_Z_CMSS),
		]:
			self.set_param(pname, pval)

	def send_body_velocity(self, vx: float, vy: float, vz: float, yaw_rate_rad_s: float) -> None:
		mask = (
			mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
			| mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
			| mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
			| mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
			| mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
			| mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
			| mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
		)
		self.master.mav.set_position_target_local_ned_send(
			0,
			self.master.target_system,
			self.master.target_component,
			mavutil.mavlink.MAV_FRAME_BODY_NED,
			mask,
			0,
			0,
			0,
			vx,
			vy,
			vz,
			0,
			0,
			0,
			0,
			yaw_rate_rad_s,
		)


def altitude_hold_vz(vehicle: MavController) -> float:
	altitude = vehicle.altitude_estimate_m()
	err = TARGET_ALT_M - altitude
	if err > 0.35:
		vz = -0.35
	elif err > 0.08:
		vz = -0.16
	elif err < -0.35:
		vz = 0.35
	elif err < -0.08:
		vz = 0.16
	else:
		vz = 0.0

	if altitude >= ALT_HARD_MAX_M:
		vz = max(vz, 0.0)
	return vz


def detect_wide_line(gray: np.ndarray) -> LineObservation:
	h, w = gray.shape
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	_, dark_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	candidates = [dark_mask]
	best_contour: Optional[np.ndarray] = None
	best_mask: Optional[np.ndarray] = None
	best_score = -1.0

	for mask_raw in candidates:
		mask = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for contour in contours:
			area = cv2.contourArea(contour)
			if area < 0.015 * h * w:
				continue
			x, y, cw, ch = cv2.boundingRect(contour)
			span_frac = max(cw / max(1.0, float(w)), ch / max(1.0, float(h)))
			if span_frac < 0.68:
				continue
			short_side = max(1.0, float(min(cw, ch)))
			long_side = float(max(cw, ch))
			elong = long_side / short_side
			if elong < 2.0:
				continue
			score = area * (0.55 + 0.45 * span_frac) * (0.50 + 0.50 * min(elong, 8.0))
			if score > best_score:
				best_score = score
				best_contour = contour
				best_mask = mask

	overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
	center = (w // 2, h // 2)

	if best_contour is None or best_mask is None:
		cv2.circle(overlay, center, 5, (255, 0, 0), -1)
		cv2.putText(overlay, "line: none", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2)
		return LineObservation(False, 0.0, 0.0, center, overlay)

	mom = cv2.moments(best_contour)
	if mom["m00"] == 0:
		cv2.circle(overlay, center, 5, (255, 0, 0), -1)
		cv2.putText(overlay, "line: invalid", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2)
		return LineObservation(False, 0.0, 0.0, center, overlay)

	cx = int(mom["m10"] / mom["m00"])
	cy = int(mom["m01"] / mom["m00"])
	line_pts = best_contour.reshape(-1, 2).astype(np.float32)
	vx, vy, _, _ = cv2.fitLine(line_pts, cv2.DIST_L2, 0, 0.01, 0.01)
	vx_f = float(np.asarray(vx).reshape(-1)[0])
	vy_f = float(np.asarray(vy).reshape(-1)[0])
	angle_deg = float(np.degrees(np.arctan2(vx_f, vy_f)))
	x_norm = (cx - center[0]) / max(1.0, (w / 2.0))

	overlay = cv2.cvtColor(best_mask, cv2.COLOR_GRAY2BGR)
	cv2.drawContours(overlay, [best_contour], -1, (0, 255, 0), 2)
	cv2.circle(overlay, center, 5, (255, 0, 0), -1)
	cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
	cv2.line(overlay, center, (cx, cy), (0, 255, 255), 2)
	cv2.putText(
		overlay,
		f"line x={x_norm:+.3f} ang={angle_deg:+.1f}",
		(8, 22),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.58,
		(255, 255, 255),
		2,
	)
	return LineObservation(True, float(x_norm), float(angle_deg), (cx, cy), overlay)


def detect_best_apriltag(detector: Any, gray: np.ndarray) -> TagObservation:
	overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
	h, w = gray.shape
	center = (w // 2, h // 2)

	detections = detector.detect(gray)
	if not detections:
		cv2.putText(overlay, "tag: none", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		return TagObservation(None, -1, 0.0, center, overlay)

	best = max(detections, key=lambda d: float(getattr(d, "decision_margin", 0.0)))
	margin = float(getattr(best, "decision_margin", 0.0))
	tag_id = int(getattr(best, "tag_id", -1))
	pts = np.array(best.corners, dtype=np.int32)
	cx = int(best.center[0])
	cy = int(best.center[1])

	cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
	cv2.circle(overlay, center, 5, (255, 0, 0), -1)
	cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
	cv2.putText(
		overlay,
		f"id={tag_id} margin={margin:.1f}",
		(8, 22),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.6,
		(0, 255, 0),
		2,
	)
	return TagObservation(best, tag_id, margin, (cx, cy), overlay)


def main() -> None:
	if Detector is None:
		raise RuntimeError(
			"Missing dependency 'pupil_apriltags'. Install with: pip install pupil-apriltags"
		) from APRILTAG_IMPORT_ERROR

	cv2.setUseOptimized(True)
	try:
		cv2.setNumThreads(1)
	except Exception:
		pass

	vehicle = MavController(MAVLINK_CONNECTION)
	camera = CameraStreamClient(CAMERA_HOST, CAMERA_PORT, CAMERA_CHANNELS)
	detector: Any = Detector(
		families="tag36h11",
		nthreads=APRIL_NTHREADS,
		quad_decimate=APRIL_QUAD_DECIMATE,
		refine_edges=0,
	)
	yaw_turn_pid = PIDController(kp=0.018, ki=0.0, kd=0.006, output_limit=TURN_MAX_YAW_RATE_RAD_S, integral_limit=0.15)
	line_lat_pid = PIDController(kp=0.24, ki=0.0, kd=0.07, output_limit=LINE_MAX_VY_MPS, integral_limit=0.20)

	mission_start = time.monotonic()
	state = MissionState.WAIT_HEARTBEAT
	state_enter_t = mission_start
	takeoff_cmd_t: Optional[float] = None
	first_tag_id: Optional[int] = None
	turn_target_yaw_deg: Optional[float] = None
	turn_stable_frames = 0
	last_tag_obs: Optional[TagObservation] = None
	last_line_obs: Optional[LineObservation] = None
	last_tag_obs_t = 0.0
	last_debug = np.zeros((240, 320, 3), dtype=np.uint8)
	frame_counter = 0
	search_start_n: Optional[float] = None
	search_start_e: Optional[float] = None
	line_pid_enabled = False
	tag_approach_n: Optional[float] = None
	tag_approach_e: Optional[float] = None
	tag_approach_active = False

	def transition(next_state: MissionState) -> None:
		nonlocal state, state_enter_t
		state = next_state
		state_enter_t = time.monotonic()

	def fresh_tag(reject_id: Optional[int] = None) -> Optional[TagObservation]:
		nonlocal last_tag_obs, last_tag_obs_t
		if last_tag_obs is None or last_tag_obs.tag is None:
			return None
		if (time.monotonic() - last_tag_obs_t) > TAG_OBS_MAX_AGE_S:
			return None
		if last_tag_obs.margin < TAG_MIN_MARGIN:
			return None
		if reject_id is not None and last_tag_obs.tag_id == reject_id:
			return None
		return last_tag_obs

	try:
		loop_dt = 1.0 / max(10.0, MAIN_LOOP_HZ)

		while state != MissionState.DONE:
			now = time.monotonic()
			if (now - mission_start) > MAX_MISSION_S:
				raise TimeoutError("Mission exceeded MAX_MISSION_S")

			vehicle.poll()

			if state.value >= MissionState.SEARCH_FIRST_TAG.value:
				gray = camera.read_latest_gray()
				if gray is not None:
					frame_counter += 1
					last_line_obs = detect_wide_line(gray)
					last_tag_obs = detect_best_apriltag(detector, gray)
					if last_tag_obs.tag is not None:
						last_tag_obs_t = now
					last_debug = last_line_obs.debug
					if last_tag_obs.tag is not None:
						cv2.putText(
							last_debug,
							f"tag id={last_tag_obs.tag_id} margin={last_tag_obs.margin:.1f}",
							(8, 44),
							cv2.FONT_HERSHEY_SIMPLEX,
							0.58,
							(0, 255, 0),
							2,
						)

			if state == MissionState.WAIT_HEARTBEAT:
				if vehicle.heartbeat_seen():
					vehicle.configure_telemetry()
					print("Heartbeat received")
					transition(MissionState.SET_GUIDED)

			elif state == MissionState.SET_GUIDED:
				vehicle.set_mode("GUIDED")
				print("GUIDED mode set")
				transition(MissionState.PREPARE_ARM)

			elif state == MissionState.PREPARE_ARM:
				vehicle.configure_sitl_prearm()
				takeoff_cmd_t = None
				transition(MissionState.ARMING)

			elif state == MissionState.ARMING:
				if vehicle.motors_armed():
					vehicle.takeoff(TARGET_ALT_M)
					takeoff_cmd_t = now
					print(f"Takeoff commanded to {TARGET_ALT_M:.2f} m")
					transition(MissionState.TAKEOFF)
				elif (now - state_enter_t) > 0.9:
					vehicle.arm_force()
					state_enter_t = now

			elif state == MissionState.TAKEOFF:
				if not vehicle.motors_armed():
					transition(MissionState.ARMING)
					continue

				alt = vehicle.altitude_estimate_m()
				if alt >= (TARGET_ALT_M - 0.15):
					camera.connect()
					print("Reached target altitude")
					transition(MissionState.WAIT_AFTER_TAKEOFF)
				elif takeoff_cmd_t is not None and (now - takeoff_cmd_t) > 3.0:
					vehicle.takeoff(TARGET_ALT_M)
					takeoff_cmd_t = now

			elif state == MissionState.WAIT_AFTER_TAKEOFF:
				vehicle.send_body_velocity(0.0, 0.0, altitude_hold_vz(vehicle), 0.0)
				if (now - state_enter_t) >= WAIT_AFTER_TAKEOFF_S:
					print("Moving forward straight until first AprilTag")
					transition(MissionState.SEARCH_FIRST_TAG)

			elif state == MissionState.SEARCH_FIRST_TAG:
				if search_start_n is None or search_start_e is None:
					search_start_n = vehicle.last_n
					search_start_e = vehicle.last_e
					line_pid_enabled = False
					line_lat_pid.reset()

				dist_from_start = float(
					np.hypot(vehicle.last_n - search_start_n, vehicle.last_e - search_start_e)
				)
				if (not line_pid_enabled) and dist_from_start >= LINE_PID_START_DIST_M:
					line_pid_enabled = True
					line_lat_pid.reset()
					print(f"Line PID enabled after {dist_from_start:.2f} m forward travel")

				vx_cmd = FORWARD_SPEED_MPS
				vy_cmd = 0.0
				yaw_rate_cmd = 0.0

				if line_pid_enabled:
					if last_line_obs is not None and last_line_obs.found:
						err_x = last_line_obs.x_norm
						if abs(err_x) <= LINE_CENTER_DEADBAND:
							err_x = 0.0
						vy_cmd = line_lat_pid.update(err_x, now)
						yaw_rate_cmd = 0.0
						vx_cmd = float(
							np.clip(
								LINE_TRACK_SPEED_MPS - 0.06 * abs(err_x),
								0.08,
								LINE_TRACK_SPEED_MPS,
							)
						)
					else:
						vx_cmd = LINE_FALLBACK_SPEED_MPS
						vy_cmd = 0.0
						yaw_rate_cmd = 0.0

				vehicle.send_body_velocity(vx_cmd, vy_cmd, altitude_hold_vz(vehicle), yaw_rate_cmd)
				tag = fresh_tag()
				if tag is not None:
					if not tag_approach_active:
						tag_approach_n = vehicle.last_n
						tag_approach_e = vehicle.last_e
						tag_approach_active = True
						first_tag_id = tag.tag_id
						print(f"FIRST APRILTAG ID: {first_tag_id}")
						print(f"Approaching tag for {TAG_APPROACH_DIST_M:.2f} m before stopping")

				if tag_approach_active and tag_approach_n is not None and tag_approach_e is not None:
					approach_dist = float(np.hypot(vehicle.last_n - tag_approach_n, vehicle.last_e - tag_approach_e))
					if approach_dist >= TAG_APPROACH_DIST_M:
						vehicle.send_body_velocity(0.0, 0.0, altitude_hold_vz(vehicle), 0.0)
						turn_target_yaw_deg = wrap_deg(vehicle.last_yaw_deg + TURN_RIGHT_DEG)
						yaw_turn_pid.reset()
						turn_stable_frames = 0
						search_start_n = None
						search_start_e = None
						line_pid_enabled = False
						tag_approach_active = False
						tag_approach_n = None
						tag_approach_e = None
						print(f"Turning right {TURN_RIGHT_DEG:.1f} degrees")
						transition(MissionState.TURN_RIGHT)

			elif state == MissionState.TURN_RIGHT:
				if turn_target_yaw_deg is None:
					turn_target_yaw_deg = wrap_deg(vehicle.last_yaw_deg + TURN_RIGHT_DEG)
				yaw_err_deg = shortest_yaw_err_deg(vehicle.last_yaw_deg, turn_target_yaw_deg)
				yaw_rate_cmd = yaw_turn_pid.update(yaw_err_deg, now)
				yaw_rate_cmd = float(np.clip(yaw_rate_cmd, -TURN_MAX_YAW_RATE_RAD_S, TURN_MAX_YAW_RATE_RAD_S))
				vehicle.send_body_velocity(0.0, 0.0, altitude_hold_vz(vehicle), yaw_rate_cmd)

				if abs(yaw_err_deg) <= YAW_TOL_DEG:
					turn_stable_frames += 1
					if turn_stable_frames >= YAW_STABLE_FRAMES:
						vehicle.send_body_velocity(0.0, 0.0, altitude_hold_vz(vehicle), 0.0)
						print("Turn complete. Moving forward to final AprilTag")
						search_start_n = None
						search_start_e = None
						line_pid_enabled = False
						line_lat_pid.reset()
						transition(MissionState.SEARCH_FINAL_TAG)
				else:
					turn_stable_frames = 0

			elif state == MissionState.SEARCH_FINAL_TAG:
				if search_start_n is None or search_start_e is None:
					search_start_n = vehicle.last_n
					search_start_e = vehicle.last_e
					line_pid_enabled = False
					line_lat_pid.reset()

				dist_from_start = float(
					np.hypot(vehicle.last_n - search_start_n, vehicle.last_e - search_start_e)
				)
				if (not line_pid_enabled) and dist_from_start >= LINE_PID_START_DIST_M:
					line_pid_enabled = True
					line_lat_pid.reset()
					print(f"Line PID enabled for final search after {dist_from_start:.2f} m forward travel")

				vx_cmd = FORWARD_SPEED_MPS
				vy_cmd = 0.0
				yaw_rate_cmd = 0.0

				if line_pid_enabled:
					if last_line_obs is not None and last_line_obs.found:
						err_x = last_line_obs.x_norm
						if abs(err_x) <= LINE_CENTER_DEADBAND:
							err_x = 0.0
						vy_cmd = line_lat_pid.update(err_x, now)
						yaw_rate_cmd = 0.0
						vx_cmd = float(
							np.clip(
								LINE_TRACK_SPEED_MPS - 0.06 * abs(err_x),
								0.08,
								LINE_TRACK_SPEED_MPS,
							)
						)
					else:
						vx_cmd = LINE_FALLBACK_SPEED_MPS
						vy_cmd = 0.0
						yaw_rate_cmd = 0.0

				vehicle.send_body_velocity(vx_cmd, vy_cmd, altitude_hold_vz(vehicle), yaw_rate_cmd)
				tag = fresh_tag(reject_id=first_tag_id)
				if tag is not None:
					if not tag_approach_active:
						tag_approach_n = vehicle.last_n
						tag_approach_e = vehicle.last_e
						tag_approach_active = True
						print(f"FINAL APRILTAG ID: {tag.tag_id}")
						print(f"Approaching final tag for {TAG_APPROACH_DIST_M:.2f} m before landing")

				if tag_approach_active and tag_approach_n is not None and tag_approach_e is not None:
					approach_dist = float(np.hypot(vehicle.last_n - tag_approach_n, vehicle.last_e - tag_approach_e))
					if approach_dist >= TAG_APPROACH_DIST_M:
						vehicle.send_body_velocity(0.0, 0.0, altitude_hold_vz(vehicle), 0.0)
						print("Final tag approached. Landing now")
						tag_approach_active = False
						tag_approach_n = None
						tag_approach_e = None
						vehicle.land()
						transition(MissionState.LANDING)

			elif state == MissionState.LANDING:
				if not vehicle.motors_armed() and vehicle.altitude_estimate_m() < 0.15:
					print("Landed and motors disarmed")
					transition(MissionState.DONE)

			if SHOW_DEBUG and frame_counter % DISPLAY_EVERY_N == 0:
				dbg = cv2.resize(last_debug, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_NEAREST)
				cv2.putText(
					dbg,
					f"state={state.name} alt={vehicle.altitude_estimate_m():.2f} yaw={vehicle.last_yaw_deg:.1f}",
					(8, 20),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					(255, 255, 255),
					2,
				)
				cv2.imshow("mission_debug", dbg)
				cv2.waitKey(1)

			time.sleep(loop_dt)

	except KeyboardInterrupt:
		print("Interrupted (Ctrl+C). Sending LAND")
		try:
			vehicle.land()
		except Exception:
			pass
		try:
			vehicle.send_body_velocity(0.0, 0.0, 0.2, 0.0)
		except Exception:
			pass
	except Exception as exc:
		print(f"Mission error: {exc}")
		try:
			vehicle.land()
		except Exception:
			pass
		raise
	finally:
		camera.close()
		vehicle.close()
		if SHOW_DEBUG:
			cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
