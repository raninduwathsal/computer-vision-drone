import os
import socket
import struct
import time
from dataclasses import dataclass
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


TARGET_ALTITUDE_M = float(os.getenv("TARGET_ALTITUDE_M", "2.5"))
TAKEOFF_TOLERANCE_M = 0.2
ALTITUDE_MAX_M = 2.95
CAMERA_HOST = os.getenv("CAMERA_HOST", "localhost")
CAMERA_PORT = int(os.getenv("CAMERA_PORT", "5599"))
FRAME_TIMEOUT_S = 8.0
LINE_STAGE_TIMEOUT_S = int(os.getenv("LINE_STAGE_TIMEOUT_S", "180"))
CENTER_TIMEOUT_S = int(os.getenv("CENTER_TIMEOUT_S", "35"))
SHOW_DEBUG = os.getenv("SHOW_CV_DEBUG", "1") == "1"
MAX_CAMERA_RECOVERY_S = float(os.getenv("MAX_CAMERA_RECOVERY_S", "20"))
CAMERA_CHANNELS = int(os.getenv("CAMERA_CHANNELS", "1"))
DEBUG_EVERY_N = int(os.getenv("DEBUG_EVERY_N", "3"))
TAG_DETECT_EVERY_N = int(os.getenv("TAG_DETECT_EVERY_N", "5"))
SLOW_LOOP_DT_S = float(os.getenv("SLOW_LOOP_DT_S", "0.18"))
LINE_ROI_TOP_RATIO = float(os.getenv("LINE_ROI_TOP_RATIO", "0.45"))
LINE_CONTROL_BAND_TOP_RATIO = float(os.getenv("LINE_CONTROL_BAND_TOP_RATIO", "0.62"))
LINE_SIDE_IGNORE_RATIO = float(os.getenv("LINE_SIDE_IGNORE_RATIO", "0.14"))


class CameraStreamClient:
	def __init__(self, host: str, port: int) -> None:
		self.host = host
		self.port = port
		self.sock: Optional[socket.socket] = None
		self.last_reconnect_attempt = 0.0

	def connect(self) -> None:
		self.sock = socket.create_connection((self.host, self.port), timeout=8)
		self.sock.settimeout(FRAME_TIMEOUT_S)
		print(f"Camera connected ({self.host}:{self.port})")
		self.last_reconnect_attempt = time.time()

	def close(self) -> None:
		if self.sock is not None:
			try:
				self.sock.close()
			except Exception:
				pass
			self.sock = None

	def _ensure_connected(self) -> None:
		if self.sock is None:
			now = time.time()
			if now - self.last_reconnect_attempt < 0.5:
				raise RuntimeError("Camera reconnect throttled")
			print("Camera reconnecting...")
			self.connect()

	def _recv_exact(self, size: int) -> bytes:
		if self.sock is None:
			raise RuntimeError("Camera socket is not connected")
		buffer = bytearray()
		while len(buffer) < size:
			try:
				chunk = self.sock.recv(size - len(buffer))
			except socket.timeout:
				raise ConnectionError("Camera frame timeout")
			if not chunk:
				raise ConnectionError("Camera stream disconnected")
			buffer.extend(chunk)
		return bytes(buffer)

	def read_frame(self) -> np.ndarray:
		self._ensure_connected()
		try:
			header = self._recv_exact(4)
			width, height = struct.unpack("<HH", header)
			if CAMERA_CHANNELS == 1:
				payload = self._recv_exact(width * height)
				gray = np.frombuffer(payload, dtype=np.uint8).reshape((height, width))
				return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
			if CAMERA_CHANNELS == 3:
				payload = self._recv_exact(width * height * 3)
				return np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3))
			raise ValueError(f"Unsupported CAMERA_CHANNELS={CAMERA_CHANNELS}; use 1 or 3")
		except (ConnectionError, socket.error, struct.error) as exc:
			self.close()
			raise ConnectionError(f"Camera stream error: {exc}")


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
			dt = 0.05
		else:
			dt = max(0.01, now - self.last_t)

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


def set_param(master: mavutil.mavfile, name: str, value: float) -> None:
	master.mav.param_set_send(
		master.target_system,
		master.target_component,
		name.encode("utf-8"),
		float(value),
		mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
	)


def set_param_and_verify(master: mavutil.mavfile, name: str, value: float, timeout_s: float = 3.0) -> bool:
	set_param(master, name, value)
	deadline = time.time() + timeout_s
	while time.time() < deadline:
		msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.4)
		if msg is None:
			continue
		pid = str(getattr(msg, "param_id", "")).strip("\x00")
		if pid == name:
			actual = float(getattr(msg, "param_value", np.nan))
			return abs(actual - float(value)) < 0.5

	master.mav.param_request_read_send(
		master.target_system,
		master.target_component,
		name.encode("utf-8"),
		-1,
	)
	deadline = time.time() + timeout_s
	while time.time() < deadline:
		msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.4)
		if msg is None:
			continue
		pid = str(getattr(msg, "param_id", "")).strip("\x00")
		if pid == name:
			actual = float(getattr(msg, "param_value", np.nan))
			return abs(actual - float(value)) < 0.5

	return False


def read_statustext(master: mavutil.mavfile) -> str:
	last_text = ""
	while True:
		msg = master.recv_match(type="STATUSTEXT", blocking=False)
		if msg is None:
			break
		text = str(getattr(msg, "text", "")).strip()
		if text:
			last_text = text
	return last_text


def send_body_velocity(master: mavutil.mavfile, vx: float, vy: float, vz: float, yaw_rate_rad_s: float) -> None:
	type_mask = (
		mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
		| mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
		| mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
		| mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
		| mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
		| mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
		| mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
	)

	master.mav.set_position_target_local_ned_send(
		0,
		master.target_system,
		master.target_component,
		mavutil.mavlink.MAV_FRAME_BODY_NED,
		type_mask,
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


def send_stop(master: mavutil.mavfile) -> None:
	send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)


def current_altitude_m(master: mavutil.mavfile) -> float:
	msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
	if msg is None:
		return 0.0
	return msg.relative_alt / 1000.0


def altitude_hold_vz(master: mavutil.mavfile, target_altitude_m: float) -> float:
	altitude = current_altitude_m(master)
	error = target_altitude_m - altitude
	if error > 0.5:
		return -0.4
	if error > 0.15:
		return -0.2
	if error < -0.5:
		return 0.4
	if error < -0.15:
		return 0.2
	return 0.0


def connect_vehicle() -> mavutil.mavfile:
	env_connection = os.getenv("MAVLINK_CONNECTION")
	connections = [env_connection] if env_connection else [
		"udp:0.0.0.0:14550",
		"udp:localhost:14550",
		"tcp:localhost:5760",
	]

	last_error = None
	for connection in connections:
		master = None
		try:
			print(f"Connecting to {connection} ...")
			master = mavutil.mavlink_connection(connection)
			deadline = time.time() + 45
			valid_hb = None
			while time.time() < deadline:
				hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
				if hb is None:
					continue
				sys_id = int(hb.get_srcSystem())
				comp_id = int(hb.get_srcComponent())
				if sys_id <= 0:
					continue
				master.target_system = sys_id
				master.target_component = comp_id
				valid_hb = hb
				break

			if valid_hb is None:
				raise TimeoutError("Timed out waiting for valid autopilot heartbeat")

			print(f"Connected (sys={master.target_system}, comp={master.target_component})")
			return master
		except Exception as exc:
			last_error = exc
			if master is not None:
				master.close()

	raise RuntimeError(f"Could not connect to vehicle: {last_error}")


def set_mode(master: mavutil.mavfile, mode_name: str) -> None:
	mode_map = master.mode_mapping()
	if mode_map is None or mode_name not in mode_map:
		raise RuntimeError(f"Unsupported mode: {mode_name}")

	master.mav.set_mode_send(
		master.target_system,
		mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
		mode_map[mode_name],
	)

	deadline = time.time() + 10
	while time.time() < deadline:
		hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
		if hb and mavutil.mode_string_v10(hb) == mode_name:
			print(f"Mode: {mode_name}")
			return

	raise TimeoutError(f"Mode change timeout: {mode_name}")


def arm_vehicle(master: mavutil.mavfile) -> None:
	print("Arming (bypassing pre-arm checks for SITL)...")
	
	# Immediately disable arming checks for SITL
	print("Setting ARMING_CHECK=0...")
	set_param(master, "ARMING_CHECK", 0)
	time.sleep(0.3)
	
	# Try multiple times with force arm
	for attempt in range(5):
		print(f"Arm attempt {attempt + 1}/5...")
		
		# Force arm with magic number
		master.mav.command_long_send(
			master.target_system,
			master.target_component,
			mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
			0,
			1,      # arm
			21196,  # force arm magic number
			0, 0, 0, 0, 0,
		)
		
		# Wait for arming
		deadline = time.time() + 3
		while time.time() < deadline:
			if master.motors_armed():
				print("Armed successfully!")
				return
			time.sleep(0.1)
		
		# Read any error messages
		last_reason = read_statustext(master)
		if last_reason:
			print(f"  Status: {last_reason}")
		
		time.sleep(0.3)
	
	raise TimeoutError("Arm failed after 5 attempts. Try restarting the simulator.")


def takeoff(master: mavutil.mavfile, altitude_m: float) -> None:
	target = min(float(altitude_m), ALTITUDE_MAX_M)
	print(f"Takeoff to {target:.2f} m")
	master.mav.command_long_send(
		master.target_system,
		master.target_component,
		mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		target,
	)

	deadline = time.time() + 45
	stable_since: Optional[float] = None
	while time.time() < deadline:
		alt = current_altitude_m(master)
		error = abs(target - alt)
		print(f"Altitude: {alt:.2f} m")

		if error < TAKEOFF_TOLERANCE_M:
			if stable_since is None:
				stable_since = time.time()
			elif time.time() - stable_since > 1.5:
				print("Stable hover altitude reached")
				return
		else:
			stable_since = None

		time.sleep(0.2)

	raise TimeoutError("Takeoff timeout")


def land(master: mavutil.mavfile) -> None:
	print("Landing...")
	set_mode(master, "LAND")
	deadline = time.time() + 90
	while time.time() < deadline:
		if not master.motors_armed():
			print("Landed and disarmed")
			return
		time.sleep(0.5)

	raise TimeoutError("Landing timeout")


def detect_yellow_line(frame: np.ndarray, prev_x_error: Optional[float] = None) -> tuple[bool, float, float, np.ndarray, np.ndarray]:
	overlay = frame.copy()
	h, w, _ = frame.shape
	roi_top = int(np.clip(LINE_ROI_TOP_RATIO, 0.20, 0.75) * h)
	control_band_top = int(np.clip(LINE_CONTROL_BAND_TOP_RATIO, 0.35, 0.90) * h)
	side_margin = int(np.clip(LINE_SIDE_IGNORE_RATIO, 0.0, 0.35) * w)
	kernel = np.ones((3, 3), np.uint8)

	if CAMERA_CHANNELS == 1:
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
		_, mask = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		mask[:roi_top, :] = 0
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	else:
		hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
		lower = np.array([15, 60, 60], dtype=np.uint8)
		upper = np.array([38, 255, 255], dtype=np.uint8)
		mask = cv2.inRange(hsv, lower, upper)
		mask[:roi_top, :] = 0
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	if side_margin > 0:
		mask[:, :side_margin] = 0
		mask[:, w - side_margin :] = 0

	control_mask = mask.copy()
	control_mask[:control_band_top, :] = 0

	contours, _ = cv2.findContours(control_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return False, 0.0, 0.0, control_mask, overlay

	best_contour = None
	best_score = -1e9
	for contour in contours:
		area = cv2.contourArea(contour)
		if area < 70:
			continue
		x, y, cw, ch = cv2.boundingRect(contour)
		if x <= 1 or (x + cw) >= (w - 2):
			continue
		if ch < 5 or cw < 3:
			continue

		mom = cv2.moments(contour)
		if mom["m00"] == 0:
			continue
		cx = float(mom["m10"] / mom["m00"])
		cy = float(mom["m01"] / mom["m00"])
		x_error = (cx - (w / 2.0)) / (w / 2.0)
		y_bias = cy / float(h)
		verticality = ch / max(cw, 1)

		continuity_penalty = 0.0
		if prev_x_error is not None:
			continuity_penalty = abs(x_error - prev_x_error)

		score = (
			0.020 * area
			+ 0.60 * y_bias
			+ 0.40 * min(verticality, 4.0)
			- 1.40 * continuity_penalty
		)
		if score > best_score:
			best_score = score
			best_contour = contour

	if best_contour is None:
		return False, 0.0, 0.0, mask, overlay

	moments = cv2.moments(best_contour)
	if moments["m00"] == 0:
		return False, 0.0, 0.0, control_mask, overlay

	cx = float(moments["m10"] / moments["m00"])
	cy = float(moments["m01"] / moments["m00"])
	x_error = (cx - (w / 2.0)) / (w / 2.0)
	y_error = (cy - (h / 2.0)) / (h / 2.0)

	cv2.drawContours(overlay, [best_contour], -1, (0, 255, 255), 2)
	cv2.circle(overlay, (int(cx), int(cy)), 5, (0, 0, 255), -1)
	cv2.line(overlay, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
	cv2.rectangle(overlay, (side_margin, control_band_top), (w - side_margin, h), (255, 255, 0), 1)
	cv2.putText(overlay, f"x_err={x_error:.2f} y_err={y_error:.2f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

	return True, float(x_error), float(y_error), control_mask, overlay


def detect_apriltag(detector: Any, frame: np.ndarray) -> tuple[Optional[int], Optional[tuple[float, float]], np.ndarray]:
	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	detections = detector.detect(gray)
	overlay = frame.copy()

	if not detections:
		return None, None, overlay

	best = max(detections, key=lambda d: float(getattr(d, "decision_margin", 0.0)))
	tag_id = int(best.tag_id)
	center = (float(best.center[0]), float(best.center[1]))

	pts = np.array(best.corners, dtype=np.int32)
	cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
	cv2.circle(overlay, (int(center[0]), int(center[1])), 4, (0, 0, 255), -1)
	cv2.putText(overlay, f"tag={tag_id}", (int(center[0]) + 8, int(center[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

	return tag_id, center, overlay


def show_debug_windows(raw_frame: np.ndarray, mask: np.ndarray, line_overlay: np.ndarray, tag_overlay: np.ndarray, status_text: str) -> None:
	if not SHOW_DEBUG:
		return

	count = int(getattr(show_debug_windows, "_count", 0)) + 1
	show_debug_windows._count = count  # type: ignore[attr-defined]
	if DEBUG_EVERY_N > 1 and (count % DEBUG_EVERY_N) != 0:
		return

	raw_vis = cv2.convertScaleAbs(raw_frame, alpha=1.35, beta=18)
	combined_overlay = cv2.cvtColor(raw_vis, cv2.COLOR_RGB2BGR)
	
	# Merge line overlay onto combined
	line_bgr = cv2.cvtColor(line_overlay, cv2.COLOR_RGB2BGR)
	line_mask_alpha = cv2.cvtColor(cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
	line_mask_alpha = (line_mask_alpha > 10).astype(np.uint8)
	combined_overlay = np.where(line_mask_alpha, line_bgr, combined_overlay)
	
	# Merge tag overlay onto combined
	tag_bgr = cv2.cvtColor(tag_overlay, cv2.COLOR_RGB2BGR)
	tag_mask_alpha = cv2.cvtColor(cv2.cvtColor(tag_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
	tag_mask_alpha = (tag_mask_alpha > 10).astype(np.uint8)
	combined_overlay = np.where(tag_mask_alpha, tag_bgr, combined_overlay)
	
	mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	mask_nonzero = int(cv2.countNonZero(mask))

	cv2.putText(combined_overlay, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	cv2.putText(combined_overlay, f"cam_ch={CAMERA_CHANNELS}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
	cv2.putText(mask_bgr, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	cv2.putText(mask_bgr, f"mask_px={mask_nonzero}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

	cv2.imshow("camera_detections", combined_overlay)
	cv2.imshow("yellow_mask", mask_bgr)
	cv2.waitKey(1)


def follow_line_to_second_pad(master: mavutil.mavfile, camera: CameraStreamClient, detector: Any) -> int:
	print("Stage 2: Follow yellow line to second pad")
	deadline = time.time() + LINE_STAGE_TIMEOUT_S
	x_pid = PIDController(kp=0.28, ki=0.002, kd=0.08, output_limit=0.10, integral_limit=0.30)
	y_pid = PIDController(kp=0.32, ki=0.0, kd=0.08, output_limit=0.16, integral_limit=0.20)
	search_direction = 1.0
	last_flip = time.time()
	confirm_id: Optional[int] = None
	confirm_count = 0
	last_log = 0.0
	camera_error_since: Optional[float] = None
	camera_error_count = 0
	frame_index = 0
	last_tag_overlay: Optional[np.ndarray] = None
	last_loop_t = time.time()
	prev_line_x_error: Optional[float] = None
	prev_yaw_cmd = 0.0
	lost_line_frames = 0

	while time.time() < deadline:
		frame_index += 1
		try:
			frame = camera.read_frame()
			camera_error_since = None
			camera_error_count = 0
		except Exception as exc:
			camera_error_count += 1
			now = time.time()
			if camera_error_since is None:
				camera_error_since = now
			if camera_error_count % 3 == 0:
				print(f"Camera error during line-follow ({camera_error_count}): {exc}")

			vz = altitude_hold_vz(master, TARGET_ALTITUDE_M)
			send_body_velocity(master, 0.06, 0.0, vz, 0.0)
			time.sleep(0.12)

			if now - camera_error_since > MAX_CAMERA_RECOVERY_S:
				raise TimeoutError("Camera unavailable for too long during line-follow")
			continue

		vz = altitude_hold_vz(master, TARGET_ALTITUDE_M)

		line_found, x_error, y_error, mask, line_overlay = detect_yellow_line(frame, prev_x_error=prev_line_x_error)

		tag_id: Optional[int] = None
		tag_center: Optional[tuple[float, float]] = None
		if frame_index % max(1, TAG_DETECT_EVERY_N) == 0:
			tag_id, tag_center, last_tag_overlay = detect_apriltag(detector, frame)
		tag_overlay = last_tag_overlay if last_tag_overlay is not None else frame.copy()

		if tag_id is not None:
			if confirm_id == tag_id:
				confirm_count += 1
			else:
				confirm_id = tag_id
				confirm_count = 1
		elif frame_index % max(1, TAG_DETECT_EVERY_N) == 0:
			confirm_id = None
			confirm_count = 0

		if confirm_count >= 1 and tag_center is not None:
			h, w, _ = frame.shape
			tag_x_error = (tag_center[0] - (w / 2.0)) / (w / 2.0)
			tag_y_error = (tag_center[1] - (h / 2.0)) / (h / 2.0)
			if abs(tag_x_error) < 0.25 and abs(tag_y_error) < 0.30:
				send_stop(master)
				time.sleep(0.3)
				show_debug_windows(frame, mask, line_overlay, tag_overlay, f"TAG LOCK ID={tag_id}")
				print(f"\n{'='*60}")
				print(f"DETECTED APRILTAG ON SECOND PAD")
				print(f"AprilTag ID: {tag_id}")
				print(f"{'='*60}\n")
				return tag_id

		now = time.time()
		loop_dt = now - last_loop_t
		last_loop_t = now
		slow_loop_scale = 0.6 if loop_dt > SLOW_LOOP_DT_S else 1.0
		
		tag_speed_factor = 1.0
		if confirm_count >= 1:
			tag_speed_factor = 0.25
		
		if line_found:
			lost_line_frames = 0
			yaw_rate = x_pid.update(x_error, now) * slow_loop_scale
			lateral_vy = y_pid.update(x_error, now)
			forward = float(np.clip(0.10 - 0.04 * abs(x_error) - 0.06 * max(-y_error, 0.0), 0.04, 0.12))
			if loop_dt > SLOW_LOOP_DT_S:
				forward = float(np.clip(forward * 0.70, 0.04, 0.10))
			forward = forward * tag_speed_factor
			yaw_rate = float(np.clip(yaw_rate, -0.10, 0.10))
			yaw_rate = 0.20 * yaw_rate + 0.80 * prev_yaw_cmd
			prev_yaw_cmd = yaw_rate
			prev_line_x_error = x_error
			send_body_velocity(master, forward, lateral_vy, vz, -yaw_rate)
			status_tag = f" [TAG_SEEN! BRAKE]" if confirm_count >= 1 else ""
			status = f"LINE x={x_error:.2f} y={y_error:.2f} fwd={forward:.2f}{status_tag}"
		else:
			lost_line_frames += 1
			x_pid.reset()
			y_pid.reset()
			prev_line_x_error = None
			if now - last_flip > 2.0:
				search_direction *= -1.0
				last_flip = now
			search_yaw = 0.05 if lost_line_frames < 20 else 0.08
			search_yaw_cmd = 0.25 * (search_yaw * search_direction) + 0.75 * prev_yaw_cmd
			prev_yaw_cmd = search_yaw_cmd
			send_body_velocity(master, 0.03, 0.0, vz, search_yaw_cmd)
			status = f"SEARCH (line lost) dt={loop_dt*1000:.0f}ms"

		if now - last_log > 1.0:
			print(status + f" | vz={vz:.2f}")
			last_log = now

		show_debug_windows(frame, mask, line_overlay, tag_overlay, status)

	raise TimeoutError("Failed to reach second pad within line-follow timeout")


def center_over_tag(master: mavutil.mavfile, camera: CameraStreamClient, detector: Any, tag_id: int) -> None:
	print(f"\n{'='*60}")
	print(f"CENTERING ABOVE APRILTAG ID: {tag_id}")
	print(f"{'='*60}\n")
	deadline = time.time() + CENTER_TIMEOUT_S
	stable_since: Optional[float] = None
	x_pid = PIDController(kp=0.40, ki=0.008, kd=0.10, output_limit=0.16, integral_limit=0.30)
	y_pid = PIDController(kp=0.40, ki=0.008, kd=0.10, output_limit=0.16, integral_limit=0.30)
	camera_error_since: Optional[float] = None
	camera_error_count = 0
	prev_vx_cmd = 0.0
	prev_vy_cmd = 0.0
	last_loop_t = time.time()
	last_log = 0.0
	lost_tag_frames = 0

	while time.time() < deadline:
		try:
			frame = camera.read_frame()
			camera_error_since = None
			camera_error_count = 0
		except Exception as exc:
			camera_error_count += 1
			now = time.time()
			if camera_error_since is None:
				camera_error_since = now
			if camera_error_count % 3 == 0:
				print(f"Camera error during tag-centering ({camera_error_count}): {exc}")

			vz = altitude_hold_vz(master, TARGET_ALTITUDE_M)
			send_body_velocity(master, 0.0, 0.0, vz, 0.08)
			time.sleep(0.12)

			if now - camera_error_since > MAX_CAMERA_RECOVERY_S:
				raise TimeoutError(f"Camera unavailable for too long while centering tag {tag_id}")
			continue

		vz = altitude_hold_vz(master, TARGET_ALTITUDE_M)
		tag_found, center, tag_overlay = detect_apriltag(detector, frame)

		line_found, x_error_line, y_error_line, mask, line_overlay = detect_yellow_line(frame)

		now = time.time()
		loop_dt = now - last_loop_t
		last_loop_t = now
		slow_loop_scale = 0.55 if loop_dt > SLOW_LOOP_DT_S else 1.0

		if tag_found == tag_id and center is not None:
			lost_tag_frames = 0
			h, w, _ = frame.shape
			x_error = (center[0] - (w / 2.0)) / (w / 2.0)
			y_error = (center[1] - (h / 2.0)) / (h / 2.0)
			
			vx_raw = -y_pid.update(y_error, now) * slow_loop_scale
			vy_raw = x_pid.update(x_error, now) * slow_loop_scale
			
			vx = float(np.clip(vx_raw, -0.18, 0.18))
			vy = float(np.clip(vy_raw, -0.18, 0.18))
			
			vx = 0.35 * vx + 0.65 * prev_vx_cmd
			vy = 0.35 * vy + 0.65 * prev_vy_cmd
			prev_vx_cmd = vx
			prev_vy_cmd = vy
			
			send_body_velocity(master, vx, vy, vz, 0.0)

			status = f"CENTER tag={tag_id} x={x_error:.2f} y={y_error:.2f} dt={loop_dt*1000:.0f}ms"
			show_debug_windows(frame, mask, line_overlay, tag_overlay, status)

			if now - last_log > 0.8:
				print(f"[AprilTag {tag_id}] x_err={x_error:.3f}, y_err={y_error:.3f}, vx={vx:.3f}, vy={vy:.3f}")
				last_log = now

			if abs(x_error) < 0.08 and abs(y_error) < 0.08:
				if stable_since is None:
					stable_since = time.time()
					print(f"AprilTag {tag_id} alignment achieved, holding position...")
				elif time.time() - stable_since > 1.8:
					send_stop(master)
					print(f"\n{'='*60}")
					print(f"SUCCESSFULLY CENTERED ABOVE APRILTAG ID: {tag_id}")
					print(f"{'='*60}\n")
					return
			else:
				stable_since = None
		else:
			lost_tag_frames += 1
			stable_since = None
			x_pid.reset()
			y_pid.reset()
			
			search_yaw = 0.10 if lost_tag_frames < 15 else 0.14
			vx_decay = prev_vx_cmd * 0.85
			vy_decay = prev_vy_cmd * 0.85
			prev_vx_cmd = vx_decay
			prev_vy_cmd = vy_decay
			
			send_body_velocity(master, vx_decay, vy_decay, vz, search_yaw)
			status = f"SEARCH TAG {tag_id} (lost {lost_tag_frames} frames)"
			show_debug_windows(frame, mask, line_overlay, tag_overlay, status)
			
			if now - last_log > 1.2:
				print(f"Searching for AprilTag {tag_id}... (lost for {lost_tag_frames} frames)")
				last_log = now

	raise TimeoutError(f"Failed to center above AprilTag {tag_id}")


def main() -> None:
	if Detector is None:
		raise RuntimeError(
			"Missing dependency 'pupil_apriltags'. Install it with: pip install pupil-apriltags"
		) from APRILTAG_IMPORT_ERROR

	master = connect_vehicle()
	camera = CameraStreamClient(CAMERA_HOST, CAMERA_PORT)
	detector: Any = Detector(families="tag36h11")

	try:
		camera.connect()
		set_mode(master, "GUIDED")
		arm_vehicle(master)
		takeoff(master, TARGET_ALTITUDE_M)

		tag_id = follow_line_to_second_pad(master, camera, detector)
		center_over_tag(master, camera, detector, tag_id)

		print("Mission step complete: takeoff + first path + second pad detection")
		land(master)
	except Exception as exc:
		print(f"Mission error: {exc}")
		try:
			land(master)
		except Exception as land_error:
			print(f"Landing after error failed: {land_error}")
		raise
	finally:
		camera.close()
		master.close()
		if SHOW_DEBUG:
			cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
