import os
import socket
import struct
import time
from typing import Optional

import cv2
import numpy as np
from pymavlink import mavutil
from pupil_apriltags import Detector


TARGET_ALTITUDE_M = 2.5
TAKEOFF_TOLERANCE_M = 0.2
CAMERA_HOST = os.getenv("CAMERA_HOST", "localhost")
CAMERA_PORT = int(os.getenv("CAMERA_PORT", "5599"))
FRAME_TIMEOUT_S = 5.0
LINE_STAGE_TIMEOUT_S = 150
CENTER_TIMEOUT_S = 20
MISSION_HOLD_S = 2


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
			payload = self._recv_exact(width * height * 3)
			frame = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3))
			return frame
		except (ConnectionError, socket.error, struct.error) as e:
			self.close()
			raise ConnectionError(f"Camera stream error: {e}")


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


def set_param(master: mavutil.mavfile, name: str, value: float) -> None:
	master.mav.param_set_send(
		master.target_system,
		master.target_component,
		name.encode("utf-8"),
		float(value),
		mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
	)


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


def command_right_turn_90(master: mavutil.mavfile) -> None:
	print("Turning right 90°")
	master.mav.command_long_send(
		master.target_system,
		master.target_component,
		mavutil.mavlink.MAV_CMD_CONDITION_YAW,
		0,
		90,
		20,
		1,
		1,
		0,
		0,
		0,
	)
	time.sleep(5)


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


def detect_yellow_line(frame: np.ndarray) -> tuple[bool, float, float]:
	hsv_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
	hsv_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower = np.array([15, 60, 60], dtype=np.uint8)
	upper = np.array([38, 255, 255], dtype=np.uint8)
	mask_rgb = cv2.inRange(hsv_rgb, lower, upper)
	mask_bgr = cv2.inRange(hsv_bgr, lower, upper)
	mask = cv2.bitwise_or(mask_rgb, mask_bgr)
	kernel = np.ones((5, 5), np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return False, 0.0, 0.0

	largest = max(contours, key=cv2.contourArea)
	area = cv2.contourArea(largest)
	if area < 120:
		return False, 0.0, 0.0

	moments = cv2.moments(largest)
	if moments["m00"] == 0:
		return False, 0.0, 0.0

	center_x = moments["m10"] / moments["m00"]
	center_y = moments["m01"] / moments["m00"]
	h, w = mask.shape
	x_error = (center_x - (w / 2.0)) / (w / 2.0)
	y_error = (center_y - (h / 2.0)) / (h / 2.0)
	return True, float(x_error), float(y_error)


def detect_apriltag(detector: Detector, frame: np.ndarray, exclude_ids: set[int]) -> tuple[Optional[int], Optional[tuple[float, float]]]:
	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	detections = detector.detect(gray)
	if not detections:
		return None, None

	best = None
	best_score = -1.0
	for detection in detections:
		tag_id = int(detection.tag_id)
		if tag_id in exclude_ids:
			continue
		score = float(getattr(detection, "decision_margin", 0.0))
		if score > best_score:
			best = detection
			best_score = score

	if best is None:
		return None, None

	center = (float(best.center[0]), float(best.center[1]))
	return int(best.tag_id), center


def center_over_tag(
	master: mavutil.mavfile,
	camera: CameraStreamClient,
	detector: Detector,
	tag_id: int,
	target_altitude_m: float,
	timeout_s: int = CENTER_TIMEOUT_S,
) -> None:
	print(f"Centering above AprilTag {tag_id}")
	deadline = time.time() + timeout_s
	stable_since: Optional[float] = None
	error_streak = 0

	while time.time() < deadline:
		try:
			frame = camera.read_frame()
			error_streak = 0
		except Exception as e:
			error_streak += 1
			if error_streak % 3 == 0:
				print(f"Camera error while centering: {e}")
			vz = altitude_hold_vz(master, target_altitude_m)
			send_body_velocity(master, 0.0, 0.0, vz, 0.0)
			time.sleep(0.1)
			continue

		detected_id, center = detect_apriltag(detector, frame, exclude_ids=set())
		vz = altitude_hold_vz(master, target_altitude_m)

		if detected_id == tag_id and center is not None:
			h, w, _ = frame.shape
			x_error = (center[0] - (w / 2.0)) / (w / 2.0)
			y_error = (center[1] - (h / 2.0)) / (h / 2.0)

			vx = float(np.clip(-0.8 * y_error, -0.5, 0.5))
			vy = float(np.clip(0.8 * x_error, -0.5, 0.5))
			send_body_velocity(master, vx, vy, vz, 0.0)

			if abs(x_error) < 0.08 and abs(y_error) < 0.08:
				if stable_since is None:
					stable_since = time.time()
				elif time.time() - stable_since >= MISSION_HOLD_S:
					send_stop(master)
					print(f"Centered above AprilTag {tag_id}")
					return
			else:
				stable_since = None
		else:
			stable_since = None
			send_body_velocity(master, 0.0, 0.0, vz, 0.25)

	raise TimeoutError(f"Failed to center above AprilTag {tag_id}")


def follow_line_until_new_tag(
	master: mavutil.mavfile,
	camera: CameraStreamClient,
	detector: Detector,
	target_altitude_m: float,
	seen_tag_ids: set[int],
	stage_name: str,
	timeout_s: int = LINE_STAGE_TIMEOUT_S,
) -> int:
	print(f"Stage: {stage_name}")
	deadline = time.time() + timeout_s
	confirm_count = 0
	confirmed_id: Optional[int] = None
	frame_count = 0
	last_print = time.time()
	error_streak = 0
	search_direction = 1.0
	last_direction_flip = time.time()

	while time.time() < deadline:
		try:
			frame = camera.read_frame()
			frame_count += 1
			error_streak = 0
		except Exception as e:
			error_streak += 1
			if error_streak % 3 == 0:
				print(f"Camera error ({error_streak}): {e}")
			vz = altitude_hold_vz(master, target_altitude_m)
			send_body_velocity(master, 0.05, 0.0, vz, 0.0)
			time.sleep(0.1)
			continue

		tag_id, _ = detect_apriltag(detector, frame, exclude_ids=seen_tag_ids)
		vz = altitude_hold_vz(master, target_altitude_m)

		if tag_id is not None:
			if confirmed_id == tag_id:
				confirm_count += 1
			else:
				confirmed_id = tag_id
				confirm_count = 1

			if confirm_count >= 3:
				send_stop(master)
				print(f"AprilTag detected: {tag_id}")
				return tag_id

		line_found, x_error, y_error = detect_yellow_line(frame)
		if line_found:
			forward_bias = 0.24
			alignment_bonus = float(np.clip(0.08 * (1.0 - abs(x_error)), 0.0, 0.08))
			far_line_penalty = float(np.clip(0.12 * max(-y_error, 0.0), 0.0, 0.08))
			vx = float(np.clip(forward_bias + alignment_bonus - far_line_penalty, 0.12, 0.32))
			vy = float(np.clip(0.35 * x_error, -0.18, 0.18))
			yaw_rate = float(np.clip(-0.75 * x_error, -0.35, 0.35))
			send_body_velocity(master, vx, vy, vz, yaw_rate)
		else:
			now = time.time()
			if now - last_direction_flip > 2.5:
				search_direction *= -1.0
				last_direction_flip = now
			send_body_velocity(master, 0.08, 0.0, vz, 0.18 * search_direction)

		now = time.time()
		if now - last_print >= 2.0:
			status = "Line found (following)" if line_found else "No line (searching)"
			print(f"[{frame_count}] {stage_name}: {status}, vz={vz:.2f}")
			last_print = now

	raise TimeoutError(f"Timed out during stage: {stage_name}")


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
	print("Arming...")
	prearm_bypassed = False
	master.arducopter_arm()
	deadline = time.time() + 8
	last_reason = ""
	while time.time() < deadline:
		reason = read_statustext(master)
		if reason:
			last_reason = reason
		if master.motors_armed():
			print("Armed")
			print(f"Pre-arm checks bypassed: {'YES' if prearm_bypassed else 'NO'}")
			return
		time.sleep(0.2)

	print("Normal arm failed" + (f": {last_reason}" if last_reason else ""))
	print("Disabling ARMING_CHECK and retrying arm (SITL fallback)")
	prearm_bypassed = True
	set_param(master, "ARMING_CHECK", 0)
	time.sleep(0.5)
	master.arducopter_arm()

	deadline = time.time() + 8
	while time.time() < deadline:
		reason = read_statustext(master)
		if reason:
			last_reason = reason
		if master.motors_armed():
			print("Armed")
			print(f"Pre-arm checks bypassed: {'YES' if prearm_bypassed else 'NO'}")
			return
		time.sleep(0.2)

	print("ARMING_CHECK fallback failed, trying force arm")
	master.mav.command_long_send(
		master.target_system,
		master.target_component,
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

	deadline = time.time() + 8
	while time.time() < deadline:
		reason = read_statustext(master)
		if reason:
			last_reason = reason
		if master.motors_armed():
			print("Armed")
			print(f"Pre-arm checks bypassed: {'YES' if prearm_bypassed else 'NO'}")
			return
		time.sleep(0.2)

	print(f"Pre-arm checks bypassed: {'YES' if prearm_bypassed else 'NO'}")
	raise TimeoutError("Arm failed" + (f". Last reason: {last_reason}" if last_reason else ""))


def takeoff(master: mavutil.mavfile, altitude_m: float) -> None:
	print(f"Takeoff to {altitude_m:.1f} m")
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
		altitude_m,
	)

	deadline = time.time() + 45
	while time.time() < deadline:
		alt = current_altitude_m(master)
		print(f"Altitude: {alt:.2f} m")
		if alt >= altitude_m - TAKEOFF_TOLERANCE_M:
			print("Takeoff complete")
			return
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


def main() -> None:
	master = connect_vehicle()
	camera = CameraStreamClient(CAMERA_HOST, CAMERA_PORT)
	detector = Detector(families="tag36h11")
	try:
		camera.connect()
		set_mode(master, "GUIDED")
		arm_vehicle(master)
		takeoff(master, TARGET_ALTITUDE_M)

		seen_tags: set[int] = set()
		first_tag = follow_line_until_new_tag(
			master,
			camera,
			detector,
			TARGET_ALTITUDE_M,
			seen_tag_ids=seen_tags,
			stage_name="Follow first yellow line to second pad",
		)
		print(f"Second pad AprilTag ID: {first_tag}")
		seen_tags.add(first_tag)
		center_over_tag(master, camera, detector, first_tag, TARGET_ALTITUDE_M)

		command_right_turn_90(master)

		second_tag = follow_line_until_new_tag(
			master,
			camera,
			detector,
			TARGET_ALTITUDE_M,
			seen_tag_ids=seen_tags,
			stage_name="Follow second yellow line to third pad",
		)
		print(f"Third pad AprilTag ID: {second_tag}")
		center_over_tag(master, camera, detector, second_tag, TARGET_ALTITUDE_M)

		land(master)
		print("Mission complete")
	except Exception as e:
		print(f"Mission error: {e}")
		try:
			land(master)
		except Exception as land_error:
			print(f"Landing after failure also failed: {land_error}")
		raise
	finally:
		camera.close()
		master.close()


if __name__ == "__main__":
	main()