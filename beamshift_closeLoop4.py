import time
import threading

import cv2
import numpy as np
import ncnn
import serial
from picamera2 import Picamera2


# ============================== SETTINGS ==============================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
YOLO_INPUT_SIZE = 320

MODEL_PATH = "/home/pizza/BeamShift/models/yolo11n_ncnn_model"
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

VEHICLE_CLASSES = {
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK",
}
YOLO_CONFIDENCE = 0.40
YOLO_NMS = 0.45

# Keep these aligned with the known-working spot_detection.py.
SPOT_ROI_TOP = 180
SPOT_ROI_BOTTOM = 430
TOPHAT_KERNEL = 21
TOPHAT_THRESHOLD = 18
SPOT_MIN_AREA = 1
SPOT_MAX_AREA = 60
SPOT_MAX_SIDE = 12
SPOT_DUPLICATE_DISTANCE = 10

SERVO_MIN = 45
SERVO_MAX = 135
SERVO_NEUTRAL = 90

# Calibration points are measured individually for every servo.
# Use a moderate spread first; extend it only if the mechanism is safe.
CALIBRATION_ANGLES = [60, 75, 90, 105, 120]
CALIBRATION_SETTLE_SECONDS = 0.45
CALIBRATION_FRAMES = 5
CALIBRATION_MATCH_RADIUS = 55

# Expand YOLO's vehicle box so the projected spot clears the car by a margin.
CAR_MARGIN_X = 24
CAR_MARGIN_Y = 20
SPOT_RADIUS_PX = 12

# When choosing a safe calibrated position, prefer the smallest servo movement.
RESET_DELAY_SECONDS = 1.2
VEHICLE_CONFIRM_FRAMES = 2
CONTROL_INTERVAL_SECONDS = 0.12
SERVO_COMMAND_DEADBAND_DEG = 2
HEARTBEAT_INTERVAL = 0.25

# Set False only if you intentionally want to keep the LEDs on during shutdown.
LIGHTS_OFF_ON_EXIT = True

# ======================================================================


class ArduinoController:
    def __init__(self):
        print("[BEAMSHIFT] Connecting to Arduino...")
        self.serial = serial.Serial(
            SERIAL_PORT, BAUD_RATE, timeout=0.05, write_timeout=0.2
        )
        time.sleep(2.0)  # Opening serial commonly resets the Uno.
        self.lock = threading.Lock()
        self.running = True
        self.last_heartbeat = 0.0
        self.angles = {ch: SERVO_NEUTRAL for ch in range(1, 6)}
        self.lights = {ch: 1 for ch in range(1, 6)}
        self.thread = threading.Thread(
            target=self._heartbeat_worker, daemon=True
        )
        self.thread.start()
        print("[BEAMSHIFT] Arduino connected.")

    def _write(self, line):
        with self.lock:
            self.serial.write((line + "\n").encode("ascii"))
            self.serial.flush()

    def _heartbeat_worker(self):
        while self.running:
            now = time.monotonic()
            if now - self.last_heartbeat >= HEARTBEAT_INTERVAL:
                try:
                    self._write("HB")
                    self.last_heartbeat = now
                except serial.SerialException:
                    pass
            time.sleep(0.025)

    def set_channel(self, channel, angle=None, light=None):
        if channel not in range(1, 6):
            raise ValueError("Arduino channel must be 1..5")
        if angle is not None:
            self.angles[channel] = int(np.clip(angle, SERVO_MIN, SERVO_MAX))
        if light is not None:
            self.lights[channel] = int(bool(light))
        self._write(
            f"Z,{channel},{self.angles[channel]},{self.lights[channel]}"
        )

    def set_all_angles(self, angle):
        for ch in range(1, 6):
            self.set_channel(ch, angle=angle)

    def set_all_lights(self, state):
        for ch in range(1, 6):
            self.set_channel(ch, light=state)

    def close(self):
        try:
            self.set_all_angles(SERVO_NEUTRAL)
            if LIGHTS_OFF_ON_EXIT:
                self.set_all_lights(0)
            time.sleep(0.35)
        except Exception:
            pass
        self.running = False
        self.thread.join(timeout=1.0)
        try:
            self.serial.close()
        except Exception:
            pass


def load_yolo_model():
    print("[BEAMSHIFT] Loading YOLO11n NCNN...")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    p = net.load_param(MODEL_PATH + "/model.ncnn.param")
    m = net.load_model(MODEL_PATH + "/model.ncnn.bin")
    if p != 0 or m != 0:
        raise RuntimeError("Could not load NCNN model from: " + MODEL_PATH)
    print("[BEAMSHIFT] YOLO model loaded.")
    return net


def yolo_preprocess(frame):
    h, w = frame.shape[:2]
    scale = min(YOLO_INPUT_SIZE / w, YOLO_INPUT_SIZE / h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh))
    pad_x = (YOLO_INPUT_SIZE - nw) // 2
    pad_y = (YOLO_INPUT_SIZE - nh) // 2
    canvas = np.full(
        (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, dtype=np.uint8
    )
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(
        rgb.transpose(2, 0, 1), dtype=np.float32
    )
    tensor /= 255.0
    return tensor, scale, pad_x, pad_y


def detect_vehicles(net, frame):
    tensor, scale, pad_x, pad_y = yolo_preprocess(frame)
    extractor = net.create_extractor()
    extractor.input("in0", ncnn.Mat(tensor))
    ret, output = extractor.extract("out0")
    if ret != 0:
        return []

    pred = np.asarray(output.numpy(), dtype=np.float32).squeeze()
    if pred.ndim != 2:
        return []
    if pred.shape[0] == 84:
        pred = pred.T
    elif pred.shape[1] != 84:
        return []

    boxes, scores, class_ids = [], [], []
    for row in pred:
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if class_id not in VEHICLE_CLASSES or confidence < YOLO_CONFIDENCE:
            continue

        cx, cy, bw, bh = row[:4]
        x1 = int(np.clip(
            (cx - bw / 2 - pad_x) / scale, 0, CAMERA_WIDTH - 1
        ))
        y1 = int(np.clip(
            (cy - bh / 2 - pad_y) / scale, 0, CAMERA_HEIGHT - 1
        ))
        x2 = int(np.clip(
            (cx + bw / 2 - pad_x) / scale, 0, CAMERA_WIDTH - 1
        ))
        y2 = int(np.clip(
            (cy + bh / 2 - pad_y) / scale, 0, CAMERA_HEIGHT - 1
        ))
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(confidence)
        class_ids.append(class_id)

    detections = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, YOLO_CONFIDENCE, YOLO_NMS
        )
        if len(indices):
            for idx in np.asarray(indices).flatten():
                x, y, w, h = boxes[int(idx)]
                detections.append({
                    "box": (x, y, w, h),
                    "class": VEHICLE_CLASSES[class_ids[int(idx)]],
                    "confidence": scores[int(idx)],
                })
    return detections


def detect_spot_candidates(frame):
    """Return bright compact candidates; coordinates are full-frame pixels."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    top = max(0, SPOT_ROI_TOP)
    bottom = min(frame.shape[0], SPOT_ROI_BOTTOM)
    roi = gray[top:bottom, :]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL, TOPHAT_KERNEL)
    )
    top_hat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(
        top_hat, TOPHAT_THRESHOLD, 255, cv2.THRESH_BINARY
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    candidates = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if not (SPOT_MIN_AREA <= area <= SPOT_MAX_AREA):
            continue
        if w > SPOT_MAX_SIDE or h > SPOT_MAX_SIDE:
            continue
        cx, cy = centroids[i]
        score = float(top_hat[labels == i].sum())
        candidates.append({
            "x": float(cx),
            "y": float(cy + top),
            "score": score,
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    for item in candidates:
        if all(
            np.hypot(item["x"] - other["x"], item["y"] - other["y"])
            >= SPOT_DUPLICATE_DISTANCE
            for other in selected
        ):
            selected.append(item)
        if len(selected) >= 12:
            break
    return selected, gray, mask


def average_spot_detections(camera, target_frames=CALIBRATION_FRAMES):
    """
    Average several frames. Used only during startup calibration while the
    physical screen must be clear of the car image.
    """
    all_frames = []
    last_gray = None
    last_mask = None

    for _ in range(target_frames):
        frame = camera.capture_array()
        candidates, gray, mask = detect_spot_candidates(frame)
        if len(candidates) >= 5:
            all_frames.append(candidates)
        last_gray, last_mask = gray, mask
        time.sleep(0.035)

    if not all_frames:
        return [], last_gray, last_mask

    # Use the strongest five from the final valid frame. The camera is static
    # during calibration, and each servo is moved separately.
    return all_frames[-1][:], last_gray, last_mask


def nearest_candidate_to(point, candidates, excluded_indices=None):
    excluded_indices = excluded_indices or set()
    best_idx, best_dist = None, float("inf")
    for idx, candidate in enumerate(candidates):
        if idx in excluded_indices:
            continue
        dist = np.hypot(
            candidate["x"] - point[0],
            candidate["y"] - point[1]
        )
        if dist < best_dist:
            best_idx, best_dist = idx, dist
    if best_idx is None:
        return None, float("inf")
    return candidates[best_idx], float(best_dist)


def robust_candidate_average(samples):
    if not samples:
        return None
    xs = [p[0] for p in samples]
    ys = [p[1] for p in samples]
    return (float(np.median(xs)), float(np.median(ys)))


def calibrate_all_servos(camera, arduino):
    """
    Builds RAM-only map:
      calibration[spot_id][angle] = (predicted_x, predicted_y)

    At each test angle only one servo moves. The other four spots are matched
    near their neutral anchors and excluded; the remaining candidate is treated
    as the moving spot. If that association is ambiguous, calibration aborts
    rather than silently assigning a wrong point.
    """
    print("\n" + "=" * 58)
    print("BEAMSHIFT STARTUP CALIBRATION")
    print("Remove the car image/object from the screen.")
    print("Keep the camera fixed and ensure all five diffused LEDs are visible.")
    print("Calibration moves one servo at a time.")
    print("=" * 58)
    input("When the screen is clear, press ENTER to begin calibration...")

    arduino.set_all_angles(SERVO_NEUTRAL)
    arduino.set_all_lights(1)
    time.sleep(0.6)

    baseline_candidates, _, _ = average_spot_detections(camera)
    if len(baseline_candidates) < 5:
        raise RuntimeError(
            "Could not detect five spots at neutral. Check lights, focus, "
            "ROI and TOPHAT_THRESHOLD, then restart."
        )

    baseline = sorted(baseline_candidates, key=lambda d: d["x"])[:5]
    anchors = {
        sid: (baseline[sid - 1]["x"], baseline[sid - 1]["y"])
        for sid in range(1, 6)
    }

    # Each spot's own servo channel is reversed in the existing hardware map.
    calibration = {
        sid: {SERVO_NEUTRAL: anchors[sid]}
        for sid in range(1, 6)
    }

    print("\n[CAL] Neutral anchors (left-to-right):")
    for sid in range(1, 6):
        print(
            f"  S{sid} -> CH{6-sid}: "
            f"({anchors[sid][0]:.1f}, {anchors[sid][1]:.1f})"
        )

    for sid in range(1, 6):
        channel = 6 - sid
        print(f"\n[CAL] Calibrating S{sid} / Arduino CH{channel}")

        # Ensure all other channels are neutral.
        for other_sid in range(1, 6):
            arduino.set_channel(6 - other_sid, angle=SERVO_NEUTRAL, light=1)
        time.sleep(0.25)

        for angle in CALIBRATION_ANGLES:
            arduino.set_channel(channel, angle=angle, light=1)
            time.sleep(CALIBRATION_SETTLE_SECONDS)

            candidates, _, _ = average_spot_detections(camera)
            if len(candidates) < 5:
                print(
                    f"[CAL] WARNING: only {len(candidates)} candidates at "
                    f"S{sid}, {angle} degrees; skipping this sample."
                )
                continue

            # Reserve detections that still match the four stationary spots.
            reserved = set()
            stationary_ok = True
            for other_sid in range(1, 6):
                if other_sid == sid:
                    continue
                candidate, distance = nearest_candidate_to(
                    anchors[other_sid], candidates, reserved
                )
                if candidate is None or distance > CALIBRATION_MATCH_RADIUS:
                    stationary_ok = False
                    break
                reserved.add(candidates.index(candidate))

            if not stationary_ok:
                print(
                    f"[CAL] WARNING: stationary spots ambiguous at "
                    f"{angle} degrees; skipping sample."
                )
                continue

            moving_candidates = [
                c for idx, c in enumerate(candidates) if idx not in reserved
            ]
            if not moving_candidates:
                print(
                    f"[CAL] WARNING: moving spot not isolated at {angle}; "
                    "skipping sample."
                )
                continue

            # The moved LED should be the remaining bright compact candidate.
            # Choose the strongest remaining one, then require it to be unique
            # enough relative to the target's neutral anchor.
            moving_candidates.sort(key=lambda d: d["score"], reverse=True)
            moved = moving_candidates[0]
            distance_from_anchor = np.hypot(
                moved["x"] - anchors[sid][0],
                moved["y"] - anchors[sid][1]
            )
            if len(moving_candidates) > 1:
                second = moving_candidates[1]
                score_gap = moved["score"] / max(second["score"], 1.0)
                if score_gap < 1.15 and distance_from_anchor > 8:
                    print(
                        f"[CAL] WARNING: ambiguous moving spot at {angle}; "
                        "skipping sample."
                    )
                    continue

            calibration[sid][angle] = (moved["x"], moved["y"])
            print(
                f"  {angle:3d} deg -> ({moved['x']:.1f}, {moved['y']:.1f})"
            )

        # Return this channel to neutral before moving to the next.
        arduino.set_channel(channel, angle=SERVO_NEUTRAL, light=1)
        time.sleep(0.35)

    arduino.set_all_angles(SERVO_NEUTRAL)

    # Each servo must have at least one measured non-neutral point.
    for sid in range(1, 6):
        non_neutral = [
            a for a in calibration[sid] if a != SERVO_NEUTRAL
        ]
        if not non_neutral:
            raise RuntimeError(
                f"Calibration failed for S{sid}/CH{6-sid}: no off-neutral "
                "position was measured. Check the spot detector and servo."
            )

    print("\n[CAL] Calibration complete. All map data is held in RAM.")
    return calibration


def interpolate_position(calibration_for_spot, angle):
    """Interpolate a predicted spot coordinate between measured servo angles."""
    points = sorted(
        (int(a), p) for a, p in calibration_for_spot.items()
    )
    if not points:
        return None

    angle = float(angle)
    if angle <= points[0][0]:
        return points[0][1]
    if angle >= points[-1][0]:
        return points[-1][1]

    for (a0, p0), (a1, p1) in zip(points, points[1:]):
        if a0 <= angle <= a1:
            t = (angle - a0) / float(a1 - a0)
            return (
                p0[0] + t * (p1[0] - p0[0]),
                p0[1] + t * (p1[1] - p0[1]),
            )
    return points[-1][1]


def expand_vehicle_boxes(detections):
    boxes = []
    for det in detections:
        x, y, w, h = det["box"]
        boxes.append((
            max(0, x - CAR_MARGIN_X),
            max(0, y - CAR_MARGIN_Y),
            min(CAMERA_WIDTH - 1, x + w + CAR_MARGIN_X),
            min(CAMERA_HEIGHT - 1, y + h + CAR_MARGIN_Y),
        ))
    return boxes


def point_inside_box(point, box, radius=SPOT_RADIUS_PX):
    x, y = point
    x1, y1, x2, y2 = box
    # Treat the LED spot as a small disk, not a dimensionless point.
    return (
        x + radius >= x1 and x - radius <= x2
        and y + radius >= y1 and y - radius <= y2
    )


def choose_safe_angle(calibration_for_spot, current_angle, vehicle_boxes):
    """
    Return the closest calibrated angle whose predicted spot footprint does
    not intersect any expanded vehicle box. None means no measured safe angle.
    """
    candidates = []
    for angle in calibration_for_spot:
        predicted = interpolate_position(calibration_for_spot, angle)
        if predicted is None:
            continue
        blocked = any(
            point_inside_box(predicted, box) for box in vehicle_boxes
        )
        if not blocked:
            candidates.append((abs(int(angle) - current_angle), int(angle)))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def draw_overlay(frame, detections, calibration, current_angles,
                 vehicle_boxes, status, fps, yolo_ms):
    view = frame.copy()

    for x1, y1, x2, y2 in vehicle_boxes:
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 165, 255), 2)

    for det in detections:
        x, y, w, h = det["box"]
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            view, f'{det["class"]} {det["confidence"]:.2f}',
            (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 255, 0), 1, cv2.LINE_AA
        )

    # Draw predicted positions from the calibration map, even if the real spot
    # is visually hidden under the vehicle image.
    for sid in range(1, 6):
        angle = current_angles[sid]
        p = interpolate_position(calibration[sid], angle)
        if p is None:
            continue
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(view, (x, y), SPOT_RADIUS_PX, (255, 255, 0), 2)
        cv2.putText(
            view, f"S{sid}/CH{6-sid} {angle}deg",
            (min(CAMERA_WIDTH - 150, x + 10), max(16, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 0), 1, cv2.LINE_AA
        )

    cv2.putText(
        view, f"FPS {fps:.1f} | YOLO {yolo_ms:.0f}ms | {status}",
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    cv2.putText(
        view, "Q quit | R reset to 90 | L lights ON | K lights OFF",
        (8, CAMERA_HEIGHT - 9), cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (255, 255, 255), 1, cv2.LINE_AA
    )
    return view


def main():
    arduino = None
    camera = None

    try:
        arduino = ArduinoController()
        net = load_yolo_model()

        camera = Picamera2()
        camera.configure(camera.create_video_configuration(
            main={
                "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                "format": "BGR888",
            },
            buffer_count=4,
        ))
        camera.options["queue"] = False
        camera.start()
        time.sleep(1.0)

        # Calibration is performed once at startup and kept in memory.
        calibration = calibrate_all_servos(camera, arduino)
        current_angles = {sid: SERVO_NEUTRAL for sid in range(1, 6)}

        # Confirm calibration visually before entering live mode.
        print("\n[BEAMSHIFT] Live mode starting. Press Q in the camera window to quit.")
        arduino.set_all_angles(SERVO_NEUTRAL)
        arduino.set_all_lights(1)

        last_vehicle_time = 0.0
        vehicle_streak = 0
        vehicle_active = False
        last_control_time = 0.0

        fps = 0.0
        fps_count = 0
        fps_start = time.monotonic()
        yolo_ms = 0.0
        status = "WAITING FOR VEHICLE"

        while True:
            frame = camera.capture_array()

            t0 = time.perf_counter()
            detections = detect_vehicles(net, frame)
            yolo_ms = (time.perf_counter() - t0) * 1000.0

            now = time.monotonic()
            if detections:
                vehicle_streak += 1
                last_vehicle_time = now
            else:
                vehicle_streak = 0

            if not vehicle_active and vehicle_streak >= VEHICLE_CONFIRM_FRAMES:
                vehicle_active = True
                status = "VEHICLE DETECTED"

            if vehicle_active and now - last_vehicle_time >= RESET_DELAY_SECONDS:
                vehicle_active = False
                status = "RESET TO NEUTRAL"
                for sid in range(1, 6):
                    ch = 6 - sid
                    arduino.set_channel(ch, angle=SERVO_NEUTRAL, light=1)
                    current_angles[sid] = SERVO_NEUTRAL
                time.sleep(0.15)

            boxes = expand_vehicle_boxes(detections)

            if vehicle_active and boxes and (
                now - last_control_time >= CONTROL_INTERVAL_SECONDS
            ):
                last_control_time = now
                for sid in range(1, 6):
                    ch = 6 - sid
                    target = choose_safe_angle(
                        calibration[sid], current_angles[sid], boxes
                    )

                    if target is None:
                        # No calibrated safe position for this spot. Hold the
                        # current position rather than guessing an angle.
                        continue

                    if abs(target - current_angles[sid]) >= SERVO_COMMAND_DEADBAND_DEG:
                        arduino.set_channel(ch, angle=target, light=1)
                        current_angles[sid] = target

                status = "ZONE AVOIDANCE ACTIVE"

            elif not vehicle_active:
                status = "WAITING FOR VEHICLE"

            fps_count += 1
            elapsed = now - fps_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count = 0
                fps_start = now

            display = draw_overlay(
                frame, detections, calibration, current_angles,
                boxes, status, fps, yolo_ms
            )
            cv2.imshow("BeamShift Autonomous - Self Calibrating", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("r"), ord("R")):
                for sid in range(1, 6):
                    arduino.set_channel(6 - sid, angle=SERVO_NEUTRAL, light=1)
                    current_angles[sid] = SERVO_NEUTRAL
                status = "MANUAL RESET"
            elif key in (ord("l"), ord("L")):
                arduino.set_all_lights(1)
            elif key in (ord("k"), ord("K")):
                arduino.set_all_lights(0)

    except KeyboardInterrupt:
        print("\n[BEAMSHIFT] Interrupted.")
    finally:
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()
        if arduino is not None:
            arduino.close()
        print("[BEAMSHIFT] Shutdown complete.")


if __name__ == "__main__":
    main()
