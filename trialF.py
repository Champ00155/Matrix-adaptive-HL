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
# Lower threshold helps small, filtered images on the laptop display.
YOLO_CONFIDENCE = 0.18
YOLO_NMS = 0.45

# Gentle software colour correction (BGR order).
# The previous correction was too strong and made skin/clothing orange.
# These near-neutral gains avoid aggressively recolouring the whole frame.
BLUE_GAIN = 0.90
GREEN_GAIN = 1.00
RED_GAIN = 1.10

# In addition to the full frame, inspect overlapping crops at higher effective
# resolution. This helps small motorcycles/trucks shown on the laptop screen.
USE_TILED_INFERENCE = True
TILE_WIDTH = 400
TILE_HEIGHT = 300
TILE_X_STARTS = (0, 240)
TILE_Y_STARTS = (0, 180)

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

# One-time neutral-position spot detection. No servo sweep.
CALIBRATION_FRAMES = 5

# Each stationary spot defines a rectangular trigger zone around it.
# Increase these if the car box passes near a spot without triggering it.
ZONE_HALF_WIDTH = 70
ZONE_HALF_HEIGHT = 65

# Servo action for a triggered zone.
AVOID_ANGLE = 135

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


def correct_camera_colour(frame):
    """Apply a deliberately mild BGR correction; preserve natural colours."""
    gains = np.array([BLUE_GAIN, GREEN_GAIN, RED_GAIN], dtype=np.float32)
    corrected = frame.astype(np.float32) * gains.reshape(1, 1, 3)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def yolo_preprocess(frame):
    h, w = frame.shape[:2]
    scale = min(YOLO_INPUT_SIZE / w, YOLO_INPUT_SIZE / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
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


def infer_yolo_region(net, region, offset_x=0, offset_y=0):
    """Run the NCNN model on one image region and return full-frame boxes."""
    tensor, scale, pad_x, pad_y = yolo_preprocess(region)
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

    h, w = region.shape[:2]
    results = []
    for row in pred:
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if class_id not in VEHICLE_CLASSES or confidence < YOLO_CONFIDENCE:
            continue

        cx, cy, bw, bh = row[:4]
        x1 = int(np.clip((cx - bw / 2 - pad_x) / scale, 0, w - 1))
        y1 = int(np.clip((cy - bh / 2 - pad_y) / scale, 0, h - 1))
        x2 = int(np.clip((cx + bw / 2 - pad_x) / scale, 0, w - 1))
        y2 = int(np.clip((cy + bh / 2 - pad_y) / scale, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        results.append({
            "box": (x1 + offset_x, y1 + offset_y,
                    x2 - x1, y2 - y1),
            "class": VEHICLE_CLASSES[class_id],
            "confidence": confidence,
        })
    return results


def detect_vehicles(net, frame):
    """Full-frame plus overlapping higher-resolution tile inference."""
    detections = infer_yolo_region(net, frame)

    if USE_TILED_INFERENCE:
        h, w = frame.shape[:2]
        for y0 in TILE_Y_STARTS:
            for x0 in TILE_X_STARTS:
                x1 = min(x0 + TILE_WIDTH, w)
                y1 = min(y0 + TILE_HEIGHT, h)
                if x1 - x0 < 160 or y1 - y0 < 160:
                    continue
                tile = frame[y0:y1, x0:x1]
                detections.extend(infer_yolo_region(net, tile, x0, y0))

    if not detections:
        return []

    # Suppress duplicate boxes from overlapping tiles. Keep the highest score.
    boxes = []
    scores = []
    for det in detections:
        x, y, bw, bh = det["box"]
        boxes.append([int(x), int(y), int(bw), int(bh)])
        scores.append(float(det["confidence"]))

    kept = []
    indices = cv2.dnn.NMSBoxes(boxes, scores, YOLO_CONFIDENCE, YOLO_NMS)
    if len(indices):
        for idx in np.asarray(indices).flatten():
            kept.append(detections[int(idx)])
    return kept


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
    Detect the five spots once with every servo at neutral.
    Return anchors[spot_id] = (x, y). No servo movement during calibration.
    """
    print("\\n" + "=" * 58)
    print("BEAMSHIFT ZONE SETUP")
    print("Remove the car image/object from the screen.")
    print("Keep the camera fixed and all five diffused LEDs visible.")
    print("All servos will remain at 90 degrees during setup.")
    print("=" * 58)
    input("When the screen is clear, press ENTER...")

    arduino.set_all_angles(SERVO_NEUTRAL)
    arduino.set_all_lights(1)
    time.sleep(0.5)

    candidates, _, _ = average_spot_detections(camera)
    if len(candidates) < 5:
        raise RuntimeError(
            f"Only {len(candidates)} light spots detected at neutral "
            f"(need 5). Check camera view, focus, ROI and spot threshold."
        )

    # Pick five strongest candidates, then label them left-to-right.
    spots = sorted(candidates, key=lambda d: d["score"], reverse=True)[:5]
    spots.sort(key=lambda d: d["x"])

    anchors = {
        sid: (float(spots[sid - 1]["x"]), float(spots[sid - 1]["y"]))
        for sid in range(1, 6)
    }

    print("\\n[ZONE] Fixed zones created from neutral spot positions:")
    for sid in range(1, 6):
        x, y = anchors[sid]
        print(
            f"  S{sid} -> CH{6-sid}: centre=({x:.1f}, {y:.1f}), "
            f"zone=+/-{ZONE_HALF_WIDTH}px X, +/-{ZONE_HALF_HEIGHT}px Y"
        )

    print("[ZONE] Setup complete. No servo sweep was performed.")
    return anchors


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


def make_zone(point):
    x, y = point
    return (
        max(0, int(round(x - ZONE_HALF_WIDTH))),
        max(0, int(round(y - ZONE_HALF_HEIGHT))),
        min(CAMERA_WIDTH - 1, int(round(x + ZONE_HALF_WIDTH))),
        min(CAMERA_HEIGHT - 1, int(round(y + ZONE_HALF_HEIGHT))),
    )


def boxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1


def triggered_spots(anchors, vehicle_boxes):
    """Return spot IDs whose fixed zones overlap any detected vehicle box."""
    active = set()
    for sid, point in anchors.items():
        zone = make_zone(point)
        if any(boxes_overlap(zone, box) for box in vehicle_boxes):
            active.add(sid)
    return active


def draw_overlay(frame, detections, anchors, active_spots,
                 vehicle_boxes, status, fps, yolo_ms):
    view = frame.copy()

    # Fixed zone outlines and spot centres.
    for sid in range(1, 6):
        x, y = anchors[sid]
        x1, y1, x2, y2 = make_zone((x, y))
        zone_color = (0, 0, 255) if sid in active_spots else (255, 255, 0)
        cv2.rectangle(view, (x1, y1), (x2, y2), zone_color, 2)
        cv2.circle(view, (int(x), int(y)), 5, zone_color, -1)
        cv2.putText(
            view, f"S{sid}/CH{6-sid}",
            (x1 + 3, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, zone_color, 1, cv2.LINE_AA
        )

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

        # Keep camera exposure and white balance automatic. Apply only a mild
        # software correction so the preview is not pushed orange or red.
        try:
            camera.set_controls({"AeEnable": True, "AwbEnable": True})
        except Exception as control_error:
            print(f"[CAMERA] Camera auto controls unavailable: {control_error}")
        time.sleep(2.0)
        print(
            f"[CAMERA] Software colour gains BGR="
            f"({BLUE_GAIN:.2f}, {GREEN_GAIN:.2f}, {RED_GAIN:.2f})"
        )
        print("[YOLO] Full-frame + overlapping tile inference enabled; "
              f"confidence={YOLO_CONFIDENCE:.2f}")

        # Detect stationary spots once; all calibration data stays in RAM.
        anchors = calibrate_all_servos(camera, arduino)

        current_angles = {sid: SERVO_NEUTRAL for sid in range(1, 6)}
        active_spots = set()
        last_vehicle_time = 0.0
        vehicle_streak = 0
        vehicle_active = False

        print("\\n[BEAMSHIFT] Live zone mode. Press Q in the camera window to quit.")

        fps = 0.0
        fps_count = 0
        fps_start = time.monotonic()
        yolo_ms = 0.0
        status = "WAITING FOR VEHICLE"

        while True:
            raw_frame = camera.capture_array()
            frame = correct_camera_colour(raw_frame)

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

            boxes = expand_vehicle_boxes(detections)

            if vehicle_active and boxes:
                active_spots = triggered_spots(anchors, boxes)

                # Only affected zones move. Every triggered servo goes directly
                # to the extreme avoidance angle (135 degrees).
                for sid in range(1, 6):
                    ch = 6 - sid
                    target = AVOID_ANGLE if sid in active_spots else SERVO_NEUTRAL
                    if current_angles[sid] != target:
                        arduino.set_channel(ch, angle=target, light=1)
                        current_angles[sid] = target

                status = (
                    "DIVERTING " + ",".join(f"S{x}" for x in sorted(active_spots))
                    if active_spots else "VEHICLE IN VIEW - NO ZONE HIT"
                )

            # Once YOLO has seen no vehicle for the configured interval,
            # return every servo to neutral.
            if vehicle_active and now - last_vehicle_time >= RESET_DELAY_SECONDS:
                vehicle_active = False
                active_spots.clear()
                for sid in range(1, 6):
                    ch = 6 - sid
                    if current_angles[sid] != SERVO_NEUTRAL:
                        arduino.set_channel(ch, angle=SERVO_NEUTRAL, light=1)
                        current_angles[sid] = SERVO_NEUTRAL
                status = "RESET TO NEUTRAL"

            elif not vehicle_active:
                active_spots.clear()
                status = "WAITING FOR VEHICLE"

            fps_count += 1
            elapsed = now - fps_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count = 0
                fps_start = now

            display = draw_overlay(
                frame, detections, anchors, active_spots,
                boxes, status, fps, yolo_ms
            )
            cv2.imshow("BeamShift Fixed-Zone Avoidance", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("r"), ord("R")):
                active_spots.clear()
                for sid in range(1, 6):
                    arduino.set_channel(6 - sid, angle=SERVO_NEUTRAL, light=1)
                    current_angles[sid] = SERVO_NEUTRAL
                vehicle_active = False
                vehicle_streak = 0
                status = "MANUAL RESET"
            elif key in (ord("l"), ord("L")):
                arduino.set_all_lights(1)
            elif key in (ord("k"), ord("K")):
                arduino.set_all_lights(0)

    except KeyboardInterrupt:
        print("\\n[BEAMSHIFT] Interrupted.")
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

