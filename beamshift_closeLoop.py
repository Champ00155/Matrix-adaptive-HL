import time
import threading
from collections import deque

import cv2
import numpy as np
import ncnn
import serial
from picamera2 import Picamera2


# --------------------------- CONFIG ---------------------------
WIDTH, HEIGHT = 640, 480
INPUT_SIZE = 320
MODEL_PATH = "/home/pizza/BeamShift/models/yolo11n_ncnn_model"
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

CONFIDENCE = 0.40
NMS_THRESHOLD = 0.45
VEHICLE_CLASSES = {2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}

# Keep these consistent with the working spot_detection.py.
ROI_TOP, ROI_BOTTOM = 180, 430
TOPHAT_KERNEL_SIZE = 21
TOPHAT_THRESHOLD = 18
MIN_AREA, MAX_AREA = 1, 60
MAX_COMPONENT_SIDE = 12

SERVO_MIN, SERVO_MAX, NEUTRAL = 45, 135, 90
SERVO_STEP = 3
MAX_SEARCH_STEPS_PER_DIRECTION = 12
MOVE_SETTLE_SECONDS = 0.20
VERIFY_FRAMES = 4

# Expand vehicle box by this margin before deciding a spot is unsafe.
SAFETY_MARGIN_X = 24
SAFETY_MARGIN_Y = 20

# Tracking
LK_WINDOW = (21, 21)
LK_LEVELS = 3
REACQUIRE_RADIUS = 32
TRACK_LOST_LIMIT = 12

# If no car is detected for this many seconds, no new avoidance movement
# is initiated. Existing servo angles are held rather than repeatedly hunting.
CAR_LOST_HOLD_SECONDS = 1.0

HEARTBEAT_INTERVAL = 0.25
DISPLAY_SCALE = 1.0
# --------------------------------------------------------------


class ArduinoLink:
    def __init__(self):
        self.ser = serial.Serial(
            SERIAL_PORT, BAUD_RATE, timeout=0.05, write_timeout=0.2
        )
        time.sleep(2.0)  # Arduino resets when serial port opens.
        self.lock = threading.Lock()
        self.running = True
        self.last_hb = 0.0
        self.angles = [NEUTRAL] * 5
        self.lights = [1] * 5
        self.thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.thread.start()

    def write_line(self, line):
        with self.lock:
            self.ser.write((line + "\n").encode("ascii"))
            self.ser.flush()

    def _heartbeat_loop(self):
        while self.running:
            now = time.monotonic()
            if now - self.last_hb >= HEARTBEAT_INTERVAL:
                try:
                    self.write_line("HB")
                    self.last_hb = now
                except serial.SerialException:
                    pass
            time.sleep(0.025)

    def command(self, channel, angle=None, light=None):
        idx = channel - 1
        if idx not in range(5):
            raise ValueError("Arduino channel must be 1..5")
        if angle is not None:
            self.angles[idx] = int(np.clip(angle, SERVO_MIN, SERVO_MAX))
        if light is not None:
            self.lights[idx] = int(bool(light))
        self.write_line(
            f"Z,{channel},{self.angles[idx]},{self.lights[idx]}"
        )

    def close(self, neutralize=True):
        try:
            if neutralize:
                for ch in range(1, 6):
                    self.command(ch, angle=NEUTRAL, light=0)
                time.sleep(0.4)
        except Exception:
            pass
        self.running = False
        self.thread.join(timeout=1.0)
        self.ser.close()


def load_yolo():
    print("[BEAMSHIFT] Loading YOLO11n NCNN...")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    r1 = net.load_param(MODEL_PATH + "/model.ncnn.param")
    r2 = net.load_model(MODEL_PATH + "/model.ncnn.bin")
    if r1 != 0 or r2 != 0:
        raise RuntimeError("Could not load YOLO NCNN model from " + MODEL_PATH)
    print("[BEAMSHIFT] YOLO model loaded")
    return net


def preprocess(frame):
    h, w = frame.shape[:2]
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh))
    px, py = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[py:py + nh, px:px + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    tensor /= 255.0
    return tensor, scale, px, py


def detect_vehicles(net, frame):
    tensor, scale, px, py = preprocess(frame)
    extractor = net.create_extractor()
    extractor.input("in0", ncnn.Mat(tensor))
    ret, output = extractor.extract("out0")
    if ret != 0:
        return []

    predictions = np.asarray(output.numpy(), dtype=np.float32).squeeze()
    if predictions.ndim != 2:
        return []
    if predictions.shape[0] == 84:
        predictions = predictions.T
    elif predictions.shape[1] != 84:
        return []

    boxes, scores, class_ids = [], [], []
    for p in predictions:
        class_scores = p[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if class_id not in VEHICLE_CLASSES or confidence < CONFIDENCE:
            continue

        cx, cy, bw, bh = p[:4]
        x1 = int(np.clip((cx - bw / 2 - px) / scale, 0, WIDTH - 1))
        y1 = int(np.clip((cy - bh / 2 - py) / scale, 0, HEIGHT - 1))
        x2 = int(np.clip((cx + bw / 2 - px) / scale, 0, WIDTH - 1))
        y2 = int(np.clip((cy + bh / 2 - py) / scale, 0, HEIGHT - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(confidence)
        class_ids.append(class_id)

    detections = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, CONFIDENCE, NMS_THRESHOLD
        )
        if len(indices):
            for i in np.asarray(indices).flatten():
                x, y, w, h = boxes[int(i)]
                detections.append({
                    "box": (x, y, w, h),
                    "class": VEHICLE_CLASSES[class_ids[int(i)]],
                    "confidence": scores[int(i)]
                })
    return detections


def detect_spots(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_top = max(0, ROI_TOP)
    roi_bottom = min(frame.shape[0], ROI_BOTTOM)
    roi = gray[roi_top:roi_bottom, :]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL_SIZE, TOPHAT_KERNEL_SIZE)
    )
    top_hat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(top_hat, TOPHAT_THRESHOLD, 255, cv2.THRESH_BINARY)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    candidates = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        if w > MAX_COMPONENT_SIDE or h > MAX_COMPONENT_SIDE:
            continue
        cx, cy = centroids[i]
        candidates.append({
            "x": float(cx),
            "y": float(cy + roi_top),
            "score": float(top_hat[labels == i].sum())
        })

    # Retain five strongest distinct candidates.
    candidates.sort(key=lambda d: d["score"], reverse=True)
    chosen = []
    for d in candidates:
        if all(np.hypot(d["x"] - q["x"], d["y"] - q["y"]) >= 10
               for q in chosen):
            chosen.append(d)
        if len(chosen) == 5:
            break
    return chosen, gray, mask


def channel_for_spot(spot_id):
    return 6 - spot_id


def expanded_vehicle_boxes(detections):
    boxes = []
    for det in detections:
        x, y, w, h = det["box"]
        boxes.append((
            max(0, x - SAFETY_MARGIN_X),
            max(0, y - SAFETY_MARGIN_Y),
            min(WIDTH - 1, x + w + SAFETY_MARGIN_X),
            min(HEIGHT - 1, y + h + SAFETY_MARGIN_Y)
        ))
    return boxes


def point_box_clearance(point, box):
    """
    Signed distance to expanded rectangle:
      positive = outside (safe)
      zero     = on the boundary
      negative = inside (unsafe)

    A signed score lets the controller make progress even when a spot starts
    well inside the car box; plain distance would be zero everywhere inside.
    """
    x, y = point
    x1, y1, x2, y2 = box
    if x1 <= x <= x2 and y1 <= y <= y2:
        return -float(min(x - x1, x2 - x, y - y1, y2 - y))
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return float(np.hypot(dx, dy))


def clearance_from_all_vehicles(point, boxes):
    if not boxes:
        return None
    return min(point_box_clearance(point, box) for box in boxes)


class SpotTracker:
    """
    Initialize IDs left-to-right once.
    Thereafter, use Lucas-Kanade optical flow to follow each individual spot.
    Detector reacquisition is local around the predicted position; IDs are never
    re-sorted after initialization.
    """

    def __init__(self):
        self.points = None
        self.previous_gray = None
        self.lost = [0] * 5
        self.initial_samples = deque(maxlen=20)

    def initialize(self, detections, gray):
        if len(detections) < 5:
            return False
        ordered = sorted(detections, key=lambda d: d["x"])[:5]
        self.points = np.array(
            [[[d["x"], d["y"]]] for d in ordered], dtype=np.float32
        )
        self.previous_gray = gray.copy()
        self.lost = [0] * 5
        return True

    def update(self, gray, detections):
        if self.points is None:
            return False

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, self.points, None,
            winSize=LK_WINDOW, maxLevel=LK_LEVELS,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        if next_points is None or status is None:
            status = np.zeros((5, 1), dtype=np.uint8)
            next_points = self.points.copy()

        used = set()
        updated = self.points.copy()

        for i in range(5):
            old_x, old_y = self.points[i, 0]
            candidate = next_points[i, 0]
            valid_flow = bool(status[i, 0]) and (
                np.hypot(candidate[0] - old_x, candidate[1] - old_y) < 100
            )

            target = candidate if valid_flow else self.points[i, 0]

            # Snap to a detected bright component near the flow prediction.
            nearest_index, nearest_distance = None, float("inf")
            for j, det in enumerate(detections):
                if j in used:
                    continue
                distance = np.hypot(det["x"] - target[0], det["y"] - target[1])
                if distance < nearest_distance:
                    nearest_distance, nearest_index = distance, j

            if nearest_index is not None and nearest_distance <= REACQUIRE_RADIUS:
                d = detections[nearest_index]
                updated[i, 0] = (d["x"], d["y"])
                used.add(nearest_index)
                self.lost[i] = 0
            elif valid_flow:
                updated[i, 0] = candidate
                self.lost[i] += 1
            else:
                self.lost[i] += 1

        self.points = updated
        self.previous_gray = gray.copy()
        return True

    def get(self, spot_id):
        if self.points is None:
            return None
        x, y = self.points[spot_id - 1, 0]
        return float(x), float(y)

    def visible(self, spot_id):
        return self.points is not None and self.lost[spot_id - 1] <= TRACK_LOST_LIMIT


def draw_overlay(frame, detections, tracker, angles, states, fps, status_text):
    view = frame.copy()

    for box in expanded_vehicle_boxes(detections):
        x1, y1, x2, y2 = box
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 165, 255), 2)

    for det in detections:
        x, y, w, h = det["box"]
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(view, f'{det["class"]} {det["confidence"]:.2f}',
                    (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)

    if tracker.points is not None:
        for sid in range(1, 6):
            if not tracker.visible(sid):
                continue
            x, y = map(int, tracker.get(sid))
            colour = (0, 255, 255) if states[sid - 1] == "MOVING" else (255, 255, 0)
            cv2.circle(view, (x, y), 9, colour, 2)
            label = f"S{sid}/CH{channel_for_spot(sid)} {angles[sid-1]}deg"
            label_y = max(18, y - 13 - (sid % 2) * 15)
            cv2.putText(view, label, (min(WIDTH - 150, x + 10), label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)

    cv2.putText(view, f"FPS {fps:.1f} | {status_text}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(view, "Q: quit | R: reset servos | L: lights on | K: lights off",
                (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (255, 255, 255), 1, cv2.LINE_AA)
    return view


def main():
    arduino = None
    camera = None
    try:
        arduino = ArduinoLink()
        net = load_yolo()

        camera = Picamera2()
        camera.configure(camera.create_video_configuration(
            main={"size": (WIDTH, HEIGHT), "format": "BGR888"},
            buffer_count=4
        ))
        camera.options["queue"] = False
        camera.start()
        time.sleep(1.5)

        angles = [NEUTRAL] * 5
        states = ["IDLE"] * 5
        tracker = SpotTracker()
        last_car_time = 0.0
        fps_count, fps_start, fps = 0, time.monotonic(), 0.0
        last_vehicle_detections = []
        search_cooldown = [0.0] * 5

        # Start at neutral with diffused LEDs enabled.
        for ch in range(1, 6):
            arduino.command(ch, angle=NEUTRAL, light=1)

        print("\n[BEAMSHIFT] Waiting for five spots to initialize left-to-right.")
        print("[BEAMSHIFT] Keep camera and servos stationary during initialization.")

        while True:
            frame = camera.capture_array()
            detections, gray, spot_mask = detect_spots(frame)

            if tracker.points is None:
                if tracker.initialize(detections, gray):
                    print("[BEAMSHIFT] Five spot IDs initialized left-to-right.")
            else:
                tracker.update(gray, detections)

            t0 = time.perf_counter()
            vehicles = detect_vehicles(net, frame)
            inference_ms = (time.perf_counter() - t0) * 1000

            now = time.monotonic()
            if vehicles:
                last_car_time = now
                last_vehicle_detections = vehicles

            boxes = expanded_vehicle_boxes(vehicles)

            status_text = "INITIALIZING SPOTS"
            if tracker.points is not None:
                status_text = "TRACKING"
            if vehicles:
                status_text += " | VEHICLE"
            elif now - last_car_time < CAR_LOST_HOLD_SECONDS:
                status_text += " | CAR LOST: HOLD"

            # Closed-loop search is deliberately sequential: one servo at a time.
            # Do not hunt when YOLO has lost the vehicle.
            if tracker.points is not None and boxes:
                for sid in range(1, 6):
                    if not tracker.visible(sid):
                        states[sid - 1] = "TRACK LOST"
                        continue

                    point = tracker.get(sid)
                    clearance = clearance_from_all_vehicles(point, boxes)
                    if clearance is None or clearance > 0:
                        states[sid - 1] = "CLEAR"
                        continue

                    if now < search_cooldown[sid - 1]:
                        continue

                    states[sid - 1] = "MOVING"
                    ch = channel_for_spot(sid)
                    original_angle = angles[sid - 1]

                    # Try + direction first, then - direction. Score is measured
                    # after each move from the camera-tracked spot to car box.
                    best_angle = original_angle
                    best_score = 0.0
                    found_clear = False

                    for direction in (1, -1):
                        test_angle = original_angle
                        for _ in range(MAX_SEARCH_STEPS_PER_DIRECTION):
                            next_angle = int(np.clip(
                                test_angle + direction * SERVO_STEP,
                                SERVO_MIN, SERVO_MAX
                            ))
                            if next_angle == test_angle:
                                break

                            arduino.command(ch, angle=next_angle, light=1)
                            angles[sid - 1] = next_angle
                            test_angle = next_angle
                            time.sleep(MOVE_SETTLE_SECONDS)

                            # Let camera/spot tracker settle for a few frames.
                            for _ in range(VERIFY_FRAMES):
                                verify_frame = camera.capture_array()
                                verify_dets, verify_gray, _ = detect_spots(verify_frame)
                                tracker.update(verify_gray, verify_dets)

                            new_point = tracker.get(sid)
                            if not tracker.visible(sid):
                                break

                            new_score = clearance_from_all_vehicles(new_point, boxes)
                            new_score = -1e6 if new_score is None else new_score

                            if new_score > best_score:
                                best_score = new_score
                                best_angle = next_angle

                            if new_score > 0:
                                found_clear = True
                                best_angle = next_angle
                                break

                        if found_clear:
                            break

                    # Keep the best observed angle. If no measurable movement
                    # occurred, restore the original angle rather than guessing.
                    arduino.command(ch, angle=best_angle, light=1)
                    angles[sid - 1] = best_angle
                    states[sid - 1] = "CLEAR" if found_clear else "LIMIT/NO CLEAR"
                    search_cooldown[sid - 1] = time.monotonic() + 1.0

                    print(
                        f"[AVOID] S{sid} / CH{ch}: {original_angle} -> "
                        f"{best_angle}; clearance={best_score:.1f}px; "
                        f"{'CLEAR' if found_clear else 'NOT CLEARED'}"
                    )

                    # Refresh the vehicle boxes after movement.
                    # Next main-loop pass will re-evaluate current car position.
                    break

            fps_count += 1
            elapsed = now - fps_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count, fps_start = 0, now

            display = draw_overlay(
                frame, vehicles, tracker, angles, states, fps,
                f"YOLO {inference_ms:.0f}ms"
            )
            cv2.imshow("BeamShift Closed-Loop Indoor Demo", display)
            cv2.imshow("Spot Detector Mask", spot_mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("r"), ord("R")):
                for sid in range(1, 6):
                    angles[sid - 1] = NEUTRAL
                    arduino.command(channel_for_spot(sid), angle=NEUTRAL, light=1)
                print("[BEAMSHIFT] All servos commanded to neutral.")
            elif key in (ord("l"), ord("L")):
                for ch in range(1, 6):
                    arduino.command(ch, light=1)
            elif key in (ord("k"), ord("K")):
                for ch in range(1, 6):
                    arduino.command(ch, light=0)

    finally:
        if camera is not None:
            camera.stop()
        cv2.destroyAllWindows()
        if arduino is not None:
            arduino.close(neutralize=True)
        print("[BEAMSHIFT] Stopped. LEDs OFF; servos commanded to neutral.")


if __name__ == "__main__":
    main()
