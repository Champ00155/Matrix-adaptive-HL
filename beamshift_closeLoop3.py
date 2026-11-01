import time
import threading

import cv2
import numpy as np
import ncnn
import serial
from picamera2 import Picamera2


# ------------------------------ CONFIG ------------------------------
WIDTH, HEIGHT = 640, 480
INPUT_SIZE = 320
MODEL_PATH = "/home/pizza/BeamShift/models/yolo11n_ncnn_model"

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

CONFIDENCE = 0.40
NMS_THRESHOLD = 0.45
VEHICLE_CLASSES = {2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}

# Match the working spot_detection.py settings.
ROI_TOP, ROI_BOTTOM = 180, 430
TOPHAT_KERNEL_SIZE = 21
TOPHAT_THRESHOLD = 18
MIN_AREA, MAX_AREA = 1, 60
MAX_COMPONENT_SIDE = 12

SERVO_MIN, SERVO_MAX, NEUTRAL = 45, 135, 90
SERVO_STEP_DEG = 2
SERVO_SETTLE_SECONDS = 0.18

SAFETY_MARGIN_X = 24
SAFETY_MARGIN_Y = 20
SAFE_CLEARANCE_PX = 22.0

# After reversing once, stop if no new best clearance is found for this many
# movements. This prevents endless hunting.
MAX_STEPS_WITHOUT_BEST = 5
MAX_TOTAL_STEPS_PER_SPOT = 28
IMPROVEMENT_EPSILON_PX = 1.0

TRACK_MAX_FLOW_PX = 100
REACQUIRE_RADIUS_PX = 28
TRACK_LOST_LIMIT = 10

HEARTBEAT_INTERVAL = 0.25
# -------------------------------------------------------------------


class ArduinoLink:
    def __init__(self):
        self.ser = serial.Serial(
            SERIAL_PORT, BAUD_RATE, timeout=0.05, write_timeout=0.2
        )
        time.sleep(2.0)
        self.lock = threading.Lock()
        self.running = True
        self.last_heartbeat = 0.0
        self.angles = [NEUTRAL] * 5
        self.lights = [1] * 5
        self.thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.thread.start()

    def _write(self, line):
        with self.lock:
            self.ser.write((line + "\n").encode("ascii"))
            self.ser.flush()

    def _heartbeat_loop(self):
        while self.running:
            now = time.monotonic()
            if now - self.last_heartbeat >= HEARTBEAT_INTERVAL:
                try:
                    self._write("HB")
                    self.last_heartbeat = now
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
        self._write(f"Z,{channel},{self.angles[idx]},{self.lights[idx]}")

    def close(self):
        try:
            for ch in range(1, 6):
                self.command(ch, angle=NEUTRAL, light=0)
            time.sleep(0.35)
        except Exception:
            pass
        self.running = False
        self.thread.join(timeout=1.0)
        self.ser.close()


def channel_for_spot(spot_id):
    return 6 - spot_id


def load_model():
    print("[BEAMSHIFT] Loading YOLO11n NCNN...")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    p = net.load_param(MODEL_PATH + "/model.ncnn.param")
    m = net.load_model(MODEL_PATH + "/model.ncnn.bin")
    if p != 0 or m != 0:
        raise RuntimeError("Could not load NCNN model: " + MODEL_PATH)
    print("[BEAMSHIFT] YOLO model loaded.")
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
    ex = net.create_extractor()
    ex.input("in0", ncnn.Mat(tensor))
    ret, out = ex.extract("out0")
    if ret != 0:
        return []

    pred = np.asarray(out.numpy(), dtype=np.float32).squeeze()
    if pred.ndim != 2:
        return []
    if pred.shape[0] == 84:
        pred = pred.T
    elif pred.shape[1] != 84:
        return []

    boxes, scores, class_ids = [], [], []
    for row in pred:
        class_scores = row[4:]
        cid = int(np.argmax(class_scores))
        conf = float(class_scores[cid])
        if cid not in VEHICLE_CLASSES or conf < CONFIDENCE:
            continue
        cx, cy, bw, bh = row[:4]
        x1 = int(np.clip((cx - bw / 2 - px) / scale, 0, WIDTH - 1))
        y1 = int(np.clip((cy - bh / 2 - py) / scale, 0, HEIGHT - 1))
        x2 = int(np.clip((cx + bw / 2 - px) / scale, 0, WIDTH - 1))
        y2 = int(np.clip((cy + bh / 2 - py) / scale, 0, HEIGHT - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(conf)
        class_ids.append(cid)

    results = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE, NMS_THRESHOLD)
        if len(indices):
            for idx in np.asarray(indices).flatten():
                x, y, w, h = boxes[int(idx)]
                results.append({
                    "box": (x, y, w, h),
                    "class": VEHICLE_CLASSES[class_ids[int(idx)]],
                    "confidence": scores[int(idx)]
                })
    return results


def detect_spots(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    top = max(0, ROI_TOP)
    bottom = min(frame.shape[0], ROI_BOTTOM)
    roi = gray[top:bottom, :]
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
            "y": float(cy + top),
            "score": float(top_hat[labels == i].sum())
        })

    candidates.sort(key=lambda d: d["score"], reverse=True)
    chosen = []
    for d in candidates:
        if all(np.hypot(d["x"] - q["x"], d["y"] - q["y"]) >= 10
               for q in chosen):
            chosen.append(d)
        if len(chosen) == 5:
            break
    return chosen, gray, mask


class SpotTracker:
    """IDs are assigned left-to-right once, then never re-sorted."""

    def __init__(self):
        self.points = None
        self.previous_gray = None
        self.lost = [0] * 5

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
            return

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, self.points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )
        if next_points is None or status is None:
            next_points = self.points.copy()
            status = np.zeros((5, 1), dtype=np.uint8)

        updated = self.points.copy()
        used = set()

        # Associate independently to each ID's optical-flow prediction.
        # No left-to-right sorting occurs after startup.
        for i in range(5):
            old = self.points[i, 0]
            flow = next_points[i, 0]
            valid = bool(status[i, 0]) and (
                np.hypot(flow[0] - old[0], flow[1] - old[1]) < TRACK_MAX_FLOW_PX
            )
            predicted = flow if valid else old

            best_j, best_dist = None, float("inf")
            for j, d in enumerate(detections):
                if j in used:
                    continue
                dist = np.hypot(d["x"] - predicted[0], d["y"] - predicted[1])
                if dist < best_dist:
                    best_j, best_dist = j, dist

            if best_j is not None and best_dist <= REACQUIRE_RADIUS_PX:
                d = detections[best_j]
                updated[i, 0] = (d["x"], d["y"])
                used.add(best_j)
                self.lost[i] = 0
            elif valid:
                updated[i, 0] = flow
                self.lost[i] += 1
            else:
                self.lost[i] += 1

        self.points = updated
        self.previous_gray = gray.copy()

    def get(self, sid):
        if self.points is None:
            return None
        x, y = self.points[sid - 1, 0]
        return float(x), float(y)

    def visible(self, sid):
        return self.points is not None and self.lost[sid - 1] <= TRACK_LOST_LIMIT


def make_safety_boxes(detections):
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


def signed_clearance(point, box):
    """Negative inside rectangle; positive outside; 0 on its boundary."""
    x, y = point
    x1, y1, x2, y2 = box
    if x1 <= x <= x2 and y1 <= y <= y2:
        return -float(min(x - x1, x2 - x, y - y1, y2 - y))
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return float(np.hypot(dx, dy))


def clearance_from_boxes(point, boxes):
    if not boxes:
        return None
    return min(signed_clearance(point, b) for b in boxes)


class IncrementalAvoider:
    """
    One active spot at a time. One servo step is issued, then the main camera
    loop continues normally. The next frames evaluate whether clearance improved.
    """

    def __init__(self, arduino):
        self.arduino = arduino
        self.angles = [NEUTRAL] * 5
        self.states = ["WAITING"] * 5
        self.active_sid = None
        self.direction = 1
        self.reversed = False
        self.best_angle = NEUTRAL
        self.best_score = -1e9
        self.score_before_move = -1e9
        self.last_move_time = 0.0
        self.steps_total = 0
        self.steps_without_best = 0
        self.pending = False
        self.last_status = "WAITING FOR VEHICLE"

    def reset(self):
        for sid in range(1, 6):
            ch = channel_for_spot(sid)
            self.angles[sid - 1] = NEUTRAL
            self.arduino.command(ch, angle=NEUTRAL, light=1)
            self.states[sid - 1] = "IDLE"
        self.active_sid = None
        self.pending = False
        self.last_status = "RESET"

    def lights(self, state):
        for ch in range(1, 6):
            self.arduino.command(ch, light=state)

    def _start(self, sid, clearance):
        self.active_sid = sid
        self.direction = 1
        self.reversed = False
        self.best_angle = self.angles[sid - 1]
        self.best_score = float(clearance)
        self.score_before_move = float(clearance)
        self.steps_total = 0
        self.steps_without_best = 0
        self.pending = False
        self.states[sid - 1] = "SEARCHING"
        self.last_status = f"SEARCH S{sid}"

    def _finish(self, status):
        sid = self.active_sid
        if sid is not None:
            ch = channel_for_spot(sid)
            # Keep the best measured angle; no repeated return-and-restart loop.
            self.arduino.command(ch, angle=self.best_angle, light=1)
            self.angles[sid - 1] = self.best_angle
            self.states[sid - 1] = status
            print(
                f"[AVOID] S{sid}/CH{ch}: hold {self.best_angle}deg; "
                f"best clearance {self.best_score:.1f}px; {status}"
            )
        self.active_sid = None
        self.pending = False
        self.last_status = status

    def update(self, tracker, boxes, now):
        if tracker.points is None or not boxes:
            self.last_status = "NO VEHICLE" if not boxes else "INITIALIZING"
            return

        # Let the physical servo/spot settle without blocking the camera loop.
        # Frames continue to be captured and YOLO continues to run during this time.
        if self.active_sid is not None and self.pending:
            if now - self.last_move_time < SERVO_SETTLE_SECONDS:
                return

        # Evaluate the result of the last commanded movement from the latest
        # camera frame. No sleeps and no extra camera captures occur here.
        if self.active_sid is not None and self.pending:
            sid = self.active_sid
            if not tracker.visible(sid):
                self.states[sid - 1] = "TRACK LOST"
                self._finish("TRACK LOST")
                return
            current_clearance = clearance_from_boxes(tracker.get(sid), boxes)
            if current_clearance is None:
                return

            if current_clearance >= SAFE_CLEARANCE_PX:
                self.best_score = current_clearance
                self.best_angle = self.angles[sid - 1]
                self._finish("CLEAR")
                return

            if current_clearance > self.best_score + IMPROVEMENT_EPSILON_PX:
                self.best_score = current_clearance
                self.best_angle = self.angles[sid - 1]
                self.steps_without_best = 0
            else:
                self.steps_without_best += 1

            # If first direction made clearance worse, reverse once.
            if (
                current_clearance < self.score_before_move - IMPROVEMENT_EPSILON_PX
                and not self.reversed
            ):
                self.direction *= -1
                self.reversed = True

            self.score_before_move = current_clearance
            self.pending = False

            if self.steps_total >= MAX_TOTAL_STEPS_PER_SPOT:
                self._finish("LIMIT")
                return
            if self.reversed and self.steps_without_best >= MAX_STEPS_WITHOUT_BEST:
                self._finish("NOT CLEAR")
                return

        # If no spot is currently being adjusted, choose the most unsafe spot.
        if self.active_sid is None:
            worst_sid, worst_score = None, float("inf")
            for sid in range(1, 6):
                if not tracker.visible(sid):
                    continue
                score = clearance_from_boxes(tracker.get(sid), boxes)
                if score is not None and score < SAFE_CLEARANCE_PX and score < worst_score:
                    worst_sid, worst_score = sid, score

            if worst_sid is None:
                for sid in range(1, 6):
                    self.states[sid - 1] = "CLEAR"
                self.last_status = "ALL SPOTS CLEAR"
                return

            self._start(worst_sid, worst_score)

        # Issue only one small movement. The main camera/YOLO loop is never paused.
        if self.active_sid is not None and not self.pending:
            sid = self.active_sid
            ch = channel_for_spot(sid)
            current_angle = self.angles[sid - 1]
            next_angle = int(np.clip(
                current_angle + self.direction * SERVO_STEP_DEG,
                SERVO_MIN, SERVO_MAX
            ))

            if next_angle == current_angle:
                if not self.reversed:
                    self.direction *= -1
                    self.reversed = True
                    next_angle = int(np.clip(
                        current_angle + self.direction * SERVO_STEP_DEG,
                        SERVO_MIN, SERVO_MAX
                    ))
                if next_angle == current_angle:
                    self._finish("LIMIT")
                    return

            self.score_before_move = clearance_from_boxes(tracker.get(sid), boxes)
            self.arduino.command(ch, angle=next_angle, light=1)
            self.angles[sid - 1] = next_angle
            self.steps_total += 1
            self.last_move_time = now
            self.pending = True
            self.states[sid - 1] = "MOVING"


def draw_overlay(frame, detections, tracker, avoider, fps, latency):
    view = frame.copy()
    boxes = make_safety_boxes(detections)

    for x1, y1, x2, y2 in boxes:
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
            state = avoider.states[sid - 1]
            colour = (0, 255, 255) if state in ("MOVING", "SEARCHING") else (255, 255, 0)
            label = f"S{sid}/CH{channel_for_spot(sid)} {avoider.angles[sid-1]}deg {state}"
            cv2.circle(view, (x, y), 9, colour, 2)
            cv2.putText(view, label, (min(WIDTH - 205, x + 10), max(18, y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, colour, 1, cv2.LINE_AA)

    cv2.putText(view, f"FPS {fps:.1f} | YOLO {latency:.0f}ms | {avoider.last_status}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(view, "Q quit | R reset | L lights ON | K lights OFF",
                (8, HEIGHT - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.39,
                (255, 255, 255), 1, cv2.LINE_AA)
    return view


def main():
    arduino = None
    camera = None
    try:
        arduino = ArduinoLink()
        net = load_model()

        camera = Picamera2()
        camera.configure(camera.create_video_configuration(
            main={"size": (WIDTH, HEIGHT), "format": "BGR888"},
            buffer_count=4
        ))
        camera.options["queue"] = False
        camera.start()
        time.sleep(1.0)

        tracker = SpotTracker()
        avoider = IncrementalAvoider(arduino)

        for ch in range(1, 6):
            arduino.command(ch, angle=NEUTRAL, light=1)

        print("\n[BEAMSHIFT V3] Waiting for five spots; keep servos neutral.")
        print("[BEAMSHIFT V3] Non-blocking control: one servo step per camera cycle.")

        fps = 0.0
        fps_count = 0
        fps_start = time.monotonic()
        last_latency = 0.0

        while True:
            frame = camera.capture_array()
            spot_detections, gray, spot_mask = detect_spots(frame)

            if tracker.points is None:
                if tracker.initialize(spot_detections, gray):
                    print("[BEAMSHIFT V3] Five spot IDs initialized left-to-right.")
            else:
                tracker.update(gray, spot_detections)

            # YOLO runs once per normal frame. No repeated inference loops or sleeps.
            t0 = time.perf_counter()
            vehicles = detect_vehicles(net, frame)
            last_latency = (time.perf_counter() - t0) * 1000.0
            safety_boxes = make_safety_boxes(vehicles)

            now = time.monotonic()
            avoider.update(tracker, safety_boxes, now)

            fps_count += 1
            elapsed = now - fps_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count = 0
                fps_start = now

            display = draw_overlay(
                frame, vehicles, tracker, avoider, fps, last_latency
            )
            cv2.imshow("BeamShift V3 - Non-blocking Closed Loop", display)
            cv2.imshow("Spot Detector Mask", spot_mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("r"), ord("R")):
                avoider.reset()
            elif key in (ord("l"), ord("L")):
                avoider.lights(1)
            elif key in (ord("k"), ord("K")):
                avoider.lights(0)

    except KeyboardInterrupt:
        print("\n[BEAMSHIFT V3] Stopping.")
    finally:
        if camera is not None:
            camera.stop()
        cv2.destroyAllWindows()
        if arduino is not None:
            arduino.close()
        print("[BEAMSHIFT V3] Stopped; LEDs OFF and servos commanded to neutral.")


if __name__ == "__main__":
    main()
