import time
import threading
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

# White balance is measured from a neutral white/grey card at startup.
# Gains are computed from the actual camera + filter + room lighting.
WB_MIN_GAIN = 0.60
WB_MAX_GAIN = 1.60

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

# Optional read-only browser dashboard. It reuses the already-rendered frame;
# it does not open another camera or run another YOLO inference.
DASHBOARD_ENABLED = True
DASHBOARD_PORT = 5000
DASHBOARD_STREAM_FPS = 5
DASHBOARD_FRAME_WIDTH = 480
DASHBOARD_JPEG_QUALITY = 65

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



DASHBOARD_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BeamShift | Live Command Centre</title>
<style>
:root{color-scheme:dark;--bg:#090d16;--panel:#111827;--panel2:#0d1421;--line:#243247;--muted:#8da0b8;--text:#edf4ff;--cyan:#52d7e8;--green:#4ce0a0;--amber:#ffbf69;--red:#ff687d}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 15% 0%,#14253c 0,transparent 42%),var(--bg);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{height:72px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 28px;background:#0a101bcc;backdrop-filter:blur(12px);position:sticky;top:0;z-index:5}
.brand{display:flex;align-items:center;gap:12px;font-weight:800;letter-spacing:.12em}.logo{width:34px;height:34px;border:1px solid #337c91;border-radius:10px;display:grid;place-items:center;color:var(--cyan);font-weight:900;background:#102b38}.brand small{display:block;color:var(--muted);font-size:10px;letter-spacing:.18em;font-weight:600;margin-top:1px}
.header-right{display:flex;align-items:center;gap:16px;color:var(--muted);font-size:12px}.live-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green);margin-right:7px}.btn{border:1px solid var(--line);background:#121d2c;color:var(--text);padding:8px 12px;border-radius:8px;cursor:pointer}.btn:hover{border-color:var(--cyan)}
main{max-width:1500px;margin:auto;padding:26px 28px 40px}.page-title{display:flex;justify-content:space-between;align-items:end;margin-bottom:20px}.eyebrow{color:var(--cyan);font-size:10px;letter-spacing:.2em;font-weight:800}.page-title h1{font-size:25px;margin:4px 0 0;letter-spacing:-.04em}.sub{color:var(--muted);margin-top:4px;font-size:12px}
.grid{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(300px,.9fr);gap:16px}.panel{background:linear-gradient(145deg,#121c2b,#0d1420);border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 12px 34px #0003}.panel-head{height:50px;padding:0 16px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between}.panel-title{font-size:11px;font-weight:800;letter-spacing:.12em}.tag{font-size:10px;color:var(--cyan);border:1px solid #285565;background:#102632;padding:4px 8px;border-radius:20px}
.feed-wrap{background:#05080d;min-height:300px;display:flex;align-items:center;justify-content:center;padding:8px}.feed-wrap img{display:block;width:100%;height:auto;max-height:68vh;object-fit:contain}.feed-placeholder{color:var(--muted);padding:70px 10px;text-align:center}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}.metric{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:13px 14px}.metric label{display:block;font-size:10px;color:var(--muted);letter-spacing:.12em;font-weight:700}.metric strong{display:block;font-size:25px;letter-spacing:-.04em;margin-top:5px}.metric span{font-size:10px;color:var(--muted)}
.side{display:grid;gap:14px;align-content:start}.status-box{padding:16px}.status-line{display:flex;align-items:center;gap:10px;font-size:13px;font-weight:700}.status-icon{width:30px;height:30px;display:grid;place-items:center;background:#102e2a;color:var(--green);border-radius:9px}.status-box p{color:var(--muted);font-size:11px;margin:10px 0 0}
.zone-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:7px;padding:14px}.zone{border:1px solid var(--line);border-radius:9px;background:#0c1420;padding:11px 5px;text-align:center}.zone b{display:block;font-size:13px}.zone small{font-size:9px;color:var(--muted)}.zone.active{border-color:#9a3e51;background:#351923;box-shadow:inset 0 0 18px #ff52651c}.zone.active b{color:#ff8290}.zone-dot{width:6px;height:6px;border-radius:50%;background:#506078;display:inline-block;margin-bottom:5px}.zone.active .zone-dot{background:var(--red);box-shadow:0 0 9px var(--red)}
.servo-list{padding:4px 16px 12px}.servo-row{display:grid;grid-template-columns:44px 1fr 48px;gap:10px;align-items:center;padding:10px 0;border-bottom:1px solid #1c293a}.servo-row:last-child{border:0}.servo-name{font-weight:800;font-size:11px}.servo-name small{display:block;color:var(--muted);font-size:9px;font-weight:500}.bar{height:6px;background:#202d40;border-radius:9px;overflow:hidden}.bar i{display:block;height:100%;background:linear-gradient(90deg,#36aeca,#61e1d0);border-radius:9px;transition:width .2s}.angle{text-align:right;font-variant-numeric:tabular-nums;color:var(--cyan);font-size:11px}
.bottom-grid{display:grid;grid-template-columns:1.05fr .95fr;gap:16px;margin-top:16px}.table-wrap{padding:4px 16px 14px}.det-row{display:grid;grid-template-columns:1fr 70px 55px;gap:8px;padding:9px 0;border-bottom:1px solid #1c293a;font-size:11px}.det-row.head{color:var(--muted);font-size:9px;letter-spacing:.1em;font-weight:800}.det-row:last-child{border:0}.confidence{color:var(--green);text-align:right}.log{padding:6px 16px 12px;max-height:195px;overflow:auto}.log-item{display:flex;gap:10px;padding:8px 0;border-bottom:1px solid #1c293a;font-size:11px}.log-item time{color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}.log-item p{margin:0}.empty{color:var(--muted);font-size:11px;padding:16px 0}
footer{color:#65768c;text-align:center;font-size:10px;margin-top:24px;letter-spacing:.05em}
@media(max-width:950px){.grid,.bottom-grid{grid-template-columns:1fr}.side{grid-template-columns:1fr 1fr}.side>.panel:last-child{grid-column:1/-1}}
@media(max-width:600px){header{padding:0 14px}.header-right .hint{display:none}main{padding:18px 12px}.page-title{align-items:start;gap:12px}.page-title h1{font-size:21px}.side{grid-template-columns:1fr}.side>.panel:last-child{grid-column:auto}.metrics{gap:6px}.metric{padding:10px}.metric strong{font-size:21px}.zone-grid{gap:4px;padding:10px 8px}.zone{padding:9px 2px}.header-right{gap:8px}}
</style></head>
<body>
<header><div class="brand"><div class="logo">B</div><div>BEAMSHIFT<small>ADAPTIVE LIGHTING SYSTEM</small></div></div><div class="header-right"><span id="connection"><i class="live-dot"></i>CONNECTING</span><span class="hint">LOCAL DEMO NETWORK</span><button class="btn" onclick="toggleFullscreen()">⛶ Presentation</button></div></header>
<main><div class="page-title"><div><div class="eyebrow">SYSTEM OVERVIEW / LIVE</div><h1>Command centre</h1><div class="sub">Computer vision · Fixed-zone detection · Servo response</div></div><div class="tag">INDOOR PROTOTYPE</div></div>
<div class="grid"><section><div class="panel"><div class="panel-head"><div class="panel-title">LIVE CAMERA FEED</div><span class="tag">YOLO OVERLAY</span></div><div class="feed-wrap"><img id="feed" src="/video_feed" alt="Waiting for camera stream" onerror="this.style.display='none';document.getElementById('feedError').style.display='block'"><div id="feedError" class="feed-placeholder" style="display:none">Camera stream unavailable</div></div></div>
<div class="metrics"><div class="metric"><label>PIPELINE FPS</label><strong id="fps">--</strong><span>End-to-end loop</span></div><div class="metric"><label>INFERENCE TIME</label><strong id="latency">--<small> ms</small></strong><span>YOLO NCNN</span></div><div class="metric"><label>VEHICLES</label><strong id="count">0</strong><span>Current frame</span></div></div></section>
<aside class="side"><div class="panel status-box"><div class="panel-head" style="padding:0 0 12px;height:auto;border:0"><div class="panel-title">SYSTEM STATUS</div><span class="tag">ACTIVE</span></div><div class="status-line"><div class="status-icon">⌁</div><span id="status">Starting…</span></div><p>Zone triggers are evaluated from the live vehicle bounding boxes.</p></div>
<div class="panel"><div class="panel-head"><div class="panel-title">ZONE ACTIVATION</div><span class="tag">5 FIXED ZONES</span></div><div class="zone-grid" id="zones"></div></div>
<div class="panel"><div class="panel-head"><div class="panel-title">SERVO CHANNELS</div><span class="tag">LIVE POSITION</span></div><div class="servo-list" id="servos"></div></div></aside></div>
<div class="bottom-grid"><section class="panel"><div class="panel-head"><div class="panel-title">VEHICLE DETECTIONS</div><span class="tag" id="detTag">0 DETECTED</span></div><div class="table-wrap" id="detections"><div class="det-row head"><span>CLASS</span><span>CONFIDENCE</span><span style="text-align:right">STATE</span></div><div class="empty">No vehicles in view.</div></div></section>
<section class="panel"><div class="panel-head"><div class="panel-title">EVENT TIMELINE</div><span class="tag">RECENT ACTIVITY</span></div><div class="log" id="events"><div class="empty">Waiting for system events…</div></div></section></div>
<footer>BEAMSHIFT · LIVE MONITORING INTERFACE · READ-ONLY DISPLAY · NOT FOR ROAD USE</footer></main>
<script>
const $=id=>document.getElementById(id);let lastEvent=0;
function toggleFullscreen(){if(!document.fullscreenElement)document.documentElement.requestFullscreen?.();else document.exitFullscreen?.()}
function initZones(){ $('zones').innerHTML=Array.from({length:5},(_,i)=>`<div class="zone" id="zone${i+1}"><i class="zone-dot"></i><b>S${i+1}</b><small>CH${5-i}</small></div>`).join(''); }
function updateEvents(items){const fresh=items.filter(e=>e.id>lastEvent);if(!fresh.length)return;lastEvent=Math.max(lastEvent,...fresh.map(e=>e.id));const html=items.slice(-18).reverse().map(e=>`<div class="log-item"><time>${e.time}</time><p>${e.message}</p></div>`).join('');$('events').innerHTML=html||'<div class="empty">No events yet.</div>'}
function update(d){$('connection').innerHTML='<i class="live-dot"></i>CONNECTED';$('fps').textContent=Number(d.fps||0).toFixed(1);$('latency').innerHTML=`${Number(d.inference_ms||0).toFixed(0)}<small> ms</small>`;$('count').textContent=d.vehicle_count||0;$('status').textContent=d.status||'—';$('detTag').textContent=`${d.vehicle_count||0} DETECTED`;
const active=new Set(d.active_spots||[]);for(let i=1;i<=5;i++)$('zone'+i).classList.toggle('active',active.has(i));
$('servos').innerHTML=(d.servos||[]).map(s=>`<div class="servo-row"><div class="servo-name">S${s.spot}<small>CH${s.channel}</small></div><div class="bar"><i style="width:${Math.max(0,Math.min(100,(s.angle-45)/90*100))}%"></i></div><div class="angle">${s.angle}°</div></div>`).join('');
const ds=d.detections||[];$('detections').innerHTML='<div class="det-row head"><span>CLASS</span><span>CONFIDENCE</span><span style="text-align:right">STATE</span></div>'+(ds.length?ds.map(x=>`<div class="det-row"><span>${x.class}</span><span>${(x.confidence*100).toFixed(0)}%</span><span class="confidence">TRACKING</span></div>`).join(''):'<div class="empty">No vehicles in view.</div>');updateEvents(d.events||[])}
async function poll(){try{const r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw new Error('offline');update(await r.json())}catch(e){$('connection').textContent='DISCONNECTED'}}
initZones();poll();setInterval(poll,1000);
</script></body></html>'''


class DashboardPublisher:
    """Low-rate MJPEG + JSON publisher. Main loop only swaps references."""

    def __init__(self, port=DASHBOARD_PORT):
        self.port = port
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_jpeg = None
        self.state = {
            "fps": 0.0, "inference_ms": 0.0, "status": "Starting…",
            "vehicle_count": 0, "detections": [], "active_spots": [],
            "servos": [], "events": [],
        }
        self.events = []
        self.event_id = 0
        self.previous_active = set()
        self.previous_classes = set()
        self.running = True
        self.encoder_thread = threading.Thread(
            target=self._encode_worker, name="BeamShift-JPEG", daemon=True
        )
        self.httpd = None
        self.server_thread = None

    def _add_event_locked(self, message):
        self.event_id += 1
        self.events.append({
            "id": self.event_id,
            "time": time.strftime("%H:%M:%S"),
            "message": message,
        })
        self.events = self.events[-40:]

    def publish(self, frame, state):
        # Intentionally lightweight: no JPEG work, network I/O or inference here.
        with self.lock:
            self.latest_frame = frame
            active = set(state.get("active_spots", []))
            classes = {d["class"] for d in state.get("detections", [])}
            if active != self.previous_active:
                if active:
                    self._add_event_locked(
                        "Avoidance zones active: " + ", ".join(f"S{x}" for x in sorted(active))
                    )
                elif self.previous_active:
                    self._add_event_locked("All avoidance zones returned to neutral")
                self.previous_active = active
            if classes and not self.previous_classes:
                self._add_event_locked("Vehicle detected: " + ", ".join(sorted(classes)))
            elif self.previous_classes and not classes:
                self._add_event_locked("Vehicle cleared from camera view")
            self.previous_classes = classes
            self.state = dict(state)
            self.state["events"] = list(self.events)

    def snapshot(self):
        with self.lock:
            result = dict(self.state)
            result["events"] = list(self.events)
            return result

    def _encode_worker(self):
        interval = 1.0 / max(1, DASHBOARD_STREAM_FPS)
        while self.running:
            started = time.monotonic()
            with self.lock:
                frame = self.latest_frame
            if frame is not None:
                try:
                    h, w = frame.shape[:2]
                    target_w = min(DASHBOARD_FRAME_WIDTH, w)
                    target_h = max(1, int(h * target_w / w))
                    small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    ok, encoded = cv2.imencode(
                        ".jpg", small,
                        [int(cv2.IMWRITE_JPEG_QUALITY), DASHBOARD_JPEG_QUALITY],
                    )
                    if ok:
                        with self.lock:
                            self.latest_jpeg = encoded.tobytes()
                except Exception as exc:
                    print(f"[DASHBOARD] Frame encoding skipped: {exc}")
            remaining = interval - (time.monotonic() - started)
            time.sleep(max(0.01, remaining))

    def start(self):
        publisher = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return  # Keep the terminal clean.

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/?"):
                    body = DASHBOARD_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith("/api/state"):
                    body = json.dumps(publisher.snapshot()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith("/video_feed"):
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while publisher.running:
                            with publisher.lock:
                                jpeg = publisher.latest_jpeg
                            if jpeg:
                                self.wfile.write(
                                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                    + str(len(jpeg)).encode("ascii")
                                    + b"\r\n\r\n" + jpeg + b"\r\n"
                                )
                                self.wfile.flush()
                            time.sleep(1.0 / max(1, DASHBOARD_STREAM_FPS))
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                else:
                    self.send_error(404)

        self.httpd = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        self.httpd.daemon_threads = True
        self.encoder_thread.start()
        self.server_thread = threading.Thread(
            target=self.httpd.serve_forever, name="BeamShift-Web", daemon=True
        )
        self.server_thread.start()
        print(f"[DASHBOARD] Running on port {self.port} (open http://<PI-IP>:{self.port})")
        print("[DASHBOARD] Find the Pi address with: hostname -I")

    def stop(self):
        self.running = False
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
        if self.encoder_thread.is_alive():
            self.encoder_thread.join(timeout=1.0)
        if self.server_thread is not None and self.server_thread.is_alive():
            self.server_thread.join(timeout=1.0)

def main():
    arduino = None
    camera = None
    dashboard = None

    try:
        arduino = ArduinoController()
        net = load_yolo_model()

        camera = Picamera2()
        camera.configure(camera.create_video_configuration(
            main={
                "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                "format": "RGB888",  # Picamera2 RGB888 is BGR byte order for OpenCV
            },
            buffer_count=4,
        ))
        camera.options["queue"] = False
        camera.start()

        # Keep the same camera colour mode that produced the approved preview.
        # Use the camera's automatic white balance/exposure only: no white-card
        # prompt, no manual RGB gains, and no software colour manipulation.
        try:
            camera.set_controls({"AeEnable": True, "AwbEnable": True})
            print("[CAMERA] Automatic white balance/exposure enabled; "
                  "skipping manual white-balance screen.")
        except Exception as control_error:
            print(f"[CAMERA] Camera auto controls unavailable: {control_error}")
        time.sleep(2.0)
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

        if DASHBOARD_ENABLED:
            try:
                dashboard = DashboardPublisher()
                dashboard.start()
            except Exception as dashboard_error:
                print(f"[DASHBOARD] Could not start; vision/control continue unchanged: {dashboard_error}")
                dashboard = None

        while True:
            # Pass camera pixels through unchanged to preserve the approved colour.
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

            # Publish the exact same annotated frame/state to the optional
            # dashboard. Encoding and browser streaming happen on worker threads.
            if dashboard is not None:
                dashboard.publish(display, {
                    "fps": fps,
                    "inference_ms": yolo_ms,
                    "status": status,
                    "vehicle_count": len(detections),
                    "detections": [
                        {"class": d["class"], "confidence": float(d["confidence"])}
                        for d in detections
                    ],
                    "active_spots": sorted(active_spots),
                    "servos": [
                        {
                            "spot": sid,
                            "channel": 6 - sid,
                            "angle": current_angles[sid],
                            "light": bool(arduino.lights[6 - sid]),
                        }
                        for sid in range(1, 6)
                    ],
                })

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
        if dashboard is not None:
            dashboard.stop()
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
