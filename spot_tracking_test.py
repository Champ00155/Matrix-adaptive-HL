import time
import threading
from collections import OrderedDict

import cv2
import numpy as np
import serial
from picamera2 import Picamera2

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
WIDTH, HEIGHT = 640, 480

ROI_TOP, ROI_BOTTOM = 180, 430
TOPHAT_KERNEL_SIZE = 21
TOPHAT_THRESHOLD = 18
MIN_AREA, MAX_AREA = 1, 60
MAX_WIDTH_HEIGHT = 12

TRACK_MAX_JUMP_PX = 220
SERVO_MIN, SERVO_MAX, SERVO_NEUTRAL = 45, 135, 90
SERVO_STEP = 2
HEARTBEAT_INTERVAL = 0.25


class ArduinoLink:
    def __init__(self):
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05,
                                 write_timeout=0.2)
        time.sleep(2.0)  # Arduino resets when serial port opens
        self.lock = threading.Lock()
        self.running = True
        self.last_hb = 0.0
        self.angles = [SERVO_NEUTRAL] * 5
        self.lights = [1] * 5
        self.thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.thread.start()

    def _write(self, message):
        with self.lock:
            self.ser.write((message + "\n").encode("ascii"))
            self.ser.flush()

    def _heartbeat_loop(self):
        while self.running:
            now = time.monotonic()
            if now - self.last_hb >= HEARTBEAT_INTERVAL:
                try:
                    self._write("HB")
                    self.last_hb = now
                except serial.SerialException:
                    pass
            time.sleep(0.03)

    def set_channel(self, channel, angle=None, light=None):
        idx = channel - 1
        if idx not in range(5):
            raise ValueError("Arduino channel must be 1..5")
        if angle is not None:
            self.angles[idx] = int(np.clip(angle, SERVO_MIN, SERVO_MAX))
        if light is not None:
            self.lights[idx] = int(bool(light))
        self._write(f"Z,{channel},{self.angles[idx]},{self.lights[idx]}")

    def set_all_angles(self, angle):
        for ch in range(1, 6):
            self.set_channel(ch, angle=angle)

    def set_all_lights(self, state):
        for ch in range(1, 6):
            self.set_channel(ch, light=state)

    def close(self):
        try:
            self.set_all_angles(SERVO_NEUTRAL)
            self.set_all_lights(0)
            time.sleep(0.5)
        except Exception:
            pass
        self.running = False
        self.thread.join(timeout=1.0)
        self.ser.close()


def detect_spots(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[max(0, ROI_TOP):min(frame.shape[0], ROI_BOTTOM), :]
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
        if w > MAX_WIDTH_HEIGHT or h > MAX_WIDTH_HEIGHT:
            continue
        cx, cy = centroids[i]
        score = float(top_hat[labels == i].sum())
        candidates.append({
            "x": float(cx), "y": float(cy + max(0, ROI_TOP)),
            "bbox": (int(x), int(y + max(0, ROI_TOP)), int(w), int(h)),
            "score": score
        })

    candidates.sort(key=lambda d: d["score"], reverse=True)
    selected = []
    for d in candidates:
        if all(np.hypot(d["x"] - p["x"], d["y"] - p["y"]) >= 10
               for p in selected):
            selected.append(d)
        if len(selected) == 5:
            break
    return selected, mask


class SpotTracker:
    def __init__(self):
        self.tracks = OrderedDict()
        self.initialized = False

    def update(self, detections):
        if not self.initialized:
            if len(detections) >= 5:
                for sid, d in enumerate(sorted(detections, key=lambda x: x["x"]), 1):
                    self.tracks[sid] = {
                        "x": d["x"], "y": d["y"], "visible": True
                    }
                self.initialized = True
            return

        pairs = []
        for sid, track in self.tracks.items():
            for di, d in enumerate(detections):
                dist = np.hypot(d["x"] - track["x"], d["y"] - track["y"])
                if dist <= TRACK_MAX_JUMP_PX:
                    pairs.append((dist, sid, di))
        pairs.sort()

        used_spots, used_detections = set(), set()
        for _, sid, di in pairs:
            if sid in used_spots or di in used_detections:
                continue
            d = detections[di]
            self.tracks[sid].update({"x": d["x"], "y": d["y"], "visible": True})
            used_spots.add(sid)
            used_detections.add(di)

        for sid in self.tracks:
            if sid not in used_spots:
                self.tracks[sid]["visible"] = False

    def ready(self):
        return self.initialized and len(self.tracks) == 5


def hardware_channel(spot_id):
    return 6 - spot_id  # reversed physical wiring


def main():
    arduino = None
    camera = None
    try:
        print("Connecting to Arduino on", SERIAL_PORT)
        arduino = ArduinoLink()

        camera = Picamera2()
        camera.configure(camera.create_video_configuration(
            main={"size": (WIDTH, HEIGHT), "format": "BGR888"}
        ))
        camera.start()
        time.sleep(1)

        tracker = SpotTracker()
        selected = 1
        angles = {sid: SERVO_NEUTRAL for sid in range(1, 6)}

        arduino.set_all_angles(SERVO_NEUTRAL)
        arduino.set_all_lights(1)

        print("\nWaiting for all five spots...")
        print("1-5 select | A/D move -/+2 degrees | R neutral | L lights ON")
        print("K lights OFF | Q/ESC quit")

        while True:
            frame = camera.capture_array()
            detections, mask = detect_spots(frame)
            tracker.update(detections)
            view = frame.copy()

            if tracker.ready():
                for sid, t in tracker.tracks.items():
                    if t["visible"]:
                        pt = (int(t["x"]), int(t["y"]))
                        colour = (0, 255, 255) if sid == selected else (0, 255, 0)
                        cv2.circle(view, pt, 11, colour, 2)
                        cv2.putText(view, f"SPOT {sid}", (pt[0] + 12, pt[1] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1)

            status = "TRACKING READY" if tracker.ready() else (
                f"INITIALIZING: {len(detections)}/5 spots detected"
            )
            cv2.putText(view, status, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (0, 255, 0), 2)
            cv2.putText(view,
                        f"Selected Spot {selected} | Arduino CH {hardware_channel(selected)}"
                        f" | Servo {angles[selected]} deg",
                        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (255, 255, 255), 1)
            cv2.imshow("BeamShift - Persistent Spot Tracking", view)
            cv2.imshow("Spot Detector Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif ord("1") <= key <= ord("5"):
                selected = key - ord("0")
            elif key in (ord("a"), ord("A"), ord("d"), ord("D")):
                if not tracker.ready():
                    print("Wait until all five spots are detected.")
                    continue
                step = -SERVO_STEP if key in (ord("a"), ord("A")) else SERVO_STEP
                angles[selected] = int(np.clip(
                    angles[selected] + step, SERVO_MIN, SERVO_MAX
                ))
                ch = hardware_channel(selected)
                arduino.set_channel(ch, angle=angles[selected], light=1)
                print(f"Spot {selected} / Arduino CH {ch} -> {angles[selected]} degrees")
            elif key in (ord("r"), ord("R")):
                angles = {sid: SERVO_NEUTRAL for sid in range(1, 6)}
                arduino.set_all_angles(SERVO_NEUTRAL)
                print("All servos commanded to 90 degrees.")
            elif key in (ord("l"), ord("L")):
                arduino.set_all_lights(1)
                print("All light outputs ON.")
            elif key in (ord("k"), ord("K")):
                arduino.set_all_lights(0)
                print("All light outputs OFF.")

    finally:
        if camera is not None:
            camera.stop()
        cv2.destroyAllWindows()
        if arduino is not None:
            arduino.close()
        print("Closed. Light outputs OFF; servos commanded to neutral.")


if __name__ == "__main__":
    main()
