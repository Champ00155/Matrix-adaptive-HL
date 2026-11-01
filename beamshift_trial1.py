import cv2
import numpy as np
import ncnn
import serial
import time

from picamera2 import Picamera2


# ==================================================
# BEAMSHIFT CONFIGURATION
# ==================================================

WIDTH = 640
HEIGHT = 480
INPUT_SIZE = 320

MODEL_PATH = (
    "/home/pizza/BeamShift/models/"
    "yolo11n_ncnn_model"
)

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

CONFIDENCE = 0.40
NMS_THRESHOLD = 0.45

NUM_ZONES = 5

MIN_ANGLE = 45
MAX_ANGLE = 135
NEUTRAL = 90

# Camera-zone order, not hardware order.
# Calibrate directions individually.
AVOID_ANGLES = [45, 45, 45, 45, 45]

VEHICLE_CLASSES = {
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK"
}


# ==================================================
# STABILITY PARAMETERS
# ==================================================

# Lower = more filtering
SMOOTHING = 0.20

# Vehicle must be absent this long before reset
RELEASE_TIMEOUT = 0.80

# Ignore extremely small detections
MIN_HEIGHT = 0.025

# Apparent size thresholds
START_DEFLECTION = 0.05
FULL_DEFLECTION = 0.38

# LED output switches OFF above this size
LIGHT_OFF_THRESHOLD = 0.38

HEARTBEAT_INTERVAL = 0.25


# ==================================================
# SERIAL CONNECTION
# ==================================================

print("[BEAMSHIFT] Connecting to Arduino...")

arduino = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=0.05,
    write_timeout=0.2
)

time.sleep(2)

print("[BEAMSHIFT] Arduino connected")


# Each entry is indexed by CAMERA ZONE.
zone_commands = [None] * 5

last_heartbeat = 0


def send_heartbeat():

    global last_heartbeat

    now = time.monotonic()

    if now - last_heartbeat >= HEARTBEAT_INTERVAL:

        arduino.write(b"HB\n")

        last_heartbeat = now


def send_zone(zone, angle, light):

    # REVERSE PHYSICAL MAPPING:
    # Camera Zone 1 -> Hardware 5
    # Camera Zone 5 -> Hardware 1

    hardware_channel = 4 - zone

    angle = int(np.clip(
        angle,
        MIN_ANGLE,
        MAX_ANGLE
    ))

    light = int(light)

    previous = zone_commands[zone]

    if previous is not None:

        if (
            previous[0] == angle
            and previous[1] == light
        ):
            return

    command = (
        f"Z,{hardware_channel + 1},{angle},{light}\n"
    )

    arduino.write(command.encode())

    zone_commands[zone] = [angle, light]


# ==================================================
# NCNN MODEL
# ==================================================

print("[BEAMSHIFT] Loading YOLO11n NCNN...")

net = ncnn.Net()

net.opt.use_vulkan_compute = False

r1 = net.load_param(
    MODEL_PATH + "/model.ncnn.param"
)

r2 = net.load_model(
    MODEL_PATH + "/model.ncnn.bin"
)

if r1 != 0 or r2 != 0:

    raise RuntimeError("Could not load NCNN model")

print("[BEAMSHIFT] Model loaded")


# ==================================================
# CAMERA
# ==================================================

print("[BEAMSHIFT] Starting camera...")

camera = Picamera2()

config = camera.create_video_configuration(
    main={
        "size": (WIDTH, HEIGHT),
        "format": "BGR888"
    },
    buffer_count=4
)

camera.configure(config)

camera.options["queue"] = False

camera.start()

time.sleep(2)

print("[BEAMSHIFT] Camera ready")


# ==================================================
# PREPROCESSING
# ==================================================

def preprocess(frame):

    h, w = frame.shape[:2]

    scale = min(
        INPUT_SIZE / w,
        INPUT_SIZE / h
    )

    nw = int(round(w * scale))
    nh = int(round(h * scale))

    resized = cv2.resize(
        frame,
        (nw, nh)
    )

    px = (INPUT_SIZE - nw) // 2
    py = (INPUT_SIZE - nh) // 2

    canvas = np.full(
        (INPUT_SIZE, INPUT_SIZE, 3),
        114,
        dtype=np.uint8
    )

    canvas[py:py+nh, px:px+nw] = resized

    rgb = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB
    )

    tensor = rgb.transpose(2, 0, 1)

    tensor = np.ascontiguousarray(
        tensor,
        dtype=np.float32
    )

    tensor /= 255.0

    return tensor, scale, px, py


# ==================================================
# DETECTION
# ==================================================

def detect_vehicles(frame):

    tensor, scale, px, py = preprocess(frame)

    mat = ncnn.Mat(tensor)

    extractor = net.create_extractor()

    extractor.input("in0", mat)

    ret, output = extractor.extract("out0")

    if ret != 0:
        raise RuntimeError("NCNN inference failed")

    predictions = np.asarray(
        output.numpy(),
        dtype=np.float32
    )

    predictions = np.squeeze(predictions)

    if predictions.shape[0] == 84:
        predictions = predictions.T

    elif predictions.shape[1] != 84:
        raise RuntimeError(
            f"Unexpected output: {predictions.shape}"
        )

    boxes = []
    scores = []
    classes = []

    for p in predictions:

        class_scores = p[4:]

        class_id = int(np.argmax(class_scores))

        confidence = float(class_scores[class_id])

        if class_id not in VEHICLE_CLASSES:
            continue

        if confidence < CONFIDENCE:
            continue

        cx, cy, bw, bh = p[:4]

        x1 = (cx - bw / 2 - px) / scale
        y1 = (cy - bh / 2 - py) / scale

        x2 = (cx + bw / 2 - px) / scale
        y2 = (cy + bh / 2 - py) / scale

        x1 = int(np.clip(x1, 0, WIDTH - 1))
        y1 = int(np.clip(y1, 0, HEIGHT - 1))

        x2 = int(np.clip(x2, 0, WIDTH - 1))
        y2 = int(np.clip(y2, 0, HEIGHT - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([
            x1, y1, x2-x1, y2-y1
        ])

        scores.append(confidence)
        classes.append(class_id)

    detections = []

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONFIDENCE,
        NMS_THRESHOLD
    )

    if len(indices) > 0:

        for i in np.array(indices).flatten():

            x, y, w, h = boxes[i]

            detections.append({
                "box": (x, y, w, h),
                "class": VEHICLE_CLASSES[classes[i]],
                "confidence": scores[i]
            })

    return detections


# ==================================================
# ZONE DETECTION
# ==================================================

def calculate_zone_heights(detections):

    heights = [0.0] * NUM_ZONES

    zone_width = WIDTH / NUM_ZONES

    for detection in detections:

        x, y, w, h = detection["box"]

        x2 = x + w

        for zone in range(NUM_ZONES):

            left = zone * zone_width
            right = (zone + 1) * zone_width

            overlap = max(
                0,
                min(x2, right) - max(x, left)
            )

            if overlap / max(w, 1) >= 0.18:

                heights[zone] = max(
                    heights[zone],
                    h / HEIGHT
                )

    return heights


# ==================================================
# STABLE ZONE CONTROLLER
# ==================================================

filtered_height = [0.0] * 5

last_seen = [0.0] * 5

zone_active = [False] * 5

current_angle = [90.0] * 5

target_angle = [90.0] * 5

current_light = [1] * 5

last_update = time.monotonic()


def update_control(raw_heights):

    global last_update

    now = time.monotonic()

    dt = min(
        now - last_update,
        0.1
    )

    last_update = now

    for zone in range(5):

        raw = raw_heights[zone]

        # --------------------------------
        # DETECTION HYSTERESIS
        # --------------------------------

        if raw >= MIN_HEIGHT:

            last_seen[zone] = now

            if not zone_active[zone]:

                zone_active[zone] = True

                filtered_height[zone] = raw

            else:

                filtered_height[zone] += (
                    SMOOTHING
                    *
                    (raw - filtered_height[zone])
                )

        elif now - last_seen[zone] > RELEASE_TIMEOUT:

            zone_active[zone] = False

            filtered_height[zone] = 0.0

        # --------------------------------
        # TARGET ANGLE
        # --------------------------------

        if zone_active[zone]:

            h = filtered_height[zone]

            progress = (
                (h - START_DEFLECTION)
                /
                (FULL_DEFLECTION - START_DEFLECTION)
            )

            progress = float(
                np.clip(progress, 0.0, 1.0)
            )

            target_angle[zone] = (
                NEUTRAL
                +
                progress
                *
                (AVOID_ANGLES[zone] - NEUTRAL)
            )

            current_light[zone] = int(
                h < LIGHT_OFF_THRESHOLD
            )

        else:

            # Return only after confirmed disappearance
            target_angle[zone] = NEUTRAL

            current_light[zone] = 1

        # --------------------------------
        # SEND TARGET TO ARDUINO
        # --------------------------------

        if abs(
            target_angle[zone] - current_angle[zone]
        ) > 0.5:

            # Pi-side movement limiter
            max_step = 35.0 * dt

            difference = (
                target_angle[zone]
                -
                current_angle[zone]
            )

            step = np.clip(
                difference,
                -max_step,
                max_step
            )

            current_angle[zone] += step

        else:

            current_angle[zone] = target_angle[zone]

        send_zone(
            zone,
            round(current_angle[zone]),
            current_light[zone]
        )


# ==================================================
# DISPLAY
# ==================================================

def draw_display(frame, detections, fps, latency):

    zone_width = WIDTH / 5

    for zone in range(5):

        x = int(zone * zone_width)

        cv2.line(
            frame,
            (x, 0),
            (x, HEIGHT),
            (255, 255, 0),
            1
        )

        cv2.putText(
            frame,
            f"ZONE {zone + 1}",
            (x + 5, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1
        )

        cv2.putText(
            frame,
            f"SERVO {5-zone}: {current_angle[zone]:.0f}",
            (x + 5, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1
        )

    for detection in detections:

        x, y, w, h = detection["box"]

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            detection["class"],
            (x, max(20, y-8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, HEIGHT-40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Latency: {latency:.1f} ms",
        (10, HEIGHT-12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    return frame


# ==================================================
# MAIN LOOP
# ==================================================

print("\n========== BEAMSHIFT ACTIVE ==========")

frame_count = 0

start_time = time.perf_counter()

try:

    while True:

        frame = camera.capture_array()

        inference_start = time.perf_counter()

        detections = detect_vehicles(frame)

        latency = (
            time.perf_counter() - inference_start
        ) * 1000

        raw_heights = calculate_zone_heights(
            detections
        )

        update_control(raw_heights)

        send_heartbeat()

        frame_count += 1

        elapsed = (
            time.perf_counter() - start_time
        )

        fps = frame_count / max(elapsed, 0.001)

        display = draw_display(
            frame,
            detections,
            fps,
            latency
        )

        cv2.imshow(
            "BEAMSHIFT ADAPTIVE HEADLIGHT",
            display
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


except KeyboardInterrupt:

    print("\nStopping BeamShift")


finally:

    print("Shutting down...")

    # Return to neutral on intentional shutdown,
    # and switch all LED outputs OFF.

    for zone in range(5):

        hardware_channel = 4 - zone

        try:

            arduino.write(
                f"Z,{hardware_channel + 1},90,0\n".encode()
            )

        except serial.SerialException:
            pass

    time.sleep(0.2)

    camera.stop()

    cv2.destroyAllWindows()

    arduino.close()

    print("BeamShift stopped")
