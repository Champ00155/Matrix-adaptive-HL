import cv2
import numpy as np
import ncnn
import serial
import time

from picamera2 import Picamera2


# =====================================================
# BEAMSHIFT - INTEGRATED VISION AND ACTUATOR CONTROL
# Raspberry Pi 5 + YOLO11n NCNN + Arduino Uno
# =====================================================


# ---------------- CONFIGURATION ----------------

CAM_WIDTH = 640
CAM_HEIGHT = 480

INPUT_SIZE = 320

MODEL_PATH = (
    "/home/pizza/BeamShift/models/"
    "yolo11n_ncnn_model"
)

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

CONF_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45

NUM_ZONES = 5

NEUTRAL_ANGLE = 90
MIN_ANGLE = 45
MAX_ANGLE = 135

# Calibrate these individually.
# These are example directions, NOT verified
# directions for your physical servo mounting.

AVOID_ANGLES = [45, 45, 45, 45, 45]

# COCO vehicle classes
VEHICLE_CLASSES = {
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK"
}

# Apparent vehicle-size thresholds
START_DEFLECTION = 0.04
FULL_DEFLECTION = 0.38

LIGHT_OFF_THRESHOLD = 0.38

# Filtering
SMOOTHING = 0.35

# Minimum fraction of vehicle bounding box
# that must overlap a zone
MIN_ZONE_OVERLAP = 0.18

# Arduino communication
HEARTBEAT_INTERVAL = 0.25

# =====================================================
# ARDUINO COMMUNICATION
# =====================================================

print("\n[BEAMSHIFT] Connecting to Arduino...")

arduino = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=0.05,
    write_timeout=0.2
)

# Arduino resets when serial connection opens
time.sleep(2)

print("[BEAMSHIFT] Arduino connected")

last_heartbeat = 0

# Previous commands sent to Arduino
zone_commands = [None] * NUM_ZONES


def send_zone(zone, angle, light):

    angle = int(
        np.clip(angle, MIN_ANGLE, MAX_ANGLE)
    )

    light = int(light)

    previous = zone_commands[zone]

    if previous is not None:

        if (
            abs(angle - previous[0]) < 2
            and light == previous[1]
        ):
            return

    command = (
        f"Z,{zone + 1},{angle},{light}\n"
    )

    arduino.write(command.encode())

    zone_commands[zone] = [angle, light]


def send_heartbeat():

    global last_heartbeat

    now = time.monotonic()

    if now - last_heartbeat >= HEARTBEAT_INTERVAL:

        arduino.write(b"HB\n")

        last_heartbeat = now


# =====================================================
# NCNN MODEL INITIALIZATION
# =====================================================

print("[BEAMSHIFT] Loading YOLO11n NCNN...")

net = ncnn.Net()

net.opt.use_vulkan_compute = False

ret1 = net.load_param(
    MODEL_PATH + "/model.ncnn.param"
)

ret2 = net.load_model(
    MODEL_PATH + "/model.ncnn.bin"
)

if ret1 != 0 or ret2 != 0:

    raise RuntimeError("NCNN model loading failed")

print("[BEAMSHIFT] YOLO11n loaded successfully")


# =====================================================
# CAMERA INITIALIZATION
# =====================================================

print("[BEAMSHIFT] Starting Pi Camera...")

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={
        "size": (CAM_WIDTH, CAM_HEIGHT),
        "format": "BGR888"
    },
    buffer_count=4
)

picam2.configure(config)

picam2.options["queue"] = False

picam2.start()

time.sleep(2)

print("[BEAMSHIFT] Camera ready")


# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess(frame):

    height, width = frame.shape[:2]

    scale = min(
        INPUT_SIZE / width,
        INPUT_SIZE / height
    )

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    pad_x = (INPUT_SIZE - new_width) // 2
    pad_y = (INPUT_SIZE - new_height) // 2

    canvas = np.full(
        (INPUT_SIZE, INPUT_SIZE, 3),
        114,
        dtype=np.uint8
    )

    canvas[
        pad_y:pad_y + new_height,
        pad_x:pad_x + new_width
    ] = resized

    rgb = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB
    )

    chw = rgb.transpose(2, 0, 1)

    chw = np.ascontiguousarray(
        chw,
        dtype=np.float32
    )

    chw /= 255.0

    return chw, scale, pad_x, pad_y


# =====================================================
# VEHICLE DETECTION
# =====================================================

def detect_vehicles(frame):

    tensor, scale, pad_x, pad_y = preprocess(frame)

    input_mat = ncnn.Mat(tensor)

    extractor = net.create_extractor()

    extractor.input("in0", input_mat)

    ret, output = extractor.extract("out0")

    if ret != 0:

        raise RuntimeError(
            f"NCNN inference failed: {ret}"
        )

    predictions = np.asarray(
        output.numpy(),
        dtype=np.float32
    )

    predictions = np.squeeze(predictions)

    # YOLO11 output: 84 x 2100
    if predictions.shape[0] == 84:

        predictions = predictions.T

    elif predictions.shape[1] != 84:

        raise RuntimeError(
            f"Unexpected output shape: {predictions.shape}"
        )

    boxes = []
    scores = []
    class_ids = []

    for prediction in predictions:

        class_scores = prediction[4:]

        class_id = int(
            np.argmax(class_scores)
        )

        confidence = float(
            class_scores[class_id]
        )

        if class_id not in VEHICLE_CLASSES:
            continue

        if confidence < CONF_THRESHOLD:
            continue

        cx, cy, w, h = prediction[:4]

        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale

        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale

        x1 = int(np.clip(x1, 0, CAM_WIDTH - 1))
        y1 = int(np.clip(y1, 0, CAM_HEIGHT - 1))

        x2 = int(np.clip(x2, 0, CAM_WIDTH - 1))
        y2 = int(np.clip(y2, 0, CAM_HEIGHT - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ])

        scores.append(confidence)
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    detections = []

    if len(indices) > 0:

        for i in np.array(indices).flatten():

            x, y, w, h = boxes[i]

            detections.append({

                "box": (x, y, w, h),

                "class": VEHICLE_CLASSES[class_ids[i]],

                "confidence": scores[i]

            })

    return detections


# =====================================================
# FIVE-ZONE ANALYSIS
# =====================================================

smoothed_heights = [0.0] * NUM_ZONES


def calculate_zone_heights(detections):

    heights = [0.0] * NUM_ZONES

    zone_width = CAM_WIDTH / NUM_ZONES

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

            if overlap / max(w, 1) >= MIN_ZONE_OVERLAP:

                ratio = h / CAM_HEIGHT

                heights[zone] = max(
                    heights[zone],
                    ratio
                )

    return heights


# =====================================================
# ADAPTIVE SERVO AND LIGHT CONTROL
# =====================================================

def update_lighting(raw_heights):

    global smoothed_heights

    for zone in range(NUM_ZONES):

        raw = raw_heights[zone]

        previous = smoothed_heights[zone]

        current = (
            (1 - SMOOTHING) * previous
            + SMOOTHING * raw
        )

        smoothed_heights[zone] = current

        # No vehicle detected
        if current < 0.025:

            angle = NEUTRAL_ANGLE

            light = 1

        else:

            progress = (
                (current - START_DEFLECTION)
                /
                (FULL_DEFLECTION - START_DEFLECTION)
            )

            progress = float(
                np.clip(progress, 0.15, 1.0)
            )

            target_angle = AVOID_ANGLES[zone]

            angle = round(
                NEUTRAL_ANGLE
                + progress * (
                    target_angle - NEUTRAL_ANGLE
                )
            )

            angle = int(
                np.clip(
                    angle,
                    MIN_ANGLE,
                    MAX_ANGLE
                )
            )

            # Switch output OFF when the
            # vehicle occupies a large image area
            light = int(
                current < LIGHT_OFF_THRESHOLD
            )

        send_zone(
            zone,
            angle,
            light
        )


# =====================================================
# DISPLAY
# =====================================================

def draw_interface(frame, detections, fps, latency):

    zone_width = CAM_WIDTH / NUM_ZONES

    # Draw five zones
    for zone in range(NUM_ZONES):

        x = int(zone * zone_width)

        cv2.line(
            frame,
            (x, 0),
            (x, CAM_HEIGHT),
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

    # Draw detections
    for detection in detections:

        x, y, w, h = detection["box"]

        zone = min(
            int((x + w / 2) / zone_width),
            4
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        label = (
            f"{detection['class']} "
            f"{detection['confidence']:.2f} "
            f"Z{zone + 1}"
        )

        cv2.putText(
            frame,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, CAM_HEIGHT - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Latency: {latency:.1f} ms",
        (10, CAM_HEIGHT - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    return frame


# =====================================================
# MAIN EXECUTION
# =====================================================

print("\n======================================")
print("       BEAMSHIFT SYSTEM ACTIVE")
print("======================================")

frame_count = 0

start_time = time.perf_counter()

inference_times = []

try:

    while True:

        frame = picam2.capture_array()

        inference_start = time.perf_counter()

        detections = detect_vehicles(frame)

        inference_end = time.perf_counter()

        latency = (
            inference_end - inference_start
        ) * 1000

        inference_times.append(latency)

        frame_count += 1

        # Analyse vehicle positions
        raw_heights = calculate_zone_heights(
            detections
        )

        # Update servo and light commands
        update_lighting(raw_heights)

        # Keep Arduino connection alive
        send_heartbeat()

        elapsed = (
            time.perf_counter() - start_time
        )

        fps = frame_count / max(elapsed, 0.001)

        display = draw_interface(
            frame,
            detections,
            fps,
            latency
        )

        cv2.imshow(
            "BEAMSHIFT - ADAPTIVE HEADLIGHT",
            display
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


except KeyboardInterrupt:

    print("\n[BEAMSHIFT] Stopped by user")


finally:

    print("[BEAMSHIFT] Shutting down...")

    # Switch all outputs OFF
    # and return servos to neutral

    for zone in range(NUM_ZONES):

        try:

            arduino.write(
                f"Z,{zone + 1},90,0\n".encode()
            )

        except serial.SerialException:
            pass

    time.sleep(0.1)

    picam2.stop()

    cv2.destroyAllWindows()

    arduino.close()

    if inference_times:

        elapsed = max(
            time.perf_counter() - start_time,
            0.001
        )

        print("\n========== RESULTS ==========")

        print("Frames:", frame_count)

        print(
            "Average FPS:",
            round(frame_count / elapsed, 2)
        )

        print(
            "Average inference latency:",
            round(np.mean(inference_times), 2),
            "ms"
        )

        print("=============================")

    print("[BEAMSHIFT] Shutdown complete")
