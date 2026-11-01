import cv2
import numpy as np
import ncnn
import time

from picamera2 import Picamera2


# ==========================================
# BEAMSHIFT CONFIGURATION
# ==========================================

CAM_WIDTH = 640
CAM_HEIGHT = 480

INPUT_SIZE = 320

MODEL_PATH = "/home/pizza/BeamShift/models/yolo11n_ncnn_model"

CONF_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45

# COCO vehicle classes
VEHICLE_CLASSES = {
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK"
}

NUM_ZONES = 5


# ==========================================
# INITIALIZE NCNN
# ==========================================

print("\nBEAMSHIFT INITIALIZING...")

net = ncnn.Net()

net.opt.use_vulkan_compute = False

param_result = net.load_param(
    MODEL_PATH + "/model.ncnn.param"
)

model_result = net.load_model(
    MODEL_PATH + "/model.ncnn.bin"
)

if param_result != 0 or model_result != 0:
    raise RuntimeError("NCNN model loading failed")

print("YOLO11n NCNN loaded successfully")


# ==========================================
# INITIALIZE CAMERA
# ==========================================

picam2 = Picamera2()

camera_config = picam2.create_video_configuration(
    main={
        "size": (CAM_WIDTH, CAM_HEIGHT),
        "format": "BGR888"
    },
    buffer_count=4
)

picam2.configure(camera_config)

picam2.options["queue"] = False

picam2.start()

time.sleep(2)

print("PI CAMERA STARTED")
print("BEAMSHIFT VISION ACTIVE")


# ==========================================
# FIVE-ZONE MAPPING
# ==========================================

def get_zone(center_x):

    zone_width = CAM_WIDTH / NUM_ZONES

    zone = int(center_x / zone_width)

    return max(0, min(zone, NUM_ZONES - 1))


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

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

    # Camera gives BGR; YOLO expects RGB
    rgb = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB
    )

    # Convert to normalized CHW float32
    chw = rgb.transpose(2, 0, 1)

    chw = np.ascontiguousarray(
        chw,
        dtype=np.float32
    )

    chw /= 255.0

    return chw, scale, pad_x, pad_y


# ==========================================
# YOLO11n INFERENCE
# ==========================================

def detect_vehicles(frame):

    input_tensor, scale, pad_x, pad_y = preprocess(frame)

    input_mat = ncnn.Mat(input_tensor)

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

    # Expected YOLO11 output:
    # 84 classes/features x 2100 predictions

    if predictions.ndim != 2:
        raise RuntimeError(
            f"Unexpected output dimensions: {predictions.shape}"
        )

    if predictions.shape[0] == 84:
        predictions = predictions.T

    elif predictions.shape[1] != 84:
        raise RuntimeError(
            f"Unexpected YOLO output shape: {predictions.shape}"
        )

    boxes = []
    scores = []
    class_ids = []

    for prediction in predictions:

        # YOLO11 output:
        # cx, cy, width, height, 80 class scores

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

        # Convert from model coordinates
        # to original camera coordinates

        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale

        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale

        x1 = max(0, min(CAM_WIDTH - 1, int(x1)))
        y1 = max(0, min(CAM_HEIGHT - 1, int(y1)))

        x2 = max(0, min(CAM_WIDTH - 1, int(x2)))
        y2 = max(0, min(CAM_HEIGHT - 1, int(y2)))

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

    # Non-maximum suppression
    # Removes duplicate detections

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    detections = []

    if len(indices) > 0:

        for index in np.array(indices).flatten():

            x, y, w, h = boxes[index]

            center_x = x + w / 2

            zone = get_zone(center_x)

            detections.append({
                "class": VEHICLE_CLASSES[class_ids[index]],
                "confidence": scores[index],
                "box": (x, y, w, h),
                "center_x": center_x,
                "zone": zone
            })

    return detections


# ==========================================
# DISPLAY AND VISUALIZATION
# ==========================================

def draw_interface(frame, detections, fps, latency):

    zone_width = CAM_WIDTH / NUM_ZONES

    # Draw five horizontal zones

    for i in range(NUM_ZONES):

        x = int(i * zone_width)

        cv2.line(
            frame,
            (x, 0),
            (x, CAM_HEIGHT),
            (255, 255, 0),
            1
        )

        cv2.putText(
            frame,
            f"ZONE {i + 1}",
            (x + 5, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1
        )

    cv2.line(
        frame,
        (CAM_WIDTH - 1, 0),
        (CAM_WIDTH - 1, CAM_HEIGHT),
        (255, 255, 0),
        1
    )

    # Draw detected vehicles

    for detection in detections:

        x, y, w, h = detection["box"]

        label = detection["class"]

        confidence = detection["confidence"]

        zone = detection["zone"]

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        text = (
            f"{label} "
            f"{confidence:.2f} "
            f"Z{zone + 1}"
        )

        cv2.putText(
            frame,
            text,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Performance information

    cv2.rectangle(
        frame,
        (0, CAM_HEIGHT - 65),
        (260, CAM_HEIGHT),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, CAM_HEIGHT - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Latency: {latency:.1f} ms",
        (10, CAM_HEIGHT - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Vehicles: {len(detections)}",
        (CAM_WIDTH - 155, CAM_HEIGHT - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    return frame


# ==========================================
# MAIN LOOP
# ==========================================

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

        elapsed = (
            inference_end - start_time
        )

        fps = frame_count / elapsed

        # Print vehicle positions

        for detection in detections:

            print(
                f"{detection['class']} | "
                f"Confidence: {detection['confidence']:.2f} | "
                f"Center X: {detection['center_x']:.0f} | "
                f"Zone: {detection['zone'] + 1}"
            )

        display = draw_interface(
            frame,
            detections,
            fps,
            latency
        )

        cv2.imshow(
            "BEAMSHIFT - ADAPTIVE HEADLIGHT VISION",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


except KeyboardInterrupt:
    print("\nBEAMSHIFT STOPPED")


finally:

    picam2.stop()

    cv2.destroyAllWindows()

    if inference_times:

        print("\n========== BEAMSHIFT RESULTS ==========")

        print("Frames processed:", frame_count)

        print(
            "Average FPS:",
            round(frame_count / max(
                time.perf_counter() - start_time,
                0.001
            ), 2)
        )

        print(
            "Average inference latency:",
            round(np.mean(inference_times), 2),
            "ms"
        )

        print(
            "Maximum inference latency:",
            round(np.max(inference_times), 2),
            "ms"
        )

        print("========================================")

