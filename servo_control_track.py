import cv2
import numpy as np
import serial
import time
import json

from picamera2 import Picamera2

# ==========================================
# BEAMSHIFT SERVO CALIBRATION
# ==========================================

WIDTH = 640
HEIGHT = 480

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

NEUTRAL = 90
TEST_ANGLE = 95

ROI_TOP = 180
ROI_BOTTOM = 430

TOPHAT_THRESHOLD = 18

# ==========================================
# CAMERA
# ==========================================

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

# ==========================================
# ARDUINO
# ==========================================

arduino = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=0.05,
    write_timeout=0.2
)

time.sleep(2)

last_heartbeat = 0


def heartbeat():

    global last_heartbeat

    now = time.monotonic()

    if now - last_heartbeat >= 0.25:

        arduino.write(b"HB\n")
        last_heartbeat = now


def send_command(channel, angle, light=1):

    command = f"Z,{channel},{angle},{light}\n"

    arduino.write(command.encode())

    heartbeat()


# ==========================================
# SPOT DETECTION
# ==========================================

def detect_spots(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    roi = gray[ROI_TOP:ROI_BOTTOM, :]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (21, 21)
    )

    tophat = cv2.morphologyEx(
        roi,
        cv2.MORPH_TOPHAT,
        kernel
    )

    _, mask = cv2.threshold(
        tophat,
        TOPHAT_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(mask)
    )

    candidates = []

    for i in range(1, count):

        x, y, w, h, area = stats[i]

        if area < 1 or area > 60:
            continue

        if w > 12 or h > 12:
            continue

        cx, cy = centroids[i]

        cy += ROI_TOP

        candidates.append({
            "x": float(cx),
            "y": float(cy),
            "area": int(area),
            "score": int(tophat[y:y+h, x:x+w].sum())
        })

    candidates.sort(
        key=lambda p: p["score"],
        reverse=True
    )

    selected = []

    for candidate in candidates:

        duplicate = False

        for existing in selected:

            distance = np.hypot(
                candidate["x"] - existing["x"],
                candidate["y"] - existing["y"]
            )

            if distance < 10:
                duplicate = True
                break

        if not duplicate:
            selected.append(candidate)

        if len(selected) == 5:
            break

    selected.sort(key=lambda p: p["x"])

    return selected, mask


# ==========================================
# STABLE POSITION MEASUREMENT
# ==========================================

def measure_spots():

    measurements = []

    for _ in range(8):

        heartbeat()

        frame = camera.capture_array()

        spots, mask = detect_spots(frame)

        if len(spots) != 5:

            print(
                f"WARNING: Only {len(spots)}/5 spots detected"
            )

            return None

        measurements.append([
            [p["x"], p["y"]]
            for p in spots
        ])

        time.sleep(0.04)

    return np.median(
        np.array(measurements),
        axis=0
    )


# ==========================================
# INITIALISE
# ==========================================

calibration = {
    "neutral_angle": NEUTRAL,
    "test_angle": TEST_ANGLE,
    "spots": {}
}

print("\n================================")
print("BEAMSHIFT SERVO CALIBRATION")
print("================================")

print("\nSwitching all channels ON at neutral.")

for channel in range(1, 6):

    send_command(channel, NEUTRAL, 1)

time.sleep(2)

print("\nDetecting initial LED positions...")

baseline = measure_spots()

if baseline is None:

    raise RuntimeError(
        "Five spots were not detected. Calibration aborted."
    )

print("\nInitial positions:")

for i, point in enumerate(baseline):

    print(
        f"LED {i+1}: "
        f"X={point[0]:.1f}, Y={point[1]:.1f}"
    )


# ==========================================
# CALIBRATE EACH SERVO
# ==========================================

try:

    for channel in range(1, 6):

        print("\n--------------------------------")
        print(f"Testing HARDWARE SERVO {channel}")
        print("--------------------------------")

        print("Keep the screen and camera stationary.")

        input("Press ENTER to move this servo by +5 degrees...")

        heartbeat()

        send_command(channel, TEST_ANGLE, 1)

        # Allow servo to settle
        time.sleep(1)

        moved = measure_spots()

        # Return to neutral
        send_command(channel, NEUTRAL, 1)

        time.sleep(1)

        if moved is None:

            print("Calibration failed for this servo.")
            continue

        # Hardware channel 5 corresponds to leftmost spot.
        # Hardware channel 1 corresponds to rightmost spot.

        spot_index = 5 - channel

        before = baseline[spot_index]
        after = moved[spot_index]

        dx = float(after[0] - before[0])
        dy = float(after[1] - before[1])

        pixels_per_degree_x = dx / (
            TEST_ANGLE - NEUTRAL
        )

        pixels_per_degree_y = dy / (
            TEST_ANGLE - NEUTRAL
        )

        calibration["spots"][str(channel)] = {

            "spot_id": spot_index + 1,

            "initial_x": float(before[0]),
            "initial_y": float(before[1]),

            "moved_x": float(after[0]),
            "moved_y": float(after[1]),

            "dx": dx,
            "dy": dy,

            "pixels_per_degree_x":
                pixels_per_degree_x,

            "pixels_per_degree_y":
                pixels_per_degree_y
        }

        print("\nCALIBRATION RESULT")

        print(f"Associated spot: LED {spot_index+1}")

        print(f"X displacement: {dx:.2f} pixels")
        print(f"Y displacement: {dy:.2f} pixels")

        print(
            f"X pixels/degree: "
            f"{pixels_per_degree_x:.2f}"
        )

        print(
            f"Y pixels/degree: "
            f"{pixels_per_degree_y:.2f}"
        )

finally:

    print("\nReturning all servos to neutral...")

    for channel in range(1, 6):

        send_command(channel, NEUTRAL, 1)

    time.sleep(1)

    arduino.close()

    camera.stop()

    cv2.destroyAllWindows()

# ==========================================
# SAVE RESULTS
# ==========================================

with open("beamshift_calibration.json", "w") as file:

    json.dump(
        calibration,
        file,
        indent=4
    )

print("\n================================")
print("CALIBRATION COMPLETE")
print("Saved: beamshift_calibration.json")
print("================================")
