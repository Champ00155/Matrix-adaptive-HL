import cv2
import numpy as np
import time
from picamera2 import Picamera2

WIDTH = 640
HEIGHT = 480

# Start camera
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

print("BEAMSHIFT SPOT DETECTION STARTED")

def detect_spots(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduce isolated pixel noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold based on image brightness
    percentile = np.percentile(blurred, 99.7)
    threshold_value = max(180, min(250, percentile))

    _, mask = cv2.threshold(
        blurred,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # Remove tiny noise
    kernel = np.ones((2, 2), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(mask)
    )

    candidates = []

    for i in range(1, count):

        x, y, w, h, area = stats[i]

        # Reject large bright objects and tiny noise
        if area < 2 or area > 100:
            continue

        if w > 15 or h > 15:
            continue

        cx, cy = centroids[i]

        candidates.append({
            "x": int(cx),
            "y": int(cy),
            "area": int(area)
        })

    # Divide image into five zones.
    # Initially select the brightest compact candidate
    # in each zone.

    selected = []

    zone_width = WIDTH / 5

    for zone in range(5):

        left = zone * zone_width
        right = (zone + 1) * zone_width

        zone_candidates = [
            p for p in candidates
            if left <= p["x"] < right
        ]

        if zone_candidates:

            best = max(
                zone_candidates,
                key=lambda p: p["area"]
            )

            selected.append(best)

        else:
            selected.append(None)

    return selected, mask


try:

    while True:

        frame = camera.capture_array()

        spots, mask = detect_spots(frame)

        display = frame.copy()

        for zone, spot in enumerate(spots):

            if spot is not None:

                x = spot["x"]
                y = spot["y"]

                cv2.circle(
                    display,
                    (x, y),
                    9,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    f"LED {zone + 1}",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1
                )

                print(
                    f"LED {zone + 1}: "
                    f"x={x}, y={y}"
                )

            else:

                print(f"LED {zone + 1}: NOT FOUND")

        cv2.imshow("BEAMSHIFT - SPOT DETECTION", display)
        cv2.imshow("BRIGHTNESS MASK", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.stop()
    cv2.destroyAllWindows()
