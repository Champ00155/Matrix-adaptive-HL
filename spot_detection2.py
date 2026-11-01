import cv2
import numpy as np
import time
from picamera2 import Picamera2

WIDTH = 640
HEIGHT = 480

# Detection parameters
ROI_TOP = 180
ROI_BOTTOM = 430

TOPHAT_THRESHOLD = 18
MAX_SPOTS = 5

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

print("BEAMSHIFT: VISUAL DETECTION ACTIVE")


def detect_spots(frame):

    # Preserve original camera frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ignore upper screen area
    roi = gray[ROI_TOP:ROI_BOTTOM, :]

    # Enhance small bright objects
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (21, 21)
    )

    tophat = cv2.morphologyEx(
        roi,
        cv2.MORPH_TOPHAT,
        kernel
    )

    # Threshold enhanced image
    _, mask = cv2.threshold(
        tophat,
        TOPHAT_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # Find connected components
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(mask)
    )

    candidates = []

    for i in range(1, count):

        x, y, w, h, area = stats[i]

        # Reject noise and large objects
        if area < 1 or area > 60:
            continue

        if w > 12 or h > 12:
            continue

        cx, cy = centroids[i]

        # Convert ROI coordinates to full-frame coordinates
        cy += ROI_TOP

        candidates.append({
            "x": float(cx),
            "y": float(cy),
            "area": int(area),
            "score": int(tophat[
                y:y+h, x:x+w
            ].sum())
        })

    # Merge detections that are very close together
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

        if len(selected) == MAX_SPOTS:
            break

    # Assign labels from left to right
    selected.sort(key=lambda p: p["x"])

    return selected, mask


try:

    while True:

        frame = camera.capture_array()

        spots, mask = detect_spots(frame)

        display = frame.copy()

        # Show detection region
        cv2.rectangle(
            display,
            (0, ROI_TOP),
            (WIDTH - 1, ROI_BOTTOM),
            (255, 255, 0),
            1
        )

        for index, spot in enumerate(spots):

            x = int(spot["x"])
            y = int(spot["y"])

            cv2.circle(
                display,
                (x, y),
                9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"LED {index + 1}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1
            )

            print(
                f"LED {index + 1}: "
                f"X={x}, Y={y}"
            )

        print(f"DETECTED: {len(spots)}/5")
        print("----------------------")

        cv2.imshow(
            "BEAMSHIFT - FIVE LED DETECTION",
            display
        )

        cv2.imshow(
            "TOPHAT DETECTION MASK",
            mask
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.stop()
    cv2.destroyAllWindows()
