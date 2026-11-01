python
from picamera2 import Picamera2
import cv2

# Initialize camera
picam2 = Picamera2()

# Configure camera
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)

picam2.configure(config)

# Start camera
picam2.start()

print("Camera started!")
print("Press Q or ESC to exit.")

try:
    while True:

        # Capture frame
        frame = picam2.capture_array()

        # Display camera output
        cv2.imshow("Raspberry Pi 5 - Live Camera", frame)

        # Exit on Q or ESC
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera stopped.")

