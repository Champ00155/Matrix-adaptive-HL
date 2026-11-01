import cv2
import time
from picamera2 import Picamera2

WIDTH = 640
HEIGHT = 480

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={
        "size": (WIDTH, HEIGHT),
        "format": "BGR888"
    },
    buffer_count=4
)

picam2.configure(config)
picam2.start()

print("BEAMSHIFT CAMERA STARTED")

time.sleep(2)

frame_count = 0
start_time = time.perf_counter()

try:
    while True:

        frame = picam2.capture_array()
        frame_count += 1

        elapsed = time.perf_counter() - start_time
        fps = frame_count / elapsed

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("BEAMSHIFT CAMERA TEST", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    picam2.stop()
    cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time

    print("\nCAMERA TEST RESULTS")
    print("Frames:", frame_count)
    print("Average FPS:", frame_count / elapsed)
    print("Duration:", elapsed)




