import cv2
import time

from Nuscenes.loader import NuScenesLoader
from Detection.detector import DetectorPipeline
from Detection.drawer import DrawEngine
from config import CAMERA_CHANNEL, OUTPUT_PATH, FRAME_RATE

# ---------------------------------------
# Initialize core modules
# ---------------------------------------
nusc_loader = NuScenesLoader()
detector = DetectorPipeline()
drawer = DrawEngine()

# ---------------------------------------
# Video writer
# ---------------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = None

start_time = time.time()
frame_count = 0

# ---------------------------------------
# Main Loop
# ---------------------------------------
for img_path in nusc_loader.load_scene_frames(CAMERA_CHANNEL):

    frame = cv2.imread(img_path)
    if frame is None:
        continue

    if video_writer is None:
        h, w = frame.shape[:2]
        video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FRAME_RATE, (w, h))

    frame_count += 1

    # ------------------------------
    # YOLO + Faster R-CNN pipeline
    # ------------------------------
    tracks = detector.process(frame)

    # ------------------------------
    # Draw detections
    # ------------------------------
    frame_out = drawer.draw(frame, tracks)

    # ------------------------------
    # Display & Save
    # ------------------------------
    cv2.imshow("YOLO + Faster R-CNN Tracking", frame_out)
    video_writer.write(frame_out)

    key = cv2.waitKey(25) & 0xFF
    if key == 27:
        break
    if key == ord(' '):
        print("Paused. Press space to resume.")
        while True:
            if cv2.waitKey(0) & 0xFF == ord(' '):
                break

# ---------------------------------------
# Cleanup
# ---------------------------------------
video_writer.release()
cv2.destroyAllWindows()

print(f"\n🔥 Video saved to:\n{OUTPUT_PATH}")
