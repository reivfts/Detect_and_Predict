# main.py

import cv2
import time
import numpy as np
import os

from config import *
from Nuscenes.loader import NuScenesLoader
from Yolo.yolo_model import YOLODetector
from Transformer.detr_model import DETRRefiner
from Detection.drawer import Drawer
from Transformer.tracker import StableIDTracker
import subprocess

def gpu_stats():
    try:
        result = subprocess.check_output("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits", shell=True)
        gpu_util, mem_used = result.decode().strip().split(", ")
        print(f"GPU Util: {gpu_util}% | Mem Used: {mem_used}MB")
    except:
        pass
    gpu_stats()

# ==========================================================
# VIDEO CONTROLLER (PAUSE, SLOW-MO, SPEED-UP, SKIP)
# ==========================================================
class VideoController:
    def __init__(self, base_delay=10):
        self.delay = base_delay           # 30ms per frame (≈33 FPS)
        self.min_delay = 1
        self.max_delay = 200

    def handle_input(self):
        key = cv2.waitKey(self.delay) & 0xFF

        # SPACE → Pause
        if key == ord(' '):
            print("⏸ Paused — press SPACE to resume")
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == ord(' '):
                    print("▶ Resumed")
                    break
                elif k == 27:
                    return "quit"

        # ',' → slowdown
        if key == ord(','):
            self.delay = min(self.delay + 10, self.max_delay)
            print(f"🐌 Slowing down — delay = {self.delay}ms")

        # '.' → speed up
        if key == ord('.'):
            self.delay = max(self.delay - 10, self.min_delay)
            print(f"⚡ Speeding up — delay = {self.delay}ms")

        # '<' → skip backwards (symbolic)
        if key == ord('<'):
            print("⏪ Jump back (not fully implemented)")

        # '>' → skip forward (symbolic)
        if key == ord('>'):
            print("⏩ Jump forward (not fully implemented)")

        # ESC → Quit
        if key == 27:
            return "quit"

        return None

# -------------------------
# IOU UTILITY
# -------------------------
def box_iou(a, b):
    x1a, y1a, x2a, y2a = a
    x1b, y1b, x2b, y2b = b

    xi1 = max(x1a, x1b)
    yi1 = max(y1a, y1b)
    xi2 = min(x2a, x2b)
    yi2 = min(y2a, y2b)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = max(0, (x2a - x1a)) * max(0, (y2a - y1a)) + \
            max(0, (x2b - x1b)) * max(0, (y2b - y1b)) - inter

    if union <= 0:
        return 0
    return inter / union


# -------------------------
# LOAD MODULES
# -------------------------
loader = NuScenesLoader(NUSCENES_ROOT)
yolo = YOLODetector(DEVICE)
detr = DETRRefiner(device=DEVICE, threshold=DETR_THRESHOLD)
drawer = Drawer()

# Video writer
video_writer = None
fps_time = time.time()
frame_idx = 0

# -------------------------
# MAIN LOOP
# -------------------------
for frame, timestamp, token in loader.frames(CAMERA_CHANNEL):


    if video_writer is None:
        h, w = frame.shape[:2]
        video_writer = cv2.VideoWriter(OUTPUT_PATH,
                                       cv2.VideoWriter_fourcc(*"mp4v"),
                                       10,
                                       (w, h))

    frame_idx += 1
    tracks = yolo.detect(frame)

    if not tracks:
        video_writer.write(frame)
        cv2.imshow("YOLO + DETR Tracking", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # Run DETR on entire frame
    detr_out = detr.predict(frame)

    # Build DETR class mapping (COCO IDs)
    DETR_CLASS = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorbike",
        6: "bus",
        8: "truck"
    }

    # -------------------------
    # REFINE EACH YOLO BOX
    # -------------------------
    final_boxes = []

    for det in tracks:
        ybox = det["box"]
        ycls = det["cls_name"]
        tid = det["stable_id"]

        if ycls not in VALID_CLASSES:
            continue

        # Match DETR boxes with same class
        best_iou = 0
        best_box = None

        for fr in detr_out:
            if fr["label"] not in DETR_CLASS:
                continue

            cls_name = DETR_CLASS[fr["label"]]
            if cls_name != ycls:
                continue

            iou_val = box_iou(ybox, fr["box"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_box = fr["box"]

        # Choose refined box or YOLO box
        if best_box is not None and best_iou > DETR_IOU_MATCH:
            final_boxes.append((best_box, ycls, tid))
        else:
            final_boxes.append((ybox, ycls, tid))

    # -------------------------
    # DRAW RESULTS
    # -------------------------
    for box, cls_name, tid in final_boxes:
        drawer.draw_box(frame, box, cls_name, tid)

    # FPS overlay
    fps = frame_idx / (time.time() - fps_time)
    cv2.putText(frame, f"FPS: {fps:.2f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    video_writer.write(frame)
    cv2.imshow("YOLO + DETR Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

# -------------------------
# CLEANUP
# -------------------------
video_writer.release()
cv2.destroyAllWindows()

print("\n🔥 DONE — Saved refined tracking to:")
print(OUTPUT_PATH)
