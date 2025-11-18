# main.py

import cv2
import time
import numpy as np
import os
import torch
from ultralytics.utils.plotting import colors
from ultralytics.data.augment import LetterBox

from config import *
from Nuscenes.loader import NuScenesLoader
from Yolo.yolo_model import YOLODetector
from Transformer.detr_model import DETRRefiner
from Detection.drawer import Drawer
from Transformer.tracker import TransformerTracker
import subprocess
from Transformer.trajectory_predictor import evaluate_trajectory, save_evaluation_summary, save_text_summary

def gpu_stats():
    try:
        result = subprocess.check_output("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits", shell=True)
        gpu_util, mem_used = result.decode().strip().split(", ")
        print(f"GPU Util: {gpu_util}% | Mem Used: {mem_used}MB")
    except:
        pass
    # Note: do not recurse. Call `gpu_stats()` from a loop or externally if periodic updates are desired.

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

    return inter / union if union > 0 else 0

class VideoController:
    def __init__(self, base_delay=10):
        self.delay = base_delay
        self.min_delay = 1
        self.max_delay = 200

    def handle_input(self):
        key = cv2.waitKey(self.delay) & 0xFF

        if key == ord(' '):
            print("⏸ Paused — press SPACE to resume")
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == ord(' '):
                    print("▶ Resumed")
                    break
                elif k == 27:
                    return "quit"

        elif key == ord(','):
            self.delay = min(self.delay + 10, self.max_delay)
            print(f"🐌 Slowing down — delay = {self.delay}ms")
        elif key == ord('.'):
            self.delay = max(self.delay - 10, self.min_delay)
            print(f"⚡ Speeding up — delay = {self.delay}ms")
        elif key == 27:
            return "quit"

        return None


loader = NuScenesLoader(NUSCENES_ROOT)
yolo = YOLODetector(DEVICE)
detr = DETRRefiner(device=DEVICE, threshold=DETR_THRESHOLD)
tracker = TransformerTracker()

video_writer = None
fps_time = time.time()
frame_idx = 0

CLASS_COLORS = {
    "car": (255, 0, 0),
    "truck": (0, 255, 0),
    "bus": (0, 0, 255),
    "person": (255, 255, 0),
    "bicycle": (255, 0, 255),
    "motorbike": (0, 255, 255)
}

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

    detr_out = detr.predict(frame)

    DETR_CLASS = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorbike",
        6: "bus",
        8: "truck"
    }

    final_dets = []

    for det in tracks:
        ybox = det["box"]
        ycls = det["cls_name"]
        tid = det["track_id"]

        if ycls not in VALID_CLASSES:
            continue

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

        final_box = best_box if best_box is not None and best_iou > DETR_IOU_MATCH else ybox

        final_dets.append({
            "box": final_box,
            "cls_name": ycls,
            "score": det.get("score", 1.0)
        })

    # Pass frame to tracker so appearance embeddings can be used when enabled
    updated = tracker.update(final_dets, frame_idx, frame=frame)

    evaluate_trajectory(tracker.track_history, updated, frame_idx)

    for obj in updated:
        box = obj["box"]
        cls = obj["cls_name"]
        if cls not in CLASS_COLORS:
            continue

        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS[cls]

        # Semi-transparent color overlay on object region
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Class label (only)
        label = f"{cls}"
        cv2.putText(
            frame,
            label,
            (x1 + 5, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )



    fps = frame_idx / (time.time() - fps_time)
    cv2.putText(frame, f"FPS: {fps:.2f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    video_writer.write(frame)
    controller = VideoController()

    cv2.imshow("YOLO + DETR Tracking", frame)

    if controller.handle_input() == "quit":
        break


video_writer.release()
cv2.destroyAllWindows()
save_evaluation_summary()
save_text_summary()

print("\n🔥 DONE — Saved refined tracking to:")
print(OUTPUT_PATH)
