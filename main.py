# main.py

"""
Detect and Predict v2: CNN-Transformer Hybrid Architecture

Implements a complete CNN-Transformer hybrid detection and tracking system:
- Step 1: YOLO (fast proposals)
- Step 2: FRCNN (CNN-based accurate localization)
- Step 3: DETR (Transformer-based global context)
- Step 4: Fusion of CNN and Transformer outputs
- Step 5: Custom TransformerTracker with Kalman filtering
- Step 6: Trajectory prediction and evaluation

Reference: "CNN-transformer mixed model for object detection" (arXiv:2212.06714)
"""

import cv2
import time
import numpy as np
import os
import torch

from config import *
from Nuscenes.loader import NuScenesLoader
from Detection.detector import DetectorPipeline
from Transformer.detr_model import DETRRefiner
from Transformer.tracker import TransformerTracker
from Transformer.fusion import fuse_cnn_transformer
from Transformer.trajectory_predictor import (
    evaluate_trajectory, 
    save_evaluation_summary, 
    save_text_summary
)


def gpu_stats():
    """Display GPU utilization statistics."""
    try:
        import subprocess
        result = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits", 
            shell=True
        )
        gpu_util, mem_used = result.decode().strip().split(", ")
        print(f"GPU Util: {gpu_util}% | Mem Used: {mem_used}MB")
    except:
        pass


class VideoController:
    """Handle video playback controls."""
    
    def __init__(self, base_delay=10):
        self.delay = base_delay
        self.min_delay = 1
        self.max_delay = 200

    def handle_input(self):
        key = cv2.waitKey(self.delay) & 0xFF

        if key == ord(' '):
            print("Paused - press SPACE to resume")
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == ord(' '):
                    print("Resumed")
                    break
                elif k == 27:
                    return "quit"

        elif key == ord(','):
            self.delay = min(self.delay + 10, self.max_delay)
            print(f"Slowing down - delay = {self.delay}ms")
        elif key == ord('.'):
            self.delay = max(self.delay - 10, self.min_delay)
            print(f"Speeding up - delay = {self.delay}ms")
        elif key == 27:
            return "quit"

        return None


def main():
    """Main processing loop for CNN-Transformer hybrid pipeline."""
    
    print("=" * 60)
    print("Detect and Predict v2: CNN-Transformer Hybrid")
    print("=" * 60)
    print(f"Fusion Mode: {FUSION_MODE}")
    print(f"Trajectory Predictor: {TRAJECTORY_PREDICTOR}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/5] Loading NuScenes dataset...")
    loader = NuScenesLoader(NUSCENES_ROOT)
    
    print("[2/5] Initializing DetectorPipeline (YOLO + FRCNN)...")
    detector_pipeline = DetectorPipeline()
    
    print("[3/5] Initializing DETR (Transformer)...")
    detr = DETRRefiner(device=DEVICE, threshold=DETR_THRESHOLD)
    
    print("[4/5] Initializing TransformerTracker...")
    tracker = TransformerTracker()
    
    print("[5/5] Setup complete. Starting processing...")
    print("=" * 60)
    
    video_writer = None
    fps_time = time.time()
    frame_idx = 0
    controller = VideoController()
    
    CLASS_COLORS = {
        "car": (255, 0, 0),
        "truck": (0, 255, 0),
        "bus": (0, 0, 255),
        "person": (255, 255, 0),
        "bicycle": (255, 0, 255),
        "motorcycle": (0, 255, 255)
    }
    
    for frame, timestamp, token in loader.frames(CAMERA_CHANNEL):
        
        if video_writer is None:
            h, w = frame.shape[:2]
            video_writer = cv2.VideoWriter(
                OUTPUT_PATH,
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (w, h)
            )
            print(f"\nVideo output: {OUTPUT_PATH} ({w}x{h})")
        
        frame_idx += 1
        
        # === STEP 1: YOLO + FRCNN Fusion (DetectorPipeline) ===
        # This gives us CNN-based detections with accurate localization
        yolo_frcnn_results = detector_pipeline.process(frame)
        
        if not yolo_frcnn_results:
            video_writer.write(frame)
            cv2.imshow("CNN-Transformer Hybrid Tracking", frame)
            if cv2.waitKey(1) == 27:
                break
            continue
        
        # Convert to standard detection format
        frcnn_dets = []
        for box, tid, cls_name in yolo_frcnn_results:
            frcnn_dets.append({
                "box": box,
                "cls_name": cls_name,
                "score": 0.9,  # DetectorPipeline doesn't return scores
                "track_id": tid
            })
        
        # === STEP 2: DETR Enhancement (Transformer stage) ===
        detr_out = detr.predict(frame)
        
        # === STEP 3: Fuse FRCNN (CNN) and DETR (Transformer) outputs ===
        if FUSION_MODE == "hybrid":
            # Full hybrid: use both CNN and Transformer
            fused_dets = fuse_cnn_transformer(
                frcnn_dets, 
                detr_out, 
                iou_thresh=FUSION_IOU_THRESHOLD,
                confidence_penalty=FUSION_CONFIDENCE_PENALTY
            )
        elif FUSION_MODE == "cnn_only":
            # CNN only: skip DETR fusion
            fused_dets = frcnn_dets
        elif FUSION_MODE == "transformer_only":
            # Transformer only: use DETR detections directly
            DETR_CLASS = {1: "person", 2: "bicycle", 3: "car", 
                         4: "motorcycle", 6: "bus", 8: "truck"}
            fused_dets = []
            for det in detr_out:
                if det["label"] in DETR_CLASS:
                    fused_dets.append({
                        "box": det["box"],
                        "cls_name": DETR_CLASS[det["label"]],
                        "score": det["score"]
                    })
        else:
            fused_dets = frcnn_dets
        
        # === STEP 4: Track with TransformerTracker ===
        updated = tracker.update(fused_dets, frame_idx, frame=frame)
        
        # === STEP 5: Trajectory prediction and evaluation ===
        evaluate_trajectory(tracker.track_history, updated, frame_idx)
        
        # === Visualization ===
        for obj in updated:
            box = obj["box"]
            cls = obj["cls_name"]
            if cls not in CLASS_COLORS:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            color = CLASS_COLORS.get(cls, (128, 128, 128))
            
            # Semi-transparent color overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            
            # Label with validation info if available
            if "validated_by" in obj:
                label = f"{cls} [{obj['validated_by'][:3]}]"
            else:
                label = f"{cls}"
            
            cv2.putText(
                frame,
                label,
                (x1 + 5, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        # Display FPS and frame info
        fps = frame_idx / (time.time() - fps_time)
        cv2.putText(
            frame, 
            f"FPS: {fps:.2f} | Frame: {frame_idx}", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, 
            (0, 255, 0), 
            2
        )
        
        # Display fusion mode
        cv2.putText(
            frame,
            f"Mode: {FUSION_MODE}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        
        video_writer.write(frame)
        cv2.imshow("CNN-Transformer Hybrid Tracking", frame)
        
        if controller.handle_input() == "quit":
            break
        
        # Print progress
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx} frames... FPS: {fps:.2f}")
    
    # Cleanup
    video_writer.release()
    cv2.destroyAllWindows()
    
    # Save evaluation results
    save_evaluation_summary()
    save_text_summary()
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Output video: {OUTPUT_PATH}")
    print(f"Total frames: {frame_idx}")
    print("=" * 60)


if __name__ == "__main__":
    main()
