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
import argparse

from config import (
    DEVICE, OUTPUT_DIR, OUTPUT_PATH, NUSCENES_ROOT, CAMERA_CHANNEL,
    FUSION_MODE, FUSION_IOU_THRESHOLD, FUSION_CONFIDENCE_PENALTY,
    USE_ANGLE_SAMPLING, PREDICTION_HORIZON_SHORT,
    TRAJECTORY_PREDICTOR, DETR_THRESHOLD, DETR_CLASS_MAP
)
from data.nuscenes import NuScenesLoader, SimpleImageLoader
from Detection.detector import DetectorPipeline
from Detection.drawer import Drawer
from Transformer.detr_model import DETRRefiner
from Transformer.tracker import TransformerTracker
from Transformer.fusion import fuse_cnn_transformer
from Transformer.trajectory_predictor import (
    evaluate_trajectory, 
    save_evaluation_summary, 
    save_text_summary
)
from Transformer.velocity_estimator import get_velocity_estimator
import tools.analyze_evaluation as analyze_evaluation

# Configuration constant for DetectorPipeline default score
DETECTOR_PIPELINE_DEFAULT_SCORE = 0.85

# Prediction horizon: 12 frames = 2 seconds at 6Hz nuScenes sampling
PREDICTION_HORIZON = 12


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
    # Parse lightweight CLI flags to override config at runtime
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--fusion-mode", type=str, default=None, help="Override fusion mode (cnn, transformer, cnn/transformer)")
    p.add_argument("--no-analyze", action="store_true", help="Skip running post-processing analyzer")
    parsed, _ = p.parse_known_args()

    # Local fusion mode used throughout main (allows runtime override)
    fusion_mode = parsed.fusion_mode if parsed.fusion_mode is not None else FUSION_MODE

    print("=" * 60)
    print("Detect and Predict v2: CNN-Transformer Hybrid")
    print("=" * 60)
    print(f"Fusion Mode: {fusion_mode}")
    print(f"Trajectory Predictor: {TRAJECTORY_PREDICTOR}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/5] Loading dataset...")
    # Check if NUSCENES_ROOT points to a direct image folder
    if os.path.isdir(NUSCENES_ROOT) and any(NUSCENES_ROOT.endswith(cam) for cam in ["CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]):
        print(f"Using SimpleImageLoader for direct folder: {NUSCENES_ROOT}")
        loader = SimpleImageLoader(NUSCENES_ROOT)
    else:
        print(f"Using NuScenesLoader for dataset: {NUSCENES_ROOT}")
        loader = NuScenesLoader(NUSCENES_ROOT)
    
    print("[2/5] Initializing DetectorPipeline (YOLO + FRCNN)...")
    detector_pipeline = DetectorPipeline()
    
    print("[3/5] Initializing DETR (Transformer)...")
    detr = DETRRefiner(device=DEVICE, threshold=DETR_THRESHOLD)
    
    print("[4/5] Initializing TransformerTracker...")
    tracker = TransformerTracker()
    
    # Initialize velocity estimator
    velocity_estimator = get_velocity_estimator(fps=10.0)  # Assuming ~10 FPS
    
    print("[5/5] Setup complete. Starting processing...")
    print("=" * 60)
    
    video_writer = None
    fps_time = time.time()
    frame_idx = 0
    controller = VideoController()
    # Drawer instance for visualization (masks, smoothing, trails)
    drawer = Drawer()
    
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
        # This gives us CNN-based detections with accurate localization (no IDs - tracker will assign)
        yolo_frcnn_results = detector_pipeline.process(frame)
        
        if not yolo_frcnn_results:
            video_writer.write(frame)
            cv2.imshow("CNN-Transformer Hybrid Tracking", frame)
            if cv2.waitKey(1) == 27:
                break
            continue
        
        # DetectorPipeline now returns dicts with box, cls_name, score (no track_id)
        frcnn_dets = []
        for det in yolo_frcnn_results:
            frcnn_dets.append({
                "box": det["box"],
                "cls_name": det["cls_name"],
                "score": det.get("score", DETECTOR_PIPELINE_DEFAULT_SCORE)
            })
        
        # === STEP 2: DETR Enhancement (Transformer stage) ===
        detr_out = detr.predict(frame)

        # === STEP 3: Fuse FRCNN (CNN) and DETR (Transformer) outputs ===
        # Accept friendly labels and legacy labels for backward compatibility
        mode = (fusion_mode or "").lower()
        if mode in ("hybrid", "cnn/transformer", "cnn_transformer"):
            # Full hybrid: use both CNN and Transformer
            fused_dets = fuse_cnn_transformer(
                frcnn_dets,
                detr_out,
                iou_thresh=FUSION_IOU_THRESHOLD,
                confidence_penalty=FUSION_CONFIDENCE_PENALTY
            )
        elif mode in ("cnn", "cnn_only"):
            # CNN only: skip DETR fusion
            fused_dets = frcnn_dets
        elif mode in ("transformer", "transformer_only"):
            # Transformer only: use DETR detections directly
            fused_dets = []
            for det in detr_out:
                if det["label"] in DETR_CLASS_MAP:
                    fused_dets.append({
                        "box": det["box"],
                        "cls_name": DETR_CLASS_MAP[det["label"]],
                        "score": det["score"]
                    })
        else:
            fused_dets = frcnn_dets

        # === STEP 4: Track with TransformerTracker ===
        updated = tracker.update(fused_dets, frame_idx, frame=frame)
        
        # === STEP 4.5: Estimate velocities for all tracked objects ===
        for obj in updated:
            track_id = obj.get("track_id")
            if track_id is not None:
                velocity = velocity_estimator.update(track_id, obj["box"], frame_idx)
                obj["velocity"] = velocity  # Add velocity to object dict

        # === STEP 5: Trajectory evaluation and prediction ===
        evaluate_trajectory(tracker.track_history, updated, frame_idx, prediction_horizon=PREDICTION_HORIZON)
        
        # === STEP 6: Generate predicted trajectories for visualization ===
        from Transformer.trajectory_predictor import KALMAN_TRACKERS
        for obj in updated:
            track_id = obj.get("track_id")
            if track_id is not None and track_id in KALMAN_TRACKERS:
                # Get velocity to check if object is moving
                velocity = obj.get("velocity")
                is_moving = False
                if velocity is not None:
                    vx, vy = velocity
                    speed = (vx**2 + vy**2)**0.5
                    is_moving = speed > 0.5  # Moving if speed > 0.5 pixels/frame
                
                # Generate predicted trajectory for all objects (will be filtered in drawer)
                predicted_traj = []
                if KALMAN_TRACKERS[track_id].initialized:
                    if USE_ANGLE_SAMPLING and is_moving:
                        # Use angle sampling for moving objects
                        predicted_traj = KALMAN_TRACKERS[track_id].predict_with_angle_sampling(steps=PREDICTION_HORIZON)
                    else:
                        # Standard prediction
                        for step in range(1, PREDICTION_HORIZON + 1):
                            pred_box = KALMAN_TRACKERS[track_id].predict(steps=step)
                            if pred_box is not None:
                                predicted_traj.append(pred_box)
                
                obj["predicted_trajectory"] = predicted_traj if predicted_traj else None
                obj["is_moving"] = is_moving
        
        # === Visualization ===
        for obj in updated:
            box = obj["box"]
            cls = obj["cls_name"]
            if cls not in CLASS_COLORS:
                continue
            # If mask available, draw it first for segmentation overlay
            mask = obj.get("mask") if isinstance(obj, dict) else None
            track_id = obj.get("track_id") if isinstance(obj, dict) else None
            score = obj.get("score") if isinstance(obj, dict) else None
            velocity = obj.get("velocity") if isinstance(obj, dict) else None
            predicted_trajectory = obj.get("predicted_trajectory") if isinstance(obj, dict) else None
            prediction_uncertainties = obj.get("prediction_uncertainties") if isinstance(obj, dict) else None
            is_moving = obj.get("is_moving", False) if isinstance(obj, dict) else False

            if mask is not None:
                try:
                    drawer.draw_mask(frame, mask, track_id, cls_name=cls)
                except Exception:
                    pass

            # Draw smoothed, labeled box with trail, velocity vector, predicted trajectory, and uncertainty
            drawer.draw_box(frame, box, cls, track_id=track_id, score=score, velocity=velocity, 
                          predicted_trajectory=predicted_trajectory,
                          prediction_uncertainties=prediction_uncertainties,
                          is_moving=is_moving)
        
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
        
        # Display fusion mode (friendly)
        disp_map = {
            'cnn': 'CNN', 'cnn_only': 'CNN',
            'transformer': 'Transformer', 'transformer_only': 'Transformer',
            'cnn/transformer': 'CNN + Transformer', 'hybrid': 'CNN + Transformer'
        }
        disp = disp_map.get((fusion_mode or '').lower(), fusion_mode)
        cv2.putText(
            frame,
            f"Mode: {disp}",
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

    # Run the analyzer to compute ADE/FDE/RMSE and per-frame/track metrics
    if not parsed.no_analyze:
        print("\nRunning post-processing analyzer (ADE/FDE/RMSE)...")
        try:
            analyze_evaluation.main()
        except Exception as e:
            print(f"Post-processing analyzer failed: {e}")
    else:
        print("\nSkipping post-processing analyzer (--no-analyze)")
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Output video: {OUTPUT_PATH}")
    print(f"Total frames: {frame_idx}")
    print("=" * 60)


if __name__ == "__main__":
    main()
