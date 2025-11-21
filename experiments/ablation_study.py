# experiments/ablation_study.py

"""
Ablation Study Framework

Compares different detection and tracking approaches:
- Experiment A: YOLO only + custom tracker
- Experiment B: YOLO + FRCNN + custom tracker
- Experiment C: YOLO + DETR + custom tracker
- Experiment D: YOLO + FRCNN + DETR (full hybrid) + custom tracker
- Experiment E: YOLO + DeepSORT (baseline)

Each experiment runs on the same NuScenes frames and logs metrics for comparison.
"""

import os
import csv
import time
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import cv2

# Import detection and tracking components
from Yolo.yolo_model import YOLODetector
from FRCNN.frcnn_model import FRCNNRefiner
from Transformer.detr_model import DETRRefiner
from Transformer.tracker import TransformerTracker
from Transformer.fusion import fuse_cnn_transformer
from Transformer.trajectory_predictor import (
    linear_extrapolate, box_center, compute_iou, evaluate_trajectory
)
from config import DEVICE, DETR_THRESHOLD, DETR_IOU_MATCH


class AblationExperiment:
    """Base class for ablation experiments."""
    
    def __init__(self, experiment_name: str, output_dir: str = "data/trackings/ablation"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.metrics = {
            "frame_times": [],
            "detections_per_frame": [],
            "tracks_per_frame": [],
            "trajectory_errors": []
        }
        
        self.track_history = defaultdict(list)
        self.evaluation_results = []
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> List[Dict]:
        """
        Process a single frame. Must be implemented by subclasses.
        
        Args:
            frame: Input frame
            frame_idx: Frame index
        
        Returns:
            List of tracked detections
        """
        raise NotImplementedError("Subclasses must implement process_frame")
    
    def evaluate_frame(self, tracked_objects: List[Dict], frame_idx: int):
        """Evaluate trajectory prediction for this frame."""
        for obj in tracked_objects:
            tid = obj["track_id"]
            actual_box = obj["box"]
            
            if tid not in self.track_history or len(self.track_history[tid]) < 2:
                continue
            
            # Get history and predict
            history = list(self.track_history[tid])[-10:]  # Last 10 frames
            pred_box = linear_extrapolate(history)
            
            if pred_box is None:
                continue
            
            # Calculate metrics
            actual_center = box_center(actual_box)
            pred_center = box_center(pred_box)
            
            center_dist = np.linalg.norm(
                np.array(actual_center) - np.array(pred_center)
            )
            iou = compute_iou(actual_box, pred_box)
            
            self.evaluation_results.append({
                "frame": frame_idx,
                "track_id": tid,
                "center_distance": center_dist,
                "iou": iou,
                "ade": center_dist,  # Average Displacement Error
                "fde": center_dist   # Final Displacement Error (same for 1-step)
            })
            
            self.metrics["trajectory_errors"].append(center_dist)
    
    def update_metrics(self, tracked_objects: List[Dict], frame_time: float):
        """Update metrics for current frame."""
        self.metrics["frame_times"].append(frame_time)
        self.metrics["detections_per_frame"].append(len(tracked_objects))
        
        # Count unique tracks
        unique_tracks = len(set(obj["track_id"] for obj in tracked_objects))
        self.metrics["tracks_per_frame"].append(unique_tracks)
    
    def save_results(self):
        """Save experiment results to CSV."""
        # Save per-frame trajectory results
        csv_path = os.path.join(self.output_dir, f"{self.experiment_name}_trajectory.csv")
        with open(csv_path, 'w', newline='') as f:
            if self.evaluation_results:
                writer = csv.DictWriter(f, fieldnames=self.evaluation_results[0].keys())
                writer.writeheader()
                writer.writerows(self.evaluation_results)
        
        # Save summary metrics
        summary_path = os.path.join(self.output_dir, f"{self.experiment_name}_summary.csv")
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            
            if self.metrics["frame_times"]:
                writer.writerow(["Avg Frame Time (s)", 
                               f"{np.mean(self.metrics['frame_times']):.4f}"])
                writer.writerow(["FPS", 
                               f"{1.0 / np.mean(self.metrics['frame_times']):.2f}"])
            
            if self.metrics["detections_per_frame"]:
                writer.writerow(["Avg Detections per Frame", 
                               f"{np.mean(self.metrics['detections_per_frame']):.2f}"])
            
            if self.metrics["tracks_per_frame"]:
                writer.writerow(["Avg Tracks per Frame", 
                               f"{np.mean(self.metrics['tracks_per_frame']):.2f}"])
            
            if self.evaluation_results:
                ades = [r["ade"] for r in self.evaluation_results]
                fdes = [r["fde"] for r in self.evaluation_results]
                ious = [r["iou"] for r in self.evaluation_results]
                
                writer.writerow(["ADE (pixels)", f"{np.mean(ades):.2f}"])
                writer.writerow(["FDE (pixels)", f"{np.mean(fdes):.2f}"])
                writer.writerow(["Avg IoU", f"{np.mean(ious):.4f}"])
        
        print(f"[{self.experiment_name}] Results saved to {self.output_dir}")


class ExperimentA(AblationExperiment):
    """YOLO only + custom tracker"""
    
    def __init__(self, output_dir: str = "data/trackings/ablation"):
        super().__init__("exp_a_yolo_only", output_dir)
        self.yolo = YOLODetector(DEVICE)
        self.tracker = TransformerTracker()
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> List[Dict]:
        start_time = time.time()
        
        # YOLO detection
        detections = self.yolo.detect(frame)
        
        # Track
        tracked = self.tracker.update(detections, frame_idx, frame=frame)
        
        # Update history
        for obj in tracked:
            self.track_history[obj["track_id"]].append(obj["box"])
        
        frame_time = time.time() - start_time
        self.update_metrics(tracked, frame_time)
        self.evaluate_frame(tracked, frame_idx)
        
        return tracked


class ExperimentB(AblationExperiment):
    """YOLO + FRCNN + custom tracker"""
    
    def __init__(self, output_dir: str = "data/trackings/ablation"):
        super().__init__("exp_b_yolo_frcnn", output_dir)
        self.yolo = YOLODetector(DEVICE)
        self.frcnn = FRCNNRefiner()
        self.tracker = TransformerTracker()
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> List[Dict]:
        start_time = time.time()
        
        # YOLO detection
        yolo_dets = self.yolo.detect(frame)
        
        if not yolo_dets:
            return []
        
        # FRCNN refinement
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frcnn_dets = self.frcnn.refine(frame_rgb)
        
        # Match YOLO tracks with FRCNN boxes
        refined_dets = []
        for yolo_det in yolo_dets:
            ybox = yolo_det["box"]
            ycls = yolo_det["cls_name"]
            
            best_iou = 0
            best_box = None
            
            for frcnn_box, frcnn_score, frcnn_cls in frcnn_dets:
                if frcnn_cls != ycls:
                    continue
                
                iou_val = compute_iou(ybox, frcnn_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_box = frcnn_box
            
            # Use FRCNN box if good match
            final_box = best_box if best_box is not None and best_iou > DETR_IOU_MATCH else ybox
            
            refined_dets.append({
                "box": final_box,
                "cls_name": ycls,
                "score": yolo_det["score"]
            })
        
        # Track
        tracked = self.tracker.update(refined_dets, frame_idx, frame=frame)
        
        # Update history
        for obj in tracked:
            self.track_history[obj["track_id"]].append(obj["box"])
        
        frame_time = time.time() - start_time
        self.update_metrics(tracked, frame_time)
        self.evaluate_frame(tracked, frame_idx)
        
        return tracked


class ExperimentC(AblationExperiment):
    """YOLO + DETR + custom tracker"""
    
    def __init__(self, output_dir: str = "data/trackings/ablation"):
        super().__init__("exp_c_yolo_detr", output_dir)
        self.yolo = YOLODetector(DEVICE)
        self.detr = DETRRefiner(device=DEVICE, threshold=DETR_THRESHOLD)
        self.tracker = TransformerTracker()
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> List[Dict]:
        start_time = time.time()
        
        # YOLO detection
        yolo_dets = self.yolo.detect(frame)
        
        if not yolo_dets:
            return []
        
        # DETR refinement
        detr_dets = self.detr.predict(frame)
        
        DETR_CLASS = {1: "person", 2: "bicycle", 3: "car", 
                     4: "motorcycle", 6: "bus", 8: "truck"}
        
        # Match YOLO with DETR
        refined_dets = []
        for yolo_det in yolo_dets:
            ybox = yolo_det["box"]
            ycls = yolo_det["cls_name"]
            
            best_iou = 0
            best_box = None
            
            for detr_det in detr_dets:
                if detr_det["label"] not in DETR_CLASS:
                    continue
                
                detr_cls = DETR_CLASS[detr_det["label"]]
                if detr_cls != ycls:
                    continue
                
                iou_val = compute_iou(ybox, detr_det["box"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_box = detr_det["box"]
            
            # Use DETR box if good match
            final_box = best_box if best_box is not None and best_iou > DETR_IOU_MATCH else ybox
            
            refined_dets.append({
                "box": final_box,
                "cls_name": ycls,
                "score": yolo_det["score"]
            })
        
        # Track
        tracked = self.tracker.update(refined_dets, frame_idx, frame=frame)
        
        # Update history
        for obj in tracked:
            self.track_history[obj["track_id"]].append(obj["box"])
        
        frame_time = time.time() - start_time
        self.update_metrics(tracked, frame_time)
        self.evaluate_frame(tracked, frame_idx)
        
        return tracked


class ExperimentD(AblationExperiment):
    """YOLO + FRCNN + DETR (full hybrid) + custom tracker"""
    
    def __init__(self, output_dir: str = "data/trackings/ablation"):
        super().__init__("exp_d_full_hybrid", output_dir)
        self.yolo = YOLODetector(DEVICE)
        self.frcnn = FRCNNRefiner()
        self.detr = DETRRefiner(device=DEVICE, threshold=DETR_THRESHOLD)
        self.tracker = TransformerTracker()
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> List[Dict]:
        start_time = time.time()
        
        # Step 1: YOLO detection
        yolo_dets = self.yolo.detect(frame)
        
        if not yolo_dets:
            return []
        
        # Step 2: FRCNN refinement
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frcnn_dets = self.frcnn.refine(frame_rgb)
        
        # Match YOLO with FRCNN
        frcnn_refined = []
        for yolo_det in yolo_dets:
            ybox = yolo_det["box"]
            ycls = yolo_det["cls_name"]
            
            best_iou = 0
            best_box = None
            best_score = yolo_det["score"]
            
            for frcnn_box, frcnn_score, frcnn_cls in frcnn_dets:
                if frcnn_cls != ycls:
                    continue
                
                iou_val = compute_iou(ybox, frcnn_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_box = frcnn_box
                    best_score = frcnn_score
            
            final_box = best_box if best_box is not None and best_iou > DETR_IOU_MATCH else ybox
            
            frcnn_refined.append({
                "box": final_box,
                "cls_name": ycls,
                "score": best_score
            })
        
        # Step 3: DETR refinement
        detr_dets = self.detr.predict(frame)
        
        # Step 4: Fuse FRCNN (CNN) and DETR (Transformer)
        fused_dets = fuse_cnn_transformer(frcnn_refined, detr_dets, iou_thresh=0.3)
        
        # Step 5: Track
        tracked = self.tracker.update(fused_dets, frame_idx, frame=frame)
        
        # Update history
        for obj in tracked:
            self.track_history[obj["track_id"]].append(obj["box"])
        
        frame_time = time.time() - start_time
        self.update_metrics(tracked, frame_time)
        self.evaluate_frame(tracked, frame_idx)
        
        return tracked


class ExperimentE(AblationExperiment):
    """YOLO + DeepSORT (baseline)"""
    
    def __init__(self, output_dir: str = "data/trackings/ablation"):
        super().__init__("exp_e_yolo_deepsort", output_dir)
        self.yolo = YOLODetector(DEVICE)
        
        try:
            from trackers.deepsort_wrapper import DeepSORTWrapper
            self.tracker = DeepSORTWrapper(max_age=30, n_init=3)
        except ImportError:
            print("Warning: DeepSORT not available. Skipping Experiment E.")
            self.tracker = None
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> List[Dict]:
        if self.tracker is None:
            return []
        
        start_time = time.time()
        
        # YOLO detection (without tracking)
        results = self.yolo.model(frame, conf=0.45, iou=0.45, imgsz=1280)[0]
        
        if results.boxes is None or len(results.boxes) == 0:
            return []
        
        boxes = results.boxes.xyxy.cpu().numpy()
        clss = results.boxes.cls.int().cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        names = self.yolo.names
        
        # Convert to detection format
        detections = []
        for box, cls_idx, score in zip(boxes, clss, scores):
            cls_name = names[int(cls_idx)]
            detections.append({
                "box": box.tolist(),
                "cls_name": cls_name,
                "score": float(score)
            })
        
        # Track with DeepSORT
        tracked = self.tracker.update(detections, frame_idx, frame=frame)
        
        # Update history
        for obj in tracked:
            self.track_history[obj["track_id"]].append(obj["box"])
        
        frame_time = time.time() - start_time
        self.update_metrics(tracked, frame_time)
        self.evaluate_frame(tracked, frame_idx)
        
        return tracked


def run_ablation_study(
    frames_generator,
    max_frames: int = 100,
    experiments: Optional[List[str]] = None
):
    """
    Run ablation study on given frames.
    
    Args:
        frames_generator: Generator yielding (frame, timestamp, token) tuples
        max_frames: Maximum number of frames to process
        experiments: List of experiment names to run (default: all)
    """
    if experiments is None:
        experiments = ["A", "B", "C", "D", "E"]
    
    # Map experiment names to classes
    exp_map = {
        "A": ExperimentA,
        "B": ExperimentB,
        "C": ExperimentC,
        "D": ExperimentD,
        "E": ExperimentE
    }
    
    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)
    
    for exp_name in experiments:
        if exp_name not in exp_map:
            print(f"Warning: Unknown experiment {exp_name}, skipping.")
            continue
        
        print(f"\nRunning Experiment {exp_name}...")
        experiment = exp_map[exp_name]()
        
        frame_idx = 0
        for frame, timestamp, token in frames_generator:
            if frame_idx >= max_frames:
                break
            
            frame_idx += 1
            experiment.process_frame(frame, frame_idx)
            
            if frame_idx % 10 == 0:
                print(f"  Processed {frame_idx} frames...")
        
        experiment.save_results()
        print(f"Experiment {exp_name} completed.")
    
    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)
