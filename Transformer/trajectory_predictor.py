# trajectory_predictor.py

import os
import csv
import numpy as np
from collections import defaultdict
from typing import List, Optional, Dict
import torch
import torch.nn as nn

# Import BASE_DIR from config for platform-independent paths
from config import BASE_DIR

SAVE_PATH = os.path.join(BASE_DIR, "data", "trackings")
CSV_PATH = os.path.join(SAVE_PATH, "evaluation.csv")
os.makedirs(SAVE_PATH, exist_ok=True)

# Global accumulator
ALL_EVALS = []


class TrajectoryTransformerPredictor(nn.Module):
    """
    Transformer-based trajectory prediction using attention over history.
    Uses sequence of past bounding boxes to predict future position.
    
    This provides an enhanced alternative to linear extrapolation by learning
    patterns in object motion through attention mechanisms.
    """
    
    def __init__(self, history_len: int = 10, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 2, dropout: float = 0.1, device: str = "cuda"):
        """
        Initialize trajectory transformer predictor.
        
        Args:
            history_len: Maximum length of trajectory history to consider
            d_model: Dimension of the model (embedding size)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dropout: Dropout rate
            device: Device to run model on ("cuda" or "cpu")
        """
        super().__init__()
        
        self.history_len = history_len
        self.d_model = d_model
        self.device = device
        
        # Embedding layer: bbox (4D: x1, y1, x2, y2) -> d_model
        self.bbox_embedding = nn.Linear(4, d_model)
        
        # Positional encoding for sequence order
        self.pos_encoding = nn.Parameter(torch.randn(history_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Prediction head: d_model -> next bbox (4D)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 4)
        )
        
        self.to(device)
        self.eval()  # Set to eval mode by default
    
    def forward(self, trajectory_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for trajectory prediction.
        
        Args:
            trajectory_sequence: Tensor of shape (batch_size, seq_len, 4)
                                where 4 is [x1, y1, x2, y2]
        
        Returns:
            Predicted next bbox of shape (batch_size, 4)
        """
        batch_size, seq_len, _ = trajectory_sequence.shape
        
        # Embed bboxes
        embedded = self.bbox_embedding(trajectory_sequence)  # (B, seq_len, d_model)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0)  # (1, seq_len, d_model)
        embedded = embedded + pos_enc
        
        # Apply transformer
        encoded = self.transformer_encoder(embedded)  # (B, seq_len, d_model)
        
        # Use last position's encoding for prediction
        last_encoded = encoded[:, -1, :]  # (B, d_model)
        
        # Predict next bbox
        predicted_bbox = self.prediction_head(last_encoded)  # (B, 4)
        
        return predicted_bbox
    
    def predict(self, trajectory_history: List[List[float]]) -> Optional[List[float]]:
        """
        Predict next bounding box given trajectory history.
        
        Args:
            trajectory_history: List of bboxes [[x1, y1, x2, y2], ...]
        
        Returns:
            Predicted bbox [x1, y1, x2, y2] or None if insufficient history
        """
        if len(trajectory_history) < 2:
            return None
        
        # Take last N frames
        history = trajectory_history[-self.history_len:]
        
        # Convert to tensor
        history_tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 4)
        history_tensor = history_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            predicted = self.forward(history_tensor)
        
        # Convert back to list
        predicted_bbox = predicted.cpu().numpy()[0].tolist()
        
        return predicted_bbox

def linear_extrapolate(trajectory, steps=1):
    if len(trajectory) < 2:
        return trajectory[-1] if trajectory else None

    box0 = trajectory[-2]
    box1 = trajectory[-1]
    dx = [b1 - b0 for b0, b1 in zip(box0, box1)]
    pred_box = [b + d * steps for b, d in zip(box1, dx)]
    return pred_box

def box_center(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2, (y1 + y2) / 2]

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea > 0 else 0.0

def evaluate_trajectory(track_history, updated_detections, frame_idx):
    global ALL_EVALS

    for det in updated_detections:
        tid = det["track_id"]
        actual_box = det["box"]

        if tid not in track_history or len(track_history[tid]) < 2:
            continue

        history = list(track_history[tid])[-2:]
        pred_box = linear_extrapolate(history)
        if pred_box is None:
            continue

        actual_center = box_center(actual_box)
        pred_center = box_center(pred_box)

        error = np.linalg.norm(np.array(actual_center) - np.array(pred_center))
        iou = compute_iou(actual_box, pred_box)

        ALL_EVALS.append({
            "frame": frame_idx,
            "track_id": tid,
            "iou": round(iou, 4),
            "center_distance": round(error, 2),
            "gt_box": actual_box,
            "pred_box": [round(x, 2) for x in pred_box]
        })

def save_evaluation_summary():
    if not ALL_EVALS:
        print("No trajectory evaluation data found.")
        return

    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "track_id", "iou", "center_distance", "gt_box", "pred_box"])
        writer.writeheader()
        for row in ALL_EVALS:
            writer.writerow(row)

        writer.writerow({})  # empty row

        ious = [row["iou"] for row in ALL_EVALS]
        dists = [row["center_distance"] for row in ALL_EVALS]
        track_ids = set(row["track_id"] for row in ALL_EVALS)
        frame_ids = set(row["frame"] for row in ALL_EVALS)

        writer.writerow({"frame": "Metrics Summary"})
        writer.writerow({"track_id": "Average IoU", "iou": round(np.mean(ious), 4)})
        writer.writerow({"track_id": "Average Center Distance", "center_distance": round(np.mean(dists), 2)})
        writer.writerow({"track_id": "Total Tracks Evaluated", "iou": len(track_ids)})
        writer.writerow({"track_id": "Total Frames Evaluated", "iou": len(frame_ids)})

def save_text_summary():
    txt_path = os.path.join(SAVE_PATH, "accuracy.txt")

    if not ALL_EVALS:
        return

    ious = [row["iou"] for row in ALL_EVALS]
    dists = [row["center_distance"] for row in ALL_EVALS]
    track_ids = set(row["track_id"] for row in ALL_EVALS)
    frame_ids = set(row["frame"] for row in ALL_EVALS)

    with open(txt_path, 'w') as f:
        f.write("Trajectory Evaluation Summary\n")
        f.write(f"Average IoU: {np.mean(ious):.4f}\n")
        f.write(f"Average Center Distance: {np.mean(dists):.2f} pixels\n")
        f.write(f"Total Tracks Evaluated: {len(track_ids)}\n")
        f.write(f"Total Frames Evaluated: {len(frame_ids)}\n")

    print(f"Accuracy summary saved to: {txt_path}")


    print(f"Trajectory evaluation saved to: {CSV_PATH}")