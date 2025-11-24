# trajectory_predictor.py

import os
import csv
import numpy as np
from collections import defaultdict
from typing import List, Optional, Dict, Tuple
import torch
import torch.nn as nn
from filterpy.kalman import KalmanFilter

# Import BASE_DIR from config for platform-independent paths
from config import (
    BASE_DIR, 
    MAX_ACCELERATION, 
    MAX_VELOCITY, 
    USE_HYBRID_PREDICTION,
    HYBRID_KALMAN_WEIGHT,
    HYBRID_TRANSFORMER_WEIGHT,
    SHOW_UNCERTAINTY,
    PREDICTION_HORIZON_SHORT,
    PREDICTION_HORIZON_LONG,
    USE_LONG_HORIZON,
    USE_ANGLE_SAMPLING,
    ANGLE_SAMPLE_RANGE,
    ANGLE_SAMPLE_STEPS,
    USE_OVERPREDICTION,
    OVERPREDICTION_HORIZONS
)

SAVE_PATH = os.path.join(BASE_DIR, "data", "trackings")
CSV_PATH = os.path.join(SAVE_PATH, "evaluation.csv")
os.makedirs(SAVE_PATH, exist_ok=True)

# Global accumulator
ALL_EVALS = []

# Kalman filters for offline accuracy computation (separate from visualization)
KALMAN_TRACKERS = {}  # Dict[track_id, KalmanTrajectoryPredictor]

# Store predictions for future evaluation
# Format: {(track_id, target_frame): {"prediction": box, "source_frame": frame_idx}}
STORED_PREDICTIONS = {}


class KalmanTrajectoryPredictor:
    """
    Enhanced Kalman Filter-based trajectory predictor with physics constraints.
    
    Uses Kalman filter to estimate track velocity and predict future positions.
    Includes physics-based constraints for realistic motion (max acceleration, velocity).
    
    State vector: [cx, cy, vx, vy, w, h] (center, velocity, size)
    """
    
    def __init__(self, process_noise=1.0, measurement_noise=1.0, ego_velocity=None):
        """
        Initialize Kalman filter predictor.
        
        Args:
            process_noise: Process noise (Q matrix scale)
            measurement_noise: Measurement noise (R matrix scale)
            ego_velocity: [vx, vy] camera velocity for ego-motion compensation
        """
        self.kf = None
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.ego_velocity = ego_velocity if ego_velocity is not None else [0.0, 0.0]
        self.initialized = False
        self.frame_count = 0
        self.last_velocity = np.array([0.0, 0.0])  # For acceleration constraint
        self.dt = 1.0 / 6.0  # Assuming 6Hz frame rate (nuScenes)
        self.velocity_history = []  # Track velocity for smoothing
        self.max_velocity_history = 10  # Increased from 5 for better smoothing
    
    def _create_kalman(self):
        """Create and initialize Kalman filter with improved settings."""
        kf = KalmanFilter(dim_x=6, dim_z=4)
        
        dt = 1.0
        # State transition: [cx, cy, vx, vy, w, h]
        kf.F = np.array([
            [1, 0, dt, 0, 0, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)
        
        # Measurement matrix: observe [cx, cy, w, h]
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)
        
        # Covariance initialization
        kf.P *= 5.0  # Reduced from 10.0 for more confident initial estimate
        kf.P[2:4, 2:4] *= 2.0  # Higher uncertainty for velocity initially
        kf.R *= self.measurement_noise
        
        # Adaptive process noise - will be adjusted based on speed in update()
        Q_scale = np.array([1.0, 1.0, 0.3, 0.3, 0.5, 0.5])  # Less noise for velocity
        kf.Q = np.diag(Q_scale) * self.process_noise
        
        return kf
    
    def _adapt_process_noise(self):
        """Adapt process noise based on current object speed.
        
        Faster objects have more unpredictable motion, need higher process noise.
        """
        if not self.initialized:
            return
        
        current_vel = self.kf.x[2:4].flatten()
        speed = np.linalg.norm(current_vel)
        
        # Scale process noise with speed: faster = more noise
        # Base noise scales from 0.5x to 2.0x based on speed
        speed_factor = 1.0 + min(1.0, speed / 10.0)  # Normalize by typical speed
        
        Q_scale = np.array([1.0, 1.0, 0.3, 0.3, 0.5, 0.5]) * speed_factor
        self.kf.Q = np.diag(Q_scale) * self.process_noise
    
    def _box_to_state(self, box):
        """Convert box [x1, y1, x2, y2] to state [cx, cy, w, h]."""
        x1, y1, x2, y2 = box
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return np.array([cx, cy, w, h], dtype=float)
    
    def _state_to_box(self, cx, cy, w, h):
        """Convert state [cx, cy, w, h] to box [x1, y1, x2, y2]."""
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=float)
    
    def update(self, box):
        """
        Update Kalman filter with new measurement and apply physics constraints.
        
        Args:
            box: [x1, y1, x2, y2] bounding box
        """
        meas = self._box_to_state(box)  # [cx, cy, w, h]
        
        if not self.initialized:
            # First measurement: initialize filter
            self.kf = self._create_kalman()
            self.kf.x = np.array([meas[0], meas[1], 0., 0., meas[2], meas[3]], 
                                dtype=float).reshape((6, 1))
            self.initialized = True
        else:
            # Predict and update
            self.kf.predict()
            self.kf.update(meas)
            
            # Track velocity history for smoothing
            current_vel = self.kf.x[2:4].flatten()
            self.velocity_history.append(current_vel.copy())
            if len(self.velocity_history) > self.max_velocity_history:
                self.velocity_history.pop(0)
            
            # Smooth velocity using outlier-robust exponential moving average
            if len(self.velocity_history) >= 3:
                # Outlier rejection using median filtering
                velocities = np.array(self.velocity_history)
                speeds = np.linalg.norm(velocities, axis=1)
                median_speed = np.median(speeds)
                std_speed = np.std(speeds) if len(speeds) > 1 else 1.0
                
                # Keep velocities within 2 std of median
                valid_mask = np.abs(speeds - median_speed) < 2 * std_speed
                valid_velocities = velocities[valid_mask]
                
                if len(valid_velocities) > 0:
                    # Exponential moving average with adaptive alpha
                    alpha = 0.5  # Increased for faster response
                    smoothed_vel = valid_velocities[0].copy()
                    for vel in valid_velocities[1:]:
                        smoothed_vel = alpha * vel + (1 - alpha) * smoothed_vel
                    self.kf.x[2] = smoothed_vel[0]
                    self.kf.x[3] = smoothed_vel[1]
            elif len(self.velocity_history) >= 2:
                # Simple smoothing for limited history
                alpha = 0.5
                smoothed_vel = self.velocity_history[0].copy()
                for vel in self.velocity_history[1:]:
                    smoothed_vel = alpha * vel + (1 - alpha) * smoothed_vel
                self.kf.x[2] = smoothed_vel[0]
                self.kf.x[3] = smoothed_vel[1]
            
            # Adapt process noise based on speed
            self._adapt_process_noise()
            
            # Apply physics constraints
            self._apply_physics_constraints()
        
        self.frame_count += 1
    
    def get_heading_angle(self):
        """Get current heading angle from velocity vector (radians)."""
        if not self.initialized:
            return 0.0
        vx, vy = self.kf.x[2], self.kf.x[3]
        velocity = np.array([vx, vy]).flatten()
        speed = np.linalg.norm(velocity)
        if speed < 0.1:  # Stationary
            return 0.0
        # Angle in image coordinates (right=0, down=90)
        return np.arctan2(vy, vx)
    
    def _apply_physics_constraints(self):
        """Apply physics-based constraints to filter state."""
        if not self.initialized:
            return
        
        # Extract velocity
        vx, vy = self.kf.x[2], self.kf.x[3]
        velocity = np.array([vx, vy]).flatten()
        
        # Constraint 1: Maximum velocity (clamp speed)
        speed = np.linalg.norm(velocity)
        if speed > MAX_VELOCITY:
            velocity = velocity / speed * MAX_VELOCITY
            self.kf.x[2] = velocity[0]
            self.kf.x[3] = velocity[1]
        
        # Constraint 2: Maximum acceleration (smooth velocity change)
        if self.frame_count > 1:
            dv = velocity - self.last_velocity
            acceleration = np.linalg.norm(dv) / self.dt
            
            if acceleration > MAX_ACCELERATION:
                # Scale velocity change to respect max acceleration
                dv_clamped = dv / acceleration * MAX_ACCELERATION * self.dt
                velocity = self.last_velocity + dv_clamped
                self.kf.x[2] = velocity[0]
                self.kf.x[3] = velocity[1]
        
        self.last_velocity = velocity
    
    def predict_with_angle_sampling(self, steps=1):
        """Predict trajectory using multiple angle hypotheses.
        
        Samples multiple heading angles around current direction and
        selects the most consistent with recent motion history.
        
        Args:
            steps: Number of time steps to predict
            
        Returns:
            List of predicted boxes [x1, y1, x2, y2]
        """
        if not self.initialized or not USE_ANGLE_SAMPLING:
            # Fallback to standard prediction
            predictions = []
            for step in range(1, steps + 1):
                pred = self.predict(steps=step, return_uncertainty=False)
                if pred is not None:
                    predictions.append(pred)
            return predictions
        
        # Get current state
        current_vel = self.kf.x[2:4].flatten()
        speed = np.linalg.norm(current_vel)
        
        # Don't sample angles for very slow objects
        if speed < 1.0:
            predictions = []
            for step in range(1, steps + 1):
                pred = self.predict(steps=step, return_uncertainty=False)
                if pred is not None:
                    predictions.append(pred)
            return predictions
        
        # Get current heading from velocity
        current_heading = np.arctan2(current_vel[1], current_vel[0])
        
        # Generate angle samples
        angle_offsets = np.linspace(-ANGLE_SAMPLE_RANGE, 
                                   ANGLE_SAMPLE_RANGE, 
                                   ANGLE_SAMPLE_STEPS)
        
        best_trajectory = None
        best_score = -np.inf
        
        for angle_offset in angle_offsets:
            # Adjusted heading for this hypothesis
            test_heading = current_heading + angle_offset
            
            # Generate trajectory with adjusted velocities
            trajectory = []
            x = self.kf.x.copy()
            
            # Adjust velocity direction while keeping speed
            vel_x = speed * np.cos(test_heading)
            vel_y = speed * np.sin(test_heading)
            x[2] = vel_x
            x[3] = vel_y
            
            for _ in range(steps):
                # Predict next position
                x[0] += x[2] * self.dt  # cx += vx * dt
                x[1] += x[3] * self.dt  # cy += vy * dt
                
                # Convert to box
                box = self._state_to_box(x[0], x[1], x[4], x[5])
                trajectory.append(box)
            
            # Score based on consistency with velocity history and acceleration patterns
            if len(self.velocity_history) >= 5:
                # Use more history for better scoring
                recent_velocities = self.velocity_history[-5:]
                recent_angles = [np.arctan2(v[1], v[0]) for v in recent_velocities if np.linalg.norm(v) > 0.1]
                recent_speeds = [np.linalg.norm(v) for v in recent_velocities]
                
                if recent_angles and len(recent_speeds) >= 2:
                    # Compute mean angle using circular mean
                    mean_recent_angle = np.arctan2(
                        np.mean([np.sin(a) for a in recent_angles]),
                        np.mean([np.cos(a) for a in recent_angles])
                    )
                    
                    # Check for acceleration/deceleration trend
                    speed_trend = recent_speeds[-1] - recent_speeds[0]
                    is_accelerating = speed_trend > 0.5
                    is_decelerating = speed_trend < -0.5
                    
                    # Angle consistency score
                    angle_diff = abs(self._angle_difference(test_heading, mean_recent_angle))
                    angle_score = -angle_diff  # Lower diff = better
                    
                    # Speed consistency: allow more deviation if accelerating (turns often involve speed changes)
                    if is_accelerating:
                        # Allow larger angle changes when accelerating (e.g., merging, turning)
                        angle_score *= 0.7  # Reduce penalty for angle changes
                    elif is_decelerating:
                        # Prefer straight ahead when decelerating (e.g., braking)
                        angle_score -= abs(angle_offset) * 0.5
                    
                    score = angle_score
                else:
                    score = -abs(angle_offset)  # Prefer straight
            elif len(self.velocity_history) >= 3:
                # Fallback to shorter history
                recent_angles = [np.arctan2(v[1], v[0]) for v in self.velocity_history[-3:] if np.linalg.norm(v) > 0.1]
                if recent_angles:
                    mean_recent_angle = np.arctan2(
                        np.mean([np.sin(a) for a in recent_angles]),
                        np.mean([np.cos(a) for a in recent_angles])
                    )
                    angle_diff = abs(self._angle_difference(test_heading, mean_recent_angle))
                    score = -angle_diff
                else:
                    score = -abs(angle_offset)
            else:
                # No history, prefer straight ahead
                score = -abs(angle_offset)
            
            if score > best_score:
                best_score = score
                best_trajectory = trajectory
        
        return best_trajectory if best_trajectory else []
    
    def _angle_difference(self, angle1, angle2):
        """Calculate smallest difference between two angles (in radians)."""
        diff = angle1 - angle2
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
    
    def predict_multi_horizon(self, horizons=None):
        """Predict at multiple temporal horizons (DONUT-inspired overprediction).
        
        Predicts trajectories at different time horizons to encourage the model
        to anticipate future motion patterns more effectively.
        
        Args:
            horizons: List of prediction steps (e.g., [1, 3, 5, 8])
            
        Returns:
            Dict mapping horizon -> predicted boxes
        """
        if horizons is None:
            from config import OVERPREDICTION_HORIZONS
            horizons = OVERPREDICTION_HORIZONS
        
        if not self.initialized:
            return {h: None for h in horizons}
        
        predictions = {}
        for horizon in horizons:
            pred = self.predict(steps=horizon, return_uncertainty=False)
            predictions[horizon] = pred
        
        return predictions
    
    def predict(self, steps=1, ego_velocity=None, return_uncertainty=False):
        """
        Predict future bounding box with optional uncertainty.
        
        Args:
            steps: Number of frames ahead to predict
            ego_velocity: Override ego velocity for this prediction
            return_uncertainty: If True, return (prediction, covariance)
        
        Returns:
            Predicted box [x1, y1, x2, y2] or tuple (box, covariance) if return_uncertainty
            Returns None if not initialized
        """
        if not self.initialized or self.kf is None:
            return None if not return_uncertainty else (None, None)
        
        # Use provided ego_velocity or stored one
        ego_v = ego_velocity if ego_velocity is not None else self.ego_velocity
        
        # Make prediction
        x = self.kf.x.copy()
        P = self.kf.P.copy()  # Covariance matrix
        F = self.kf.F
        Q = self.kf.Q
        
        for _ in range(steps):
            x = F.dot(x)
            P = F.dot(P).dot(F.T) + Q
        
        # Extract state: [cx, cy, vx, vy, w, h]
        cx, cy, vx, vy, w, h = x.flatten()
        
        # Compensate for ego motion
        cx -= ego_v[0] * steps
        cy -= ego_v[1] * steps
        
        # Convert to box
        pred_box = self._state_to_box(cx, cy, w, h)
        
        # Clamp to reasonable bounds
        pred_box = np.maximum(pred_box, 0)
        
        if return_uncertainty:
            # Return position covariance (2x2 matrix for cx, cy)
            position_cov = P[:2, :2]
            return pred_box.tolist(), position_cov
        
        return pred_box.tolist()
    
    def get_velocity(self):
        """Get estimated velocity [vx, vy]. Returns None if not initialized."""
        if not self.initialized or self.kf is None:
            return None
        return self.kf.x[2:4].flatten().tolist()
    
    def get_uncertainty(self):
        """Get position uncertainty as 2x2 covariance matrix. Returns None if not initialized."""
        if not self.initialized or self.kf is None:
            return None
        return self.kf.P[:2, :2]


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


class HybridTrajectoryPredictor:
    """
    Hybrid predictor combining Kalman filter (physics-based) and Transformer (learning-based).
    Uses weighted combination for robust predictions.
    """
    
    def __init__(self, kalman_predictor, transformer_predictor=None, 
                 kalman_weight=0.7, transformer_weight=0.3):
        """
        Initialize hybrid predictor.
        
        Args:
            kalman_predictor: KalmanTrajectoryPredictor instance
            transformer_predictor: TrajectoryTransformerPredictor instance (optional)
            kalman_weight: Weight for Kalman prediction
            transformer_weight: Weight for Transformer prediction
        """
        self.kalman = kalman_predictor
        self.transformer = transformer_predictor
        self.kalman_weight = kalman_weight
        self.transformer_weight = transformer_weight
        self.trajectory_history = []
    
    def update(self, box):
        """Update both predictors with new observation."""
        self.kalman.update(box)
        self.trajectory_history.append(box)
        if len(self.trajectory_history) > 30:  # Keep last 30 frames
            self.trajectory_history = self.trajectory_history[-30:]
    
    def predict(self, steps=1, return_uncertainty=False):
        """
        Hybrid prediction combining Kalman and Transformer.
        
        Args:
            steps: Number of frames ahead to predict
            return_uncertainty: If True, return (prediction, covariance)
        
        Returns:
            Predicted box [x1, y1, x2, y2] or tuple (box, covariance)
        """
        # Get Kalman prediction
        if return_uncertainty:
            kalman_pred, cov = self.kalman.predict(steps, return_uncertainty=True)
        else:
            kalman_pred = self.kalman.predict(steps)
            cov = None
        
        if kalman_pred is None:
            return None if not return_uncertainty else (None, None)
        
        # If no Transformer or insufficient history, return Kalman only
        if self.transformer is None or len(self.trajectory_history) < 2:
            return kalman_pred if not return_uncertainty else (kalman_pred, cov)
        
        # Get Transformer prediction (only 1-step ahead supported)
        if steps == 1:
            try:
                transformer_pred = self.transformer.predict(self.trajectory_history)
                if transformer_pred is not None:
                    # Weighted combination
                    kalman_arr = np.array(kalman_pred)
                    transformer_arr = np.array(transformer_pred)
                    combined = (self.kalman_weight * kalman_arr + 
                               self.transformer_weight * transformer_arr)
                    result = combined.tolist()
                    return result if not return_uncertainty else (result, cov)
            except Exception as e:
                print(f"[Hybrid] Transformer prediction failed: {e}, using Kalman only")
        
        # Fallback to Kalman for multi-step or if Transformer failed
        return kalman_pred if not return_uncertainty else (kalman_pred, cov)
    
    def get_velocity(self):
        """Get velocity from Kalman filter."""
        return self.kalman.get_velocity()
    
    def get_uncertainty(self):
        """Get uncertainty from Kalman filter."""
        return self.kalman.get_uncertainty()


def linear_extrapolate(trajectory, steps=1):
    """Legacy linear extrapolation. For new code, use KalmanTrajectoryPredictor."""
    if len(trajectory) < 2:
        return trajectory[-1] if trajectory else None

    box0 = trajectory[-2]
    box1 = trajectory[-1]
    dx = [b1 - b0 for b0, b1 in zip(box0, box1)]
    pred_box = [b + d * steps for b, d in zip(box1, dx)]
    return pred_box

def box_center(box):
    """Compute (cx, cy) center of bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def estimate_object_distance(box, frame_height=900):
    """Estimate relative distance from camera based on bbox position and size.
    
    Objects lower in frame and smaller are typically further away.
    Returns normalized distance weight (0-1, where 1 = closest/most important).
    """
    x1, y1, x2, y2 = box
    bbox_height = y2 - y1
    bbox_area = (x2 - x1) * bbox_height
    
    # Vertical position (normalized): higher y = closer (bottom of frame)
    vertical_pos = (y1 + y2) / 2.0
    vertical_weight = vertical_pos / frame_height
    
    # Size weight: larger objects are typically closer
    # Normalize by typical car size in pixels
    size_weight = min(1.0, bbox_area / (100 * 50))  # 100x50 px baseline
    
    # Combined distance weight (prioritize both size and position)
    distance_weight = 0.6 * vertical_weight + 0.4 * size_weight
    return min(1.0, max(0.1, distance_weight))  # Clamp to [0.1, 1.0]

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

def evaluate_trajectory(track_history, updated_detections, frame_idx, prediction_horizon=None):
    """
    Enhanced trajectory prediction evaluation with hybrid prediction and uncertainty.
    
    At EVERY frame:
    1. Make N-step-ahead predictions for all tracks (using Kalman or Hybrid)
    2. Store predictions with uncertainty for future evaluation
    3. Compare current position against predictions made N frames ago
    4. Calculate continuous accuracy metrics
    
    Args:
        track_history: Dictionary of track histories
        updated_detections: Current frame detections with track IDs
        frame_idx: Current frame number
        prediction_horizon: Number of frames to predict ahead (default from config)
    """
    global ALL_EVALS, KALMAN_TRACKERS, STORED_PREDICTIONS
    
    # Use configured horizon
    if prediction_horizon is None:
        prediction_horizon = PREDICTION_HORIZON_LONG if USE_LONG_HORIZON else PREDICTION_HORIZON_SHORT

    for det in updated_detections:
        tid = det["track_id"]
        actual_box = det["box"]

        if tid not in track_history or len(track_history[tid]) < 2:
            continue

        # === STEP 1: Initialize predictor (Kalman or Hybrid) ===
        if tid not in KALMAN_TRACKERS:
            kalman = KalmanTrajectoryPredictor(
                process_noise=0.5,  # Reduced from 1.0 for smoother predictions
                measurement_noise=0.8  # Slightly reduced to trust measurements more
            )
            
            # Optionally create hybrid predictor
            if USE_HYBRID_PREDICTION:
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    transformer = TrajectoryTransformerPredictor(device=device)
                    KALMAN_TRACKERS[tid] = HybridTrajectoryPredictor(
                        kalman, transformer,
                        kalman_weight=HYBRID_KALMAN_WEIGHT,
                        transformer_weight=HYBRID_TRANSFORMER_WEIGHT
                    )
                except Exception as e:
                    print(f"[Hybrid] Failed to create Transformer, using Kalman only: {e}")
                    KALMAN_TRACKERS[tid] = kalman
            else:
                KALMAN_TRACKERS[tid] = kalman
        
        KALMAN_TRACKERS[tid].update(actual_box)
        
        # === STEP 2: Generate predictions (with overprediction if enabled) ===
        predicted_trajectory = []
        uncertainties = []
        
        if USE_OVERPREDICTION:
            # DONUT-style overprediction: predict at key horizons for better anticipation
            multi_horizon_preds = KALMAN_TRACKERS[tid].predict_multi_horizon(OVERPREDICTION_HORIZONS)
            
            # Store multi-horizon predictions for evaluation
            for horizon, pred_box in multi_horizon_preds.items():
                if pred_box is not None:
                    target_frame = frame_idx + horizon
                    key = (tid, target_frame, horizon)
                    STORED_PREDICTIONS[key] = {
                        "prediction": pred_box,
                        "source_frame": frame_idx,
                        "step_ahead": horizon,
                        "predicted_at": frame_idx,
                        "uncertainty": None
                    }
        
        # Generate continuous trajectory for visualization
        for step in range(1, prediction_horizon + 1):
            target_frame = frame_idx + step
            
            # Get prediction with uncertainty if enabled
            if SHOW_UNCERTAINTY:
                result = KALMAN_TRACKERS[tid].predict(steps=step, return_uncertainty=True)
                if result[0] is not None:
                    pred_box, cov = result
                    predicted_trajectory.append(pred_box)
                    uncertainties.append(cov)
                else:
                    pred_box = None
            else:
                pred_box = KALMAN_TRACKERS[tid].predict(steps=step)
                if pred_box is not None:
                    predicted_trajectory.append(pred_box)
            
            # Store prediction if not using overprediction (avoid duplicate storage)
            if pred_box is not None and not USE_OVERPREDICTION:
                # Store each step's prediction with unique key
                key = (tid, target_frame, step)
                STORED_PREDICTIONS[key] = {
                    "prediction": pred_box,
                    "source_frame": frame_idx,
                    "step_ahead": step,
                    "predicted_at": frame_idx,
                    "uncertainty": cov.tolist() if SHOW_UNCERTAINTY and len(uncertainties) > 0 else None
                }
        
        # Store trajectory and uncertainty for visualization
        det["predicted_trajectory"] = predicted_trajectory
        det["prediction_uncertainties"] = uncertainties if SHOW_UNCERTAINTY else None
        
        # Get velocity for visualization
        velocity = KALMAN_TRACKERS[tid].get_velocity()
        det["velocity"] = velocity
        
        # === STEP 3: Evaluate against stored predictions ===
        # Check if we have a N-step prediction from N frames ago
        if frame_idx >= prediction_horizon:
            source_frame = frame_idx - prediction_horizon
            key = (tid, frame_idx, prediction_horizon)
            
            if key in STORED_PREDICTIONS:
                stored = STORED_PREDICTIONS[key]
                pred_box = stored["prediction"]
                
                # Compute error between prediction and actual
                actual_center = box_center(actual_box)
                pred_center = box_center(pred_box)
                error = np.linalg.norm(np.array(actual_center) - np.array(pred_center))
                iou = compute_iou(actual_box, pred_box)
                
                # Compute distance-weighted error (prioritize nearby objects)
                distance_weight = estimate_object_distance(actual_box)
                weighted_error = error * (2.0 - distance_weight)  # Closer = lower acceptable error
                
                # Store evaluation result
                eval_dict = {
                    "frame": frame_idx,
                    "track_id": tid,
                    "prediction_horizon": prediction_horizon,
                    "predicted_at_frame": source_frame,
                    "iou": round(iou, 4),
                    "center_distance": round(error, 2),
                    "weighted_distance": round(weighted_error, 2),
                    "distance_weight": round(distance_weight, 3),
                    "gt_box": actual_box,
                    "pred_box": [round(x, 2) for x in pred_box],
                    "full_trajectory": [[round(x, 2) for x in box] for box in predicted_trajectory]
                }
                
                # Add velocity if available
                if velocity is not None:
                    speed = np.linalg.norm(velocity)
                    eval_dict["velocity"] = round(speed, 2)
                
                ALL_EVALS.append(eval_dict)
                
                # Clean up evaluated prediction
                del STORED_PREDICTIONS[key]
        
        # === STEP 4: Also evaluate intermediate horizons (1 to N-1 steps) ===
        for step in range(1, prediction_horizon):
            if frame_idx >= step:
                source_frame = frame_idx - step
                key = (tid, frame_idx, step)
                
                if key in STORED_PREDICTIONS:
                    stored = STORED_PREDICTIONS[key]
                    pred_box = stored["prediction"]
                    
                    actual_center = box_center(actual_box)
                    pred_center = box_center(pred_box)
                    error = np.linalg.norm(np.array(actual_center) - np.array(pred_center))
                    iou = compute_iou(actual_box, pred_box)
                    
                    ALL_EVALS.append({
                        "frame": frame_idx,
                        "track_id": tid,
                        "prediction_horizon": step,
                        "predicted_at_frame": source_frame,
                        "iou": round(iou, 4),
                        "center_distance": round(error, 2),
                        "gt_box": actual_box,
                        "pred_box": [round(x, 2) for x in pred_box],
                        "full_trajectory": []  # Only store for max horizon
                    })
                    
                    del STORED_PREDICTIONS[key]
        
        # Clean up very old predictions (keep last 100 frames)
        keys_to_delete = [k for k in STORED_PREDICTIONS.keys() 
                         if k[1] < frame_idx - 100]
        for k in keys_to_delete:
            del STORED_PREDICTIONS[k]

def save_evaluation_summary():
    if not ALL_EVALS:
        print("No trajectory evaluation data found.")
        return

    with open(CSV_PATH, mode='w', newline='') as f:
        # Include new fields in CSV header
        writer = csv.DictWriter(f, fieldnames=["frame", "track_id", "prediction_horizon", "predicted_at_frame", "iou", "center_distance", "gt_box", "pred_box", "full_trajectory"])
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
    
    # Group by prediction horizon for detailed analysis
    by_horizon = defaultdict(list)
    for row in ALL_EVALS:
        horizon = row.get("prediction_horizon", 1)
        by_horizon[horizon].append(row)
    
    # Per-track aggregation for ADE/FDE/RMSE
    tracks = defaultdict(list)
    for row in ALL_EVALS:
        tracks[row['track_id']].append(row)

    per_track_ade = []
    per_track_fde = []
    per_track_rmse = []
    for tid, rows in tracks.items():
        # sort by frame
        rows_sorted = sorted(rows, key=lambda x: x['frame'])
        dlist = [r['center_distance'] for r in rows_sorted if r.get('center_distance') is not None]
        if not dlist:
            continue
        ade = float(np.mean(dlist))
        fde = float(dlist[-1])
        rmse = float(np.sqrt(np.mean([v * v for v in dlist])))
        per_track_ade.append(ade)
        per_track_fde.append(fde)
        per_track_rmse.append(rmse)

    # Overall stats
    overall = {
        'avg_iou': float(np.mean(ious)) if ious else None,
        'median_iou': float(np.median(ious)) if ious else None,
        'avg_center_distance': float(np.mean(dists)) if dists else None,
        'median_center_distance': float(np.median(dists)) if dists else None,
        'overall_ade': float(np.mean(per_track_ade)) if per_track_ade else None,
        'overall_fde': float(np.mean(per_track_fde)) if per_track_fde else None,
        'overall_rmse': float(np.mean(per_track_rmse)) if per_track_rmse else None,
        'tracks_evaluated': len(per_track_ade),
        'frames_evaluated': len(frame_ids),
        'total_rows': len(ALL_EVALS)
    }

    # Top worst tracks by ADE and by FDE
    worst_by_ade = sorted([(tid, float(np.mean([r['center_distance'] for r in rs]))) for tid, rs in tracks.items() if rs], key=lambda x: -x[1])[:5]
    worst_by_fde = sorted([(tid, float(sorted([r['center_distance'] for r in rs])[-1])) for tid, rs in tracks.items() if rs], key=lambda x: -x[1])[:5]

    with open(txt_path, 'w') as f:
        f.write("Trajectory Evaluation Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total evaluation rows: {overall['total_rows']}\n")
        f.write(f"Tracks evaluated (with distances): {overall['tracks_evaluated']}\n")
        f.write(f"Frames evaluated: {overall['frames_evaluated']}\n\n")

        f.write("Overall Metrics (All Horizons Combined):\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Average IoU: {overall['avg_iou']:.4f}\n")
        f.write(f"  Median IoU: {overall['median_iou']:.4f}\n")
        f.write(f"  Average Center Distance: {overall['avg_center_distance']:.2f} pixels\n")
        f.write(f"  Median Center Distance: {overall['median_center_distance']:.2f} pixels\n\n")

        f.write("Trajectory Errors (per-track averages):\n")
        f.write(f"  Overall ADE (mean of per-track ADE): {overall['overall_ade']:.2f} pixels\n")
        f.write(f"  Overall FDE (mean of per-track FDE): {overall['overall_fde']:.2f} pixels\n")
        f.write(f"  Overall RMSE (mean of per-track RMSE): {overall['overall_rmse']:.2f} pixels\n\n")

        # Per-horizon statistics
        f.write("Performance by Prediction Horizon:\n")
        f.write("-" * 60 + "\n")
        for horizon in sorted(by_horizon.keys()):
            rows = by_horizon[horizon]
            h_dists = [r['center_distance'] for r in rows]
            h_ious = [r['iou'] for r in rows]
            f.write(f"  {horizon}-step ahead ({horizon/6:.2f}s @ 6Hz):\n")
            f.write(f"    Samples: {len(rows)}\n")
            f.write(f"    Mean Error: {np.mean(h_dists):.2f} px\n")
            f.write(f"    Median Error: {np.median(h_dists):.2f} px\n")
            f.write(f"    Mean IoU: {np.mean(h_ious):.4f}\n\n")

        f.write("Top 5 Worst Tracks by ADE:\n")
        f.write("-" * 60 + "\n")
        for tid, val in worst_by_ade:
            f.write(f"  Track {tid}: {val:.2f} pixels\n")
        
        f.write("\nTop 5 Worst Tracks by FDE:\n")
        f.write("-" * 60 + "\n")
        for tid, val in worst_by_fde:
            f.write(f"  Track {tid}: {val:.2f} pixels\n")

    print(f"Accuracy summary saved to: {txt_path}")
    print(f"Trajectory evaluation saved to: {CSV_PATH}")