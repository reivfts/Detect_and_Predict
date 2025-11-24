"""
Velocity Estimation Module for Trajectory Prediction Enhancement.

Implements temporal velocity estimation using frame-to-frame differences,
as recommended by the nuScenes paper finding that velocity estimation
significantly improves trajectory prediction accuracy.

Provides:
    - Velocity estimation from bounding box movements
    - Smoothed velocity using exponential moving average
    - Velocity-aware trajectory prediction
"""

import numpy as np
from collections import defaultdict, deque


class VelocityEstimator:
    """
    Estimates velocity for tracked objects using temporal differences.
    
    Maintains velocity history per track and provides smoothed estimates
    using exponential moving average.
    """
    
    def __init__(self, fps=10.0, smoothing_alpha=0.3, history_size=5):
        """
        Initialize velocity estimator.
        
        Args:
            fps: Frames per second (for dt calculation)
            smoothing_alpha: EMA smoothing factor (0=no smoothing, 1=instant response)
            history_size: Number of velocity samples to keep for statistics
        """
        self.dt = 1.0 / fps  # time step in seconds
        self.smoothing_alpha = smoothing_alpha
        self.history_size = history_size
        
        # Per-track data
        self.last_positions = {}  # track_id -> (cx, cy, frame_idx)
        self.velocities = {}  # track_id -> (vx, vy) current estimate
        self.velocity_history = defaultdict(lambda: deque(maxlen=history_size))
    
    def update(self, track_id, box, frame_idx):
        """
        Update velocity estimate for a track.
        
        Args:
            track_id: Unique track identifier
            box: [x1, y1, x2, y2] bounding box
            frame_idx: Current frame number
        
        Returns:
            velocity: (vx, vy) in pixels/second, or None if not enough data
        """
        # Compute center position
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        
        # Check if we have previous position
        if track_id not in self.last_positions:
            # Initialize
            self.last_positions[track_id] = (cx, cy, frame_idx)
            self.velocities[track_id] = (0.0, 0.0)
            return None
        
        # Get previous position
        prev_cx, prev_cy, prev_frame_idx = self.last_positions[track_id]
        
        # Compute time difference
        frame_diff = frame_idx - prev_frame_idx
        if frame_diff <= 0:
            return self.velocities[track_id]
        
        time_diff = frame_diff * self.dt
        
        # Compute raw velocity
        vx_raw = (cx - prev_cx) / time_diff
        vy_raw = (cy - prev_cy) / time_diff
        
        # Smooth with EMA
        if track_id in self.velocities:
            prev_vx, prev_vy = self.velocities[track_id]
            vx_smooth = self.smoothing_alpha * vx_raw + (1 - self.smoothing_alpha) * prev_vx
            vy_smooth = self.smoothing_alpha * vy_raw + (1 - self.smoothing_alpha) * prev_vy
        else:
            vx_smooth = vx_raw
            vy_smooth = vy_raw
        
        # Update state
        self.last_positions[track_id] = (cx, cy, frame_idx)
        self.velocities[track_id] = (vx_smooth, vy_smooth)
        self.velocity_history[track_id].append((vx_smooth, vy_smooth))
        
        return (vx_smooth, vy_smooth)
    
    def get_velocity(self, track_id):
        """
        Get current velocity estimate for a track.
        
        Returns:
            (vx, vy) or (0, 0) if track not found
        """
        return self.velocities.get(track_id, (0.0, 0.0))
    
    def get_speed(self, track_id):
        """
        Get speed (velocity magnitude) for a track.
        
        Returns:
            speed in pixels/second
        """
        vx, vy = self.get_velocity(track_id)
        return np.sqrt(vx**2 + vy**2)
    
    def get_heading(self, track_id):
        """
        Get heading direction (angle) for a track.
        
        Returns:
            angle in radians (0 = right, pi/2 = down, pi = left, -pi/2 = up)
        """
        vx, vy = self.get_velocity(track_id)
        return np.arctan2(vy, vx)
    
    def get_velocity_stats(self, track_id):
        """
        Get velocity statistics (mean, std) from history.
        
        Returns:
            dict with 'mean', 'std', 'speed_mean', 'speed_std'
        """
        if track_id not in self.velocity_history:
            return None
        
        history = list(self.velocity_history[track_id])
        if not history:
            return None
        
        vx_vals = [v[0] for v in history]
        vy_vals = [v[1] for v in history]
        speeds = [np.sqrt(vx**2 + vy**2) for vx, vy in history]
        
        return {
            'mean': (np.mean(vx_vals), np.mean(vy_vals)),
            'std': (np.std(vx_vals), np.std(vy_vals)),
            'speed_mean': np.mean(speeds),
            'speed_std': np.std(speeds)
        }
    
    def cleanup_track(self, track_id):
        """Remove data for lost track."""
        if track_id in self.last_positions:
            del self.last_positions[track_id]
        if track_id in self.velocities:
            del self.velocities[track_id]
        if track_id in self.velocity_history:
            del self.velocity_history[track_id]
    
    def predict_position(self, track_id, box, num_steps):
        """
        Predict future positions using current velocity.
        
        Args:
            track_id: Track identifier
            box: Current bounding box [x1, y1, x2, y2]
            num_steps: Number of future steps to predict
        
        Returns:
            predicted_boxes: list of (num_steps) predicted boxes
        """
        vx, vy = self.get_velocity(track_id)
        
        # Current center and size
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        predicted_boxes = []
        for step in range(1, num_steps + 1):
            # Predict center position
            pred_cx = cx + vx * self.dt * step
            pred_cy = cy + vy * self.dt * step
            
            # Assume constant size
            pred_box = [
                pred_cx - w/2,
                pred_cy - h/2,
                pred_cx + w/2,
                pred_cy + h/2
            ]
            predicted_boxes.append(pred_box)
        
        return predicted_boxes


class VelocityAwareTrajectoryPredictor:
    """
    Trajectory predictor that incorporates velocity estimation.
    
    Uses VelocityEstimator to enhance trajectory predictions with
    temporal velocity information.
    """
    
    def __init__(self, fps=10.0, smoothing_alpha=0.3):
        self.velocity_estimator = VelocityEstimator(
            fps=fps,
            smoothing_alpha=smoothing_alpha
        )
    
    def update(self, track_id, box, frame_idx):
        """Update velocity estimate."""
        return self.velocity_estimator.update(track_id, box, frame_idx)
    
    def predict(self, track_id, box, num_steps=10, use_acceleration=False):
        """
        Predict future trajectory with velocity awareness.
        
        Args:
            track_id: Track identifier
            box: Current box [x1, y1, x2, y2]
            num_steps: Prediction horizon
            use_acceleration: Whether to estimate acceleration from velocity history
        
        Returns:
            trajectory: (num_steps, 4) predicted boxes
            velocity: (vx, vy) current velocity
        """
        vx, vy = self.velocity_estimator.get_velocity(track_id)
        
        # Current state
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        # Estimate acceleration if requested
        ax, ay = 0.0, 0.0
        if use_acceleration:
            velocity_stats = self.velocity_estimator.get_velocity_stats(track_id)
            if velocity_stats and len(self.velocity_estimator.velocity_history[track_id]) >= 2:
                # Simple finite difference for acceleration
                history = list(self.velocity_estimator.velocity_history[track_id])
                v_recent = history[-1]
                v_prev = history[-2]
                dt = self.velocity_estimator.dt
                ax = (v_recent[0] - v_prev[0]) / dt
                ay = (v_recent[1] - v_prev[1]) / dt
        
        # Predict trajectory
        trajectory = []
        dt = self.velocity_estimator.dt
        
        for step in range(1, num_steps + 1):
            # Kinematic prediction: x = x0 + v*t + 0.5*a*t^2
            t = step * dt
            pred_cx = cx + vx * t + 0.5 * ax * t**2
            pred_cy = cy + vy * t + 0.5 * ay * t**2
            
            # Assume constant size
            pred_box = [
                pred_cx - w/2,
                pred_cy - h/2,
                pred_cx + w/2,
                pred_cy + h/2
            ]
            trajectory.append(pred_box)
        
        return np.array(trajectory), (vx, vy)
    
    def cleanup_track(self, track_id):
        """Cleanup track data."""
        self.velocity_estimator.cleanup_track(track_id)


# Global velocity estimator instance for main pipeline
GLOBAL_VELOCITY_ESTIMATOR = None

def get_velocity_estimator(fps=10.0):
    """Get or create global velocity estimator instance."""
    global GLOBAL_VELOCITY_ESTIMATOR
    if GLOBAL_VELOCITY_ESTIMATOR is None:
        GLOBAL_VELOCITY_ESTIMATOR = VelocityEstimator(fps=fps)
    return GLOBAL_VELOCITY_ESTIMATOR
