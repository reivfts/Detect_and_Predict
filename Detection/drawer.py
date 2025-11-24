import cv2
import numpy as np
from ultralytics.utils.plotting import colors
from config import (
    SHOW_UNCERTAINTY, 
    UNCERTAINTY_SCALE, 
    VELOCITY_COLOR_CODING,
    VELOCITY_SLOW_THRESHOLD,
    VELOCITY_FAST_THRESHOLD
)


class Drawer:
    """Improved drawer with translucent boxes, smoothed positions and track trails.

    Display-only changes: does not modify detection outputs or tracking IDs.
    """

    def __init__(self, line_width=2, box_alpha=0.3, smoothing_alpha=1.0, max_trail=15):
        self.line_width = line_width
        self.box_alpha = box_alpha  # Increased from 0.2 to 0.3 for better visibility
        # smoothing for box positions: new = alpha*new + (1-alpha)*old
        # alpha=1.0 -> no smoothing, boxes snap exactly to detections (no lag)
        self.smoothing_alpha = smoothing_alpha
        # track history for smoothing and trails: {track_id: last_box}
        self.history = {}
        # track trails for trajectory visualization: {track_id: [centers...]}
        self.trails = {}
        self.max_trail = max_trail  # Reduced from 30 to 15 for cleaner view
        # Track velocity for directional prediction visualization
        self.velocity_cache = {}  # {track_id: (vx, vy)}
        # Track last position to compute actual movement
        self.last_positions = {}  # {track_id: (x, y)}
        self.movement_threshold = 1.0  # pixels per frame to consider "moving"

    def _color_for_id(self, track_id_or_name, velocity=None):
        """Get color for track, optionally modulated by velocity."""
        # Fixed colors per class for visual consistency
        class_colors = {
            "car": (255, 0, 0),          # Blue
            "truck": (0, 255, 0),        # Green
            "bus": (0, 165, 255),        # Orange
            "person": (0, 0, 255),       # Red
            "motorcycle": (255, 0, 255), # Magenta
        }
        
        # Base color
        if isinstance(track_id_or_name, str):
            base_color = class_colors.get(track_id_or_name.lower(), (128, 128, 128))
        else:
            col = colors(track_id_or_name, bgr=True)
            base_color = tuple(int(c) for c in col)
        
        # Apply velocity-based color coding if enabled
        if VELOCITY_COLOR_CODING and velocity is not None:
            vx, vy = velocity
            speed = np.sqrt(vx**2 + vy**2)  # pixels per frame
            
            # Color coding: green (slow) -> yellow (medium) -> red (fast)
            if speed < VELOCITY_SLOW_THRESHOLD:
                # Slow - greenish tint
                return tuple(int(c * 0.7) if i != 1 else int(min(255, c * 1.3)) 
                           for i, c in enumerate(base_color))
            elif speed > VELOCITY_FAST_THRESHOLD:
                # Fast - reddish tint
                return tuple(int(c * 0.7) if i != 2 else int(min(255, c * 1.3)) 
                           for i, c in enumerate(base_color))
            else:
                # Medium speed - yellowish tint
                factor = (speed - VELOCITY_SLOW_THRESHOLD) / (VELOCITY_FAST_THRESHOLD - VELOCITY_SLOW_THRESHOLD)
                return tuple(int(c * (1.0 - 0.3 * factor)) if i == 0 else int(min(255, c * (1.0 + 0.3 * factor)))
                           for i, c in enumerate(base_color))
        
        return base_color

    def _smooth_box(self, track_id, box):
        """Exponential smoothing of box coordinates per track"""
        if track_id is None:
            return box
        prev = self.history.get(track_id)
        # store floats internally for smoother transitions, convert to int when drawing
        if prev is None:
            self.history[track_id] = [float(v) for v in box]
            return [int(v) for v in box]

        alpha = self.smoothing_alpha
        sm = [alpha * float(b) + (1 - alpha) * float(p) for b, p in zip(box, prev)]
        self.history[track_id] = sm
        return [int(v) for v in sm]

    def draw_box(self, frame, box, cls_name, track_id=None, score=None, velocity=None, 
                 predicted_trajectory=None, prediction_uncertainties=None, is_moving=False):
        """Draw a translucent, smoothed bounding box with label, trail, and predicted trajectory.

        Args:
            frame: BGR image
            box: [x1,y1,x2,y2]
            cls_name: class label string
            track_id: optional track id (int)
            score: optional confidence score (float)
            velocity: optional (vx, vy) velocity tuple (pixels/second)
            predicted_trajectory: optional list of predicted boxes [[x1,y1,x2,y2], ...]
            prediction_uncertainties: optional list of 2-by-2 covariance matrices for each prediction
            is_moving: whether object is currently moving (affects trajectory display)
        """
        x1, y1, x2, y2 = map(int, box)

        # smoothing
        sm_box = self._smooth_box(track_id, [x1, y1, x2, y2])
        x1, y1, x2, y2 = sm_box

        color = self._color_for_id(cls_name, velocity)  # Use class name and velocity for color

        # Translucent filled rectangle with reduced alpha for cleaner appearance
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, self.box_alpha, frame, 1 - self.box_alpha, 0, frame)

        # border with thinner line
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, self.line_width))

        # Display class name label for clarity
        label = cls_name

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Label background with compact padding for better visibility
        lx1, ly1 = x1, max(0, y1 - th - 6)
        lx2, ly2 = x1 + tw + 8, y1
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.putText(frame, label, (lx1 + 4, ly2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw velocity vector if available
        if velocity is not None:
            vx, vy = velocity
            if vx != 0 or vy != 0:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                # Scale velocity for visualization (assuming ~10 FPS, show next ~5 frames)
                scale = 0.5  # Adjust for visual clarity
                end_x = int(center[0] + vx * scale)
                end_y = int(center[1] + vy * scale)
                # Draw arrow
                cv2.arrowedLine(frame, center, (end_x, end_y), color, 2, tipLength=0.3)

        # update trail (ACTUAL PAST TRAJECTORY - solid for "ground truth")
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Use provided is_moving flag, or calculate if not provided
        if not is_moving and track_id is not None:
            # Fallback calculation if is_moving not provided
            if track_id in self.last_positions:
                last_pos = self.last_positions[track_id]
                movement = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                is_moving = movement > self.movement_threshold
            
        if track_id is not None:
            self.last_positions[track_id] = center
            
            # Only add to trail if moving
            if is_moving:
                pts = self.trails.get(track_id, [])
                pts.append(center)
                if len(pts) > self.max_trail:
                    pts = pts[-self.max_trail:]
                self.trails[track_id] = pts
            else:
                # Clear trail for stopped objects
                if track_id in self.trails and len(self.trails[track_id]) > 0:
                    self.trails[track_id] = [center]  # Keep only current position
            # draw ACTUAL trail only if object has been moving (solid line with gradient)
            pts = self.trails.get(track_id, [])
            if len(pts) > 2:  # Need at least 3 points for meaningful trail
                for i in range(1, len(pts)):
                    thickness = max(1, int(2 * (i / len(pts))))  # Reduced from 2-3 to 1-2
                    cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)
                
                # Mark latest actual position with "A" (Actual) - smaller
                cv2.circle(frame, pts[-1], 3, color, 1)  # Reduced from 5,2 to 3,1
                cv2.putText(frame, "A", (pts[-1][0]-3, pts[-1][1]+3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
        
        # Draw predicted trajectory ONLY for moving objects
        if predicted_trajectory is not None and len(predicted_trajectory) > 0 and is_moving:
            # Store velocity for this track
            if velocity is not None and track_id is not None:
                self.velocity_cache[track_id] = velocity
            
            # Collect predicted path centers
            trajectory_points = []
            for pred_box in predicted_trajectory:
                pred_x1, pred_y1, pred_x2, pred_y2 = map(int, pred_box)
                pred_center = ((pred_x1 + pred_x2) // 2, (pred_y1 + pred_y2) // 2)
                trajectory_points.append(pred_center)
            
            # Draw PREDICTED trajectory showing where object is heading
            if trajectory_points and len(trajectory_points) >= 2:
                # All points: current + predicted future
                all_points = [center] + trajectory_points
                
                # Draw direction arrow at start (shows movement direction)
                if track_id in self.velocity_cache:
                    vx, vy = self.velocity_cache[track_id]
                    speed = np.sqrt(vx**2 + vy**2)
                    if speed > 0.5:  # Only if actually moving
                        # Larger arrow for better visibility
                        arrow_length = min(40, int(speed * 3))  # Scale with speed
                        arrow_end = (int(center[0] + vx * arrow_length / speed), 
                                   int(center[1] + vy * arrow_length / speed))
                        # Draw thick arrow showing direction
                        cv2.arrowedLine(frame, center, arrow_end, (255, 255, 255), 4, 
                                      tipLength=0.3, line_type=cv2.LINE_AA)
                        cv2.arrowedLine(frame, center, arrow_end, color, 2, 
                                      tipLength=0.3, line_type=cv2.LINE_AA)
                
                # Draw smooth predicted path line (solid, thinner than arrow)
                for i in range(len(all_points) - 1):
                    # Fade opacity along the path (more confident near, less far)
                    alpha = 1.0 - (i / len(all_points)) * 0.5
                    thickness = 2
                    
                    # Draw white outline then colored line
                    cv2.line(frame, all_points[i], all_points[i + 1], 
                           (255, 255, 255), thickness + 2, cv2.LINE_AA)
                    cv2.line(frame, all_points[i], all_points[i + 1], 
                           color, thickness, cv2.LINE_AA)
                
                # Draw endpoint with "P" marker
                endpoint = trajectory_points[-1]
                cv2.circle(frame, endpoint, 5, (255, 255, 255), -1)
                cv2.circle(frame, endpoint, 4, color, -1)
                cv2.putText(frame, "P", (endpoint[0]-3, endpoint[1]+3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    
    def _draw_uncertainty_ellipse(self, frame, center, covariance, color, alpha=0.3):
        """Draw uncertainty ellipse from 2-by-2 covariance matrix."""
        try:
            # Convert to numpy array if needed
            if isinstance(covariance, list):
                cov = np.array(covariance)
            else:
                cov = covariance
            
            # Ensure 2-by-2 shape
            if cov.shape != (2, 2):
                return
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            
            # Get angle and axes lengths (2-sigma = 95% confidence)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width = 2 * np.sqrt(eigenvalues[0]) * UNCERTAINTY_SCALE
            height = 2 * np.sqrt(eigenvalues[1]) * UNCERTAINTY_SCALE
            
            # Draw ellipse
            overlay = frame.copy()
            cv2.ellipse(overlay, center, (int(width), int(height)), angle, 
                       0, 360, color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw ellipse outline
            cv2.ellipse(frame, center, (int(width), int(height)), angle,
                       0, 360, color, 2)
        except Exception as e:
            # Silently skip if ellipse drawing fails
            pass

    def draw_mask(self, frame, mask, track_id, cls_name=None):
        """Overlay a semi-transparent mask on the frame."""
        color = self._color_for_id(cls_name if cls_name else track_id)
        color = np.array(color, dtype=np.uint8)
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[mask > 0] = color
        alpha = 0.4
        cv2.addWeighted(colored_mask, alpha, frame, 1 - alpha, 0, frame)

    def draw_id_label(self, frame, box, track_id):
        x1, y1, x2, y2 = map(int, box)
        cv2.putText(
            frame,
            f"ID:{track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    def draw_fps(self, frame, fps, pos=(20, 30)):
        cv2.putText(frame, f"FPS: {fps:.1f}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def draw_ego_trajectory(self, frame, ego_history, future_steps=12):
        """
        Draw camera/ego vehicle trajectory prediction.
        Shows where the vehicle (camera) is heading based on recent motion.
        
        Args:
            frame: Image to draw on
            ego_history: List of recent ego positions [(x, y), ...] in image space
            future_steps: Number of future positions to predict
        """
        if len(ego_history) < 3:
            return
        
        # Compute camera motion vector from recent history
        recent = ego_history[-3:]
        velocities = []
        for i in range(1, len(recent)):
            vx = recent[i][0] - recent[i-1][0]
            vy = recent[i][1] - recent[i-1][1]
            velocities.append((vx, vy))
        
        # Average velocity
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)
        
        # Predict future camera positions
        h, w = frame.shape[:2]
        current_pos = ego_history[-1]
        
        # Draw ego trajectory as a lane on the bottom of the frame
        ego_color = (0, 255, 255)  # Yellow for ego trajectory
        
        predicted_positions = []
        for step in range(1, future_steps + 1):
            pred_x = int(current_pos[0] + avg_vx * step)
            pred_y = int(current_pos[1] + avg_vy * step)
            
            # Clamp to frame bounds
            pred_x = max(0, min(w - 1, pred_x))
            pred_y = max(0, min(h - 1, pred_y))
            
            predicted_positions.append((pred_x, pred_y))
        
        # Draw predicted ego path
        if predicted_positions:
            # Draw base of predicted path
            for i in range(len(predicted_positions) - 1):
                cv2.line(frame, predicted_positions[i], predicted_positions[i + 1], 
                        (255, 255, 255), 8, cv2.LINE_AA)
                cv2.line(frame, predicted_positions[i], predicted_positions[i + 1], 
                        ego_color, 5, cv2.LINE_AA)
            
            # Draw markers
            for i, pos in enumerate(predicted_positions):
                if i % 3 == 0:  # Every 3rd position
                    cv2.circle(frame, pos, 8, (255, 255, 255), -1)
                    cv2.circle(frame, pos, 6, ego_color, -1)
        
        # Add label
        label_pos = (20, h - 30)
        cv2.putText(frame, "EGO TRAJECTORY", label_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, "EGO TRAJECTORY", label_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ego_color, 2, cv2.LINE_AA)

