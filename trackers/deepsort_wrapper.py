# trackers/deepsort_wrapper.py

"""
DeepSORT Tracker Wrapper

Integrates standard DeepSORT as a baseline for comparison with TransformerTracker.
Uses deep_sort_realtime library with built-in MobileNet feature extractor.
"""

from typing import List, Dict, Optional
import numpy as np


class DeepSORTWrapper:
    """
    Wrapper for DeepSORT tracker to maintain consistent interface with TransformerTracker.
    
    DeepSORT combines:
    - Kalman filtering for motion prediction
    - Deep appearance features (CNN embeddings)
    - Hungarian algorithm for data association
    """
    
    def __init__(self, max_age: int = 30, n_init: int = 3, max_iou_distance: float = 0.7):
        """
        Initialize DeepSORT tracker.
        
        Args:
            max_age: Maximum frames to keep track alive without detection
            n_init: Number of consecutive detections before track is confirmed
            max_iou_distance: Maximum IoU distance for matching
        """
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError:
            raise ImportError(
                "deep-sort-realtime not installed. "
                "Install with: pip install deep-sort-realtime"
            )
        
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder="mobilenet",
            embedder_gpu=True,  # Use GPU if available
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
        # For tracking history (to maintain compatibility with TransformerTracker)
        self.track_history = {}
    
    def update(
        self, 
        detections: List[Dict], 
        frame_idx: int, 
        frame: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of dicts with keys: box, cls_name, score
            frame_idx: Current frame index
            frame: Optional frame image (required for DeepSORT embeddings)
        
        Returns:
            List of tracked objects with track_id assigned
        """
        if frame is None:
            # DeepSORT needs frame for appearance embedding
            # Create a dummy frame if not provided (fall back to position-only tracking)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Convert detections to DeepSORT format
        # DeepSORT expects: ([left, top, width, height], confidence, class_name)
        raw_detections = []
        for det in detections:
            box = det["box"]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            
            # Convert to [left, top, width, height]
            left = x1
            top = y1
            width = x2 - x1
            height = y2 - y1
            
            raw_detections.append((
                [left, top, width, height],
                det["score"],
                det["cls_name"]
            ))
        
        # Update tracker (frame is guaranteed to be not None at this point)
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        
        # Convert back to our format
        output = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltwh = track.to_ltwh()  # [left, top, width, height]
            
            # Convert back to [x1, y1, x2, y2]
            x1 = ltwh[0]
            y1 = ltwh[1]
            x2 = ltwh[0] + ltwh[2]
            y2 = ltwh[1] + ltwh[3]
            
            box = [x1, y1, x2, y2]
            
            # Update history
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(box)
            
            # Keep only recent history
            if len(self.track_history[track_id]) > 10:
                self.track_history[track_id].pop(0)
            
            output.append({
                "track_id": track_id,
                "box": box,
                "cls_name": track.get_det_class(),
                "score": track.get_det_conf()
            })
        
        return output
