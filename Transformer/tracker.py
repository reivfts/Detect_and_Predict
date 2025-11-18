import numpy as np

class StableIDTracker:
    """
    Maintains stable IDs across frames using simple IoU matching.
    This does NOT replace YOLO tracking — it stabilizes flickering IDs.
    """
    def __init__(self, iou_threshold=0.4):
        self.iou_threshold = iou_threshold
        self.prev_tracks = {}   # track_id → {box, cls_name}
        self.next_stable_id = 1
        self.yolo_to_stable = {}  # YOLO ID → stable ID

    def _iou(self, boxA, boxB):
        x1A, y1A, x2A, y2A = boxA
        x1B, y1B, x2B, y2B = boxB

        xi1 = max(x1A, x1B)
        yi1 = max(y1A, y1B)
        xi2 = min(x2A, x2B)
        yi2 = min(y2A, y2B)

        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter = inter_w * inter_h

        areaA = max(0, x2A - x1A) * max(0, y2A - y1A)
        areaB = max(0, x2B - x1B) * max(0, y2B - y1B)

        union = areaA + areaB - inter
        if union <= 0:
            return 0.0

        return inter / union

    def update(self, detections, frame_idx):
        """
        Convert YOLO track IDs → stable IDs, based on IoU match to previous frame.
        """

        updated = []

        for det in detections:
            yolo_id = det["track_id"]
            box = det["box"]

            # --- If YOLO ID already seen, use its stable ID ---
            if yolo_id in self.yolo_to_stable:
                stable_id = self.yolo_to_stable[yolo_id]

            else:
                # Try to match by IoU to old tracks
                best_iou = 0
                best_sid = None

                for sid, prev in self.prev_tracks.items():
                    i = self._iou(prev["box"], box)
                    if i > best_iou:
                        best_iou = i
                        best_sid = sid

                if best_iou > self.iou_threshold:
                    # Reuse old stable ID
                    stable_id = best_sid
                else:
                    # Assign brand-new stable ID
                    stable_id = self.next_stable_id
                    self.next_stable_id += 1

                # Map YoloID → stableID
                self.yolo_to_stable[yolo_id] = stable_id

            # Write stable ID into detection
            det["stable_id"] = stable_id
            updated.append(det)

        # Save detections for next frame
        self.prev_tracks = {det["stable_id"]: {"box": det["box"]} for det in updated}

        return updated
