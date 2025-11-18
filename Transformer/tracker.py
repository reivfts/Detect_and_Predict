# tracker.py

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque
import os

class TransformerTracker:
    def __init__(self, iou_thresh=0.5, max_lost=5, history_len=10, log_path="C:\\Users\\rayra\\OneDrive\\Desktop\\Detect_and_Predict\\data\\trackings\\trajectory_eval.txt"):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.next_id = 0
        self.tracks = dict()
        self.track_history = defaultdict(lambda: deque(maxlen=history_len))
        self.log_path = log_path

        # Clear existing file
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, 'w') as f:
            f.write("track_id,frame,x_pred,y_pred,x_actual,y_actual,error\n")

    def iou(self, boxA, boxB):
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

    def predict_next_center(self, history):
        if len(history) < 2:
            return None
        (x1, y1, x2, y2) = history[-2]
        (x1_, y1_, x2_, y2_) = history[-1]
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        cx0, cy0 = (x1_ + x2_) / 2, (y1_ + y2_) / 2
        dx, dy = cx1 - cx0, cy1 - cy0
        return (cx1 + dx, cy1 + dy)

    def update(self, detections, frame_idx):
        updated_tracks = dict()
        updated_ids = set()

        det_boxes = [d["box"] for d in detections]
        det_clss = [d["cls_name"] for d in detections]
        det_scores = [d["score"] for d in detections]

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]["box"] for tid in track_ids]
        track_clss = [self.tracks[tid]["cls_name"] for tid in track_ids]

        if det_boxes and track_boxes:
            cost_matrix = np.ones((len(track_boxes), len(det_boxes)), dtype=np.float32)

            for i, (tb, tc) in enumerate(zip(track_boxes, track_clss)):
                for j, (db, dc) in enumerate(zip(det_boxes, det_clss)):
                    if tc != dc:
                        continue
                    iou_val = self.iou(tb, db)
                    cost_matrix[i, j] = 1 - iou_val

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_dets = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < (1 - self.iou_thresh):
                    tid = track_ids[r]
                    self.tracks[tid] = {
                        "box": det_boxes[c],
                        "cls_name": det_clss[c],
                        "score": det_scores[c],
                        "last_seen": frame_idx
                    }
                    self.track_history[tid].append(det_boxes[c])
                    updated_tracks[tid] = self.tracks[tid]
                    updated_ids.add(tid)
                    assigned_dets.add(c)

                    pred = self.predict_next_center(self.track_history[tid])
                    if pred is not None:
                        x1, y1, x2, y2 = det_boxes[c]
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        error = np.sqrt((pred[0] - cx) ** 2 + (pred[1] - cy) ** 2)
                        with open(self.log_path, 'a') as f:
                            f.write(f"{tid},{frame_idx},{pred[0]:.2f},{pred[1]:.2f},{cx:.2f},{cy:.2f},{error:.4f}\n")

            for j, (box, cls_name, score) in enumerate(zip(det_boxes, det_clss, det_scores)):
                if j not in assigned_dets:
                    tid = self.next_id
                    self.next_id += 1
                    self.tracks[tid] = {
                        "box": box,
                        "cls_name": cls_name,
                        "score": score,
                        "last_seen": frame_idx
                    }
                    self.track_history[tid].append(box)
                    updated_tracks[tid] = self.tracks[tid]
                    updated_ids.add(tid)

        else:
            for j, (box, cls_name, score) in enumerate(zip(det_boxes, det_clss, det_scores)):
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "box": box,
                    "cls_name": cls_name,
                    "score": score,
                    "last_seen": frame_idx
                }
                self.track_history[tid].append(box)
                updated_tracks[tid] = self.tracks[tid]
                updated_ids.add(tid)

        to_remove = [tid for tid, trk in self.tracks.items() if frame_idx - trk["last_seen"] > self.max_lost]
        for tid in to_remove:
            del self.tracks[tid]
            if tid in self.track_history:
                del self.track_history[tid]

        self.tracks = updated_tracks

        output = []
        for tid in updated_ids:
            trk = self.tracks[tid]
            output.append({
                "track_id": tid,
                "box": trk["box"],
                "cls_name": trk["cls_name"],
                "score": trk["score"]
            })
        return output