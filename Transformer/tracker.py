# tracker.py

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque
import os

# Kalman filter for motion prediction
from filterpy.kalman import KalmanFilter
import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
from config import APPEARANCE_MATCHING, APPEARANCE_WEIGHT, IOU_WEIGHT, EMBEDDING_DEVICE

class TransformerTracker:
    def __init__(self, iou_thresh=0.5, max_lost=10, history_len=10, log_path=None):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.next_id = 0
        self.tracks = dict()
        self.track_history = defaultdict(lambda: deque(maxlen=history_len))
        
        # Default log path inside project data folder when not provided
        if log_path is None:
            from config import BASE_DIR
            log_path = os.path.join(BASE_DIR, "data", "trackings", "trajectory_eval.txt")
        
        self.log_path = log_path

        # Clear existing file and write header
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w') as f:
            f.write("track_id,frame,x_pred,y_pred,x_actual,y_actual,error\n")

    # Helper: Kalman filter creation
    def _create_kalman(self, cx, cy, w, h):
        # State: [cx, cy, vx, vy, w, h]
        kf = KalmanFilter(dim_x=6, dim_z=4)

        dt = 1.0
        kf.F = np.array([
            [1, 0, dt, 0, 0, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)

        # Measurement matrix: z = [cx, cy, w, h]
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)

        # Initial state
        kf.x = np.array([cx, cy, 0., 0., w, h], dtype=float).reshape((6, 1))

        # Covariances
        kf.P *= 10.0
        kf.R *= 1.0
        q = 1.0
        kf.Q = np.eye(6) * q

        return kf

    def _box_to_z(self, box):
        # box: [x1,y1,x2,y2] -> z: [cx,cy,w,h]
        x1, y1, x2, y2 = box
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return np.array([cx, cy, w, h], dtype=float)

    def _init_embed_model(self):
        # Initialize a small ResNet18 feature extractor (remove final fc)
        self.embed_device = EMBEDDING_DEVICE if EMBEDDING_DEVICE is not None else "cpu"
        model = resnet18(pretrained=True)
        # remove classifier head -> output of avgpool (512)
        modules = list(model.children())[:-1]
        self.embed_model = torch.nn.Sequential(*modules)
        self.embed_model.eval()
        self.embed_model.to(self.embed_device)

        # transform: resize -> center crop -> to tensor -> normalize
        self.embed_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _z_to_box(self, z):
        # z: [cx,cy,w,h] -> box: [x1,y1,x2,y2]
        cx, cy, w, h = z
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return [x1, y1, x2, y2]

    def _clamp_box(self, box):
        # Ensure boxes have non-negative coords and sensible sizes
        x1, y1, x2, y2 = box
        x1 = max(0.0, float(x1))
        y1 = max(0.0, float(y1))
        x2 = max(x1 + 1.0, float(x2))
        y2 = max(y1 + 1.0, float(y2))
        return [x1, y1, x2, y2]

    def _mahalanobis_distance(self, kf, meas):
        # meas: [cx,cy,w,h]
        # innovation y = z - Hx
        H = kf.H
        x = kf.x.reshape((-1, 1))
        z = np.array(meas, dtype=float).reshape((-1, 1))
        y = z - H.dot(x)
        S = H.dot(kf.P).dot(H.T) + kf.R
        try:
            invS = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return np.inf
        d2 = float(y.T.dot(invS).dot(y))
        return d2

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
        # history[-2] is older, history[-1] is most recent
        old = history[-2]
        new = history[-1]

        cx_old = (old[0] + old[2]) / 2
        cy_old = (old[1] + old[3]) / 2

        cx_new = (new[0] + new[2]) / 2
        cy_new = (new[1] + new[3]) / 2

        # velocity = new - old; predicted center = new + velocity
        dx = cx_new - cx_old
        dy = cy_new - cy_old

        return (cx_new + dx, cy_new + dy)

    def predict_tracks(self):
        """Predict current track positions using internal Kalman filters without
        modifying track state (safe for visualization when detections are delayed).

        Returns:
            List of dicts: [{"track_id", "box", "cls_name", "score"}, ...]
        """
        output = []
        for tid, trk in list(self.tracks.items()):
            kf = trk.get("kf")
            if kf is not None:
                # Predict state but do not call update on stored KF (we call predict locally)
                # Use a copy of the state to avoid side-effects
                try:
                    x_copy = kf.x.copy()
                    F = kf.F.copy()
                    x_pred = F.dot(x_copy)
                    z_pred = np.array([x_pred[0, 0], x_pred[1, 0], x_pred[4, 0], x_pred[5, 0]])
                    pred_box = self._z_to_box(z_pred)
                except Exception:
                    pred_box = trk["box"]
            else:
                pred_box = trk["box"]

            pred_box = self._clamp_box(pred_box)
            output.append({
                "track_id": tid,
                "box": pred_box,
                "cls_name": trk.get("cls_name"),
                "score": trk.get("score", 1.0)
            })

        return output

    def update(self, detections, frame_idx, frame=None):
        updated_tracks = dict()
        updated_ids = set()

        # Lazy init feature extractor if appearance matching enabled
        if APPEARANCE_MATCHING and not hasattr(self, "embed_model"):
            self._init_embed_model()

        det_boxes = [d["box"] for d in detections]
        det_clss = [d["cls_name"] for d in detections]
        det_scores = [d["score"] for d in detections]

        # Clamp detection boxes to avoid negative coords / empty crops
        det_boxes = [self._clamp_box(box) for box in det_boxes]

        # Compute appearance embeddings for detections if frame provided
        det_embeds = [None] * len(det_boxes)
        if APPEARANCE_MATCHING and frame is not None and len(det_boxes) > 0:
            for i, box in enumerate(det_boxes):
                x1, y1, x2, y2 = map(int, box)
                # Clamp
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))
                try:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    img = Image.fromarray(crop[:, :, ::-1])
                    inp = self.embed_transform(img).unsqueeze(0).to(self.embed_device)
                    with torch.no_grad():
                        feat = self.embed_model(inp).squeeze()  # shape (C,)
                    feat = feat.cpu().numpy()
                    # L2 normalize
                    nrm = np.linalg.norm(feat)
                    if nrm > 0:
                        feat = feat / nrm
                    det_embeds[i] = feat
                except Exception:
                    det_embeds[i] = None

        track_ids = list(self.tracks.keys())
        # Predict each track forward using its Kalman filter (if present)
        track_boxes = []
        track_clss = []
        for tid in track_ids:
            trk = self.tracks[tid]
            kf = trk.get("kf")
            if kf is not None:
                kf.predict()
                z_pred = np.array([kf.x[0, 0], kf.x[1, 0], kf.x[4, 0], kf.x[5, 0]])
                pred_box = self._z_to_box(z_pred)
            else:
                pred_box = trk["box"]
            pred_box = self._clamp_box(pred_box)
            track_boxes.append(pred_box)
            track_clss.append(self.tracks[tid]["cls_name"])

        if det_boxes and track_boxes:
            cost_matrix = np.ones((len(track_boxes), len(det_boxes)), dtype=np.float32)

            for i, (tb, tc) in enumerate(zip(track_boxes, track_clss)):
                for j, (db, dc) in enumerate(zip(det_boxes, det_clss)):
                    if tc != dc:
                        continue

                    # IoU similarity (higher is better)
                    iou_val = self.iou(tb, db)

                    # Appearance distance (0 best, 1 worst) if available
                    app_cost = None
                    if APPEARANCE_MATCHING:
                        track_embed = self.tracks[track_ids[i]].get("embed")
                        det_embed = det_embeds[j]
                        if track_embed is not None and det_embed is not None:
                            # cosine distance
                            cos_sim = np.dot(track_embed, det_embed)
                            cos_sim = np.clip(cos_sim, -1.0, 1.0)
                            app_cost = 1.0 - cos_sim

                    # Mahalanobis gating: if the track has a KF, compute distance
                    gate_thresh = 9.488  # chi2 0.95 for df=4
                    mahalanobis = None
                    kf = self.tracks[track_ids[i]].get("kf")
                    if kf is not None:
                        meas = self._box_to_z(db)
                        mahalanobis = self._mahalanobis_distance(kf, meas)

                    if mahalanobis is not None and mahalanobis > gate_thresh:
                        # too far -> prohibit matching by setting large cost
                        cost = 1e6
                    else:
                        if app_cost is not None:
                            # fused cost (appearance + iou)
                            cost = APPEARANCE_WEIGHT * app_cost + IOU_WEIGHT * (1.0 - iou_val)
                        else:
                            cost = 1.0 - iou_val

                    cost_matrix[i, j] = cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_dets = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < (1 - self.iou_thresh):
                    tid = track_ids[r]
                    # Update Kalman filter for matched track
                    meas = self._box_to_z(det_boxes[c])
                    kf = self.tracks[tid].get("kf")
                    if kf is not None:
                        # filterpy expects a 1D measurement vector
                        kf.update(meas)
                        z_upd = np.array([kf.x[0, 0], kf.x[1, 0], kf.x[4, 0], kf.x[5, 0]])
                        box_upd = self._z_to_box(z_upd)
                    else:
                        box_upd = det_boxes[c]

                    # clamp updated box before storing
                    self.tracks[tid]["box"] = self._clamp_box(box_upd)
                    self.tracks[tid]["cls_name"] = det_clss[c]
                    self.tracks[tid]["score"] = det_scores[c]
                    self.tracks[tid]["last_seen"] = frame_idx
                    self.track_history[tid].append(det_boxes[c])
                    # update appearance embedding if available
                    if APPEARANCE_MATCHING and det_embeds[c] is not None:
                        self.tracks[tid]["embed"] = det_embeds[c]
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
                    # initialize Kalman filter for new track
                    z = self._box_to_z(box)
                    kf = self._create_kalman(z[0], z[1], z[2], z[3])

                    self.tracks[tid] = {
                        "box": self._clamp_box(box),
                        "cls_name": cls_name,
                        "score": score,
                        "last_seen": frame_idx,
                        "kf": kf,
                        "embed": None
                    }
                    # set initial embed if available
                    if APPEARANCE_MATCHING and det_embeds[j] is not None:
                        self.tracks[tid]["embed"] = det_embeds[j]
                    self.track_history[tid].append(box)
                    updated_tracks[tid] = self.tracks[tid]
                    updated_ids.add(tid)

        else:
            for j, (box, cls_name, score) in enumerate(zip(det_boxes, det_clss, det_scores)):
                tid = self.next_id
                self.next_id += 1
                z = self._box_to_z(box)
                kf = self._create_kalman(z[0], z[1], z[2], z[3])
                self.tracks[tid] = {
                    "box": self._clamp_box(box),
                    "cls_name": cls_name,
                    "score": score,
                    "last_seen": frame_idx,
                    "kf": kf,
                    "embed": None
                }
                # set embedding for new track if available
                if APPEARANCE_MATCHING and det_embeds[j] is not None:
                    self.tracks[tid]["embed"] = det_embeds[j]
                self.track_history[tid].append(box)
                updated_tracks[tid] = self.tracks[tid]
                updated_ids.add(tid)

        # Remove stale tracks
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