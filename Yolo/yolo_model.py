import torch
from ultralytics import YOLO
from config import VALID_CLASSES
from Transformer.tracker import TransformerTracker


class YOLODetector:
    def __init__(self,device):
        self.device = device
        self.model = YOLO("yolo11m-seg.pt").to(device)
        self.id_tracker = TransformerTracker()
        self.frame_idx = 0
        self.names = self.model.names

    def detect(self, frame):
        """
        Runs YOLO detection + tracking and returns cleaned results.
        """
        self.frame_idx += 1

        results = self.model.track(
            frame,
            persist=True,
            conf=0.45,
            iou=0.45,
            imgsz=1280
        )[0]

        detections = []

        if results.boxes.id is None:
            return []

        boxes = results.boxes.xyxy.cpu().numpy()
        tids = results.boxes.id.int().cpu().numpy()
        clss = results.boxes.cls.int().cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        for box, tid, cls_idx, score in zip(boxes, tids, clss, scores):
            cls_name = self.names[int(cls_idx)]

            if cls_name not in VALID_CLASSES:
                continue
            if score < 0.45:
                continue

            detections.append({
                "box": box.tolist(),
                "track_id": int(tid),
                "cls_name": cls_name,
                "score": float(score)
            })

        # STEP 2: ASSIGN STABLE TRANSFORMER-BASED IDs
        # Pass the raw frame to the tracker so appearance embeddings can be computed (optional)
        detections = self.id_tracker.update(
            detections,
            frame_idx=self.frame_idx,
            frame=frame
        )

        return detections
