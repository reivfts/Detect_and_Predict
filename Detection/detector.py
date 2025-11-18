import cv2
from Yolo.yolo_model import YOLODetector
from FRCNN.frcnn_model import FRCNNRefiner

from config import VALID_CLASSES, DEVICE, DETR_IOU_MATCH


# ===========================
# IOU 
# ===========================
def iou(boxA, boxB):
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    xi1 = max(x1A, x1B)
    yi1 = max(y1A, y1B)
    xi2 = min(x2A, x2B)
    yi2 = min(y2A, y2B)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    areaA = max(0, (x2A - x1A)) * max(0, (y2A - y1A))
    areaB = max(0, (x2B - x1B)) * max(0, (y2B - y1B))

    union = areaA + areaB - inter
    return inter / union if union > 0 else 0


# ===========================
# DETECTOR PIPELINE
# ===========================
class DetectorPipeline:
    def __init__(self):
        # ensure detector constructed with proper device
        self.yolo = YOLODetector(device=DEVICE)
        self.frcnn = FRCNNRefiner()

    def process(self, frame):
        # YOLO detection with tracking
        yolo_res = self.yolo.detect_and_track(frame)

        if yolo_res.boxes.id is None:
            return []

        y_boxes = yolo_res.boxes.xyxy.cpu().numpy()
        y_ids = yolo_res.boxes.id.int().cpu().numpy()
        y_clss = yolo_res.boxes.cls.int().cpu().numpy()
        names = self.yolo.names

        # Filter YOLO tracks
        yolo_tracks = []
        for box, tid, cls_idx in zip(y_boxes, y_ids, y_clss):
            cls_name = names[int(cls_idx)]

            if cls_name in VALID_CLASSES:
                yolo_tracks.append((box, int(tid), cls_name))

        if not yolo_tracks:
            return []

        # FRCNN refinement
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frcnn_dets = self.frcnn.refine(frame_rgb)

        refined = []
        for box, tid, cls_name in yolo_tracks:

            best_iou = 0
            best_box = None

            for fr_box, _, fr_cls in frcnn_dets:
                if fr_cls != cls_name:
                    continue

                iou_val = iou(box, fr_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_box = fr_box

            # Use FRCNN box if strong match
            if best_box is not None and best_iou >= DETR_IOU_MATCH:
                refined.append((best_box, tid, cls_name))
            else:
                refined.append((box, tid, cls_name))

        return refined
