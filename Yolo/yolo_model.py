from ultralytics import YOLO
from config import DEVICE, YOLO_CONF, YOLO_IOU


class YOLODetector:
    def __init__(self):
        self.model = YOLO("yolo11m-seg.pt").to(DEVICE)
        self.names = self.model.names

    def detect_and_track(self, frame):
        return self.model.track(
            frame,
            imgsz=1280,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            persist=True
        )[0]
