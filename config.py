# CONFIGURATION FILE

import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "trackings")
NUSCENES_ROOT = r"C:\Users\rayra\OneDrive\Desktop\v1.0-mini"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "nuscenes_yolo_detr_tracking.mp4")

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Camera
CAMERA_CHANNEL = "CAM_FRONT"

# Classes to keep (COCO + NuScenes relevant)
VALID_CLASSES = ["person", "bicycle", "car", "motorbike", "bus", "truck"]

# YOLO settings
YOLO_CONF = 0.4
YOLO_IOU = 0.45
YOLO_IMGSZ = 1280

# Appearance (ReID) matching - optional. Disabled by default to keep tests lightweight.
APPEARANCE_MATCHING = False
# Weighting between appearance distance and IoU when computing assignment cost.
# cost = APPEARANCE_WEIGHT * appearance_distance + IOU_WEIGHT * (1 - iou)
APPEARANCE_WEIGHT = 0.6
IOU_WEIGHT = 0.4

# Device for embedding model (defaults to main DEVICE)
EMBEDDING_DEVICE = DEVICE

# DETR settings
DETR_THRESHOLD = 0.50
DETR_IOU_MATCH = 0.30
