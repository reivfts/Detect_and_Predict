# ================================
# CONFIGURATION FILE
# ================================

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

# DETR settings
DETR_THRESHOLD = 0.50
DETR_IOU_MATCH = 0.30
