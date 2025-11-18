import os
import torch

# ============================================
# PATHS
# ============================================
OUTPUT_DIR = r"C:\Users\rayra\OneDrive\Desktop\Detect_and_Predict\data\trackings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "nuscenes_yolo_frcnn_tracking.mp4")

# THIS IS THE MISSING VARIABLE
NUSCENES_ROOT = r"C:\Users\rayra\OneDrive\Desktop\v1.0-mini"

CAMERA_CHANNEL = "CAM_FRONT"

# ============================================
# DEVICE
# ============================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# DETECTION SETTINGS
# ============================================
VALID_CLASSES = ["person", "bicycle", "car", "motorbike", "bus", "truck"]

FRCNN_ID2NAME = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorbike",
    6: "bus",
    8: "truck",
}

YOLO_CONF = 0.40
YOLO_IOU = 0.45

FRCNN_SCORE_THRESH = 0.50
REFINE_IOU_THRESH = 0.30

FRAME_RATE = 10
