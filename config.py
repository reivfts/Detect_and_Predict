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

# Classes for COCO and NuScenes
VALID_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

# FRCNN settings  
FRCNN_SCORE_THRESH = 0.5

# Standard COCO category names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Map the torchvision/COCO 
# torchvision's detection models return labels that index into the list above.
FRCNN_ID2NAME = {
    i: name for i, name in enumerate(COCO_INSTANCE_CATEGORY_NAMES) if name in VALID_CLASSES
}

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

# DETR COCO class mapping (shared across modules)
DETR_CLASS_MAP = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck"
}

# =========================================
# CNN-TRANSFORMER FUSION (v2)
# =========================================
# Fusion parameters
FUSION_IOU_THRESHOLD = 0.3
FUSION_MODE = "hybrid"  # Options: "cnn_only", "transformer_only", "hybrid"
FUSION_CONFIDENCE_PENALTY = 0.7  # Penalty for unvalidated CNN detections

# Trajectory Prediction
TRAJECTORY_PREDICTOR = "linear"  # Options: "linear", "kalman", "transformer"
TRAJECTORY_HISTORY_LEN = 10

# Ablation Study
EXPERIMENT_MODE = "full_hybrid"  # Options: A, B, C, D, E, or "full_hybrid"
SAVE_INTERMEDIATE_RESULTS = True
