import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "trackings")

# NOTE: Update this path to your NuScenes dataset location
# For Windows: r"C:\path\to\nuscenes\v1.0-mini"
# For Linux/Mac: "/path/to/nuscenes/v1.0-mini"
# Or point to a direct image folder like: r"C:\path\to\samples\CAM_FRONT"
NUSCENES_ROOT = os.getenv("NUSCENES_ROOT", r"C:\Users\rayra\OneDrive\Desktop\v1.0-trainval01_blobs\samples\CAM_FRONT")

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "nuscenes_yolo_detr_tracking.mp4")

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Camera
# Options: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
CAMERA_CHANNEL = "CAM_FRONT"

# Classes for COCO and NuScenes
VALID_CLASSES = ["person", "car", "motorcycle", "bus", "truck"]

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
# Path to YOLO weights in repo root (default: bundled `yolo11m-seg.pt`)
YOLO_WEIGHTS = os.path.join(BASE_DIR, "yolo11m-seg.pt")

# Appearance (ReID) matching - optional. Disabled by default to keep tests lightweight.
APPEARANCE_MATCHING = True
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
# Friendly mode names: use shorter, more readable labels.
# Options (both current and legacy accepted):
#  - "cnn" or legacy "cnn_only": use CNN detections only
#  - "transformer" or legacy "transformer_only": use Transformer/DETR only
#  - "cnn/transformer" or legacy "hybrid": fuse CNN + Transformer
FUSION_IOU_THRESHOLD = 0.3
FUSION_MODE = "cnn/transformer"  # Options: "cnn", "transformer", "cnn/transformer"
FUSION_CONFIDENCE_PENALTY = 0.7  # Penalty for unvalidated CNN detections

# Trajectory Prediction
TRAJECTORY_PREDICTOR = "kalman"  # Options: "linear", "kalman", "transformer", "hybrid"
TRAJECTORY_HISTORY_LEN = 10

# Prediction Horizon (optimized for front camera view)
PREDICTION_HORIZON_SHORT = 8   # 1.3 seconds @ 6Hz (immediate planning)
PREDICTION_HORIZON_LONG = 15   # 2.5 seconds @ 6Hz (strategic planning)
USE_LONG_HORIZON = False        # Enable for highway scenarios

# Trajectory Sampling (for better turn prediction)
USE_ANGLE_SAMPLING = True       # Sample multiple heading angles for turn prediction
ANGLE_SAMPLE_RANGE = 0.3        # ±radians to sample (±17 degrees)
ANGLE_SAMPLE_STEPS = 3          # Number of angle samples per prediction

# Overprediction Strategy (inspired by DONUT - ICCV 2025)
USE_OVERPREDICTION = True       # Predict multiple horizons simultaneously
OVERPREDICTION_HORIZONS = [1, 3, 5, 8]  # Predict at these step intervals
OVERPREDICTION_WEIGHT = 0.3     # Weight for auxiliary overprediction loss

# Kalman Filter settings (used when TRAJECTORY_PREDICTOR = "kalman")
KALMAN_PROCESS_NOISE = 0.5     # Reduced from 1.0 for smoother, more accurate predictions
KALMAN_MEASUREMENT_NOISE = 0.8  # Slightly reduced to trust measurements more

# Physics Constraints (adjusted for front camera urban scenarios)
MAX_ACCELERATION = 4.0          # m/s^2 (urban driving)
MAX_VELOCITY = 20.0             # m/s (~72 km/h, typical urban)
MIN_TURN_RADIUS = 5.0           # meters (vehicle kinematic constraint)

# Uncertainty Visualization
SHOW_UNCERTAINTY = False        # Visualize prediction confidence (disable for cleaner view)
UNCERTAINTY_SCALE = 1.0         # Scale covariance ellipses (reduced from 2.0)

# Multi-Model Prediction
USE_HYBRID_PREDICTION = False   # Combine Kalman + Transformer (disabled for performance)
HYBRID_KALMAN_WEIGHT = 0.7      # Weight for Kalman (physics-based)
HYBRID_TRANSFORMER_WEIGHT = 0.3 # Weight for Transformer (learning-based)

# Visualization
VELOCITY_COLOR_CODING = False   # Color-code trajectories by speed (disabled for cleaner view)
VELOCITY_SLOW_THRESHOLD = 2.0   # m/s (green)
VELOCITY_FAST_THRESHOLD = 10.0  # m/s (red)

# Ablation Study
EXPERIMENT_MODE = "full_hybrid"  # Options: A, B, C, D, E, or "full_hybrid"
SAVE_INTERMEDIATE_RESULTS = True

# === PERFORMANCE TUNING ===
# Enable async inference to boost display FPS (inference in background thread)
# DISABLED: Async causes visual lag when worker is delayed. Use sync for accurate box alignment.
ASYNC_INFERENCE = False  # Set to True to enable
# For async inference: queue size (larger = more latency, more buffering)
ASYNC_QUEUE_SIZE = 2
