# Multi-Stage Object Detection, Tracking, and Trajectory Prediction

A real-time deep learning pipeline for autonomous driving perception using the nuScenes dataset. This system combines YOLO, Faster R-CNN, and DETR for detection, applies physics-constrained Kalman filtering with hybrid trajectory prediction to forecast vehicle positions up to 5 seconds ahead.

## System Architecture

```
Input Frame (nuScenes Camera or Image Folder)
         |
         v
┌───────────────────────────────────────────────────────────┐
│                   DETECTION PIPELINE                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  YOLO11m-seg │ -> │ Faster R-CNN │    │     DETR     │ │
│  │   (Segment)  │    │ (ResNet-50)  │    │ (ResNet-50)  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         |                    |                    |       │
│         └──────┬─────────────┘                    |       | 
│                v                                  |       │
│      ┌─────────────────┐                          |       │
│      │ DetectorPipeline│ (YOLO + FRCNN Fusion)    |       │
│      └─────────────────┘                          |       │
│                |                                  |       │
│                └─────────────┬────────────────────┘       │
│                              v                            │
│                    ┌──────────────────┐                   │
│                    │  Fusion Module   │                   │
│                    │ (CNN+Transformer)│                   │
│                    └──────────────────┘                   │
└───────────────────────────────────────────────────────────┘
         |
         v
┌───────────────────────────────────────────────────────────┐
│                 TRACKING & ID ASSIGNMENT                  │
│                ┌─────────────────────┐                    │
│                │ TransformerTracker  │                    │
│                │  - Kalman Filter    │                    │
│                │  - Hungarian Algo   │                    │
│                │  - ResNet-18 (opt)  │                    │
│                └─────────────────────┘                    │
└───────────────────────────────────────────────────────────┘
         |
         v
┌───────────────────────────────────────────────────────────┐
│              TRAJECTORY PREDICTION PIPELINE               │
│                                                           │
│  ┌────────────────────┐                                   │
│  │ Velocity Estimator │ (EMA smoothing, outlier rejection)│
│  └────────────────────┘                                   │
│           |                                               │
│           v                                               │
│  ┌────────────────────┐                                   │
│  │  Policy LSTM Net   │ (6-class maneuver prediction)     │
│  └────────────────────┘                                   │
│           |                                               │
│           v                                               │
│  ┌────────────────────────────────────────────────────┐   │
│  │         Hybrid Trajectory Predictor                |   │
│  │  ┌──────────────────┐  ┌──────────────────────┐    │   │
│  │  │  Kalman Physics  │  │ Transformer Attention│    │   │
│  │  │   (Weight: 0.7)  │  │    (Weight: 0.3)     │    │   │
│  │  └──────────────────┘  └──────────────────────┘    │   │
│  │           |                       |                │   │
│  │           └───────────┬───────────┘                │   │
│  │                       v                            │   │
│  │         ┌─────────────────────────┐                │   │
│  │         │ Uncertainty Quantify    │                │   │
│  │         │ (Covariance Ellipses)   │                │   │
│  │         └─────────────────────────┘                │   │
│  └────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
         |
         v
   Predicted Trajectories
  (12-30 frames, 2-5 sec)
```

## Quick Start

```powershell
# Install dependencies
pip install -r requirements.txt

# Run pipeline with default settings (2-second hybrid prediction)
python main.py

# Specify scene and camera
python main.py --scene-idx 5 --camera CAM_FRONT_LEFT --max-frames 150
```

## Direct Image Folder Support

Load raw image folders directly without NuScenes metadata. Set `NUSCENES_ROOT` in `config.py` to your image directory:

```python
NUSCENES_ROOT = r"C:\path\to\your\images\CAM_FRONT"
```

The pipeline automatically detects and uses `SimpleImageLoader` for direct folders. Images are processed in filename order. 

**Note:** Evaluation metrics (ADE, FDE, IoU) are only computed for frames with valid ground truth and predictions. This is why 429 frames may be evaluated from a 3376-frame video.

## Models and Components

### Detection Models

1. **YOLO11m-seg** (Ultralytics)
   - Fast region proposals with instance segmentation
   - Weights: `yolo11m-seg.pt`
   - Config: Confidence 0.4, IoU 0.45, Image size 1280

2. **Faster R-CNN with ResNet-50 + FPN** (torchvision)
   - Accurate bounding box localization
   - Pretrained on COCO dataset
   - Score threshold: 0.5

3. **DETR with ResNet-50 Backbone** (HuggingFace)
   - Transformer-based detection with global context
   - Model: `facebook/detr-resnet-50`
   - Detection threshold: 0.5

### Tracking Models

4. **ResNet-18 Appearance Embeddings** (torchvision, optional)
   - Feature extraction for object re-identification
   - Pretrained on ImageNet
   - Final FC layer removed for embeddings
   - Enabled via `APPEARANCE_MATCHING = True`

5. **Kalman Filter** (filterpy)
   - 6D state space: [cx, cy, vx, vy, w, h]
   - Physics-constrained (max velocity 20 m/s, acceleration 4 m/s²)
   - Constant velocity motion model
   - Hungarian algorithm for optimal data association

### Prediction Models

6. **Policy Anticipation LSTM** (custom PyTorch)
   - 2-layer LSTM with 128 hidden units
   - Predicts 6 maneuver classes:
     - Forward
     - Yield (stop/slow)
     - Turn left/right
     - Lane change left/right
   - 40-frame observation window
   - Located in `Transformer/policy_network.py`

7. **Transformer Trajectory Predictor** (custom)
   - 4-head attention over 10-frame history
   - Combines with Kalman predictions (70% physics, 30% learning)
   - Extended horizon: 2-5 second predictions

## Pipeline Stages

### Stage 1: Three-Stage Detection
- YOLO11m-seg: Fast initial detection and segmentation
- Faster R-CNN: Refined bounding boxes with region proposals
- DETR: Transformer-based validation with global context

### Stage 2: Fusion
- Matches detections by class and IoU (threshold 0.3)
- Averages confidence when both CNN and Transformer agree
- Reduces confidence for CNN-only detections

### Stage 3: Tracking
- Multi-hypothesis tracker with appearance embeddings
- Kalman filter with physics constraints
- Hungarian algorithm for optimal assignment
- Mahalanobis distance gating

### Stage 4: Velocity Estimation
- 10-frame exponential moving average (α=0.5)
- Median-based outlier rejection (2σ threshold)
- Acceleration/deceleration detection
- Visual velocity arrows (color-coded by speed)

### Stage 5: Trajectory Prediction
- Policy LSTM anticipates high-level maneuvers
- Hybrid ensemble: Kalman physics + Transformer attention
- Uncertainty quantification with covariance ellipses
- Multi-horizon predictions: [1, 3, 5, 8] steps

## Performance Metrics

From evaluation on nuScenes v1.0-mini (429 evaluated frames, 303 tracks):

- **Overall ADE**: 4.11 pixels
- **Overall FDE**: 4.69 pixels
- **Overall RMSE**: 4.38 pixels
- **Average IoU**: 0.7937
- **Median IoU**: 0.8546
- **Processing Speed**: 4-8 FPS (GPU dependent)

### By Prediction Horizon
- 1-step (0.17s): 3.08 px error, 0.8204 IoU
- 3-step (0.50s): 4.77 px error, 0.7616 IoU
- 5-step (0.83s): 5.72 px error, 0.7326 IoU
- 8-step (1.33s): 8.47 px error, 0.6442 IoU

## Key Features

1. **Physics-Constrained Tracking**: Enforces realistic velocity/acceleration limits
2. **Hybrid Prediction**: Combines physics-based and learning-based approaches
3. **Uncertainty Quantification**: Displays confidence ellipses for predictions
4. **Multi-Stage Fusion**: Reduces false positives through model agreement
5. **Velocity Visualization**: Color-coded trajectories (green=slow, red=fast)
6. **Policy-Aware Prediction**: Anticipates driver intentions (turns, lane changes)
7. **Flexible Data Loading**: Supports both nuScenes datasets and raw image folders

## Dataset

Uses nuScenes v1.0-mini or v1.0-trainval with 6 camera views:
- CAM_FRONT (default)
- CAM_FRONT_LEFT, CAM_FRONT_RIGHT
- CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT

Alternatively, load images directly from any folder without metadata.

## Evaluation

Run evaluation analysis:
```powershell
python tools/analyze_evaluation.py
```

Metrics computed:
- **ADE** (Average Displacement Error): Mean Euclidean distance across predicted points
- **FDE** (Final Displacement Error): Distance at prediction horizon
- **RMSE**: Root mean squared error
- **IoU**: Bounding box overlap for tracking quality

Results saved to:
- `data/trackings/evaluation.csv` - Per-frame predictions
- `data/trackings/accuracy.txt` - Summary statistics

## Configuration

Edit `config.py` to customize:

```python
# Dataset
NUSCENES_ROOT = r"C:\path\to\dataset"

# Detection thresholds
YOLO_CONF = 0.4
FRCNN_SCORE_THRESH = 0.5
DETR_THRESHOLD = 0.5

# Tracking
APPEARANCE_MATCHING = True  # Enable ResNet-18 embeddings
MAX_LOST_FRAMES = 10

# Prediction
TRAJECTORY_PREDICTOR = "hybrid"  # Options: "linear", "kalman", "transformer", "hybrid"
PREDICTION_HORIZON = 12  # frames (2 seconds at 6Hz)
```

## Dependencies

- PyTorch 2.0+
- torchvision
- transformers (HuggingFace)
- ultralytics (YOLO)
- opencv-python
- nuscenes-devkit (optional, for full dataset)
- filterpy (Kalman filter)
- numpy, scipy, pandas
- Pillow

Install all:
```powershell
pip install -r requirements.txt
```

## Research References

This project builds on research from multiple papers:

1. **nuScenes Dataset**  
   Caesar et al., "nuScenes: A Multimodal Dataset for Autonomous Driving" (CVPR 2020)

2. **Two-Level Trajectory Prediction**  
   Xue et al., "SS-LSTM: A Hierarchical LSTM Model for Pedestrian Trajectory Prediction" (WACV 2018)

3. **CNN-Transformer Hybrid**  
   Feng et al., "CNN-Transformer Mixed Model for Object Detection" (arXiv 2022)

## License

For academic and research use. See individual component licenses:

- **YOLO11**: AGPL-3.0 (Ultralytics)
- **Faster R-CNN**: BSD-3-Clause (torchvision/PyTorch)
- **DETR**: Apache-2.0 (Facebook Research)
- **ResNet**: BSD-3-Clause (torchvision/PyTorch)
- **nuScenes Dataset**: CC BY-NC-SA 4.0 (non-commercial)

This project is released under **AGPL-3.0** to comply with YOLO's license.

## Citation

If you use this code, please cite the underlying models and datasets:

```bibtex
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics}
}

@inproceedings{carion2020end,
  title={End-to-End Object Detection with Transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and others},
  booktitle={ECCV},
  year={2020}
}

@inproceedings{ren2015faster,
  title={Faster R-CNN: Towards Real-Time Object Detection},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  booktitle={NeurIPS},
  year={2015}
}

@inproceedings{he2016deep,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR},
  year={2016}
}

@inproceedings{caesar2020nuscenes,
  title={nuScenes: A Multimodal Dataset for Autonomous Driving},
  author={Caesar, Holger and Bankiti, Varun and Lang, Alex H. and others},
  booktitle={CVPR},
  year={2020}
}
```

## Acknowledgments

This project incorporates models and methodologies from the open-source community. We thank the authors of YOLO, Faster R-CNN, DETR, ResNet, and the nuScenes dataset for making their work publicly available.

## Contact

For questions or issues, please open a GitHub issue in the repository.
