# Detect and Predict v2: CNN-Transformer Hybrid Architecture

## Overview

Version 2 implements a complete CNN-Transformer hybrid detection and tracking system following the approach from "CNN-transformer mixed model for object detection" (arXiv:2212.06714).

The system combines the strengths of CNN-based and Transformer-based architectures:
- **CNN (FRCNN)**: Provides accurate localization through local feature extraction
- **Transformer (DETR)**: Provides global context and contextual validation
- **Fusion**: Intelligently combines both for improved robustness

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Input Frame (NuScenes)                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
        ┌───────▼────────┐             ┌────────▼────────┐
        │  YOLO Detector │             │                 │
        │  (Fast Props)  │             │                 │
        └───────┬────────┘             │                 │
                │                      │                 │
        ┌───────▼────────┐             │                 │
        │ Faster R-CNN   │             │  DETR Model    │
        │ (CNN Refiner)  │             │ (Transformer)   │
        └───────┬────────┘             └────────┬────────┘
                │                               │
                └──────────┬────────────────────┘
                           │
                 ┌─────────▼──────────┐
                 │  Fusion Module     │
                 │ (CNN+Transformer)  │
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │ TransformerTracker │
                 │  (Kalman Filter)   │
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │ Trajectory Predict │
                 │  (Linear/Transform)│
                 └─────────┬──────────┘
                           │
                    ┌──────▼───────┐
                    │   Output     │
                    │  + Metrics   │
                    └──────────────┘
```

## Pipeline Stages

### Stage 1: YOLO Detection
- Fast region proposal generation
- Initial object detection with tracking IDs
- Filters for relevant classes (car, truck, bus, person, bicycle, motorcycle)

### Stage 2: FRCNN Refinement (CNN)
- Accurate bounding box localization
- Leverages ResNet-50 FPN backbone
- Local feature extraction

### Stage 3: DETR Enhancement (Transformer)
- Global context through self-attention
- Validates detections using scene understanding
- Reduces false positives

### Stage 4: CNN-Transformer Fusion
- Matches FRCNN and DETR detections by class and IoU
- When both agree (IoU > 0.3): uses FRCNN box, averages confidence
- When only CNN detected: keeps detection but reduces confidence
- Marks each detection with validation status

### Stage 5: Tracking
- TransformerTracker with Kalman filtering
- Motion prediction using constant velocity model
- Optional appearance embedding (ResNet-18)
- Hungarian algorithm for data association

### Stage 6: Trajectory Prediction
- Linear extrapolation (default)
- Transformer-based prediction (optional)
- Evaluation metrics: ADE, FDE, IoU, center distance

## Key Files

### New in v2
- `Transformer/fusion.py` - CNN-Transformer fusion logic
- `Transformer/trajectory_predictor.py` - Enhanced with transformer predictor
- `trackers/deepsort_wrapper.py` - DeepSORT baseline integration
- `experiments/ablation_study.py` - Comparative evaluation framework
- `Nuscenes/loader.py` - Fixed to yield (frame, timestamp, token)
- `README_v2.md` - This file

### Updated in v2
- `main.py` - Now uses full pipeline (YOLO → FRCNN → DETR → Fusion → Track)
- `config.py` - Added v2 parameters (fusion, trajectory predictor, experiments)
- `requirements.txt` - Cleaned up, added supervision, removed unused packages
- `data/nuscenes.py` - Fixed to match usage pattern

### Existing (v1)
- `Detection/detector.py` - DetectorPipeline (YOLO + FRCNN) [now used in main]
- `Transformer/tracker.py` - TransformerTracker with Kalman filter
- `Transformer/detr_model.py` - DETR wrapper
- `FRCNN/frcnn_model.py` - Faster R-CNN wrapper
- `Yolo/yolo_model.py` - YOLO wrapper

## Configuration

Edit `config.py` to customize behavior:

```python
# Fusion mode
FUSION_MODE = "hybrid"  # Options: "cnn_only", "transformer_only", "hybrid"
FUSION_IOU_THRESHOLD = 0.3
FUSION_CONFIDENCE_PENALTY = 0.7

# Trajectory prediction
TRAJECTORY_PREDICTOR = "linear"  # Options: "linear", "kalman", "transformer"
TRAJECTORY_HISTORY_LEN = 10

# Ablation study
EXPERIMENT_MODE = "full_hybrid"  # Which experiment configuration
SAVE_INTERMEDIATE_RESULTS = True
```

## Running the Code

### Basic Usage

```bash
python main.py
```

This runs the full CNN-Transformer hybrid pipeline on NuScenes frames.

### Ablation Study

Run comparative experiments:

```python
from experiments.ablation_study import run_ablation_study
from Nuscenes.loader import NuScenesLoader
from config import NUSCENES_ROOT, CAMERA_CHANNEL

loader = NuScenesLoader(NUSCENES_ROOT)
frames = loader.frames(CAMERA_CHANNEL)

# Run all experiments (A, B, C, D, E)
run_ablation_study(frames, max_frames=100)

# Or run specific experiments
run_ablation_study(frames, max_frames=100, experiments=["A", "D", "E"])
```

### Experiment Configurations

- **Experiment A**: YOLO only + custom tracker
- **Experiment B**: YOLO + FRCNN + custom tracker
- **Experiment C**: YOLO + DETR + custom tracker
- **Experiment D**: YOLO + FRCNN + DETR (full hybrid) + custom tracker
- **Experiment E**: YOLO + DeepSORT (baseline)

Results are saved to `data/trackings/ablation/`:
- `exp_X_trajectory.csv` - Per-frame trajectory metrics
- `exp_X_summary.csv` - Aggregate metrics

## Evaluation Metrics

### Tracking Metrics
- **FPS**: Frames per second (processing speed)
- **Detections per Frame**: Average number of detections
- **Tracks per Frame**: Average number of active tracks

### Trajectory Prediction Metrics
- **ADE** (Average Displacement Error): Mean distance between predicted and actual centers
- **FDE** (Final Displacement Error): Distance at final prediction step
- **IoU**: Intersection over Union between predicted and actual boxes
- **Center Distance**: Pixel distance between predicted and actual centers

## Comparison with v1

### What's New in v2
1. **Full Pipeline Integration**: Now uses YOLO → FRCNN → DETR → Fusion
2. **CNN-Transformer Fusion**: Intelligent combination of local and global features
3. **Validation Status**: Each detection marked with validation source
4. **Ablation Study Framework**: Systematic comparison of approaches
5. **DeepSORT Baseline**: Standard tracker for comparison
6. **Transformer Predictor**: Optional transformer-based trajectory prediction
7. **Cleaner Code**: Better modularization and documentation

### What Stayed from v1
1. **TransformerTracker**: Core tracking logic with Kalman filtering
2. **Appearance Matching**: Optional ResNet-18 embeddings
3. **Linear Trajectory Prediction**: Fast baseline predictor
4. **NuScenes Integration**: Dataset loading and processing

## Expected Results

With the full hybrid pipeline (Experiment D), you should see:
- **Improved Accuracy**: Fusion reduces false positives
- **Better Localization**: FRCNN provides precise boxes
- **Contextual Validation**: DETR confirms detections
- **Slower Speed**: More models = more computation (trade-off)

Typical results on NuScenes v1.0-mini:
- FPS: 2-5 (depending on GPU)
- Average IoU: 0.6-0.8
- ADE: 10-30 pixels
- Validation Rate: 60-80% (detections confirmed by both CNN and Transformer)

## Troubleshooting

### NuScenes Not Found
Edit `config.py` and set `NUSCENES_ROOT` to your NuScenes dataset path.

### Out of Memory
Reduce batch size or disable appearance matching:
```python
APPEARANCE_MATCHING = False
```

### DeepSORT Import Error
Install deep-sort-realtime:
```bash
pip install deep-sort-realtime
```

### Slow Processing
Try different fusion modes:
```python
FUSION_MODE = "cnn_only"  # Faster, skip DETR
```

## Future Enhancements

Potential improvements for v3:
1. **End-to-End Training**: Joint training of fusion module
2. **Attention Visualization**: Show where transformer focuses
3. **Multi-Camera Fusion**: Combine multiple camera views
4. **Online Learning**: Adapt to scene-specific patterns
5. **Efficient Transformers**: Use lighter transformer architectures

## References

- **Main Paper**: Xiao et al., "CNN-transformer mixed model for object detection" (arXiv:2212.06714)
- **DETR**: Carion et al., "End-to-End Object Detection with Transformers" (ECCV 2020)
- **Faster R-CNN**: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection" (NeurIPS 2015)
- **YOLO**: Jocher et al., "Ultralytics YOLO" (2023)
- **DeepSORT**: Wojke et al., "Simple Online and Realtime Tracking with a Deep Association Metric" (ICIP 2017)

## License

Same as v1. See repository root for license information.

## Contact

For questions or issues, please open a GitHub issue in the repository.
