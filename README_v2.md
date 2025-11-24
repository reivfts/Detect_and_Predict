# Detect and Predict: CNN-Transformer Hybrid Architecture

## Project Overview

**Object Detection and Segmentation with real-time Trajectory Prediction using Faster R-CNN and Transformers**

Version 2 implements a complete CNN-Transformer hybrid detection and tracking system with research-driven trajectory prediction improvements from nuScenes and two-level prediction papers.

The system combines the strengths of CNN-based and Transformer-based architectures:
- **CNN (FRCNN)**: Provides accurate localization through local feature extraction
- **Transformer (DETR)**: Provides global context and contextual validation
- **Fusion**: Intelligently combines both for improved robustness
- **Velocity Estimation**: Temporal velocity tracking with visual arrows
- **Policy Anticipation**: LSTM-based high-level maneuver prediction
- **Cost Map Optimization**: Multi-layer context-aware trajectory prediction


## Direct Image Folder Support (2025 Update)

This project now supports loading raw image folders directly, without requiring NuScenes metadata. Use the `SimpleImageLoader` to process images from a directory such as `samples/CAM_FRONT`. This is useful for large datasets or custom video sources.

**How to use:**
1. Set `NUSCENES_ROOT` in `config.py` to your image folder, e.g.:
  ```python
  NUSCENES_ROOT = r"C:\Users\rayra\OneDrive\Desktop\v1.0-trainval01_blobs\samples\CAM_FRONT"
  ```
2. The pipeline will automatically detect and use the simple loader for direct folders.
3. No NuScenes metadata is required; images are loaded and processed in filename order.

**Note:** Evaluation metrics (ADE, FDE, IoU, etc.) are only computed for frames with valid ground truth and predictions. If your video has 3376 frames but only 429 are evaluated, this is expected due to filtering and available data.

### Research-Driven Improvements
Based on findings from:
- **nuScenes paper** (Caesar et al., 2020): Velocity estimation, multi-modal dataset insights
- **Two-level trajectory prediction** (Xue et al., 2018): LSTM policy + cost map optimization


**Implemented Features:**
- Velocity Estimation (`Transformer/velocity_estimator.py`)
  - Temporal velocity tracking from bbox movements
  - EMA smoothing, velocity history, speed/heading computation
  - Visual velocity arrows in output video

- Policy Anticipation Network (`Transformer/policy_network.py`)
  - LSTM-based policy classification (6 classes)
  - Addresses multimodal trajectory prediction
  - Expected 30-40 percent improvement over linear prediction

- Cost Map Optimization (`Transformer/cost_map.py`)
  - Multi-layer cost map (static, moving obstacles, context, nonholonomic)
  - Optimization-based trajectory prediction
  - Expected 2-3x better accuracy with context awareness

**Documentation**:
- `RESEARCH_IMPROVEMENTS.md`: Detailed improvement guide
- `IMPLEMENTATION_STATUS.md`: Implementation details and test results
- `QUICK_REFERENCE.md`: Usage guide and API reference
- `examples/integration_example.py`: Working code examples

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
- Fast region proposal generation with segmentation masks
- Initial object detection (no internal tracking)
- Filters for relevant classes (car, truck, bus, person, motorcycle)
- Mask-based bbox tightening for improved accuracy

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

### Stage 5: Unified Tracking
- Single TransformerTracker for all ID assignment
- Kalman filtering for motion prediction (6D state: cx, cy, vx, vy, w, h)
- Optional appearance embedding (ResNet-18) for robust re-identification
- Hungarian algorithm for data association
- Mahalanobis distance gating

### Stage 6: Velocity Estimation
- Temporal velocity tracking from frame-to-frame bbox movements
- Exponential Moving Average (EMA) smoothing (alpha=0.3)
- Speed and heading computation
- Visual velocity vectors displayed on output video

### Stage 7: Trajectory Prediction (Enhanced)
- Linear extrapolation (real-time visualization)
- Kalman filter prediction (offline accuracy metrics)
- Policy Anticipation LSTM: High-level maneuver classification (forward, yield, turn_left, turn_right, lane_change_left, lane_change_right)
- Cost Map Optimization: Context-aware trajectory prediction with multi-layer cost map (static, moving obstacles, context, nonholonomic)
- Evaluation metrics: ADE, FDE, RMSE, IoU, center distance


## Key Files

### Direct Image Loader
- `data/nuscenes.py` - Contains `SimpleImageLoader` for direct image folder support

### Research-Driven Improvements
- `Transformer/velocity_estimator.py` - Temporal velocity estimation with EMA smoothing
- `Transformer/policy_network.py` - LSTM-based policy anticipation network
- `Transformer/cost_map.py` - Multi-layer cost map for optimization-based prediction
- `examples/integration_example.py` - Complete integration examples
- `RESEARCH_IMPROVEMENTS.md` - Detailed improvement documentation
- `IMPLEMENTATION_STATUS.md` - Test results and status
- `QUICK_REFERENCE.md` - Usage guide and API reference

### Core Pipeline
- `Transformer/fusion.py` - CNN-Transformer fusion logic
- `Transformer/trajectory_predictor.py` - Enhanced with Kalman filter metrics
- `Detection/drawer.py` - Visual rendering with velocity arrows, trails, smooth motion
- `trackers/deepsort_wrapper.py` - DeepSORT baseline integration
- `experiments/ablation_study.py` - Comparative evaluation framework
- `Nuscenes/loader.py` - nuScenes dataset loader
- `tools/async_inference.py` - Background inference for FPS boost

### Main Components
- `main.py` - Full pipeline with velocity estimation integration
- `config.py` - Comprehensive configuration (detection, tracking, trajectory, visualization)
- `Detection/detector.py` - DetectorPipeline (YOLO + FRCNN)
- `Transformer/tracker.py` - TransformerTracker with Kalman filter
- `Transformer/detr_model.py` - DETR wrapper
- `FRCNN/frcnn_model.py` - Faster R-CNN wrapper
- `Yolo/yolo_model.py` - YOLO11m-seg wrapper with mask support

## Quick Start

### Basic Usage (With Velocity Estimation)
```bash
python main.py
```

Velocity vectors will be automatically displayed on tracked objects.

### Advanced Usage (Two-Level Prediction)
```python
from Transformer.velocity_estimator import get_velocity_estimator
from Transformer.cost_map import TwoLevelTrajectoryPredictor

# Initialize
velocity_estimator = get_velocity_estimator(fps=10.0)
predictor = TwoLevelTrajectoryPredictor(pred_steps=10)

# In your tracking loop
for obj in tracked_objects:
    velocity = velocity_estimator.update(obj["track_id"], obj["box"], frame_idx)
    predictor.update_observation(obj["track_id"], obj["box"], velocity)
    
    # Predict with context
    context = {'obstacles': [...], 'speed_limit': 20.0, ...}
    result = predictor.predict(obj["track_id"], obj, context)
    trajectory = result['trajectory']  # (10, 2)
    policy = result['policy_name']     # e.g., 'turn_left'
```

See `examples/integration_example.py` for complete working examples.

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

# Create loader instance (NOT generator)
loader = NuScenesLoader(NUSCENES_ROOT)

# Run all experiments (A, B, C, D, E)
# The loader will create fresh generators for each experiment
run_ablation_study(loader, max_frames=100)

# Or run specific experiments
run_ablation_study(loader, max_frames=100, experiments=["A", "D", "E"])
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
