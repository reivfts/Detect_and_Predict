# Multi-Stage Object Detection, Tracking, and Trajectory Prediction

A real-time deep learning pipeline for autonomous driving perception using the nuScenes dataset. This system combines YOLO, Faster R-CNN, and DETR for detection, applies physics-constrained Kalman filtering with hybrid Transformer-based trajectory prediction to forecast vehicle positions up to 5 seconds ahead.

## System Architecture

```
Input Frame (nuScenes Camera)
         |
         v
    [YOLO11m-seg]  Fast detection + segmentation
         |
         v
  [Faster R-CNN]  Refined bounding boxes
         |
         v
      [DETR]  Transformer-based detection
         |
         v
  [Fusion Module] --> Combined Detections
         |
         v
[Multi-Hypothesis Tracker]
    |            |
    |    [ResNet-18 Appearance]
    |            |
    v            v
[Kalman Filter] + [Hungarian Association]
  (Physics-Constrained)
         |
         v
  [Velocity Estimator] (EMA, Outlier Rejection)
         |
         v
[Hybrid Trajectory Predictor]
    |                    |
    v                    v
[Kalman Physics]   [Transformer Attention]
(Weight: 0.7)         (Weight: 0.3)
    |                    |
    +--------------------+
              |
              v
    [Uncertainty Quantification]
       (Covariance Ellipses)
              |
              v
      Predicted Trajectories
   (12-30 frames, 2-5 seconds)
```

### Pipeline Stages

**Three-Stage Detection:**
1. YOLO11m-seg: Fast initial detection and instance segmentation
2. Faster R-CNN (ResNet-50 + FPN): Refined bounding boxes with region proposals
3. DETR (ResNet-50): Transformer-based detection for global context

**Tracking and Prediction:**
- Multi-hypothesis tracker with ResNet-18 appearance embeddings
- Physics-constrained Kalman filter (enforces max velocity 20 m/s, acceleration 4 m/s²)
- Hybrid prediction: Combines Kalman physics-based (70%) and Transformer learning-based (30%)
- Uncertainty quantification: Covariance ellipses show prediction confidence
- Extended horizon: Configurable 2-5 second predictions (12-30 frames at 6Hz)
- Velocity-based visualization: Color-coded trajectories (green=slow, yellow=medium, red=fast)

## Quick Start

```powershell
# Install dependencies
pip install -r requirements.txt

# Run pipeline with default settings (2-second hybrid prediction)
python main.py

# Specify scene and camera
python main.py --scene-idx 5 --camera CAM_FRONT_LEFT --max-frames 150
```

## Models and Techniques

**Detection Models:**
- YOLO11m-seg (Ultralytics): Instance segmentation
- Faster R-CNN (torchvision): Region-based detection with ResNet-50 + FPN
- DETR (HuggingFace): Transformer encoder-decoder with ResNet-50

**Tracking:**
- ResNet-18: Appearance embeddings
- Kalman Filter (filterpy): Physics-constrained constant velocity model (6D state: cx, cy, vx, vy, w, h)
- Hungarian algorithm: Optimal data association

**Trajectory Prediction:**
- Linear extrapolation (baseline)
- Kalman filter prediction
- Angle sampling: 3 hypotheses (straight, ±17°)
- DONUT overprediction: Multi-horizon forecasting [1, 3, 5, 8 steps]
- Transformer: 4-head attention over 10-frame history
- Hybrid ensemble: Weighted combination of Kalman + Transformer

## Performance Metrics

- **Speed**: 4 FPS on CUDA (scene-dependent)
- **Accuracy**: 1-step ADE 3.42px (0.17s ahead), 3-step ADE 6.64px (0.50s ahead)
- **Tracking**: Overall IoU 0.7213
- **Dataset**: nuScenes v1.0-mini (10 scenes, 23,830 annotations)
- **Prediction Horizon**: 2-5 seconds (configurable)

## Configuration

Edit `config.py` for model selection, thresholds, and prediction settings:

```python
# Detection Configuration
USE_YOLO = True
USE_FRCNN = True
USE_DETR = True
YOLO_CONF = 0.25
FRCNN_CONF = 0.6
DETR_CONF = 0.5

# Tracking Configuration
TRACK_PATIENCE = 30
IOU_THRESHOLD = 0.3
APPEARANCE_WEIGHT = 0.3
MOTION_WEIGHT = 0.7

# Prediction Configuration
TRAJECTORY_PREDICTOR = "kalman"  # Options: "kalman", "transformer", "hybrid"
USE_HYBRID_PREDICTION = True
HYBRID_KALMAN_WEIGHT = 0.7
HYBRID_TRANSFORMER_WEIGHT = 0.3

# Angle Sampling
USE_ANGLE_SAMPLING = True
ANGLE_SAMPLE_RANGE = 0.3  # ±17° in radians
ANGLE_SAMPLE_STEPS = 3

# Overprediction
USE_OVERPREDICTION = True
OVERPREDICTION_HORIZONS = [1, 3, 5, 8]

# Physics Constraints
MAX_ACCELERATION = 4.0  # m/s²
MAX_VELOCITY = 20.0  # m/s

# Visualization
SHOW_UNCERTAINTY = True
VELOCITY_COLOR_CODING = True
```

## Project Structure

```
main.py                   # Pipeline orchestration
config.py                 # Configuration parameters
Detection/
  detector.py             # Three-stage detection pipeline
  drawer.py               # Visualization (uncertainty, velocity colors)
FRCNN/
  frcnn_model.py          # Faster R-CNN wrapper
Yolo/
  yolo_model.py           # YOLO wrapper
Transformer/
  detr_model.py           # DETR wrapper
  tracker.py              # Multi-hypothesis tracker
  fusion.py               # Detection fusion logic
  trajectory_predictor.py # Kalman, Transformer, and Hybrid predictors
Nuscenes/
  loader.py               # Dataset loader
data/
  nuscenes.py             # nuScenes data interface
  trackings/              # Evaluation outputs
tools/
  analyze_evaluation.py   # Compute ADE/FDE/RMSE metrics
  visualize_worst.py      # Visualize worst-performing tracks
```

## Research-Based Improvements

This implementation includes eight enhancements for real-world deployment:

### 1. Physics Constraints
Kalman filter enforces realistic motion limits:
- Max velocity: 20 m/s (urban/highway scenarios)
- Max acceleration: 4 m/s² (comfortable driving limits)
- Prevents trajectory jumps and unrealistic speed changes

### 2. Hybrid Prediction Ensemble
Combines physics-based and learning-based approaches:
- Kalman filter (70% weight): Reliable short-term, physics-grounded
- Transformer (30% weight): Captures complex patterns (turns, stops)
- Weighted combination provides reliability and adaptability
- Automatic fallback to Kalman on Transformer failure

### 3. Angle-Based Trajectory Sampling
Tests multiple heading hypotheses for turn prediction:
- 3 hypotheses: straight, +17°, -17°
- Scores based on velocity history and acceleration patterns
- Improves prediction accuracy on curved trajectories
- Inspired by DONUT (ICCV 2025) angle sampling

### 4. Multi-Horizon Overprediction
DONUT-inspired prediction strategy:
- Generates forecasts at horizons: [1, 3, 5, 8] steps
- Provides predictions at multiple time scales
- Enables both immediate collision avoidance and strategic planning

### 5. Distance-Weighted Error Metrics
Prioritizes nearby objects for accuracy:
- Error weighting inversely proportional to distance
- Focus computational resources on safety-critical near objects
- More realistic evaluation for autonomous driving

### 6. Enhanced Velocity Estimation
Robust velocity tracking with:
- 10-frame exponential moving average (α=0.5)
- Median-based outlier rejection (2σ threshold)
- Acceleration/deceleration detection for angle scoring
- Improved tracking persistence (max_lost increased from 5 to 10 frames)

### 7. Adaptive Noise Tuning
Speed-based Kalman filter adaptation:
- Process noise scales 1.0x-2.0x with object speed
- Higher uncertainty for faster-moving objects
- Better handling of dynamic motion patterns

### 8. Uncertainty Quantification
Covariance-based confidence visualization:
- Propagates Kalman filter covariance through predictions
- Displays 2-sigma ellipses (95% confidence intervals)
- Uncertainty grows with prediction horizon
- Enables risk-aware decision making

**Impact**: These improvements enhance robustness, physical plausibility, and real-world applicability while maintaining real-time performance (4 FPS). Average displacement error improved from baseline linear extrapolation to 3.42px at 1-step ahead.

## Dataset

Uses nuScenes v1.0-mini with 6 camera views:
- CAM_FRONT (default)
- CAM_FRONT_LEFT
- CAM_FRONT_RIGHT
- CAM_BACK
- CAM_BACK_LEFT
- CAM_BACK_RIGHT

Each scene contains approximately 40 frames at 6Hz sampling with 3D bounding box annotations.

## Dependencies

- PyTorch 2.0+
- torchvision
- ultralytics (YOLO)
- transformers (HuggingFace)
- opencv-python
- nuscenes-devkit
- filterpy (Kalman filter)
- numpy
- pandas
- Pillow

## Evaluation

The system computes standard trajectory prediction metrics:
- **ADE** (Average Displacement Error): Mean Euclidean distance across all predicted points
- **FDE** (Final Displacement Error): Euclidean distance at prediction horizon
- **RMSE**: Root mean squared error of predictions
- **IoU**: Bounding box overlap for tracking quality

Results are saved to `data/trackings/evaluation.csv` with per-track and per-frame metrics.

Run evaluation analysis:
```powershell
python tools/analyze_evaluation.py
```

## Research References

1. **DONUT**: "DONUT: Rethinking Dual-stage Methodology for End-to-end Autonomous Driving" (ICCV 2025)
2. **Trajectron++**: "Trajectron++: Dynamically-Feasible Trajectory Forecasting" (Salzmann et al., 2020)
3. **AgentFormer**: "Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting" (Yuan et al., 2021)
4. **MultiPath**: "Multiple Probabilistic Anchor Trajectory Hypotheses" (Chai et al., 2019)

## License

For academic and research use. See individual model licenses:
- YOLO: AGPL-3.0
- DETR: Apache-2.0
- Faster R-CNN: BSD-3-Clause
