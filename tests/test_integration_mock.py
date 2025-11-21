"""
Simple integration test for v2 pipeline without requiring NuScenes dataset.
Run with: python tests/test_integration_mock.py
"""
import sys
import os
import numpy as np

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("Testing v2 Pipeline Integration (Mock Data)...")
print("=" * 60)

# Create mock frame
print("\n[1/5] Creating mock frame...")
mock_frame = np.zeros((600, 800, 3), dtype=np.uint8)
mock_frame[:] = (100, 100, 100)  # Gray background
print("✓ Mock frame created (800x600)")

# Test fusion with mock detections
print("\n[2/5] Testing fusion module...")
try:
    from Transformer.fusion import fuse_cnn_transformer
    
    # Mock FRCNN detections
    frcnn_dets = [
        {"box": [100, 100, 200, 200], "cls_name": "car", "score": 0.9, "track_id": 1},
        {"box": [300, 150, 400, 250], "cls_name": "person", "score": 0.85, "track_id": 2}
    ]
    
    # Mock DETR detections (with overlap)
    detr_dets = [
        {"box": [105, 102, 205, 202], "label": 3, "score": 0.88},  # Car (overlaps track 1)
        {"box": [500, 400, 600, 500], "label": 1, "score": 0.75}   # Person (no overlap)
    ]
    
    fused = fuse_cnn_transformer(frcnn_dets, detr_dets, iou_thresh=0.3)
    
    print(f"  Input: {len(frcnn_dets)} FRCNN + {len(detr_dets)} DETR")
    print(f"  Output: {len(fused)} fused detections")
    for det in fused:
        print(f"    - {det['cls_name']}: validated by {det['validated_by']}")
    
    print("✓ Fusion module working")
    
except Exception as e:
    print(f"✗ Fusion test failed: {e}")
    import traceback
    traceback.print_exc()

# Test tracker
print("\n[3/5] Testing TransformerTracker...")
try:
    from Transformer.tracker import TransformerTracker
    
    tracker = TransformerTracker(iou_thresh=0.5, max_lost=5)
    
    # Frame 1
    detections = [
        {"box": [100, 100, 200, 200], "cls_name": "car", "score": 0.9}
    ]
    tracked = tracker.update(detections, frame_idx=1, frame=mock_frame)
    
    print(f"  Frame 1: {len(tracked)} tracks")
    
    # Frame 2 - same object moved
    detections = [
        {"box": [110, 105, 210, 205], "cls_name": "car", "score": 0.88}
    ]
    tracked = tracker.update(detections, frame_idx=2, frame=mock_frame)
    
    print(f"  Frame 2: {len(tracked)} tracks")
    assert len(tracked) == 1, "Should maintain 1 track"
    
    print("✓ Tracker working")
    
except Exception as e:
    print(f"✗ Tracker test failed: {e}")
    import traceback
    traceback.print_exc()

# Test trajectory prediction
print("\n[4/5] Testing trajectory prediction...")
try:
    from Transformer.trajectory_predictor import linear_extrapolate
    
    trajectory = [
        [100, 100, 200, 200],
        [110, 105, 210, 205],
        [120, 110, 220, 210]
    ]
    
    predicted = linear_extrapolate(trajectory, steps=1)
    print(f"  Input: {len(trajectory)} boxes")
    print(f"  Predicted box: [{predicted[0]:.1f}, {predicted[1]:.1f}, {predicted[2]:.1f}, {predicted[3]:.1f}]")
    
    print("✓ Trajectory prediction working")
    
except Exception as e:
    print(f"✗ Trajectory prediction test failed: {e}")
    import traceback
    traceback.print_exc()

# Test configuration
print("\n[5/5] Testing configuration...")
try:
    from config import (
        FUSION_MODE, 
        FUSION_IOU_THRESHOLD,
        TRAJECTORY_PREDICTOR,
        EXPERIMENT_MODE
    )
    
    print(f"  FUSION_MODE: {FUSION_MODE}")
    print(f"  FUSION_IOU_THRESHOLD: {FUSION_IOU_THRESHOLD}")
    print(f"  TRAJECTORY_PREDICTOR: {TRAJECTORY_PREDICTOR}")
    print(f"  EXPERIMENT_MODE: {EXPERIMENT_MODE}")
    
    print("✓ Configuration loaded")
    
except Exception as e:
    print(f"✗ Configuration test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Integration tests completed!")
print("=" * 60)
print("\nNext steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Configure NuScenes path in config.py")
print("3. Run main pipeline: python main.py")
print("4. Run ablation study (if NuScenes available)")
