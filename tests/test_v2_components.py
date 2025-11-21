"""
Test script for v2 components - fusion and trajectory predictor.
Run with: python tests/test_v2_components.py
"""
import sys
import os

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("Testing v2 Components...")
print("=" * 60)

# Test 1: Fusion module
print("\n[Test 1] Testing CNN-Transformer Fusion...")
try:
    from Transformer.fusion import fuse_cnn_transformer, compute_iou
    
    # Test IoU calculation
    box1 = [10, 10, 50, 50]
    box2 = [10, 10, 50, 50]
    iou = compute_iou(box1, box2)
    assert iou == 1.0, f"Expected IoU=1.0 for identical boxes, got {iou}"
    
    box3 = [100, 100, 150, 150]
    iou = compute_iou(box1, box3)
    assert iou == 0.0, f"Expected IoU=0.0 for non-overlapping boxes, got {iou}"
    
    # Test fusion with matching detections
    frcnn_dets = [
        {"box": [10, 10, 50, 50], "cls_name": "car", "score": 0.9, "track_id": 1}
    ]
    detr_dets = [
        {"box": [12, 11, 52, 51], "label": 3, "score": 0.85}  # label 3 = car
    ]
    
    fused = fuse_cnn_transformer(frcnn_dets, detr_dets, iou_thresh=0.3)
    
    assert len(fused) == 1, f"Expected 1 fused detection, got {len(fused)}"
    assert fused[0]["validated_by"] == "cnn+transformer", \
        f"Expected cnn+transformer validation, got {fused[0]['validated_by']}"
    
    # Test fusion with non-matching detection
    detr_dets_far = [
        {"box": [100, 100, 150, 150], "label": 3, "score": 0.85}
    ]
    fused2 = fuse_cnn_transformer(frcnn_dets, detr_dets_far, iou_thresh=0.3)
    assert fused2[0]["validated_by"] == "cnn_only", \
        f"Expected cnn_only validation, got {fused2[0]['validated_by']}"
    
    print("✓ Fusion module working correctly")
    
except Exception as e:
    print(f"✗ Fusion test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Trajectory predictor (linear)
print("\n[Test 2] Testing Linear Trajectory Prediction...")
try:
    from Transformer.trajectory_predictor import linear_extrapolate, box_center
    
    # Test linear extrapolation
    trajectory = [
        [10, 10, 50, 50],  # Moving right and down
        [15, 12, 55, 52]
    ]
    
    predicted = linear_extrapolate(trajectory, steps=1)
    assert predicted is not None, "Expected prediction, got None"
    
    # Check that prediction follows motion direction
    center1 = box_center(trajectory[0])
    center2 = box_center(trajectory[1])
    center_pred = box_center(predicted)
    
    # Should continue moving right and down
    assert center_pred[0] > center2[0], "Prediction should move right"
    assert center_pred[1] > center2[1], "Prediction should move down"
    
    print("✓ Linear trajectory prediction working correctly")
    
except Exception as e:
    print(f"✗ Trajectory prediction test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Transformer predictor (basic instantiation)
print("\n[Test 3] Testing Transformer Trajectory Predictor...")
try:
    from Transformer.trajectory_predictor import TrajectoryTransformerPredictor
    
    # Try to create predictor (may fail if torch not available)
    try:
        predictor = TrajectoryTransformerPredictor(
            history_len=5, 
            d_model=32, 
            nhead=2,
            device="cpu"
        )
        
        # Test prediction with sample trajectory
        trajectory = [
            [10, 10, 50, 50],
            [15, 12, 55, 52],
            [20, 14, 60, 54],
            [25, 16, 65, 56],
            [30, 18, 70, 58]
        ]
        
        predicted = predictor.predict(trajectory)
        assert predicted is not None, "Expected prediction, got None"
        assert len(predicted) == 4, f"Expected 4D bbox, got {len(predicted)}"
        
        print("✓ Transformer trajectory predictor working correctly")
        
    except ImportError as ie:
        print(f"⚠ Torch not available, skipping transformer predictor: {ie}")
    
except Exception as e:
    print(f"✗ Transformer predictor test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: DeepSORT wrapper (basic instantiation)
print("\n[Test 4] Testing DeepSORT Wrapper...")
try:
    from trackers.deepsort_wrapper import DeepSORTWrapper
    
    try:
        tracker = DeepSORTWrapper(max_age=30, n_init=3)
        print("✓ DeepSORT wrapper instantiated successfully")
    except ImportError as ie:
        print(f"⚠ DeepSORT not available: {ie}")
        print("  Install with: pip install deep-sort-realtime")
    
except Exception as e:
    print(f"✗ DeepSORT wrapper test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Component tests completed!")
print("=" * 60)
