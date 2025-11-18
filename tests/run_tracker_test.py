"""
Simple runnable test for TransformerTracker behavior.
Run with: python tests/run_tracker_test.py
"""
import sys
import os

# Ensure project root is on sys.path when running the test directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Transformer.tracker import TransformerTracker

# Create tracker with small max_lost so it prunes quickly
tracker = TransformerTracker(iou_thresh=0.3, max_lost=2)

# Frame 1: one detection
frame_idx = 1
detections = [
    {"box": [10, 10, 50, 50], "cls_name": "car", "score": 0.9}
]
updated = tracker.update(detections, frame_idx)
if len(updated) != 1:
    raise SystemExit(f"Test failed: expected 1 updated track after frame 1, got {len(updated)}")
first_id = updated[0]["track_id"]
print(f"Frame 1 assigned track id: {first_id}")

# Frame 2: same object moved slightly
frame_idx = 2
detections = [
    {"box": [12, 11, 52, 51], "cls_name": "car", "score": 0.88}
]
updated = tracker.update(detections, frame_idx)
if len(updated) != 1:
    raise SystemExit(f"Test failed: expected 1 updated track after frame 2, got {len(updated)}")
second_id = updated[0]["track_id"]
print(f"Frame 2 assigned track id: {second_id}")

if first_id != second_id:
    raise SystemExit(f"Test failed: track id changed across frames: {first_id} -> {second_id}")

print("PASS: Tracker maintained same ID across two frames")

# Frame 3: object disappears (no detections)
frame_idx = 3
detections = []
updated = tracker.update(detections, frame_idx)
print("Frame 3 (no detections) updated length:", len(updated))

print("All tracker tests completed.")
