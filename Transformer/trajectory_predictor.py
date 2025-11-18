# trajectory_predictor.py

import os
import csv
import numpy as np
from collections import defaultdict

SAVE_PATH = r"C:\Users\rayra\OneDrive\Desktop\Detect_and_Predict\data\trackings"
CSV_PATH = os.path.join(SAVE_PATH, "evaluation.csv")
os.makedirs(SAVE_PATH, exist_ok=True)

# Global accumulator
ALL_EVALS = []

def linear_extrapolate(trajectory, steps=1):
    if len(trajectory) < 2:
        return trajectory[-1] if trajectory else None

    box0 = trajectory[-2]
    box1 = trajectory[-1]
    dx = [b1 - b0 for b0, b1 in zip(box0, box1)]
    pred_box = [b + d * steps for b, d in zip(box1, dx)]
    return pred_box

def box_center(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2, (y1 + y2) / 2]

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea > 0 else 0.0

def evaluate_trajectory(track_history, updated_detections, frame_idx):
    global ALL_EVALS

    for det in updated_detections:
        tid = det["track_id"]
        actual_box = det["box"]

        if tid not in track_history or len(track_history[tid]) < 2:
            continue

        history = list(track_history[tid])[-2:]
        pred_box = linear_extrapolate(history)
        if pred_box is None:
            continue

        actual_center = box_center(actual_box)
        pred_center = box_center(pred_box)

        error = np.linalg.norm(np.array(actual_center) - np.array(pred_center))
        iou = compute_iou(actual_box, pred_box)

        ALL_EVALS.append({
            "frame": frame_idx,
            "track_id": tid,
            "iou": round(iou, 4),
            "center_distance": round(error, 2),
            "gt_box": actual_box,
            "pred_box": [round(x, 2) for x in pred_box]
        })

def save_evaluation_summary():
    if not ALL_EVALS:
        print("No trajectory evaluation data found.")
        return

    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "track_id", "iou", "center_distance", "gt_box", "pred_box"])
        writer.writeheader()
        for row in ALL_EVALS:
            writer.writerow(row)

        writer.writerow({})  # empty row

        ious = [row["iou"] for row in ALL_EVALS]
        dists = [row["center_distance"] for row in ALL_EVALS]
        track_ids = set(row["track_id"] for row in ALL_EVALS)
        frame_ids = set(row["frame"] for row in ALL_EVALS)

        writer.writerow({"frame": "Metrics Summary"})
        writer.writerow({"track_id": "Average IoU", "iou": round(np.mean(ious), 4)})
        writer.writerow({"track_id": "Average Center Distance", "center_distance": round(np.mean(dists), 2)})
        writer.writerow({"track_id": "Total Tracks Evaluated", "iou": len(track_ids)})
        writer.writerow({"track_id": "Total Frames Evaluated", "iou": len(frame_ids)})

def save_text_summary():
    txt_path = os.path.join(SAVE_PATH, "accuracy.txt")

    if not ALL_EVALS:
        return

    ious = [row["iou"] for row in ALL_EVALS]
    dists = [row["center_distance"] for row in ALL_EVALS]
    track_ids = set(row["track_id"] for row in ALL_EVALS)
    frame_ids = set(row["frame"] for row in ALL_EVALS)

    with open(txt_path, 'w') as f:
        f.write("=== Trajectory Evaluation Summary ===\n")
        f.write(f"Average IoU: {np.mean(ious):.4f}\n")
        f.write(f"Average Center Distance: {np.mean(dists):.2f} pixels\n")
        f.write(f"Total Tracks Evaluated: {len(track_ids)}\n")
        f.write(f"Total Frames Evaluated: {len(frame_ids)}\n")

    print(f"📄 Accuracy summary saved to: {txt_path}")


    print(f"✅ Trajectory evaluation saved to: {CSV_PATH}")