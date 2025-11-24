# Transformer/fusion.py

"""
CNN-Transformer Fusion Module

Implements fusion logic between CNN-based detectors (FRCNN) and 
Transformer-based detectors (DETR) following the paper:
"CNN-transformer mixed model for object detection" (arXiv:2212.06714)

Strategy:
- FRCNN provides accurate localization (local features)
- DETR provides contextual validation (global features)
- Use DETR to filter false positives
- Use FRCNN boxes when both agree (IoU > threshold)
"""

import numpy as np
from typing import List, Dict, Optional
from config import DETR_CLASS_MAP


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        boxA: [x1, y1, x2, y2]
        boxB: [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB
    
    xi1 = max(x1A, x1B)
    yi1 = max(y1A, y1B)
    xi2 = min(x2A, x2B)
    yi2 = min(y2A, y2B)
    
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    areaA = max(0, x2A - x1A) * max(0, y2A - y1A)
    areaB = max(0, x2B - x1B) * max(0, y2B - y1B)
    
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def fuse_cnn_transformer(
    frcnn_dets: List[Dict],
    detr_dets: List[Dict],
    iou_thresh: float = 0.3,
    confidence_penalty: float = 0.7
) -> List[Dict]:
    """
    Fuse FRCNN (CNN local features) and DETR (Transformer global context).
    
    Strategy:
    - FRCNN provides accurate localization
    - DETR provides contextual validation
    - Use DETR to filter false positives
    - Use FRCNN boxes when both agree (IoU > threshold)
    - Return detections with validation status
    
    Args:
        frcnn_dets: List of dicts with keys: box, cls_name, score, track_id (optional)
        detr_dets: List of dicts with keys: box, label, score
        iou_thresh: IoU threshold for matching (default: 0.3)
        confidence_penalty: Multiplier for unvalidated FRCNN detections (default: 0.7)
    
    Returns:
        List of dicts with: box, cls_name, score, track_id (if present), validated_by
        
    validated_by values:
        - "cnn+transformer": Both FRCNN and DETR agree (IoU > threshold)
        - "cnn_only": Only FRCNN detected, no DETR confirmation
    """
    
    fused_results = []
    
    for frcnn_det in frcnn_dets:
        frcnn_box = frcnn_det["box"]
        frcnn_cls = frcnn_det["cls_name"]
        frcnn_score = frcnn_det["score"]
        frcnn_tid = frcnn_det.get("track_id", None)
        
        # Find best matching DETR detection
        best_iou = 0.0
        best_detr_score = 0.0
        
        for detr_det in detr_dets:
            detr_label = detr_det.get("label")
            if detr_label not in DETR_CLASS_MAP:
                continue
            
            detr_cls = DETR_CLASS_MAP[detr_label]
            
            # Only match same class
            if detr_cls != frcnn_cls:
                continue
            
            detr_box = detr_det["box"]
            iou_val = compute_iou(frcnn_box, detr_box)
            
            if iou_val > best_iou:
                best_iou = iou_val
                best_detr_score = detr_det.get("score", 0.0)
        
        # Determine validation status and final confidence
        if best_iou > iou_thresh:
            # Both CNN and Transformer agree
            validated_by = "cnn/transformer"
            # Average the confidence scores
            final_score = (frcnn_score + best_detr_score) / 2.0
        else:
            # Only CNN detected (no Transformer confirmation)
            validated_by = "cnn"
            # Reduce confidence due to lack of validation
            final_score = frcnn_score * confidence_penalty
        
        fused_det = {
            "box": frcnn_box,
            "cls_name": frcnn_cls,
            "score": final_score,
            "validated_by": validated_by
        }
        
        # Preserve track_id if present
        if frcnn_tid is not None:
            fused_det["track_id"] = frcnn_tid
        
        fused_results.append(fused_det)
    
    return fused_results


def filter_by_validation(
    fused_dets: List[Dict],
    require_transformer: bool = False
) -> List[Dict]:
    """
    Filter fused detections based on validation status.
    
    Args:
        fused_dets: List of fused detections from fuse_cnn_transformer
        require_transformer: If True, only keep detections validated by transformer
    
    Returns:
        Filtered list of detections
    """
    if not require_transformer:
        return fused_dets
    
    return [det for det in fused_dets if det["validated_by"] == "cnn+transformer"]
