import cv2
from Yolo.yolo_model import YOLODetector
from FRCNN.frcnn_model import FRCNNRefiner
import numpy as np

from config import VALID_CLASSES, DEVICE, DETR_IOU_MATCH


# IOU calculation
def iou(boxA, boxB):
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    xi1 = max(x1A, x1B)
    yi1 = max(y1A, y1B)
    xi2 = min(x2A, x2B)
    yi2 = min(y2A, y2B)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    areaA = max(0, (x2A - x1A)) * max(0, (y2A - y1A))
    areaB = max(0, (x2B - x1B)) * max(0, (y2B - y1B))

    union = areaA + areaB - inter
    return inter / union if union > 0 else 0


# DETECTOR PIPELINE
class DetectorPipeline:
    """Detection pipeline: YOLO detection -> FRCNN refinement (no tracking - handled by main pipeline)."""
    
    def __init__(self):
        # ensure detector constructed with proper device
        self.yolo = YOLODetector(device=DEVICE)
        self.frcnn = FRCNNRefiner()

    def process(self, frame):
        """
        Process frame: YOLO detection to FRCNN refinement.
        Returns list of dicts: [{"box": [x1,y1,x2,y2], "cls_name": str, "score": float}, ...]
        No IDs returned; pipeline tracker (TransformerTracker in main.py) assigns IDs.
        """
        # YOLO detection only (no tracking)
        yolo_dets = self.yolo.detect(frame)

        if not yolo_dets:
            return []

        # FRCNN refinement
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frcnn_dets = self.frcnn.refine(frame_rgb)

        refined = []
        for yolo_det in yolo_dets:
            box = yolo_det["box"]
            cls_name = yolo_det["cls_name"]
            score = yolo_det["score"]
            mask = yolo_det.get("mask")

            best_iou = 0
            best_box = None

            # Find matching FRCNN box of same class
            for fr_box, _, fr_cls in frcnn_dets:
                if fr_cls != cls_name:
                    continue

                iou_val = iou(box, fr_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_box = fr_box

            # Choose refined box: prefer FRCNN when it strongly matches
            chosen_box = None
            if best_box is not None and best_iou >= DETR_IOU_MATCH:
                chosen_box = best_box
            else:
                chosen_box = box

            # If YOLO provided a segmentation mask, use its tight bbox when sensible
            try:
                if mask is not None:
                    m = mask
                    m_arr = None
                    try:
                        m_arr = np.array(m)
                    except Exception:
                        m_arr = None

                    if m_arr is not None:
                        # If mask size doesn't match frame, resize mask to frame size
                        h, w = frame.shape[:2]
                        if m_arr.ndim == 2 and (m_arr.shape[0] != h or m_arr.shape[1] != w):
                            try:
                                m_resized = cv2.resize((m_arr.astype('uint8') * 255), (w, h), interpolation=cv2.INTER_NEAREST)
                                m_bool = m_resized > 0
                            except Exception:
                                m_bool = m_arr > 0
                        else:
                            m_bool = m_arr > 0

                        ys, xs = np.where(m_bool)
                        if len(xs) > 0 and len(ys) > 0:
                            mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max(), ys.max()
                            mask_box = [float(mx1), float(my1), float(mx2), float(my2)]
                            # only use mask_box if it overlaps reasonably with chosen_box
                            iou_with_chosen = iou(chosen_box, mask_box)
                            if iou_with_chosen > 0.05:
                                chosen_box = mask_box
            except Exception:
                pass

            refined.append({"box": chosen_box, "cls_name": cls_name, "score": score, "mask": mask})

        return refined
