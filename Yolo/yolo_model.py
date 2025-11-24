import torch
from ultralytics import YOLO
from config import VALID_CLASSES, YOLO_CONF, YOLO_IOU, YOLO_IMGSZ, YOLO_WEIGHTS
import numpy as np


class YOLODetector:
    """YOLO detector for object detection only. ID assignment is handled by pipeline tracker (TransformerTracker)."""

    def __init__(self, device):
        self.device = device
        # use configured weights and inference settings
        self.model = YOLO(YOLO_WEIGHTS).to(device)
        self.frame_idx = 0
        self.names = self.model.names

    def detect_and_track(self, frame):
        """
        Runs YOLO detection only (no tracking - pipeline tracker handles IDs) and returns raw results object.
        Used by DetectorPipeline.
        """
        # detection-only: no built-in tracking
        results = self.model(frame, conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ)[0]

        # Attempt to tighten boxes using segmentation masks when present
        try:
            masks_obj = getattr(results, "masks", None)
            if masks_obj is not None:
                # ultralytics mask data may live in .data or .masks; try both
                mask_arrays = None
                for attr in ("data", "masks"):
                    if hasattr(masks_obj, attr):
                        mask_arrays = getattr(masks_obj, attr)
                        break

                if mask_arrays is not None:
                    # convert tensor-like to numpy if needed
                    try:
                        if hasattr(mask_arrays, "cpu"):
                            mask_arrays = mask_arrays.cpu().numpy()
                    except Exception:
                        pass

                    # if mask_arrays is list-like and per-detection, iterate
                    if isinstance(mask_arrays, (list, tuple)) or (hasattr(mask_arrays, 'ndim') and mask_arrays.ndim >= 2):
                        boxes = results.boxes.xyxy.cpu().numpy()
                        new_boxes = []
                        for i, box in enumerate(boxes):
                            try:
                                m = mask_arrays[i]
                                # mask may be HxW boolean or float array
                                if hasattr(m, 'astype'):
                                    m = np.array(m)
                                # if mask is not 2D, try to squeeze
                                if m.ndim == 3:
                                    m = m.squeeze()
                                ys, xs = np.where(m > 0)
                                if len(xs) > 0 and len(ys) > 0:
                                    mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max(), ys.max()
                                    # ensure within image
                                    h, w = m.shape[:2]
                                    mx1 = max(0, mx1); my1 = max(0, my1)
                                    mx2 = min(w - 1, mx2); my2 = min(h - 1, my2)
                                    new_box = [float(mx1), float(my1), float(mx2), float(my2)]
                                    new_boxes.append(new_box)
                                else:
                                    new_boxes.append(box.tolist())
                            except Exception:
                                new_boxes.append(box.tolist())

                        # attach new boxes back to results if lengths match
                        if len(new_boxes) == len(boxes):
                            try:
                                # set results.boxes.xyxy to tightened boxes if possible
                                import torch
                                results.boxes.xyxy = torch.tensor(new_boxes, device=results.boxes.xyxy.device)
                            except Exception:
                                pass
        except Exception:
            # fail-safe: do not prevent detection if mask tightening errors occur
            pass

        return results

    def detect(self, frame):
        """
        Runs YOLO detection only and returns cleaned results dict list.
        ID assignment happens in the main pipeline via TransformerTracker.
        """
        self.frame_idx += 1
        results = self.model(frame, conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ)[0]

        detections = []
        boxes = results.boxes.xyxy.cpu().numpy()
        clss = results.boxes.cls.int().cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        # Try to extract per-detection masks (if the model provides segmentation)
        masks = None
        try:
            masks_obj = getattr(results, "masks", None)
            if masks_obj is not None:
                # ultralytics mask data may live in .data or be a tensor/list
                if hasattr(masks_obj, "data"):
                    masks = masks_obj.data
                elif hasattr(masks_obj, "cpu"):
                    masks = masks_obj.cpu().numpy()
                else:
                    masks = masks_obj
        except Exception:
            masks = None

        for i, (box, cls_idx, score) in enumerate(zip(boxes, clss, scores)):
            cls_name = self.names[int(cls_idx)]

            if cls_name not in VALID_CLASSES:
                continue
            if score < YOLO_CONF:
                continue

            det = {
                "box": box.tolist(),
                "track_id": None,
                "cls_name": cls_name,
                "score": float(score),
                "mask": None
            }

            # attach mask if available and shapes align
            try:
                if masks is not None:
                    # masks may be a list/ndarray of per-detection boolean arrays
                    m = None
                    if isinstance(masks, (list, tuple)):
                        if i < len(masks):
                            m = masks[i]
                    else:
                        # assume ndarray with shape (N,H,W) or (H,W,N)
                        m_arr = np.array(masks)
                        if m_arr.ndim == 3:
                            # (N,H,W)
                            if i < m_arr.shape[0]:
                                m = m_arr[i]
                        elif m_arr.ndim == 2:
                            # single mask
                            m = m_arr

                    if m is not None:
                        # ensure boolean mask
                        m_np = np.array(m)
                        if m_np.dtype != bool:
                            m_np = m_np > 0.5
                        det["mask"] = m_np
            except Exception:
                det["mask"] = None

            detections.append(det)

        return detections
