import torch
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_pil_image

from config import DEVICE, FRCNN_SCORE_THRESH, FRCNN_ID2NAME

class FRCNNRefiner:
    def __init__(self):
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.model = fasterrcnn_resnet50_fpn(weights=weights).to(DEVICE)
        self.model.eval()

    def refine(self, frame_rgb):
        pil = to_pil_image(frame_rgb)

        with torch.no_grad():
            inp = self.preprocess(pil).to(DEVICE)
            out = self.model([inp])[0]

        refined = []
        for box, score, label in zip(out["boxes"], out["scores"], out["labels"]):
            if score < FRCNN_SCORE_THRESH:
                continue
            if int(label) in FRCNN_ID2NAME:
                refined.append((
                    box.cpu().numpy(),
                    float(score),
                    FRCNN_ID2NAME[int(label)]
                ))

        return refined
