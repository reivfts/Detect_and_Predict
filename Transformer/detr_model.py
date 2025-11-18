# Transformer/detr_model.py

import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

class DETRRefiner:
    """
    Runs DETR object detection on a full image and returns refined bounding boxes.
    """

    def __init__(self, device="cuda", threshold=0.50):
        self.device = device
        self.threshold = threshold

        model_name = "facebook/detr-resnet-50"
        print(f"[DETR] Loading {model_name}...")

        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name).to(device)
        self.model.eval()

    def predict(self, image_bgr):
        """
        image_bgr: numpy array in BGR (OpenCV)
        Returns: list of {box, label, score}
        """

        # Convert BGR → RGB for DETR
        image_rgb = image_bgr[:, :, ::1][:, :, [2,1,0]]

        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image_rgb.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.threshold
        )[0]

        boxes = results["boxes"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        scores = results["scores"].cpu().numpy()

        refined = []
        for b, lab, s in zip(boxes, labels, scores):
            refined.append({
                "box": b,
                "label": int(lab),
                "score": float(s)
            })

        return refined
