import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge, RFDETRSegXLarge, RFDETRSeg2XLarge
from rfdetr.assets.coco_classes import COCO_CLASSES

MODEL_MAP = {
    "small":   (RFDETRSegSmall,   "models/rf-detr-seg-small.pt"),
    "medium":  (RFDETRSegMedium,  "models/rf-detr-seg-medium.pt"),
    "large":   (RFDETRSegLarge,   "models/rf-detr-seg-large.pt"),
    "xlarge":  (RFDETRSegXLarge,  "models/rf-detr-seg-xlarge.pt"),
    "2xlarge": (RFDETRSeg2XLarge, "models/rf-detr-seg-2xlarge.pt"),
}


class Detector:
    def __init__(self, model_size: str = "small", threshold: float = 0.5):
        if model_size not in MODEL_MAP:
            raise ValueError(f"model_size must be one of {list(MODEL_MAP)}")
        self.threshold = threshold
        cls, weights_path = MODEL_MAP[model_size]
        self.model = cls(pretrain_weights=weights_path)
        self.model.optimize_for_inference(compile=False, dtype="float16")

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        """Run detection on a BGR frame, return sv.Detections with masks."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.model.predict(frame_rgb, threshold=self.threshold)

    @staticmethod
    def labels(detections: sv.Detections) -> list[str]:
        return [COCO_CLASSES[class_id] for class_id in detections.class_id]
