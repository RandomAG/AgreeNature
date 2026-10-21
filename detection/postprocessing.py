"""
Detection post-processing for the proposed pipeline.

Converts raw object-detection predictions into a standardized representation used by the temporal validation and threat assessment stages.
"""

from dataclasses import dataclass
from typing import List
import torch


@dataclass
class Detection:
    """Standard representation of a single detection"""

    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]


class DetectionPostProcessor:
    """Filtering and standardizing detector predictions"""

    def __init__(self, class_names=None, confidence_threshold=0.25):
        # Harmonized classes for weapon detection
        self.class_names = class_names or [
            "Pistol",
            "Knife",
            "Rifle"
        ]
        self.confidence_threshold = confidence_threshold

    def process(self, predictions):
        """
        Converting model predictions into Detection objects.

        Expected prediction format per detection:
            [x1, y1, x2, y2, confidence, class_id]
        """
        detections = []

        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()

        for prediction in predictions:
            x1, y1, x2, y2, confidence, class_id = prediction
            confidence = float(confidence)

            if confidence < self.confidence_threshold:
                continue

            class_id = int(class_id)

            if 0 <= class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = "Unknown"

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=[float(x1), float(y1), float(x2), float(y2)]
                )
            )

        return detections