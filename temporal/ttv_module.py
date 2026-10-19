"""
Temporal Threat Validation Module (TTVM)

Temporal validation of frame-level weapon detections:
    1. Multi-instance greedy spatial association via IoU >= tau_iou (0.50)
    2. Sliding FIFO observation window of size N (default = 5)
    3. Persistence computation: P_t = n_p / N >= tau_p (default = 0.60)
    4. Categorical majority voting: c* = Mode(C_t)
    5. EMA bounding box smoothing: b_t = alpha * b_t^curr + (1 - alpha) * b_{t-1}

Output:
    V_t = {b_t, c*, P_t}
"""

from collections import Counter, deque
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ValidatedDetection:
    """Validated temporal threat instance"""
    bbox: List[float]
    class_name: str
    confidence: float
    persistence: float


class TemporalThreatValidator:
    """
    Lightweight FIFO-based temporal threat validation with spatial IoU
    tracking and EMA bounding box stabilization.
    """

    def __init__(
        self,
        window_size: int = 5,
        persistence_threshold: float = 0.60,
        tau_iou: float = 0.50,
        alpha: float = 0.80
    ):
        if window_size < 1:
            raise ValueError("window_size must be positive.")
        if not 0.0 <= persistence_threshold <= 1.0:
            raise ValueError("persistence_threshold must be in [0, 1].")

        self.window_size = window_size
        self.persistence_threshold = persistence_threshold
        self.tau_iou = tau_iou
        self.alpha = alpha

        # Sliding FIFO window buffer (stores lists of frame detections)
        self.buffer = deque(maxlen=window_size)
        
        # Track previous smoothed bounding box for EMA filtering 
        self.prev_smoothed_bbox = None

    @staticmethod
    def _compute_iou(boxA: List[float], boxB: List[float]) -> float:
        """Compute Intersection-over-Union (IoU) between two bounding boxes (Eq. 37)."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea

        return interArea / unionArea if unionArea > 0.0 else 0.0

    def update(self, detections) -> Optional[ValidatedDetection]:
        """
        Updating the sliding window buffer, evaluating IoU-based cross-frame persistence,
        performing majority voting, and applying EMA box smoothing.
        """
        self.buffer.append(list(detections) if detections else [])

        if not detections:
            return None

        # Select candidate detection with highest confidence in current frame
        candidate = max(detections, key=lambda item: item.confidence)

        # Multi-Instance Temporal Association across window buffer (Section 4.6)
        matched_classes = [candidate.class_name]
        matched_frames = 1

        for past_frame in list(self.buffer)[:-1]:
            # Find best overlapping detection in historical frame
            best_iou = 0.0
            best_match = None
            for past_det in past_frame:
                iou = self._compute_iou(candidate.bbox, past_det.bbox)
                if iou >= self.tau_iou and iou > best_iou:
                    best_iou = iou
                    best_match = past_det

            if best_match is not None:
                matched_frames += 1
                matched_classes.append(best_match.class_name)

        # Persistence score computation: P_t = n_p / N (Eq. 38)
        observed_window = len(self.buffer)
        persistence = matched_frames / float(observed_window)

        # Validate persistence against operational threshold (Eq. 39)
        if persistence < self.persistence_threshold:
            return None

        # Temporal majority voting for class consistency: c* = Mode(C_t) (Eq. 40)
        majority_class = Counter(matched_classes).most_common(1)[0][0]

        # EMA spatial coordinate filter: b_t = alpha * b_t^curr + (1 - alpha) * b_{t-1} (Eq. 41)
        curr_bbox = [float(v) for v in candidate.bbox]
        if self.prev_smoothed_bbox is not None and len(self.prev_smoothed_bbox) == 4:
            smoothed_bbox = [
                self.alpha * curr + (1.0 - self.alpha) * prev
                for curr, prev in zip(curr_bbox, self.prev_smoothed_bbox)
            ]
        else:
            smoothed_bbox = curr_bbox

        self.prev_smoothed_bbox = smoothed_bbox

        return ValidatedDetection(
            bbox=smoothed_bbox,
            class_name=majority_class,
            confidence=candidate.confidence,
            persistence=persistence
        )

    def reset(self):
        """Reseting temporal FIFO window and smoothed state."""
        self.buffer.clear()
        self.prev_smoothed_bbox = None

    @property
    def current_window_size(self):
        return len(self.buffer)