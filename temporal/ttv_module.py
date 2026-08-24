"""
Temporal Threat Validation Module (TTVM)

Temporal validation of frame-level weapon detections.

Pipeline:

    Last N detections
        |
        +--> Persistence computation
        |
        +--> Majority voting
        |
        +--> Validated threat instance

Output:

    V_t = {b_t, c*_t, P_t}
"""

from collections import Counter, deque
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ValidatedDetection:
    """Validated temporal threat instance."""

    bbox: List[float]
    class_name: str
    confidence: float
    persistence: float


class TemporalThreatValidator:
    """
    FIFO-based temporal validation.

    P_t = n_p / N

    where:
        n_p = number of frames containing the candidate class
        N   = number of frames in the temporal window
    """

    def __init__(
        self,
        window_size=5,
        persistence_threshold=0.60
    ):
        if window_size < 1:
            raise ValueError(
                "window_size must be positive."
            )

        if not 0.0 <= persistence_threshold <= 1.0:
            raise ValueError(
                "persistence_threshold must be in [0, 1]."
            )

        self.window_size = window_size
        self.persistence_threshold = (
            persistence_threshold
        )

        self.buffer = deque(
            maxlen=window_size
        )

    def update(self, detections):
        """
        Update the temporal window and validate the current
        detection set.
        """

        self.buffer.append(
            list(detections)
        )

        if not detections:
            return None

        # Build class sequence across the temporal window
        class_sequence = []

        for frame in self.buffer:
            class_sequence.extend(
                detection.class_name
                for detection in frame
            )

        if not class_sequence:
            return None


        # Majority voting
        majority_class, _ = Counter(
            class_sequence
        ).most_common(1)[0]

        # Persistence is calculated per frame, not per detection
        observed_frames = len(
            self.buffer
        )

        confirmed_frames = sum(
            1
            for frame in self.buffer
            if any(
                detection.class_name
                == majority_class
                for detection in frame
            )
        )

        persistence = (
            confirmed_frames
            / observed_frames
        )


        # Temporal validation threshold
        if persistence < self.persistence_threshold:
            return None

        # Select the current-frame instance belonging to the validated majority class
        candidates = [
            detection
            for detection in detections
            if detection.class_name
            == majority_class
        ]

        if not candidates:
            return None

        # Highest-confidence current-frame instance
        selected = max(
            candidates,
            key=lambda item: item.confidence
        )

        return ValidatedDetection(
            bbox=selected.bbox,
            class_name=majority_class,
            confidence=selected.confidence,
            persistence=persistence
        )

    def reset(self):
        """Reset the temporal observation window."""

        self.buffer.clear()

    @property
    def current_window_size(self):
        """Number of frames currently in the FIFO window."""

        return len(self.buffer)