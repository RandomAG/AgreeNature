"""
Temporal Threat Validation Module (TTVM)

Reference implementation of the TTVM described in the
manuscript.

The module performs:
    1. FIFO temporal buffering
    2. Persistence scoring
    3. Persistence thresholding
    4. Majority voting over predicted weapon classes

The validated detection is represented as:

    V_t = {b_t, c*_t, P_t}

where:
    b_t  = validated bounding-box location
    c*_t = temporally validated weapon category
    P_t  = persistence score
"""

from collections import Counter, deque
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ValidatedDetection:
    """
    Temporally validated weapon detection.

    Attributes
    ----------
    bbox : list
        Bounding-box coordinates for the current validated detection.

    class_name : str
        Majority-voted weapon category.

    persistence : float
        Proportion of frames in the temporal window in which
        the weapon category was detected.
    """

    bbox: List[float]
    class_name: str
    persistence: float


class TemporalThreatValidator:
    """
    FIFO-based Temporal Threat Validation Module.

    Persistence is calculated as:

        P_t = n_p / N

    where:
        n_p = number of frames in which the weapon is detected
        N   = number of frames in the observation window

    A detection is considered temporally valid when:

        P_t >= persistence_threshold
    """

    def __init__(self, window_size: int = 5, persistence_threshold: float = 0.6):
        if window_size <= 0:
            raise ValueError(
                "window_size must be greater than zero."
            )

        if not 0.0 <= persistence_threshold <= 1.0:
            raise ValueError(
                "persistence_threshold must be between 0 and 1."
            )

        self.window_size = window_size
        self.persistence_threshold = persistence_threshold

        # Sliding FIFO observation window.
        self.buffer = deque(maxlen=window_size)

    def update(self, detections):
        """
        Add the detections from the current frame and perform
        temporal validation.

        Parameters
        ----------
        detections : list
            List of Detection objects produced by the detection
            post-processing stage.

        Returns
        -------
        ValidatedDetection or None
            Temporally validated detection when the persistence
            condition is satisfied; otherwise None.
        """

        # Store the current frame's detections.
        self.buffer.append(detections)

        if not detections:
            return None

        # Collect weapon categories observed in the temporal window.
        class_sequence = []

        for frame_detections in self.buffer:
            for detection in frame_detections:
                class_sequence.append(
                    detection.class_name
                )

        if not class_sequence:
            return None

        # Majority voting:
        #
        # c* = Mode(C_t)
        majority_class, _ = Counter(
            class_sequence
        ).most_common(1)[0]

        # Persistence calculation:
        #
        # n_p = number of frames in which the majority weapon
        #       category was detected.
        #
        # N = number of frames in the current temporal window.
        frame_count = len(self.buffer)

        detected_frames = sum(
            1
            for frame_detections in self.buffer
            if any(
                detection.class_name == majority_class
                for detection in frame_detections
            )
        )

        persistence = detected_frames / frame_count

        # Temporal validity condition:
        #
        # P_t >= tau_p
        if persistence < self.persistence_threshold:
            return None

        # Select the current-frame detection corresponding to
        # the temporally validated class. The paper represents
        # the validated location as b_t.
        current_detection = next(
            (
                detection
                for detection in reversed(detections)
                if detection.class_name == majority_class
            ),
            None
        )

        if current_detection is None:
            return None

        return ValidatedDetection(
            bbox=current_detection.bbox,
            class_name=majority_class,
            persistence=persistence
        )

    def reset(self):
        """Clear the temporal FIFO buffer."""

        self.buffer.clear()

    def buffer_size(self) -> int:
        """Return the number of frames currently stored."""

        return len(self.buffer)