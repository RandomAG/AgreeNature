"""
Threat Assessment Network (TAN)

Reference implementation of the lightweight TAN described in the manuscript.

The threat score is defined as:

    R_t = W_c * c_t * P_t

where:
    W_c = predefined weapon severity coefficient
    c_t = detection confidence
    P_t = temporal persistence

TAN performs only lightweight CPU-side operations:
severity mapping, scalar multiplication, threshold comparison,
and conditional decision rules.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ThreatAssessmentResult:
    """Output of the Threat Assessment Network."""

    bbox: List[float]
    weapon_class: str
    confidence: float
    persistence: float
    threat_score: float
    threat_level: str
    alert_priority: int


class ThreatAssessment:
    """
    Lightweight threat assessment module.

    The numerical severity coefficients and threat thresholds are
    configurable because their exact experimental values are not
    specified in the manuscript.
    """

    def __init__(
        self,
        severity_weights: Dict[str, float],
        thresholds: Dict[str, float],
        priorities: Dict[str, int] = None
    ):
        self.severity_weights = severity_weights
        self.thresholds = thresholds

        self.priorities = priorities or {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4
        }

        required_levels = {
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        }

        if not required_levels.issubset(thresholds):
            raise ValueError(
                "Thresholds must define MEDIUM, HIGH and CRITICAL."
            )

    def compute_score(
        self,
        weapon_class: str,
        confidence: float,
        persistence: float
    ) -> float:
        """Compute R_t = W_c * c_t * P_t."""

        severity = self.severity_weights.get(
            weapon_class,
            0.0
        )

        confidence = max(0.0, min(1.0, confidence))
        persistence = max(0.0, min(1.0, persistence))

        score = severity * confidence * persistence

        return max(0.0, min(1.0, score))

    def classify(self, score: float) -> str:
        """Map a threat score to one of four threat levels."""

        if score >= self.thresholds["CRITICAL"]:
            return "CRITICAL"

        if score >= self.thresholds["HIGH"]:
            return "HIGH"

        if score >= self.thresholds["MEDIUM"]:
            return "MEDIUM"

        return "LOW"

    def evaluate(
        self,
        bbox: List[float],
        weapon_class: str,
        confidence: float,
        persistence: float
    ) -> ThreatAssessmentResult:
        """Generate the final TAN output."""

        score = self.compute_score(
            weapon_class,
            confidence,
            persistence
        )

        level = self.classify(score)

        priority = self.priorities[level]

        return ThreatAssessmentResult(
            bbox=bbox,
            weapon_class=weapon_class,
            confidence=confidence,
            persistence=persistence,
            threat_score=score,
            threat_level=level,
            alert_priority=priority
        )