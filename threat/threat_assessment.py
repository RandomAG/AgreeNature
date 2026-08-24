"""
Threat Assessment Network (TAN)

Context-aware threat scoring and risk stratification for validated
weapon detections.

Threat score:

    R_t = W_c * c_t * P_t

Severity mapping:

    Rifle / Assault Rifle -> 1.00
    Pistol / Handgun      -> 0.75
    Knife / Blade         -> 0.50
    Other / Blunt Object  -> 0.25

Risk thresholds:

    R < 0.25          -> Low
    0.25 <= R < 0.50  -> Medium
    0.50 <= R < 0.75  -> High
    R >= 0.75         -> Critical
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ThreatAssessmentResult:
    """Final threat intelligence produced by TAN."""

    bbox: List[float]
    weapon_class: str
    confidence: float
    persistence: float
    threat_score: float
    threat_level: str
    alert_priority: int


class ThreatAssessment:
    """Lightweight CPU-side Threat Assessment Network."""

    SEVERITY_WEIGHTS = {
        "rifle": 1.00,
        "assault rifle": 1.00,
        "pistol": 0.75,
        "handgun": 0.75,
        "knife": 0.50,
        "blade": 0.50,
        "other": 0.25,
        "blunt object": 0.25
    }

    THRESHOLDS = {
        "medium": 0.25,
        "high": 0.50,
        "critical": 0.75
    }

    PRIORITIES = {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4
    }

    def _severity(self, weapon_class):
        """
        Map a validated weapon class to its severity coefficient.
        """

        normalized = weapon_class.lower().strip()

        return self.SEVERITY_WEIGHTS.get(
            normalized,
            self.SEVERITY_WEIGHTS["other"]
        )

    def compute_score(
        self,
        weapon_class,
        confidence,
        persistence
    ):
        """
        Compute:

            R_t = W_c * c_t * P_t
        """

        confidence = max(
            0.0,
            min(1.0, float(confidence))
        )

        persistence = max(
            0.0,
            min(1.0, float(persistence))
        )

        severity = self._severity(
            weapon_class
        )

        score = (
            severity
            * confidence
            * persistence
        )

        return max(
            0.0,
            min(1.0, score)
        )

    def classify(self, score):
        """Assign the four-level threat category."""

        if score >= self.THRESHOLDS["critical"]:
            return "CRITICAL"

        if score >= self.THRESHOLDS["high"]:
            return "HIGH"

        if score >= self.THRESHOLDS["medium"]:
            return "MEDIUM"

        return "LOW"

    def evaluate(
        self,
        bbox,
        weapon_class,
        confidence,
        persistence
    ):
        """Generate contextual threat intelligence."""

        score = self.compute_score(
            weapon_class,
            confidence,
            persistence
        )

        level = self.classify(
            score
        )

        return ThreatAssessmentResult(
            bbox=bbox,
            weapon_class=weapon_class,
            confidence=float(confidence),
            persistence=float(persistence),
            threat_score=score,
            threat_level=level,
            alert_priority=self.PRIORITIES[level]
        )