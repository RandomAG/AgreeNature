"""
Threat Assessment Network (TAN)

Deterministic, context-aware threat scoring and risk stratification:

Threat Score:
    R_t = W_c * c_t * P_t

Severity Mapping (W_c):
    Rifle / Assault Rifle -> 1.00
    Pistol / Handgun      -> 0.75
    Knife / Blade         -> 0.50
    Other / Blunt Object  -> 0.25

Operational Thresholds & Priorities:
    R_t >= 0.75          -> Critical (Priority Level 4)
    0.50 <= R_t < 0.75   -> High     (Priority Level 3)
    0.25 <= R_t < 0.50   -> Medium   (Priority Level 2)
    R_t < 0.25           -> Low      (Priority Level 1)
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ThreatAssessmentResult:
    """Standardizing threat intelligence representation"""
    bbox: List[float]
    weapon_class: str
    confidence: float
    persistence: float
    threat_score: float
    threat_level: str
    alert_priority: int


class ThreatAssessment:
    """Lightweight deterministic Threat Assessment Network (TAN)"""

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
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Critical": 4
    }

    def _severity(self, weapon_class: str) -> float:
        """Map validated weapon category to its severity coefficient W_c"""
        normalized = weapon_class.lower().strip()
        return self.SEVERITY_WEIGHTS.get(normalized, self.SEVERITY_WEIGHTS["other"])

    def compute_score(self, weapon_class: str, confidence: float, persistence: float) -> float:
        """Compute deterministic threat score: R_t = W_c * c_t * P_t"""
        c_t = max(0.0, min(1.0, float(confidence)))
        p_t = max(0.0, min(1.0, float(persistence)))
        w_c = self._severity(weapon_class)

        score = w_c * c_t * p_t
        return max(0.0, min(1.0, score))

    def classify(self, score: float) -> str:
        """Stratify score into operational threat levels"""
        if score >= self.THRESHOLDS["critical"]:
            return "Critical"
        if score >= self.THRESHOLDS["high"]:
            return "High"
        if score >= self.THRESHOLDS["medium"]:
            return "Medium"
        return "Low"

    def evaluate(self, bbox: List[float], weapon_class: str, confidence: float, persistence: float) -> ThreatAssessmentResult:
        """Transforming validated detection into structured threat intelligence."""
        score = self.compute_score(weapon_class, confidence, persistence)
        level = self.classify(score)

        return ThreatAssessmentResult(
            bbox=bbox,
            weapon_class=weapon_class,
            confidence=float(confidence),
            persistence=float(persistence),
            threat_score=score,
            threat_level=level,
            alert_priority=self.PRIORITIES[level]
        )