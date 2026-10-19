"""
Compact threat metadata generation.

Raw image and video data are strictly excluded from the packet to enforce
Edge-Only Visual Processing (EOVP) and data minimization.
"""

import json
from datetime import datetime, timezone


class MetadataGenerator:
    """Generating the privacy-preserving metadata-only threat packet."""

    def generate(self, assessment):
        """
        Converting threat assessment into standardized metadata schema.
        
        Parameters
        ----------
        assessment : ThreatAssessment
            Object containing evaluated threat attributes.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        return {
            "timestamp": timestamp,                                   # T_s
            "class": assessment.weapon_class,                         # c*
            "threat_score": round(float(assessment.threat_score), 4),  # R_t
            "threat_level": assessment.threat_level,                  # L_t
            "confidence": round(float(assessment.confidence), 4),     # c_t
            "persistence": round(float(assessment.persistence), 4),   # P_t
            "alert_priority": assessment.alert_priority,
            "bbox": [
                round(float(coord), 2)
                for coord in assessment.bbox                          # b_t
            ]
        }

    def serialize(self, metadata):
        """Converting metadata to a compact JSON packet."""
        return json.dumps(metadata, separators=(",", ":"))