"""
Threat metadata generation.

The metadata structure follows the representation M_t described
in the manuscript.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json


class MetadataGenerator:
    """Generate compact threat metadata packets."""

    def generate(self, assessment):
        """
        Convert a ThreatAssessmentResult into a metadata dictionary.
        
        """

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        metadata = {
            "timestamp": timestamp,
            "weapon_category": assessment.weapon_class,
            "threat_score": round(
                assessment.threat_score,
                4
            ),
            "threat_level": assessment.threat_level,
            "confidence": round(
                assessment.confidence,
                4
            ),
            "bounding_box": assessment.bbox,
            "alert_priority": assessment.alert_priority
        }

        return metadata

    def serialize(self, metadata):
        """Serialize metadata for transmission."""

        return json.dumps(
            metadata,
            separators=(",", ":")
        )