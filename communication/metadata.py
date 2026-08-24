"""
Compact threat metadata generation.

Raw image/video data are deliberately excluded from the packet.
"""

import json
from datetime import datetime, timezone


class MetadataGenerator:
    """Generate the metadata-only threat packet."""

    def generate(self, assessment):
        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        return {
            "timestamp": timestamp,
            "weapon_type": assessment.weapon_class,
            "confidence": round(
                assessment.confidence,
                4
            ),
            "persistence": round(
                assessment.persistence,
                4
            ),
            "threat_score": round(
                assessment.threat_score,
                4
            ),
            "threat_level": assessment.threat_level,
            "alert_priority": assessment.alert_priority,
            "bbox": [
                round(float(value), 2)
                for value in assessment.bbox
            ]
        }

    def serialize(self, metadata):
        """Convert metadata to a compact JSON packet."""

        return json.dumps(
            metadata,
            separators=(",", ":")
        )