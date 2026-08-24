"""
MQTT metadata publisher.

Only compact threat metadata is transmitted.
Raw surveillance frames are never passed to this client.
"""

import json


class MQTTMetadataPublisher:
    """
    Lightweight MQTT publisher for threat metadata.

    The actual MQTT connection is injected through the client
    object so that deployment-specific configuration remains
    outside the framework logic.
    """

    def __init__(
        self,
        client,
        topic="pewtaf/threats"
    ):
        self.client = client
        self.topic = topic

    def publish(self, metadata):
        """Publish a metadata dictionary."""

        payload = json.dumps(metadata)

        return self.client.publish(
            self.topic,
            payload
        )