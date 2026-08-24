"""
MQTT metadata communication.

Only compact threat metadata is transmitted outside the edge
device. Raw surveillance frames are never passed to this class.
"""


class MQTTMetadataPublisher:
    """Publish threat metadata through MQTT."""

    def __init__(
        self,
        mqtt_client,
        topic="pewtaf/threat"
    ):
        self.client = mqtt_client
        self.topic = topic

    def publish(self, metadata):
        """
        Publish metadata using the configured MQTT client.

        TLS configuration, authentication, certificates and broker
        details remain deployment-specific.
        """

        return self.client.publish(
            self.topic,
            payload=metadata
        )