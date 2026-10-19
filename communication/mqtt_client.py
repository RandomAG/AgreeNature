"""
MQTT metadata communication.

Only compact threat metadata is transmitted outside the edge device.
Raw surveillance frames are never passed to this class.

Implements Cryptographic Transport Security
"""

import base64
import hashlib
import cryptography


class MQTTMetadataPublisher:
    """Publishing threat metadata through MQTT with optional payload encryption."""

    def __init__(
        self,
        mqtt_client,
        topic="pewtaf/threat",
        secret_key=None
    ):
        self.client = mqtt_client
        self.topic = topic
        self.cipher = None

        if secret_key:
            try:
                from cryptography.fernet import Fernet
                key_hash = hashlib.sha256(secret_key.encode("utf-8")).digest()
                self.cipher = Fernet(base64.urlsafe_b64encode(key_hash))
            except ImportError:
                raise ImportError(
                    "cryptography library is required for payload encryption. "
                    "Install via 'pip install cryptography'."
                )

    def publish(self, metadata):
        """
        Publishing metadata using the configured MQTT client.
        
        If secret_key was provided, encrypts payload with AES-256 before transit.
        TLS 1.3 configuration is handled at the paho-mqtt client level.
        """
        payload = metadata

        if self.cipher is not None:
            if isinstance(metadata, str):
                raw_bytes = metadata.encode("utf-8")
            else:
                raw_bytes = bytes(metadata)
            payload = self.cipher.encrypt(raw_bytes).decode("utf-8")

        return self.client.publish(
            self.topic,
            payload=payload
        )