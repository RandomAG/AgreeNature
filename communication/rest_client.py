"""
REST-based threat metadata communication.

The client transmits metadata only and never accepts raw video
or surveillance frames.
"""

import requests


class RESTMetadataClient:
    """HTTP REST client for threat metadata."""

    def __init__(
        self,
        endpoint,
        timeout=5
    ):
        self.endpoint = endpoint
        self.timeout = timeout

    def send(self, metadata):
        """Send one metadata packet to the configured endpoint."""

        response = requests.post(
            self.endpoint,
            json=metadata,
            timeout=self.timeout
        )

        response.raise_for_status()

        return response