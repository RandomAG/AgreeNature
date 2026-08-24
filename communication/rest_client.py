"""
REST metadata client.

Provides an HTTP interface for transmitting compact threat
metadata to an authorized monitoring service.
"""

import requests


class RESTMetadataClient:
    """REST client for threat metadata."""

    def __init__(
        self,
        endpoint,
        timeout=5
    ):
        self.endpoint = endpoint
        self.timeout = timeout

    def send(self, metadata):
        """
        Send threat metadata to the configured endpoint.

        No raw image or video data are transmitted.
        """

        response = requests.post(
            self.endpoint,
            json=metadata,
            timeout=self.timeout
        )

        response.raise_for_status()

        return response