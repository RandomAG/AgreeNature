"""
Ephemeral Frame Processing Mechanism (EFPM)

Temporary in-memory handling of surveillance frames.

Frames are not persisted by this component and are released
after metadata generation.
"""

import gc


class EphemeralFrameProcessor:

    def __init__(self):
        self._frame = None

    def store(self, frame):
        """Store the current frame only in volatile memory."""

        self._frame = frame

    def get(self):
        """Return the current temporary frame."""

        return self._frame

    def discard(self):
        """Release the current frame from the processing buffer."""

        self._frame = None
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):
        self.discard()