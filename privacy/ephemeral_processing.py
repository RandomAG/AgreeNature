"""
Ephemeral Frame Processing Mechanism (EFPM)

Implements the temporary frame-handling behavior described in
the proposed deployment architecture.

Raw frames are retained only for the duration required for:
    - preprocessing
    - inference
    - temporal validation
    - threat assessment
    - metadata generation

After processing, the frame reference is released.
"""

import gc


class EphemeralFrameProcessor:
    """
    Lightweight volatile-memory frame manager.

    The class intentionally does not provide persistent storage
    operations for surveillance frames.
    """

    def __init__(self):
        self._frame = None

    def store(self, frame):
        """Temporarily retain a frame in memory."""

        self._frame = frame

    def get(self):
        """Return the currently buffered frame."""

        return self._frame

    def discard(self):
        """
        Release the current frame from the processing buffer.

        No frame is written to persistent storage.
        """

        self._frame = None
        gc.collect()

    def has_frame(self):
        """Check whether a frame is currently buffered."""

        return self._frame is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.discard()