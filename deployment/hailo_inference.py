"""
Hailo-8L inference interface.

The manuscript specifies that the trained
YOLOv10-MobileViTv3-MSHSA network is compiled into HEF format
and executed on the Hailo-8L accelerator.

The actual Hailo runtime/pipeline is supplied by the Hailo
software stack. This class provides the project-level interface
between the Raspberry Pi application and that backend.
"""


class HailoInference:
    """
    Project-level interface for Hailo-8L inference.

    The concrete inference implementation is provided by the
    installed Hailo runtime/application pipeline.
    """

    def __init__(self, hef_path, runtime=None):
        self.hef_path = hef_path
        self.runtime = runtime
        self.initialized = False

    def initialize(self):
        """
        Initialize the Hailo inference backend.

        The exact runtime initialization is deployment-specific
        and is delegated to the Hailo software environment.
        """

        if self.runtime is None:
            raise RuntimeError(
                "A Hailo runtime backend must be supplied."
            )

        self.runtime.load_model(self.hef_path)
        self.initialized = True

    def infer(self, input_tensor):
        """Run inference using the configured Hailo backend."""

        if not self.initialized:
            raise RuntimeError(
                "Hailo inference has not been initialized."
            )

        return self.runtime.infer(input_tensor)

    def close(self):
        """Release the Hailo inference backend."""

        if self.runtime is not None:
            self.runtime.close()

        self.initialized = False