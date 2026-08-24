"""
Hailo-8L inference adapter.

The trained Hybrid YOLOv10-MobileViTv3-MSHSA model is deployed
through a Hailo Executable Format (HEF) model.

The Hailo runtime/application pipeline is treated as the hardware
inference backend. This adapter isolates the rest of PEWTAF from
Hailo-specific runtime details.
"""


class HailoInference:
    """
    Interface between PEWTAF and the Hailo inference backend.

    Parameters
    ----------
    hef_path : str
        Path to the compiled HEF model.

    runtime : object
        Hailo runtime/pipeline object exposing:
            load_model()
            infer()
            close()
    """

    def __init__(
        self,
        hef_path,
        runtime
    ):
        self.hef_path = hef_path
        self.runtime = runtime
        self.initialized = False

    def initialize(self):
        """Load and initialize the HEF model."""

        self.runtime.load_model(
            self.hef_path
        )

        self.initialized = True

    def infer(self, input_tensor):
        """Execute inference on the Hailo accelerator."""

        if not self.initialized:
            raise RuntimeError(
                "Hailo backend has not been initialized."
            )

        return self.runtime.infer(
            input_tensor
        )

    def close(self):
        """Release the Hailo backend."""

        if self.initialized:
            self.runtime.close()

        self.initialized = False