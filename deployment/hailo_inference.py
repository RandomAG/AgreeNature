"""
Native Hailo-8L PCIe Runtime Adapter for Raspberry Pi 5.

Directly binds to HailoRT v4.17+ using the official `hailo_platform` library.
Handles HEF loading, VDevice allocation, input stream transformation and output tensor retrieval on the Raspberry Pi 5.
"""

import numpy as np

try:
    from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False


class HailoInference:
    """Production Hailo-8L runtime interface for Raspberry Pi 5."""

    def __init__(self, hef_path: str):
        if not HAILO_AVAILABLE:
            raise ImportError("HailoRT library ('hailo_platform') not found on this system. Ensure HailoRT v4.17+ is installed via 'sudo apt install hailo-all' on Raspberry Pi OS.")
        self.hef_path = hef_path
        self.target = None
        self.network_group = None
        self.network_group_params = None
        self.input_vstream_info = None
        self.output_vstream_info = None
        self.activated_network_group = None
        self.infer_pipeline = None
        self.initialized = False

    def initialize(self):
        """Configure VDevice, parse HEF, and initialize input/output virtual streams."""
        # 1. Load compiled HEF binary
        self.hef = HEF(self.hef_path)
        
        # 2. Allocate VDevice (PCIe interface on Raspberry Pi 5)
        self.target = VDevice()
        
        # 3. Configure Network Group
        configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        # 4. Stream parameters (UINT8 input for quantized Hailo graph, FLOAT32 output)
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=FormatType.UINT8)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

        # 5. Activate network group context on accelerator
        self.activated_network_group = self.network_group.activate(self.network_group_params)
        self.infer_pipeline = InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params)
        self.infer_pipeline.__enter__()
        self.initialized = True

    def infer(self, input_data):
        """
        Execute accelerated tensor forward pass on Hailo-8L NPU.
        
        Parameters
        ----------
        input_data : np.ndarray
            Input image array with shape (1, 640, 640, 3) in UINT8 / RGB format.
        """
        if not self.initialized:
            raise RuntimeError("Hailo-8L inference engine is not initialized.")

        # Ensure array is contiguous and shaped correctly for Hailo input buffer
        if not isinstance(input_data, np.ndarray):
            input_data = input_data.detach().cpu().numpy()
            
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)

        # Feed to Hailo Virtual Streams
        input_name = self.hef.get_input_vstream_infos()[0].name
        raw_outputs = self.infer_pipeline.infer({input_name: input_data})
        
        # Unpack primary output prediction tensor
        output_name = self.hef.get_output_vstream_infos()[0].name
        return raw_outputs[output_name]

    def close(self):
        """Safely deallocate PCIe buffers and shutdown VDevice."""
        if self.initialized:
            if self.infer_pipeline:
                self.infer_pipeline.__exit__(None, None, None)
            if self.activated_network_group:
                self.activated_network_group = None
            if self.target:
                self.target.release()
            self.initialized = False