"""
Input preprocessing for the proposed detection pipeline.

The Raspberry Pi performs frame acquisition and preprocessing
before forwarding the input to the neural accelerator.
"""

import cv2
import numpy as np
import torch


class FramePreprocessor:
    """Preparing RGB video frames for neural-network inference."""

    def __init__(self, image_size=640):
        self.image_size = image_size

    def resize(self, frame):
        """Resizing an input frame to the model resolution."""
        return cv2.resize(
            frame,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

    def normalize(self, frame):
        """Normalizing pixel values from [0, 255] to [0, 1]."""
        return frame.astype(np.float32) / 255.0

    def to_tensor(self, frame):
        """Converting HWC image format to BCHW tensor format."""
        tensor = torch.from_numpy(frame)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)

        return tensor

    def process(self, frame):
        """
        Complete preprocessing pipeline.

        Input:
            BGR frame obtained from OpenCV.

        Output:
            Normalized BCHW tensor.
        """

        frame = self.resize(frame)

        # OpenCV uses BGR; neural models conventionally use RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = self.normalize(frame)
        frame = self.to_tensor(frame)

        return frame


if __name__ == "__main__":
    image = cv2.imread("sample.jpg")

    if image is None:
        raise FileNotFoundError("sample.jpg was not found.")

    processor = FramePreprocessor(image_size=640)
    tensor = processor.process(image)

    print("Input tensor shape:", tuple(tensor.shape))