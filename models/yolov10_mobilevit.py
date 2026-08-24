"""
Hybrid YOLOv10-MobileViTv3 feature extraction interface.

The manuscript describes MobileViTv3 blocks incorporated into
intermediate YOLOv10 backbone stages. The resulting local and
global representations are fused before being passed to MSHSA.

This file provides a compact reference representation of that
architectural organization. The deployed neural network is
compiled separately into Hailo Executable Format (HEF).
"""

import torch
import torch.nn as nn

from .mshsa import MSHSA


class LocalFeatureExtractor(nn.Module):
    """
    Lightweight convolutional representation corresponding to
    the local feature extraction component of the hybrid backbone.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class GlobalContextExtractor(nn.Module):
    """
    Reference representation of the contextual component provided
    by the MobileViTv3 stage.

    The manuscript describes transformer-derived contextual
    representations being generated and subsequently fused with
    convolutional representations.
    """

    def __init__(self, channels):
        super().__init__()

        self.projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        # Compact contextual representation.
        context = torch.mean(
            x,
            dim=(2, 3),
            keepdim=True
        )

        return self.projection(context)


class HybridFeatureStage(nn.Module):
    """
    Local/global feature fusion stage.

    Local convolutional features are combined with the contextual
    representation before propagation to the next backbone stage.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.local = LocalFeatureExtractor(
            in_channels,
            out_channels,
            stride=stride
        )

        self.global_context = GlobalContextExtractor(
            out_channels
        )

        self.fusion = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        local_features = self.local(x)

        global_features = self.global_context(
            local_features
        )

        fused = local_features + global_features

        return self.activation(
            self.fusion(fused)
        )


class HybridYOLOv10MobileViT(nn.Module):
    """
    Reference organization of the hybrid YOLOv10-MobileViTv3
    feature extraction pipeline.

    The architecture follows the manuscript's stated sequence:

        YOLOv10-style convolutional feature extraction
            -> MobileViTv3 contextual processing
            -> local/global feature fusion
            -> MSHSA feature refinement
            -> detection head

    The actual YOLOv10 detection head and Hailo deployment model
    are handled separately.
    """

    def __init__(
        self,
        in_channels=3,
        channels=(32, 64, 128, 256)
    ):
        super().__init__()

        self.stage1 = LocalFeatureExtractor(
            in_channels,
            channels[0],
            stride=2
        )

        self.stage2 = HybridFeatureStage(
            channels[0],
            channels[1],
            stride=2
        )

        self.stage3 = HybridFeatureStage(
            channels[1],
            channels[2],
            stride=2
        )

        self.stage4 = HybridFeatureStage(
            channels[2],
            channels[3],
            stride=2
        )

        # MSHSA is applied after hybrid feature extraction
        # and before the detection head.
        self.mshsa = MSHSA(
            channels[3]
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        refined_features = self.mshsa(x)

        return refined_features