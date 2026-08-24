"""
Multi-Scale Hybrid Spatial Attention (MSHSA)

Reference implementation of the MSHSA module described in the manuscript.

Processing:
    Input feature map
        -> 3x3 / 5x5 / 7x7 depth-wise convolutions
        -> channel-wise concatenation
        -> channel attention
        -> spatial attention
        -> 1x1 projection
        -> residual addition
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel attention using global average pooling followed by
    a lightweight MLP and sigmoid activation.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.mlp(self.pool(x))
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial attention using average-pooling and max-pooling
    along the channel dimension followed by convolution and
    sigmoid activation.
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)

        pooled = torch.cat(
            [avg_map, max_map],
            dim=1
        )

        attention = self.activation(self.conv(pooled))

        return x * attention


class MSHSA(nn.Module):
    """
    Multi-Scale Hybrid Spatial Attention module.

    The input feature representation is processed using three
    parallel depth-wise convolution branches with 3x3, 5x5,
    and 7x7 kernels. The resulting representations are
    concatenated and refined using channel and spatial attention.
    A 1x1 convolution then restores the original channel
    dimensionality, followed by a residual connection.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        # Multi-scale depth-wise feature extraction
        self.dwconv3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )

        self.dwconv5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=False
        )

        self.dwconv7 = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False
        )

        # Concatenation produces 3C channels.
        multi_scale_channels = channels * 3

        # Attention mechanisms operate on the multi-scale
        # representation before channel projection.
        self.channel_attention = ChannelAttention(
            multi_scale_channels,
            reduction=reduction
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=7
        )

        # Restore the original channel dimensionality.
        self.projection = nn.Conv2d(
            multi_scale_channels,
            channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        residual = x

        # Equations (17)-(19):
        f3 = self.dwconv3(x)
        f5 = self.dwconv5(x)
        f7 = self.dwconv7(x)

        # Equation (20):
        multi_scale = torch.cat(
            [f3, f5, f7],
            dim=1
        )

        # Channel refinement:
        refined = self.channel_attention(multi_scale)

        # Spatial refinement:
        refined = self.spatial_attention(refined)

        # Equation (31): restore C channels.
        projected = self.projection(refined)

        # Equation (33): residual refinement.
        return projected + residual