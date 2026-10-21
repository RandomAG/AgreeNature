"""
Multi-Scale Hybrid Spatial Attention (MSHSA)

Reference implementation of the MSHSA module described in the proposed framework.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel attention operating on the concatenated 3C feature map.

    M_c = sigmoid(MLP(GAP(F_cat)))
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.mlp(self.pool(x))
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial attention based on channel-wise average and maximum pooling followed by a 7x7 convolution.
    """

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.sigmoid(self.conv(pooled))
        return x * attention


class MSHSA(nn.Module):
    """
    Multi-Scale Hybrid Spatial Attention module.

    Input: C x H x W

    Multi-scale branches:
        3x3 depth-wise convolution
        5x5 depth-wise convolution
        7x7 depth-wise convolution

    Concatenation: 3C x H x W

    Attention refinement:
        Channel attention
        Spatial attention

    Projection:
        1x1 convolution
        Batch normalization
        SiLU activation

    Output: C x H x W
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.channels = channels

        # Multi-scale depth-wise convolution branches
        self.dwconv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.dwconv5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.dwconv7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)

        # Attention operates on the concatenated 3C representation
        multi_scale_channels = channels * 3
        self.channel_attention = ChannelAttention(multi_scale_channels, reduction=reduction)
        self.spatial_attention = SpatialAttention()
       
        # Feature refinement
        # 3C -> C
        
        self.refinement = nn.Sequential(nn.Conv2d(multi_scale_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        residual = x

        # Multi-scale depth-wise feature extraction.
        f3 = self.dwconv3(x)
        f5 = self.dwconv5(x)
        f7 = self.dwconv7(x)

        # Concatenate multi-scale representations.
        f_cat = torch.cat([f3, f5, f7], dim=1)

        # Channel attention.
        f_channel = self.channel_attention(f_cat)

        # Spatial attention.
        f_spatial = self.spatial_attention(f_channel)

        # Point-wise projection + normalization + activation.
        f_refined = self.refinement(f_spatial)

        # Residual connection.
        return f_refined + residual