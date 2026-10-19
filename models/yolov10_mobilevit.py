"""
Hybrid YOLOv10-MobileViTv3 Architecture

Reference implementation of the hybrid feature extraction and
multi-scale fusion architecture described in manuscript
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mshsa import MSHSA


class ConvBNAct(nn.Module):
    """Standard convolution + batch normalization + SiLU."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class C2FBlock(nn.Module):
    """Lightweight C2f-style feature aggregation block"""

    def __init__(self, channels, expansion=0.5, repeats=2):
        super().__init__()
        hidden = max(int(channels * expansion), 1)
        self.cv1 = ConvBNAct(channels, hidden * 2, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(hidden, hidden, kernel_size=3),
                    ConvBNAct(hidden, hidden, kernel_size=3)
                )
                for _ in range(repeats)
            ]
        )
        self.cv2 = ConvBNAct(hidden * (2 + repeats), channels, kernel_size=1)

    def forward(self, x):
        parts = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.cv2(torch.cat(parts, dim=1))


class MobileViTBlock(nn.Module):
    """MobileViTv3-style local-global representation block"""

    def __init__(self, channels, num_heads=4, expansion=2):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads.")

        hidden = channels * expansion

        # Local representation
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

        # Transformer encoder
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels)
        )

        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        local = self.local(x)
        batch, channels, height, width = local.shape

        tokens = local.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm2(tokens))

        global_features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        global_features = self.project(global_features)

        return global_features + residual


class BackboneStage(nn.Module):
    """YOLO-style downsampling stage followed by MobileViTv3"""

    def __init__(self, in_channels, out_channels, num_heads=4, blocks=1):
        super().__init__()
        self.downsample = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=2)
        self.mobilevit = nn.Sequential(
            *[MobileViTBlock(out_channels, num_heads=num_heads) for _ in range(blocks)]
        )

    def forward(self, x):
        x = self.downsample(x)
        return self.mobilevit(x)


class FeatureFusion(nn.Module):
    """Top-down multi-scale feature fusion block"""

    def __init__(self, high_channels, low_channels, output_channels):
        super().__init__()
        self.reduce = ConvBNAct(high_channels, low_channels, kernel_size=1)
        self.fusion = C2FBlock(low_channels * 2, repeats=2)
        self.output = ConvBNAct(low_channels * 2, output_channels, kernel_size=1)

    def forward(self, high, low):
        high = self.reduce(high)
        high = F.interpolate(high, size=low.shape[-2:], mode="nearest")
        fused = torch.cat([high, low], dim=1)
        fused = self.fusion(fused)
        return self.output(fused)


class HybridYOLOv10MobileViT(nn.Module):
    """
    Hybrid YOLOv10-MobileViTv3 backbone with MSHSA refinement and multi-scale fusion
    """

    def __init__(
        self,
        in_channels=3,
        channels=(32, 64, 128, 256, 512)
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = channels

        # Stem: P2 (stride 2, H/2 x W/2)
        self.stem = ConvBNAct(in_channels, c1, kernel_size=3, stride=2)

        # Stage 1: P4 (stride 4, H/4 x W/4)
        self.stage1 = BackboneStage(c1, c2, num_heads=4, blocks=1)

        # Stage 2: P8 (stride 8, H/8 x W/8)
        self.stage2 = BackboneStage(c2, c3, num_heads=4, blocks=1)

        # Stage 3: P16 (stride 16, H/16 x W/16)
        self.stage3 = BackboneStage(c3, c4, num_heads=8, blocks=2)

        # Stage 4: P32 (stride 32, H/32 x W/32)
        self.stage4 = BackboneStage(c4, c5, num_heads=8, blocks=2)

        # Deepest feature attention refinement
        self.mshsa = MSHSA(c5)

        # Top-down feature fusion pathways
        self.fuse_p16 = FeatureFusion(c5, c4, c4)
        self.fuse_p8 = FeatureFusion(c4, c3, c3)
        self.fuse_p4 = FeatureFusion(c3, c2, c2)

        # Multi-scale detection head projections
        self.p4_proj = ConvBNAct(c2, c2, kernel_size=1)
        self.p8_proj = ConvBNAct(c3, c3, kernel_size=1)
        self.p16_proj = ConvBNAct(c4, c4, kernel_size=1)
        self.p32_proj = ConvBNAct(c5, c5, kernel_size=1)

    def forward(self, x):
        # Forward backbone
        x_stem = self.stem(x)         # P2
        p4 = self.stage1(x_stem)      # P4: (B, c2, H/4, W/4)
        p8 = self.stage2(p4)          # P8: (B, c3, H/8, W/8)
        p16 = self.stage3(p8)         # P16: (B, c4, H/16, W/16)
        p32 = self.stage4(p16)        # P32: (B, c5, H/32, W/32)

        # MSHSA attention refinement on deepest feature
        p32_refined = self.mshsa(p32)

        # Top-down aggregation
        p16_fused = self.fuse_p16(p32_refined, p16)
        p8_fused = self.fuse_p8(p16_fused, p8)
        p4_fused = self.fuse_p4(p8_fused, p4)

        return {
            "p4": self.p4_proj(p4_fused),
            "p8": self.p8_proj(p8_fused),
            "p16": self.p16_proj(p16_fused),
            "p32": self.p32_proj(p32_refined)
        }