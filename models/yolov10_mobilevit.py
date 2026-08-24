"""
Hybrid YOLOv10-MobileViTv3 Architecture

Reference implementation of the hybrid feature extraction and
multi-scale fusion architecture described in the proposed framework.

The final Hailo deployment uses a compiled HEF representation.
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
    """
    Lightweight C2f-style feature aggregation block.

    The block preserves a shortcut representation while processing
    a partial feature stream through convolutional bottlenecks.
    """

    def __init__(
        self,
        channels,
        expansion=0.5,
        repeats=2
    ):
        super().__init__()

        hidden = max(
            int(channels * expansion),
            1
        )

        self.cv1 = ConvBNAct(
            channels,
            hidden * 2,
            kernel_size=1
        )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(
                        hidden,
                        hidden,
                        kernel_size=3
                    ),
                    ConvBNAct(
                        hidden,
                        hidden,
                        kernel_size=3
                    )
                )
                for _ in range(repeats)
            ]
        )

        self.cv2 = ConvBNAct(
            hidden * (2 + repeats),
            channels,
            kernel_size=1
        )

    def forward(self, x):
        parts = list(
            self.cv1(x).chunk(2, dim=1)
        )

        for block in self.blocks:
            parts.append(
                block(parts[-1])
            )

        return self.cv2(
            torch.cat(parts, dim=1)
        )


class MobileViTBlock(nn.Module):
    """
    MobileViTv3-style local-global representation block.

    Local representation:
        3x3 depth-wise convolution

    Global representation:
        Multi-head self-attention

    Followed by:
        Feed-forward network
        Residual fusion
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        expansion=2
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                "channels must be divisible by num_heads."
            )

        hidden = channels * expansion

        # Local representation.
        self.local = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

        # Transformer projection.
        self.norm1 = nn.LayerNorm(channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        # Feed-forward network.
        self.norm2 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(
            nn.Linear(
                channels,
                hidden
            ),
            nn.SiLU(inplace=True),
            nn.Linear(
                hidden,
                channels
            )
        )

        # Feature projection after global processing.
        self.project = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        residual = x

        # Local representation.
        local = self.local(x)

        batch, channels, height, width = local.shape

        # Convert feature map into transformer tokens.
        tokens = local.flatten(
            2
        ).transpose(
            1,
            2
        )

        # Transformer encoder.
        normalized = self.norm1(tokens)

        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False
        )

        tokens = tokens + attended

        # Feed-forward network.
        tokens = tokens + self.ffn(
            self.norm2(tokens)
        )

        # Return to spatial representation.
        global_features = tokens.transpose(
            1,
            2
        ).reshape(
            batch,
            channels,
            height,
            width
        )

        global_features = self.project(
            global_features
        )

        return global_features + residual


class BackboneStage(nn.Module):
    """YOLO-style downsampling stage followed by MobileViTv3."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads=4,
        blocks=1
    ):
        super().__init__()

        self.downsample = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2
        )

        self.mobilevit = nn.Sequential(
            *[
                MobileViTBlock(
                    out_channels,
                    num_heads=num_heads
                )
                for _ in range(blocks)
            ]
        )

    def forward(self, x):
        x = self.downsample(x)
        return self.mobilevit(x)


class FeatureFusion(nn.Module):
    """
    Top-down multi-scale feature fusion.

    Higher-level features are upsampled and concatenated with
    lower-level representations before C2f-style refinement.
    """

    def __init__(
        self,
        high_channels,
        low_channels,
        output_channels
    ):
        super().__init__()

        self.reduce = ConvBNAct(
            high_channels,
            low_channels,
            kernel_size=1
        )

        self.fusion = C2FBlock(
            low_channels * 2,
            repeats=2
        )

        self.output = ConvBNAct(
            low_channels * 2,
            output_channels,
            kernel_size=1
        )

    def forward(self, high, low):
        high = self.reduce(high)

        high = F.interpolate(
            high,
            size=low.shape[-2:],
            mode="nearest"
        )

        fused = torch.cat(
            [high, low],
            dim=1
        )

        fused = self.fusion(fused)

        return self.output(fused)


class HybridYOLOv10MobileViT(nn.Module):
    """
    Hybrid YOLOv10-MobileViTv3 backbone with multi-scale feature
    fusion and MSHSA refinement.

    Returns feature maps at multiple pyramid levels suitable for
    integration with a YOLO-style detection head.
    """

    def __init__(
        self,
        in_channels=3,
        channels=(32, 64, 128, 256, 512)
    ):
        super().__init__()

        c1, c2, c3, c4, c5 = channels

        
        # YOLO-style CNN backbone

        self.stem = ConvBNAct(
            in_channels,
            c1,
            kernel_size=3,
            stride=2
        )

        self.stage1 = BackboneStage(
            c1,
            c2,
            num_heads=4,
            blocks=1
        )

        self.stage2 = BackboneStage(
            c2,
            c3,
            num_heads=4,
            blocks=1
        )

        self.stage3 = BackboneStage(
            c3,
            c4,
            num_heads=8,
            blocks=2
        )

        self.stage4 = BackboneStage(
            c4,
            c5,
            num_heads=8,
            blocks=2
        )

        # MSHSA refinement at the deepest representation
        self.mshsa = MSHSA(c5)

        # Top-down feature fusion
        self.fuse4 = FeatureFusion(
            c5,
            c4,
            c4
        )

        self.fuse3 = FeatureFusion(
            c4,
            c3,
            c3
        )

        self.fuse2 = FeatureFusion(
            c3,
            c2,
            c2
        )

        # Output projections for detection pyramid
        self.p4_projection = ConvBNAct(
            c4,
            c4,
            kernel_size=1
        )

        self.p8_projection = ConvBNAct(
            c3,
            c3,
            kernel_size=1
        )

        self.p16_projection = ConvBNAct(
            c2,
            c2,
            kernel_size=1
        )

        self.p32_projection = ConvBNAct(
            c5,
            c5,
            kernel_size=1
        )

    def forward(self, x):
        # P1/2
        x = self.stem(x)

        # P1/4
        p4 = self.stage1(x)

        # P1/8
        p8 = self.stage2(p4)

        # P1/16
        p16 = self.stage3(p8)

        # P1/32
        p32 = self.stage4(p16)

        # Deep feature refinement.
        p32_refined = self.mshsa(p32)

        # Top-down fusion.
        p16_fused = self.fuse4(
            p32_refined,
            p16
        )

        p8_fused = self.fuse3(
            p16_fused,
            p8
        )

        p4_fused = self.fuse2(
            p8_fused,
            p4
        )

        return {
            "p4": self.p4_projection(p4_fused),
            "p8": self.p8_projection(p8_fused),
            "p16": self.p16_projection(p16_fused),
            "p32": self.p32_projection(p32_refined)
        }