"""
Standard U-Net baseline for polyp segmentation.

Implements the classic U-Net encoder–decoder architecture with skip connections
for comparison against ResUNet++.

Reference: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
           Image Segmentation", MICCAI 2015.

This is a standard, unmodified U-Net with:
    - 4-level encoder with double convolutions + max pooling
    - Bottleneck double convolution
    - 4-level decoder with transposed convolution upsampling + skip concatenation
    - 1×1 segmentation head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive 3×3 convolutions, each followed by BN and ReLU."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """Encoder block: MaxPool2d → DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.pool_conv(x)


class DecoderBlock(nn.Module):
    """Decoder block: ConvTranspose2d upsample → Concatenate skip → DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                      kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle spatial dimension mismatches from odd input sizes
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for binary segmentation.

    Architecture:
        Input → Enc0 → Enc1 → Enc2 → Enc3 → Bottleneck
              ↓ skip   ↓ skip  ↓ skip  ↓ skip
              Dec3 ← Dec2 ← Dec1 ← Dec0 → 1×1 Conv → Output

    Args:
        in_channels: Number of input channels (default: 3 for RGB).
        out_channels: Number of output classes (default: 1 for binary).
        features: List of filter counts for each encoder level
                  (default: [64, 128, 256, 512]).
    """

    def __init__(self, in_channels=3, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        # Initial double convolution (no pooling)
        self.inc = DoubleConv(in_channels, features[0])

        # Encoder path
        self.enc1 = EncoderBlock(features[0], features[1])
        self.enc2 = EncoderBlock(features[1], features[2])
        self.enc3 = EncoderBlock(features[2], features[3])

        # Bottleneck
        self.bottleneck = EncoderBlock(features[3], features[3] * 2)

        # Decoder path
        self.dec3 = DecoderBlock(features[3] * 2, features[3])
        self.dec2 = DecoderBlock(features[3], features[2])
        self.dec1 = DecoderBlock(features[2], features[1])
        self.dec0 = DecoderBlock(features[1], features[0])

        # Segmentation head
        self.seg_head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Segmentation logits of shape (B, out_channels, H, W).
        """
        input_size = x.shape[2:]

        # Encoder
        s0 = self.inc(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        # Bottleneck
        b = self.bottleneck(s3)

        # Decoder with skip connections
        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)

        # Segmentation head
        out = self.seg_head(d0)

        # Ensure output matches input spatial dimensions
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear",
                                align_corners=True)

        return out


def get_unet(in_channels=3, out_channels=1, features=None):
    """Factory function to create a U-Net model."""
    return UNet(in_channels=in_channels, out_channels=out_channels, features=features)


if __name__ == "__main__":
    # Quick smoke test
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    assert y.shape == (2, 1, 256, 256), f"Unexpected output shape: {y.shape}"
    assert torch.isfinite(y).all(), "Output contains non-finite values"
    print("U-Net smoke test passed ✓")
