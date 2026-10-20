"""
ResUNet++: An Advanced Architecture for Medical Image Segmentation

Faithful PyTorch implementation based on:
    Jha et al., "ResUNet++: An Advanced Architecture for Medical Image Segmentation"
    arXiv: 1911.07067, 2019

Architecture components:
    1. Stem Block with residual connection
    2. Encoders (×3) — ResidualConv + Squeeze-and-Excitation (SE)
    3. ASPP Bridge — Multi-scale feature extraction at bottleneck
    4. Decoders (×3) — Attention Gates (AG) + Upsample + ResidualConv
    5. Output ASPP — Multi-scale context aggregation
    6. Segmentation Head — 1×1 Conv with Sigmoid probability output

Default filter progression: [32, 64, 128, 256, 512]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class SqueezeAndExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise feature recalibration.
    Hu et al., CVPR 2018.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


# Alias for backward compatibility
SqueezeExciteBlock = SqueezeAndExcitation


class ResidualConvSE(nn.Module):
    """
    Residual convolution block with Squeeze-and-Excitation.
    Structure: BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv3x3 + SE + Skip.
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )
        self.se = SqueezeAndExcitation(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        return self.shortcut(x) + self.se(self.conv_block(x))


# Alias for backward compatibility
ResidualConv = ResidualConvSE


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    Parallel dilated convolutions with sampling rates [1, 6, 12, 18].
    """

    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch,
                    kernel_size=1 if r == 1 else 3,
                    padding=0 if r == 1 else r,
                    dilation=r,
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for r in rates
        ])
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * len(rates), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        return self.project(torch.cat(outs, dim=1))


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections.
    Filters encoder features (x) using the gating signal (g) from the decoder.
    Oktay et al., 2018.
    """

    def __init__(self, g_ch=None, x_ch=None, inter_ch=None, F_g=None, F_l=None, F_int=None):
        super().__init__()
        # Support both (g_ch, x_ch, inter_ch) and (F_g, F_l, F_int) parameter naming
        g_dim = g_ch if g_ch is not None else F_g
        x_dim = x_ch if x_ch is not None else F_l
        int_dim = inter_ch if inter_ch is not None else F_int

        self.W_g = nn.Sequential(
            nn.Conv2d(g_dim, int_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(int_dim),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_dim, int_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(int_dim),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(int_dim, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)


# ---------------------------------------------------------------------------
# ResUNet++ Full Architecture
# ---------------------------------------------------------------------------

class ResUNetPlusPlus(nn.Module):
    """
    ResUNet++ — Full hybrid architecture for colorectal polyp segmentation.
    Matches the trained Colab checkpoint weights state_dict.
    """

    def __init__(self, in_channels=3, out_channels=1, filters=None):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128, 256, 512]
        assert len(filters) == 5, "ResUNet++ requires exactly 5 filter levels"
        f = filters

        # Stem Block
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, f[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(f[0], f[0], kernel_size=3, padding=1, bias=False),
        )
        self.stem_sc = nn.Conv2d(in_channels, f[0], kernel_size=1, bias=False)

        # Encoder Path
        self.enc1 = ResidualConvSE(f[0], f[1], stride=2)
        self.enc2 = ResidualConvSE(f[1], f[2], stride=2)
        self.enc3 = ResidualConvSE(f[2], f[3], stride=2)

        # ASPP Bridge
        self.bridge_aspp = ASPP(f[3], f[4])

        # Decoder 3
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag3 = AttentionGate(g_ch=f[4], x_ch=f[2], inter_ch=f[2] // 2)
        self.dec3 = ResidualConvSE(f[4] + f[2], f[3])

        # Decoder 2
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag2 = AttentionGate(g_ch=f[3], x_ch=f[1], inter_ch=f[1] // 2)
        self.dec2 = ResidualConvSE(f[3] + f[1], f[2])

        # Decoder 1
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag1 = AttentionGate(g_ch=f[2], x_ch=f[0], inter_ch=f[0] // 2)
        self.dec1 = ResidualConvSE(f[2] + f[0], f[1])

        # Output ASPP & Segmentation Head
        self.out_aspp = ASPP(f[1], f[0])
        self.head = nn.Conv2d(f[0], out_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]

        # Stem
        x0 = self.stem(x) + self.stem_sc(x)  # [B, 32, H, W]

        # Encoders
        x1 = self.enc1(x0)                   # [B, 64, H/2, W/2]
        x2 = self.enc2(x1)                   # [B, 128, H/4, W/4]
        x3 = self.enc3(x2)                   # [B, 256, H/8, W/8]

        # Bridge
        b = self.bridge_aspp(x3)             # [B, 512, H/8, W/8]

        # Decoder 3
        u3 = self.up3(b)                     # [B, 512, H/4, W/4]
        if u3.shape[2:] != x2.shape[2:]:
            u3 = F.interpolate(u3, size=x2.shape[2:], mode="bilinear", align_corners=True)
        s2 = self.ag3(g=u3, x=x2)            # [B, 128, H/4, W/4]
        d3 = self.dec3(torch.cat([u3, s2], dim=1))

        # Decoder 2
        u2 = self.up2(d3)                    # [B, 256, H/2, W/2]
        if u2.shape[2:] != x1.shape[2:]:
            u2 = F.interpolate(u2, size=x1.shape[2:], mode="bilinear", align_corners=True)
        s1 = self.ag2(g=u2, x=x1)            # [B, 64, H/2, W/2]
        d2 = self.dec2(torch.cat([u2, s1], dim=1))

        # Decoder 1
        u1 = self.up1(d2)                    # [B, 128, H, W]
        if u1.shape[2:] != x0.shape[2:]:
            u1 = F.interpolate(u1, size=x0.shape[2:], mode="bilinear", align_corners=True)
        s0 = self.ag1(g=u1, x=x0)            # [B, 32, H, W]
        d1 = self.dec1(torch.cat([u1, s0], dim=1))

        # Output ASPP and Head
        oa = self.out_aspp(d1)               # [B, 32, H, W]
        out = torch.sigmoid(self.head(oa))   # [B, 1, H, W]

        # Ensure exact match with input spatial dimensions
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=True)

        return out


def get_resunetpp(in_channels=3, out_channels=1, filters=None):
    """Factory function to create a ResUNet++ model."""
    return ResUNetPlusPlus(
        in_channels=in_channels,
        out_channels=out_channels,
        filters=filters,
    )
