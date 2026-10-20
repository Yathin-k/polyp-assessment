"""
Tests for ResUNet++ and U-Net models.

Verifies:
    - Model initialization
    - Forward pass output shape
    - Output finiteness
    - Loss computation and backpropagation
    - Parameter count sanity
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn as nn

from src.models.resunetpp import ResUNetPlusPlus, SqueezeExciteBlock, ASPP, AttentionGate
from src.models.unet import UNet
from src.training.losses import DiceLoss, BCEDiceLoss


# ---------------------------------------------------------------------------
# ResUNet++ Tests
# ---------------------------------------------------------------------------

class TestResUNetPlusPlus:
    """Tests for the ResUNet++ architecture."""

    def test_initialization(self):
        """Model initializes without errors."""
        model = ResUNetPlusPlus(in_channels=3, out_channels=1)
        assert model is not None

    def test_forward_shape(self):
        """Output shape matches input spatial dimensions."""
        model = ResUNetPlusPlus(in_channels=3, out_channels=1)
        x = torch.randn(2, 3, 256, 256)
        y = model(x)
        assert y.shape == (2, 1, 256, 256), f"Got {y.shape}"

    def test_forward_different_sizes(self):
        """Model handles various input sizes."""
        model = ResUNetPlusPlus(in_channels=3, out_channels=1)
        for size in [128, 192, 256, 320]:
            x = torch.randn(1, 3, size, size)
            y = model(x)
            assert y.shape == (1, 1, size, size), \
                f"Size {size}: got {y.shape}"

    def test_output_finite(self):
        """Output values are finite (no NaN or Inf)."""
        model = ResUNetPlusPlus(in_channels=3, out_channels=1)
        x = torch.randn(2, 3, 256, 256)
        y = model(x)
        assert torch.isfinite(y).all(), "Output contains non-finite values"

    def test_backward_pass(self):
        """Loss computation and backpropagation work correctly."""
        model = ResUNetPlusPlus(in_channels=3, out_channels=1)
        x = torch.randn(2, 3, 128, 128)
        target = torch.randint(0, 2, (2, 1, 128, 128)).float()

        y = model(x)
        loss_fn = BCEDiceLoss()
        loss = loss_fn(y, target)

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        loss.backward()

        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "No gradients computed"

    def test_parameter_count(self):
        """Model has reasonable parameter count."""
        model = ResUNetPlusPlus(in_channels=3, out_channels=1)
        n_params = sum(p.numel() for p in model.parameters())
        # ResUNet++ with [32,64,128,256,512] should have ~5-20M params
        assert 1_000_000 < n_params < 50_000_000, \
            f"Unexpected param count: {n_params:,}"

    def test_custom_filters(self):
        """Model works with custom filter configuration."""
        model = ResUNetPlusPlus(
            in_channels=3, out_channels=1,
            filters=[16, 32, 64, 128, 256],
        )
        x = torch.randn(1, 3, 128, 128)
        y = model(x)
        assert y.shape == (1, 1, 128, 128)


# ---------------------------------------------------------------------------
# U-Net Tests
# ---------------------------------------------------------------------------

class TestUNet:
    """Tests for the standard U-Net baseline."""

    def test_initialization(self):
        """U-Net initializes without errors."""
        model = UNet(in_channels=3, out_channels=1)
        assert model is not None

    def test_forward_shape(self):
        """Output shape matches input spatial dimensions."""
        model = UNet(in_channels=3, out_channels=1)
        x = torch.randn(2, 3, 256, 256)
        y = model(x)
        assert y.shape == (2, 1, 256, 256), f"Got {y.shape}"

    def test_output_finite(self):
        """Output values are finite."""
        model = UNet(in_channels=3, out_channels=1)
        x = torch.randn(2, 3, 256, 256)
        y = model(x)
        assert torch.isfinite(y).all()

    def test_backward_pass(self):
        """Loss and backprop work correctly."""
        model = UNet(in_channels=3, out_channels=1)
        x = torch.randn(2, 3, 128, 128)
        target = torch.randint(0, 2, (2, 1, 128, 128)).float()

        y = model(x)
        loss_fn = BCEDiceLoss()
        loss = loss_fn(y, target)
        loss.backward()

        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads


# ---------------------------------------------------------------------------
# Building Block Tests
# ---------------------------------------------------------------------------

class TestBuildingBlocks:
    """Tests for individual ResUNet++ components."""

    def test_squeeze_excite(self):
        """SE block preserves shape."""
        se = SqueezeExciteBlock(64)
        x = torch.randn(2, 64, 32, 32)
        y = se(x)
        assert y.shape == x.shape

    def test_aspp(self):
        """ASPP produces correct output channels."""
        aspp = ASPP(256, 128)
        x = torch.randn(2, 256, 16, 16)
        y = aspp(x)
        assert y.shape == (2, 128, 16, 16)

    def test_attention_gate(self):
        """Attention gate produces correct shape."""
        attn = AttentionGate(F_g=128, F_l=64, F_int=32)
        g = torch.randn(2, 128, 8, 8)
        x = torch.randn(2, 64, 8, 8)
        y = attn(g, x)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Loss Function Tests
# ---------------------------------------------------------------------------

class TestLosses:
    """Tests for loss functions."""

    def test_dice_loss_range(self):
        """Dice loss is in [0, 1]."""
        loss_fn = DiceLoss()
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        assert 0.0 <= loss.item() <= 1.0

    def test_dice_loss_perfect(self):
        """Dice loss is near 0 for perfect prediction."""
        loss_fn = DiceLoss()
        target = torch.ones(1, 1, 64, 64)
        # Very high logits → sigmoid ≈ 1
        pred = torch.ones(1, 1, 64, 64) * 10.0
        loss = loss_fn(pred, target)
        assert loss.item() < 0.05

    def test_bce_dice_loss(self):
        """BCEDice loss computes without errors."""
        loss_fn = BCEDiceLoss()
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)
        loss.backward()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
