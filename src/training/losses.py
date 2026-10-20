"""
Segmentation loss functions.

Implements:
    - DiceLoss: Differentiable Dice coefficient loss
    - BCEDiceLoss: Combined Binary Cross-Entropy + Dice loss (recommended)

All losses operate on logits (pre-sigmoid) for numerical stability.
The sigmoid is applied internally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Differentiable Dice coefficient loss for binary segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    DiceLoss = 1 - Dice

    Uses smooth factor to prevent division by zero when both
    prediction and target are empty.
    """

    def __init__(self, smooth=1.0, from_logits=True):
        """
        Args:
            smooth: Smoothing factor to avoid zero division.
            from_logits: If True, applies sigmoid to input. Set False if
                         input is already probabilities.
        """
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted tensor (B, 1, H, W) — logits or probabilities.
            target: Ground truth tensor (B, 1, H, W) — binary float.

        Returns:
            Scalar Dice loss value.
        """
        if self.from_logits:
            pred = torch.sigmoid(pred)

        # Flatten spatial dimensions
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice loss.

    This is the recommended loss for polyp segmentation:
        L = α * BCE(pred, target) + β * DiceLoss(pred, target)

    BCE handles pixel-level classification while Dice addresses class
    imbalance (polyps are typically a small fraction of the image).

    Operates on logits for numerical stability.
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        """
        Args:
            alpha: Weight for BCE loss.
            beta: Weight for Dice loss.
            smooth: Smoothing factor for Dice computation.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth, from_logits=True)

    def forward(self, pred, target):
        """
        Args:
            pred: Logits tensor (B, 1, H, W).
            target: Ground truth binary mask (B, 1, H, W).

        Returns:
            Weighted sum of BCE and Dice losses.
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.alpha * bce_loss + self.beta * dice_loss
