"""
Reusable training loop for polyp segmentation models.

Supports both ResUNet++ and U-Net training with:
    - Configurable optimizer and learning rate schedule
    - BCEDice loss
    - Validation monitoring and early stopping
    - Best checkpoint saving
    - Training curve logging
    - Reproducible seeds

Designed to be imported by the Colab notebook or run standalone.
"""

import os
import sys
import time
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.resunetpp import ResUNetPlusPlus
from src.models.unet import UNet
from src.training.losses import BCEDiceLoss, DiceLoss
from src.training.metrics import dice_coefficient, iou_score


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU available)")
    return device


def create_model(model_name="resunetpp", in_channels=3, out_channels=1, filters=None):
    """
    Factory function to create a segmentation model.

    Args:
        model_name: "resunetpp" or "unet".
        in_channels: Number of input channels.
        out_channels: Number of output classes.
        filters: Custom filter sizes (optional).

    Returns:
        nn.Module instance.
    """
    if model_name.lower() in ("resunetpp", "resunet++", "resunetplusplus"):
        return ResUNetPlusPlus(
            in_channels=in_channels,
            out_channels=out_channels,
            filters=filters or [32, 64, 128, 256, 512],
        )
    elif model_name.lower() == "unet":
        return UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=filters or [64, 128, 256, 512],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'resunetpp' or 'unet'.")


class Trainer:
    """
    Training manager for segmentation models.

    Handles the complete training workflow: forward/backward pass,
    validation, early stopping, checkpointing, and logging.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device=None,
        lr=1e-4,
        weight_decay=1e-5,
        epochs=100,
        patience=15,
        checkpoint_dir="artifacts/checkpoints",
        model_name="resunetpp",
        scheduler_type="reduce_on_plateau",
    ):
        """
        Args:
            model: PyTorch model (ResUNet++ or UNet).
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            device: torch.device (auto-detected if None).
            lr: Initial learning rate.
            weight_decay: L2 regularization weight.
            epochs: Maximum training epochs.
            patience: Early stopping patience (epochs without improvement).
            checkpoint_dir: Directory to save checkpoints.
            model_name: Name for labeling saved files.
            scheduler_type: "reduce_on_plateau" or "cosine".
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.patience = patience
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = BCEDiceLoss(alpha=0.5, beta=0.5)

        # Optimizer
        self.optimizer = Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        # Learning rate scheduler
        if scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5,
                patience=7, verbose=True,
            )
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-7,
            )
        else:
            self.scheduler = None

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "lr": [],
        }

        # Early stopping state
        self.best_val_dice = 0.0
        self.epochs_without_improvement = 0
        self.best_epoch = 0

    def train_epoch(self):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for images, masks, _ in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self):
        """Run validation and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        dice_scores = []
        iou_scores = []
        n_batches = 0

        for images, masks, _ in self.val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            total_loss += loss.item()
            n_batches += 1

            # Compute per-image metrics
            probs = torch.sigmoid(outputs)
            for i in range(probs.shape[0]):
                dice_scores.append(
                    dice_coefficient(probs[i], masks[i])
                )
                iou_scores.append(
                    iou_score(probs[i], masks[i])
                )

        avg_loss = total_loss / max(n_batches, 1)
        avg_dice = np.mean(dice_scores) if dice_scores else 0.0
        avg_iou = np.mean(iou_scores) if iou_scores else 0.0

        return avg_loss, avg_dice, avg_iou

    def save_checkpoint(self, epoch, val_dice, is_best=False):
        """Save model checkpoint with metadata."""
        checkpoint = {
            "epoch": epoch,
            "model_name": self.model_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_dice": val_dice,
            "history": self.history,
            "config": {
                "model_name": self.model_name,
                "in_channels": 3,
                "out_channels": 1,
                "img_size": [256, 256],
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
                "threshold": 0.5,
            },
        }

        # Save latest checkpoint
        path = self.checkpoint_dir / f"{self.model_name}_latest.pth"
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"  ★ Best model saved: {best_path} (Dice: {val_dice:.4f})")

    def train(self):
        """
        Run the complete training loop.

        Returns:
            Training history dictionary.
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")
        print(f"Device:  {self.device}")
        print(f"Epochs:  {self.epochs}")
        print(f"Patience: {self.patience}")
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Parameters: {n_params:,}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_dice, val_iou = self.validate()

            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_dice)
            elif self.scheduler is not None:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(val_dice)
            self.history["val_iou"].append(val_iou)
            self.history["lr"].append(current_lr)

            # Check for improvement
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_dice, is_best)

            # Print progress
            elapsed = time.time() - epoch_start
            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Dice: {val_dice:.4f} | "
                f"Val IoU: {val_iou:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
                + (" ★" if is_best else "")
            )

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\n⏹ Early stopping at epoch {epoch} "
                      f"(best: epoch {self.best_epoch}, "
                      f"Dice: {self.best_val_dice:.4f})")
                break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 60:.1f} minutes")
        print(f"Best validation Dice: {self.best_val_dice:.4f} "
              f"(epoch {self.best_epoch})")

        # Save history
        history_path = self.checkpoint_dir / f"{self.model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
