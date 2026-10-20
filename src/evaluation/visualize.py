"""
Visualization utilities for polyp segmentation.

Generates:
    - Sample image-mask-overlay grids
    - Training/validation loss curves
    - Metric comparison charts
    - Prediction visualization with overlays
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image


def denormalize(tensor, mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)):
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor: Normalized image tensor (C, H, W).

    Returns:
        Denormalized numpy array (H, W, C) in [0, 1].
    """
    if isinstance(tensor, torch.Tensor):
        img = tensor.detach().cpu().clone()
    else:
        img = torch.tensor(tensor).clone()

    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]

    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def create_overlay(image, mask, color=(0, 255, 0), alpha=0.4):
    """
    Create a transparent overlay of the mask on the image.

    Args:
        image: RGB image as numpy array (H, W, 3) in [0, 1] or [0, 255].
        mask: Binary mask as numpy array (H, W) with values 0/1.
        color: Overlay color as (R, G, B) tuple.
        alpha: Overlay transparency (0 = fully transparent, 1 = opaque).

    Returns:
        Overlaid image as numpy array (H, W, 3) in [0, 1].
    """
    if image.max() > 1.0:
        image = image / 255.0

    overlay = image.copy()
    mask_bool = mask.squeeze() > 0.5

    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            overlay[:, :, c] * (1 - alpha) + (color[c] / 255.0) * alpha,
            overlay[:, :, c],
        )

    return np.clip(overlay, 0, 1)


def plot_sample_grid(images, masks, n_samples=8, save_path=None,
                     title="Dataset Samples"):
    """
    Plot a grid of images with their ground-truth masks and overlays.

    Args:
        images: List of image tensors or arrays.
        masks: List of mask tensors or arrays.
        n_samples: Number of samples to display.
        save_path: Path to save the figure.
        title: Figure title.
    """
    n = min(n_samples, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img = images[i]
        msk = masks[i]

        # Convert to numpy if needed
        if isinstance(img, torch.Tensor):
            img = denormalize(img)
        if isinstance(msk, torch.Tensor):
            msk = msk.detach().cpu().squeeze().numpy()

        overlay = create_overlay(img, msk)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(msk, cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()


def plot_predictions(images, gt_masks, pred_masks, metrics_list=None,
                     n_samples=6, save_path=None, title="Model Predictions"):
    """
    Plot prediction results: image, ground truth, prediction, overlay.

    Args:
        images: List of image tensors.
        gt_masks: List of ground truth mask tensors.
        pred_masks: List of predicted mask tensors.
        metrics_list: Optional list of metric dicts per image.
        n_samples: Number of samples to show.
        save_path: Path to save figure.
    """
    n = min(n_samples, len(images))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img = images[i]
        gt = gt_masks[i]
        pred = pred_masks[i]

        if isinstance(img, torch.Tensor):
            img = denormalize(img)
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().squeeze().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().squeeze().numpy()

        overlay = create_overlay(img, pred, color=(0, 200, 255))

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="gray")
        pred_title = "Prediction"
        if metrics_list and i < len(metrics_list):
            m = metrics_list[i]
            pred_title += f"\nDice: {m.get('dice', 0):.3f}, IoU: {m.get('iou', 0):.3f}"
        axes[i, 2].set_title(pred_title, fontsize=9)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Overlay")
        axes[i, 3].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()


def plot_training_curves(history, model_name="Model", save_path=None):
    """
    Plot training and validation loss/metric curves.

    Args:
        history: Training history dictionary with keys:
                 train_loss, val_loss, val_dice, val_iou, lr
        save_path: Path to save figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Dice curve
    axes[0, 1].plot(epochs, history["val_dice"], "g-", label="Val Dice")
    best_epoch = np.argmax(history["val_dice"]) + 1
    best_dice = max(history["val_dice"])
    axes[0, 1].axvline(x=best_epoch, color="r", linestyle="--", alpha=0.5,
                        label=f"Best: {best_dice:.4f} (ep {best_epoch})")
    axes[0, 1].set_title("Validation Dice Coefficient")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Dice")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # IoU curve
    axes[1, 0].plot(epochs, history["val_iou"], "m-", label="Val IoU")
    axes[1, 0].set_title("Validation IoU")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("IoU")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(epochs, history["lr"], "k-", label="Learning Rate")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} — Training Curves", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()


def plot_comparison_bar(comparison_data, save_path=None):
    """
    Plot side-by-side bar chart comparing model metrics.

    Args:
        comparison_data: Dict mapping model names to metric dicts.
        save_path: Path to save figure.
    """
    models = list(comparison_data.keys())
    metrics = ["dice", "iou", "precision", "recall", "specificity"]

    x = np.arange(len(metrics))
    width = 0.35
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800"]

    for i, model in enumerate(models):
        values = [comparison_data[model].get(m, 0) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model,
                      color=colors[i % len(colors)], alpha=0.85)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()
