"""
Model evaluation on the test set.

Runs the final trained model on the held-out test split and reports:
    - Dice, IoU, Precision, Recall, Specificity (per-image and aggregated)
    - Comparison table for U-Net vs ResUNet++
    - Inference latency statistics

This module is used ONLY after model selection is complete.
The test set must NOT be used for hyperparameter tuning or model selection.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.training.metrics import (
    compute_all_metrics, aggregate_metrics, format_metrics_table,
)
from src.training.train import create_model, get_device


def load_model_from_checkpoint(checkpoint_path, device=None):
    """
    Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pth checkpoint.
        device: Target device.

    Returns:
        (model, config) tuple.
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model_name = config.get("model_name", "resunetpp")
    model = create_model(
        model_name=model_name,
        in_channels=config.get("in_channels", 3),
        out_channels=config.get("out_channels", 1),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_name} from {checkpoint_path}")
    print(f"  Best validation Dice: {checkpoint.get('val_dice', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model, config


@torch.no_grad()
def evaluate_model(model, test_loader, device=None, threshold=0.5,
                   model_name="Model"):
    """
    Evaluate a model on the test set.

    Args:
        model: Trained PyTorch model in eval mode.
        test_loader: DataLoader for the test split.
        device: Computation device.
        threshold: Binarization threshold for predictions.
        model_name: Label for reporting.

    Returns:
        Dictionary with:
            - per_image: List of per-image metric dicts
            - aggregated: Aggregated metric dict
            - latency: Inference time statistics
    """
    if device is None:
        device = get_device()

    model.eval()
    per_image_metrics = []
    latencies = []

    for images, masks, meta in test_loader:
        images = images.to(device)

        # Measure inference time
        start = time.time()
        outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

        probs = torch.sigmoid(outputs).cpu()
        latencies.append(elapsed / images.shape[0])  # Per-image latency

        # Per-image metrics
        for i in range(probs.shape[0]):
            metrics = compute_all_metrics(probs[i], masks[i], threshold)
            metrics["filename"] = meta["filename"][i] if "filename" in meta else f"img_{len(per_image_metrics)}"
            per_image_metrics.append(metrics)

    # Aggregate
    agg = aggregate_metrics(per_image_metrics)
    latency_stats = {
        "mean_ms": float(np.mean(latencies)) * 1000,
        "std_ms": float(np.std(latencies)) * 1000,
        "min_ms": float(np.min(latencies)) * 1000,
        "max_ms": float(np.max(latencies)) * 1000,
    }

    # Print results
    table = format_metrics_table(agg, model_name)
    print(table)
    print(f"\n  Inference latency: {latency_stats['mean_ms']:.1f} ± "
          f"{latency_stats['std_ms']:.1f} ms/image "
          f"(on {device})")

    return {
        "per_image": per_image_metrics,
        "aggregated": agg,
        "latency": latency_stats,
        "device": str(device),
        "threshold": threshold,
        "n_images": len(per_image_metrics),
    }


def compare_models(results_dict, save_path=None):
    """
    Generate a comparison table for multiple models.

    Args:
        results_dict: Dict mapping model names to evaluation results.
        save_path: Optional path to save the comparison as JSON.

    Returns:
        Formatted comparison string.
    """
    lines = [
        f"\n{'='*75}",
        f"  MODEL COMPARISON — Test Set Evaluation",
        f"{'='*75}",
        f"  {'Model':<18} {'Dice':>8} {'IoU':>8} {'Prec':>8} "
        f"{'Recall':>8} {'Spec':>8} {'ms/img':>8}",
        f"  {'-'*70}",
    ]

    comparison_data = {}
    for name, result in results_dict.items():
        agg = result["aggregated"]
        lat = result["latency"]
        lines.append(
            f"  {name:<18} "
            f"{agg['dice']['mean']:>8.4f} "
            f"{agg['iou']['mean']:>8.4f} "
            f"{agg['precision']['mean']:>8.4f} "
            f"{agg['recall']['mean']:>8.4f} "
            f"{agg['specificity']['mean']:>8.4f} "
            f"{lat['mean_ms']:>8.1f}"
        )
        comparison_data[name] = {
            "dice": agg["dice"]["mean"],
            "iou": agg["iou"]["mean"],
            "precision": agg["precision"]["mean"],
            "recall": agg["recall"]["mean"],
            "specificity": agg["specificity"]["mean"],
            "latency_ms": lat["mean_ms"],
        }

    lines.append(f"{'='*75}")
    lines.append(f"  Note: Metrics are mean values over the test set.")
    lines.append(f"  Aggregation: per-image mean.")
    lines.append(f"{'='*75}")

    table_str = "\n".join(lines)
    print(table_str)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(comparison_data, f, indent=2)
        print(f"\nComparison saved to {save_path}")

    return table_str


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data-dir", default="data/processed",
                        help="Path to processed dataset")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="artifacts/results/evaluation.json",
                        help="Path to save evaluation results")

    args = parser.parse_args()

    device = get_device()
    model, config = load_model_from_checkpoint(args.checkpoint, device)

    # Import dataset loader
    from src.data.dataset import get_dataloaders
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    if "test" not in loaders:
        print("⚠ Test split not found. Cannot evaluate.")
        return

    results = evaluate_model(
        model, loaders["test"], device,
        threshold=args.threshold,
        model_name=config.get("model_name", "Model"),
    )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
