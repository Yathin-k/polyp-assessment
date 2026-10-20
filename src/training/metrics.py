"""
Segmentation evaluation metrics.

Computes per-image and dataset-level metrics:
    - Dice Coefficient (F1 Score)
    - Intersection over Union (IoU / Jaccard Index)
    - Precision
    - Recall (Sensitivity)
    - Specificity

All metrics operate on binary masks (0/1). Handles edge cases:
    - Empty prediction + empty ground truth → metric = 1.0 (correct)
    - Empty prediction + non-empty ground truth → metric = 0.0
    - Non-empty prediction + empty ground truth → precision = 0.0, recall = 1.0
"""

import numpy as np
import torch


def _to_numpy_binary(tensor, threshold=0.5):
    """Convert tensor to binary numpy array."""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.array(tensor)

    # Flatten to 2D if needed
    arr = arr.squeeze()
    if arr.ndim > 2:
        arr = arr[0]  # Take first channel

    return (arr > threshold).astype(np.float32)


def dice_coefficient(pred, target, threshold=0.5, smooth=1e-7):
    """
    Compute Dice coefficient (F1 Score) for binary segmentation.

    Dice = 2 * TP / (2*TP + FP + FN)

    Returns 1.0 when both pred and target are empty (correct rejection).
    """
    pred = _to_numpy_binary(pred, threshold)
    target = _to_numpy_binary(target, threshold)

    # Handle both-empty case
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0

    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, threshold=0.5, smooth=1e-7):
    """
    Compute Intersection over Union (Jaccard Index).

    IoU = TP / (TP + FP + FN)

    Returns 1.0 when both pred and target are empty.
    """
    pred = _to_numpy_binary(pred, threshold)
    target = _to_numpy_binary(target, threshold)

    if pred.sum() == 0 and target.sum() == 0:
        return 1.0

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def precision_score(pred, target, threshold=0.5, smooth=1e-7):
    """
    Compute Precision = TP / (TP + FP).

    Returns 1.0 when both are empty. Returns 0.0 when pred has foreground
    but target is empty.
    """
    pred = _to_numpy_binary(pred, threshold)
    target = _to_numpy_binary(target, threshold)

    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0:
        return 0.0

    tp = (pred * target).sum()
    return (tp + smooth) / (pred.sum() + smooth)


def recall_score(pred, target, threshold=0.5, smooth=1e-7):
    """
    Compute Recall (Sensitivity) = TP / (TP + FN).

    Returns 1.0 when both are empty. Returns 0.0 when target has foreground
    but pred is empty.
    """
    pred = _to_numpy_binary(pred, threshold)
    target = _to_numpy_binary(target, threshold)

    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if target.sum() == 0:
        return 1.0  # No positives to miss

    tp = (pred * target).sum()
    return (tp + smooth) / (target.sum() + smooth)


def specificity_score(pred, target, threshold=0.5, smooth=1e-7):
    """
    Compute Specificity = TN / (TN + FP).

    Measures the true negative rate (correct background identification).
    """
    pred = _to_numpy_binary(pred, threshold)
    target = _to_numpy_binary(target, threshold)

    neg_pred = 1.0 - pred
    neg_target = 1.0 - target

    tn = (neg_pred * neg_target).sum()
    total_neg = neg_target.sum()

    if total_neg == 0:
        return 1.0  # No negatives in target

    return (tn + smooth) / (total_neg + smooth)


def compute_all_metrics(pred, target, threshold=0.5):
    """
    Compute all segmentation metrics for a single prediction-target pair.

    Args:
        pred: Predicted mask (tensor or numpy array).
        target: Ground truth mask.
        threshold: Binarization threshold.

    Returns:
        Dictionary with all metric values.
    """
    return {
        "dice": dice_coefficient(pred, target, threshold),
        "iou": iou_score(pred, target, threshold),
        "precision": precision_score(pred, target, threshold),
        "recall": recall_score(pred, target, threshold),
        "specificity": specificity_score(pred, target, threshold),
    }


def aggregate_metrics(metric_list, method="mean"):
    """
    Aggregate per-image metrics to dataset-level.

    Args:
        metric_list: List of metric dictionaries (from compute_all_metrics).
        method: Aggregation method ("mean" or "median").

    Returns:
        Dictionary with aggregated metric values and standard deviations.
    """
    if not metric_list:
        return {}

    keys = metric_list[0].keys()
    agg_fn = np.mean if method == "mean" else np.median
    result = {}

    for key in keys:
        values = [m[key] for m in metric_list]
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return result


def format_metrics_table(aggregated, model_name="Model"):
    """Format aggregated metrics as a printable table."""
    lines = [
        f"\n{'='*60}",
        f"  {model_name} — Evaluation Results",
        f"{'='*60}",
        f"  {'Metric':<15} {'Mean':>8} {'±Std':>8} {'Median':>8} "
        f"{'Min':>8} {'Max':>8}",
        f"  {'-'*55}",
    ]

    for key in ["dice", "iou", "precision", "recall", "specificity"]:
        if key in aggregated:
            v = aggregated[key]
            lines.append(
                f"  {key.capitalize():<15} {v['mean']:>8.4f} "
                f"{v['std']:>8.4f} {v['median']:>8.4f} "
                f"{v['min']:>8.4f} {v['max']:>8.4f}"
            )

    lines.append(f"{'='*60}")
    return "\n".join(lines)
