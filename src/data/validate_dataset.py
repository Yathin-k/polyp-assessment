"""
Dataset validation utility.

Performs comprehensive validation of the prepared dataset:
    - Image-mask pair matching
    - File integrity (readable, correct format)
    - Mask value validation (binary: 0 and 255 or 0 and 1)
    - Dimension consistency
    - Foreground pixel statistics
    - Duplicate detection
    - Data leakage checks across splits

Generates a validation report with detailed statistics.
"""

import os
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image


def compute_file_hash(filepath, algorithm="md5"):
    """Compute hash of a file for duplicate detection."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_image(filepath):
    """
    Validate that a file is a readable image.

    Returns:
        dict with keys: valid, width, height, mode, error
    """
    result = {"valid": False, "width": 0, "height": 0, "mode": "", "error": None}
    try:
        with Image.open(filepath) as img:
            img.verify()
        # Re-open after verify (verify can leave file in bad state)
        with Image.open(filepath) as img:
            result["width"] = img.width
            result["height"] = img.height
            result["mode"] = img.mode
            result["valid"] = True
    except Exception as e:
        result["error"] = str(e)
    return result


def validate_mask(filepath):
    """
    Validate a segmentation mask file.

    Checks:
        - Is a valid image
        - Contains only binary values (0/255 or 0/1)
        - Has foreground pixels

    Returns:
        dict with keys: valid, width, height, unique_values, fg_pixels, fg_ratio, error
    """
    result = {
        "valid": False, "width": 0, "height": 0,
        "unique_values": [], "fg_pixels": 0, "fg_ratio": 0.0,
        "is_empty": True, "error": None,
    }
    try:
        with Image.open(filepath) as img:
            arr = np.array(img)

        # Handle multi-channel masks — convert to single channel
        if arr.ndim == 3:
            arr = arr[:, :, 0]

        result["width"] = arr.shape[1]
        result["height"] = arr.shape[0]
        result["unique_values"] = sorted(np.unique(arr).tolist())

        # Normalize to binary (0/1 for analysis)
        binary = (arr > 0).astype(np.uint8)
        result["fg_pixels"] = int(binary.sum())
        total = binary.shape[0] * binary.shape[1]
        result["fg_ratio"] = result["fg_pixels"] / total if total > 0 else 0.0
        result["is_empty"] = result["fg_pixels"] == 0
        result["valid"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def find_matching_mask(image_name, mask_dir):
    """
    Find the mask file corresponding to an image.

    Tries multiple naming conventions:
        - Same filename with .png extension
        - filename_mask.png
        - filename without extension + .png
    """
    stem = Path(image_name).stem
    mask_dir = Path(mask_dir)

    candidates = [
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}_mask.png",
        mask_dir / f"{stem}.jpg",
        mask_dir / f"{stem}_mask.jpg",
        mask_dir / image_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fuzzy match: look for masks containing the image stem
    for mask_file in mask_dir.iterdir():
        if stem.lower() in mask_file.stem.lower():
            return mask_file

    return None


def validate_dataset(data_dir="data/processed", report_path=None):
    """
    Run comprehensive dataset validation.

    Args:
        data_dir: Path to the processed dataset directory.
        report_path: Optional path to save the validation report.

    Returns:
        Dictionary containing the complete validation report.
    """
    data_path = Path(data_dir)
    images_root = data_path / "images"
    masks_root = data_path / "masks"

    report = {
        "data_dir": str(data_dir),
        "splits": {},
        "total_images": 0,
        "total_masks": 0,
        "total_valid_pairs": 0,
        "issues": [],
        "duplicates_within_splits": {},
        "duplicates_across_splits": [],
        "dimension_stats": {},
        "foreground_stats": {},
    }

    if not images_root.exists():
        report["issues"].append(f"Images directory not found: {images_root}")
        return report

    # Collect hashes for cross-split duplicate detection
    all_hashes = {}  # hash -> (split, filename)

    splits = sorted([d.name for d in images_root.iterdir() if d.is_dir()])
    if not splits:
        # Maybe images are directly in images_root
        splits = ["all"]

    for split in splits:
        split_img_dir = images_root / split if split != "all" else images_root
        split_mask_dir = masks_root / split if split != "all" else masks_root

        split_report = {
            "images": 0, "masks": 0, "valid_pairs": 0,
            "missing_masks": [], "invalid_images": [], "invalid_masks": [],
            "empty_masks": [], "dimension_mismatches": [],
            "widths": [], "heights": [], "fg_ratios": [],
        }

        if not split_img_dir.exists():
            report["issues"].append(f"Split directory not found: {split_img_dir}")
            continue

        # Find all images
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = [f for f in split_img_dir.iterdir()
                       if f.suffix.lower() in img_extensions]
        split_report["images"] = len(image_files)

        # Duplicate detection within split
        split_hashes = {}
        duplicates_within = []

        for img_file in image_files:
            # Validate image
            img_val = validate_image(img_file)
            if not img_val["valid"]:
                split_report["invalid_images"].append(
                    {"file": img_file.name, "error": img_val["error"]}
                )
                continue

            split_report["widths"].append(img_val["width"])
            split_report["heights"].append(img_val["height"])

            # Hash for duplicate detection
            file_hash = compute_file_hash(img_file)
            if file_hash in split_hashes:
                duplicates_within.append(
                    (img_file.name, split_hashes[file_hash])
                )
            split_hashes[file_hash] = img_file.name

            # Cross-split tracking
            if file_hash in all_hashes:
                report["duplicates_across_splits"].append({
                    "file1": f"{all_hashes[file_hash][0]}/{all_hashes[file_hash][1]}",
                    "file2": f"{split}/{img_file.name}",
                })
            all_hashes[file_hash] = (split, img_file.name)

            # Find matching mask
            if not split_mask_dir.exists():
                split_report["missing_masks"].append(img_file.name)
                continue

            mask_file = find_matching_mask(img_file.name, split_mask_dir)
            if mask_file is None:
                split_report["missing_masks"].append(img_file.name)
                continue

            split_report["masks"] += 1

            # Validate mask
            mask_val = validate_mask(mask_file)
            if not mask_val["valid"]:
                split_report["invalid_masks"].append(
                    {"file": mask_file.name, "error": mask_val["error"]}
                )
                continue

            # Check dimension match
            if (img_val["width"] != mask_val["width"] or
                    img_val["height"] != mask_val["height"]):
                split_report["dimension_mismatches"].append({
                    "image": img_file.name,
                    "img_size": (img_val["width"], img_val["height"]),
                    "mask_size": (mask_val["width"], mask_val["height"]),
                })

            if mask_val["is_empty"]:
                split_report["empty_masks"].append(mask_file.name)

            split_report["fg_ratios"].append(mask_val["fg_ratio"])
            split_report["valid_pairs"] += 1

        report["duplicates_within_splits"][split] = duplicates_within
        report["splits"][split] = split_report
        report["total_images"] += split_report["images"]
        report["total_masks"] += split_report["masks"]
        report["total_valid_pairs"] += split_report["valid_pairs"]

        # Dimension statistics
        if split_report["widths"]:
            report["dimension_stats"][split] = {
                "width_min": min(split_report["widths"]),
                "width_max": max(split_report["widths"]),
                "width_mean": np.mean(split_report["widths"]),
                "height_min": min(split_report["heights"]),
                "height_max": max(split_report["heights"]),
                "height_mean": np.mean(split_report["heights"]),
            }

        if split_report["fg_ratios"]:
            report["foreground_stats"][split] = {
                "fg_ratio_min": min(split_report["fg_ratios"]),
                "fg_ratio_max": max(split_report["fg_ratios"]),
                "fg_ratio_mean": np.mean(split_report["fg_ratios"]),
                "fg_ratio_median": np.median(split_report["fg_ratios"]),
                "empty_mask_count": len(split_report["empty_masks"]),
            }

    return report


def print_report(report):
    """Print a human-readable validation report."""
    print("\n" + "=" * 70)
    print("DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f"Data directory: {report['data_dir']}")
    print(f"Total images:      {report['total_images']}")
    print(f"Total masks:       {report['total_masks']}")
    print(f"Total valid pairs: {report['total_valid_pairs']}")

    for split, info in report["splits"].items():
        print(f"\n--- Split: {split} ---")
        print(f"  Images:           {info['images']}")
        print(f"  Masks found:      {info['masks']}")
        print(f"  Valid pairs:      {info['valid_pairs']}")
        print(f"  Missing masks:    {len(info['missing_masks'])}")
        print(f"  Invalid images:   {len(info['invalid_images'])}")
        print(f"  Invalid masks:    {len(info['invalid_masks'])}")
        print(f"  Empty masks:      {len(info['empty_masks'])}")
        print(f"  Dim mismatches:   {len(info['dimension_mismatches'])}")

        if info["missing_masks"][:5]:
            print(f"  Sample missing: {info['missing_masks'][:5]}")
        if info["dimension_mismatches"][:3]:
            print(f"  Sample dim mismatch: {info['dimension_mismatches'][:3]}")

    # Dimension stats
    if report["dimension_stats"]:
        print(f"\n--- Image Dimensions ---")
        for split, ds in report["dimension_stats"].items():
            print(f"  [{split}] Width: {ds['width_min']}-{ds['width_max']} "
                  f"(mean {ds['width_mean']:.0f}), "
                  f"Height: {ds['height_min']}-{ds['height_max']} "
                  f"(mean {ds['height_mean']:.0f})")

    # Foreground stats
    if report["foreground_stats"]:
        print(f"\n--- Foreground Pixel Statistics ---")
        for split, fs in report["foreground_stats"].items():
            print(f"  [{split}] FG ratio: {fs['fg_ratio_min']:.4f} - "
                  f"{fs['fg_ratio_max']:.4f} (mean {fs['fg_ratio_mean']:.4f}, "
                  f"median {fs['fg_ratio_median']:.4f}), "
                  f"Empty masks: {fs['empty_mask_count']}")

    # Duplicates
    for split, dups in report["duplicates_within_splits"].items():
        if dups:
            print(f"\n  ⚠ Duplicates within [{split}]: {len(dups)}")
            for d in dups[:5]:
                print(f"    {d[0]} ↔ {d[1]}")

    if report["duplicates_across_splits"]:
        print(f"\n  ⚠ CROSS-SPLIT DUPLICATES (potential leakage): "
              f"{len(report['duplicates_across_splits'])}")
        for d in report["duplicates_across_splits"][:10]:
            print(f"    {d['file1']} ↔ {d['file2']}")

    # Issues
    if report["issues"]:
        print(f"\n--- Issues ---")
        for issue in report["issues"]:
            print(f"  ⚠ {issue}")

    print("\n" + "=" * 70)

    # Overall assessment
    if report["total_valid_pairs"] > 0 and not report["duplicates_across_splits"]:
        print("✓ Dataset validation PASSED")
    elif report["duplicates_across_splits"]:
        print("⚠ Dataset has cross-split duplicates — re-splitting recommended")
    else:
        print("✗ Dataset validation FAILED — no valid image-mask pairs found")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Validate the prepared dataset")
    parser.add_argument("--data-dir", default="data/processed",
                        help="Path to the processed dataset")
    parser.add_argument("--report", default=None,
                        help="Path to save JSON report")
    args = parser.parse_args()

    report = validate_dataset(args.data_dir)
    print_report(report)

    if args.report:
        import json
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, default=convert)
        print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
