"""
Reproducible dataset splitting utility.

Creates train/validation/test splits with:
    - Fixed random seed for reproducibility
    - Configurable split ratios (default: 70/15/15)
    - Duplicate detection across splits
    - Split manifest saved for reproduction

The splitter works on the processed dataset (images + masks already paired)
and creates symlinks or copies in the appropriate split directories.
"""

import os
import json
import random
import hashlib
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


def compute_file_hash(filepath):
    """Compute MD5 hash for duplicate detection."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_paired_files(images_dir, masks_dir):
    """
    Collect all valid image-mask pairs from directories.

    Matches images to masks by filename stem (ignoring extension).

    Returns:
        List of (image_path, mask_path) tuples.
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    # Build mask lookup by stem
    mask_lookup = {}
    if masks_dir.exists():
        for mask_file in masks_dir.iterdir():
            if mask_file.suffix.lower() in img_extensions:
                mask_lookup[mask_file.stem.lower()] = mask_file

    pairs = []
    if images_dir.exists():
        for img_file in sorted(images_dir.iterdir()):
            if img_file.suffix.lower() in img_extensions:
                stem = img_file.stem.lower()
                if stem in mask_lookup:
                    pairs.append((img_file, mask_lookup[stem]))

    return pairs


def create_splits(data_dir="data/processed",
                  output_dir="data/processed",
                  train_ratio=0.70,
                  val_ratio=0.15,
                  test_ratio=0.15,
                  seed=42,
                  manifest_path="data/split_manifest.json"):
    """
    Create reproducible train/validation/test splits.

    If the dataset already has Roboflow-provided splits, this merges all
    images and re-splits from scratch (to prevent potential leakage).

    Args:
        data_dir: Root of the processed dataset.
        output_dir: Where to write the split directories.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed for reproducibility.
        manifest_path: Path to save the split assignments.

    Returns:
        Dictionary with split statistics.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Collect ALL pairs from all existing splits
    all_pairs = []

    images_root = data_path / "images"
    masks_root = data_path / "masks"

    if images_root.exists():
        # Check for split subdirectories
        subdirs = [d for d in images_root.iterdir() if d.is_dir()]
        if subdirs:
            for subdir in subdirs:
                split_name = subdir.name
                mask_subdir = masks_root / split_name
                pairs = collect_paired_files(subdir, mask_subdir)
                all_pairs.extend(pairs)
        else:
            # Images directly in images_root
            pairs = collect_paired_files(images_root, masks_root)
            all_pairs.extend(pairs)

    print(f"Total image-mask pairs found: {len(all_pairs)}")
    if len(all_pairs) == 0:
        print("⚠ No pairs found. Cannot create splits.")
        return {"error": "No pairs found"}

    # Deduplicate by image hash
    seen_hashes = {}
    unique_pairs = []
    duplicates = []

    for img_path, mask_path in all_pairs:
        img_hash = compute_file_hash(img_path)
        if img_hash in seen_hashes:
            duplicates.append((str(img_path), seen_hashes[img_hash]))
        else:
            seen_hashes[img_hash] = str(img_path)
            unique_pairs.append((img_path, mask_path))

    if duplicates:
        print(f"⚠ Found {len(duplicates)} duplicate images — removed from splits")
        for dup in duplicates[:5]:
            print(f"  Duplicate: {dup[0]} ↔ {dup[1]}")

    # Shuffle with fixed seed
    random.seed(seed)
    random.shuffle(unique_pairs)

    # Calculate split sizes
    n = len(unique_pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val  # Remainder goes to test

    train_pairs = unique_pairs[:n_train]
    val_pairs = unique_pairs[n_train:n_train + n_val]
    test_pairs = unique_pairs[n_train + n_val:]

    print(f"\nSplit sizes (seed={seed}):")
    print(f"  Train: {len(train_pairs)} ({len(train_pairs)/n*100:.1f}%)")
    print(f"  Val:   {len(val_pairs)} ({len(val_pairs)/n*100:.1f}%)")
    print(f"  Test:  {len(test_pairs)} ({len(test_pairs)/n*100:.1f}%)")

    # Create output directories and copy files
    manifest = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "total_pairs": n,
        "duplicates_removed": len(duplicates),
        "splits": {},
    }

    for split_name, pairs in [("train", train_pairs),
                               ("val", val_pairs),
                               ("test", test_pairs)]:
        img_out = output_path / "images" / split_name
        mask_out = output_path / "masks" / split_name

        # Clear existing split directory
        if img_out.exists():
            shutil.rmtree(img_out)
        if mask_out.exists():
            shutil.rmtree(mask_out)

        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)

        split_files = []
        for img_path, mask_path in pairs:
            # Copy image
            dst_img = img_out / img_path.name
            shutil.copy2(str(img_path), str(dst_img))

            # Copy mask (ensure .png extension)
            mask_name = img_path.stem + ".png"
            dst_mask = mask_out / mask_name
            shutil.copy2(str(mask_path), str(dst_mask))

            split_files.append({
                "image": img_path.name,
                "mask": mask_name,
            })

        manifest["splits"][split_name] = {
            "count": len(pairs),
            "files": split_files,
        }

    # Save manifest
    manifest_file = Path(manifest_path)
    os.makedirs(manifest_file.parent, exist_ok=True)
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSplit manifest saved to: {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Create reproducible train/val/test splits"
    )
    parser.add_argument("--data-dir", default="data/processed",
                        help="Path to processed dataset")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Output directory for splits")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", default="data/split_manifest.json",
                        help="Path to save split manifest")

    args = parser.parse_args()

    create_splits(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
