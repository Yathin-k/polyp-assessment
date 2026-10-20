"""
Dataset preparation pipeline for the Roboflow Kvasir-SEG dataset.

Downloads the dataset from:
    https://universe.roboflow.com/teste-mhypc/kvasir-seg-gouny

Supports multiple Roboflow export formats and converts annotations to
binary segmentation masks (background=0, polyp=1).

IMPORTANT: Never hardcode API keys. The user must supply their Roboflow
API key via environment variable or interactive prompt.
"""

import os
import json
import glob
import shutil
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# Roboflow SDK is required for download — installed via `pip install roboflow`
try:
    import roboflow
    HAS_ROBOFLOW = True
except ImportError:
    HAS_ROBOFLOW = False


# ---------------------------------------------------------------------------
# Roboflow Download
# ---------------------------------------------------------------------------

def get_api_key():
    """
    Retrieve the Roboflow API key from environment variable or user prompt.
    Never hardcodes or logs the key.
    """
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        print("=" * 60)
        print("Roboflow API key not found in environment.")
        print("Set it via: export ROBOFLOW_API_KEY=your_key_here")
        print("Or enter it below (it will NOT be stored).")
        print("=" * 60)
        key = input("Roboflow API Key: ").strip()
    if not key:
        raise ValueError("No API key provided. Cannot download the dataset.")
    return key


def download_dataset(api_key, output_dir="data/raw",
                     workspace="teste-mhypc",
                     project_name="kvasir-seg-gouny",
                     version=4,
                     export_format="png-mask"):
    """
    Download the Kvasir-SEG dataset from Roboflow Universe.

    The 'png-mask' format provides images and their corresponding
    segmentation mask PNGs, which is the most direct format for our pipeline.

    If 'png-mask' is unavailable for the project version, falls back to
    'coco' format and converts annotations to masks.

    Args:
        api_key: Roboflow API key.
        output_dir: Local directory to save the dataset.
        workspace: Roboflow workspace identifier.
        project_name: Roboflow project identifier.
        version: Dataset version number on Roboflow.
        export_format: Desired export format ('png-mask' or 'coco').

    Returns:
        Path to the downloaded dataset directory.
    """
    if not HAS_ROBOFLOW:
        raise ImportError(
            "The 'roboflow' package is required to download the dataset.\n"
            "Install it with: pip install roboflow"
        )

    rf = roboflow.Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)

    print(f"Project: {project_name}")
    print(f"Type: {project.type}")
    print(f"Downloading version {version} in '{export_format}' format...")

    ds_version = project.version(version)

    try:
        dataset = ds_version.download(
            model_format=export_format,
            location=output_dir,
        )
        print(f"Dataset downloaded to: {output_dir}")
        return output_dir
    except Exception as e:
        print(f"Failed to download in '{export_format}' format: {e}")
        if export_format != "coco":
            print("Retrying with 'coco' format...")
            return download_dataset(
                api_key, output_dir, workspace, project_name,
                version, export_format="coco",
            )
        raise


# ---------------------------------------------------------------------------
# COCO JSON → Binary Mask Conversion
# ---------------------------------------------------------------------------

def coco_to_binary_masks(coco_json_path, images_dir, masks_output_dir):
    """
    Convert COCO-format polygon annotations to binary PNG masks.

    Each image gets a single binary mask where:
        - Background pixels = 0
        - Polyp foreground pixels = 255 (saved as uint8; normalized to 1 during loading)

    Args:
        coco_json_path: Path to _annotations.coco.json.
        images_dir: Directory containing the source images.
        masks_output_dir: Directory to save generated mask PNGs.

    Returns:
        Number of masks generated.
    """
    try:
        from pycocotools.coco import COCO
        HAS_PYCOCOTOOLS = True
    except ImportError:
        HAS_PYCOCOTOOLS = False

    os.makedirs(masks_output_dir, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    count = 0

    if HAS_PYCOCOTOOLS:
        # Use official COCO API for accurate polygon rasterization
        coco = COCO(coco_json_path)
        for img_id in coco.getImgIds():
            img_info = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            h, w = img_info["height"], img_info["width"]
            mask = np.zeros((h, w), dtype=np.uint8)
            for ann in anns:
                m = coco.annToMask(ann)
                mask = np.maximum(mask, m)

            mask[mask > 0] = 255
            mask_path = os.path.join(masks_output_dir, img_info["file_name"])
            # Save with same filename as source image
            base = os.path.splitext(mask_path)[0]
            Image.fromarray(mask).save(base + ".png")
            count += 1
    else:
        # Fallback: manual polygon rasterization using PIL
        from PIL import ImageDraw

        # Build lookup structures
        id_to_img = {img["id"]: img for img in coco_data["images"]}
        img_to_anns = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            img_to_anns.setdefault(img_id, []).append(ann)

        for img_id, img_info in id_to_img.items():
            h, w = img_info["height"], img_info["width"]
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)

            anns = img_to_anns.get(img_id, [])
            for ann in anns:
                segmentation = ann.get("segmentation", [])
                for poly in segmentation:
                    if isinstance(poly, list) and len(poly) >= 6:
                        # Convert flat list to list of (x, y) tuples
                        points = [(poly[i], poly[i + 1])
                                  for i in range(0, len(poly), 2)]
                        draw.polygon(points, fill=255)

            base = os.path.splitext(img_info["file_name"])[0]
            mask.save(os.path.join(masks_output_dir, base + ".png"))
            count += 1

    print(f"Generated {count} binary masks in {masks_output_dir}")
    return count


# ---------------------------------------------------------------------------
# Dataset Structure Preparation
# ---------------------------------------------------------------------------

def prepare_dataset(raw_dir="data/raw", processed_dir="data/processed"):
    """
    Organize the downloaded dataset into a standardized structure:

        data/processed/
            images/
                train/
                val/
                test/
            masks/
                train/
                val/
                test/

    Detects the Roboflow export format and converts as needed.

    Args:
        raw_dir: Path to the raw downloaded dataset.
        processed_dir: Path to store the processed dataset.

    Returns:
        Dictionary with dataset statistics.
    """
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)

    # Detect format
    stats = {"format_detected": None, "splits": {}}

    # Check for Roboflow split directories (train, valid, test)
    splits_found = []
    for split_name in ["train", "valid", "test"]:
        split_dir = raw_path / split_name
        if split_dir.exists():
            splits_found.append(split_name)

    if not splits_found:
        # Check if images are directly in raw_dir
        images = list(raw_path.glob("*.jpg")) + list(raw_path.glob("*.png"))
        if images:
            splits_found = ["all"]
        else:
            # Try one level deeper
            for subdir in raw_path.iterdir():
                if subdir.is_dir():
                    imgs = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
                    if imgs:
                        splits_found.append(subdir.name)

    print(f"Found splits: {splits_found}")

    # Process each split
    for split in splits_found:
        split_dir = raw_path / split if split != "all" else raw_path
        # Normalize split name
        out_split = "val" if split == "valid" else split

        # Check for COCO annotations
        coco_file = split_dir / "_annotations.coco.json"
        has_coco = coco_file.exists()

        # Check for mask subdirectory (png-mask format)
        mask_dirs = [d for d in split_dir.iterdir()
                     if d.is_dir() and "mask" in d.name.lower()]

        # Collect images
        img_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        image_files = []
        for ext in img_extensions:
            image_files.extend(split_dir.glob(ext))
        # Exclude mask files if they're in the same directory
        image_files = [f for f in image_files
                       if "_mask" not in f.stem.lower()
                       and "mask" not in f.stem.lower()]

        if has_coco:
            stats["format_detected"] = "coco"
            print(f"  [{split}] COCO format detected — converting to binary masks")
            out_mask_dir = proc_path / "masks" / out_split
            out_img_dir = proc_path / "images" / out_split
            os.makedirs(out_mask_dir, exist_ok=True)
            os.makedirs(out_img_dir, exist_ok=True)

            # Convert COCO annotations to masks
            coco_to_binary_masks(
                str(coco_file),
                str(split_dir),
                str(out_mask_dir),
            )

            # Copy images
            for img_file in image_files:
                shutil.copy2(str(img_file), str(out_img_dir / img_file.name))

        elif mask_dirs:
            stats["format_detected"] = "png-mask"
            print(f"  [{split}] PNG mask format detected")
            out_mask_dir = proc_path / "masks" / out_split
            out_img_dir = proc_path / "images" / out_split
            os.makedirs(out_mask_dir, exist_ok=True)
            os.makedirs(out_img_dir, exist_ok=True)

            # Copy masks
            mask_dir = mask_dirs[0]
            for mask_file in mask_dir.glob("*.png"):
                shutil.copy2(str(mask_file), str(out_mask_dir / mask_file.name))

            # Copy images
            for img_file in image_files:
                shutil.copy2(str(img_file), str(out_img_dir / img_file.name))

        else:
            # Check for paired image-mask naming convention
            # Common pattern: image.jpg and image_mask.png
            stats["format_detected"] = "paired-naming"
            print(f"  [{split}] Attempting paired-naming detection")
            out_mask_dir = proc_path / "masks" / out_split
            out_img_dir = proc_path / "images" / out_split
            os.makedirs(out_mask_dir, exist_ok=True)
            os.makedirs(out_img_dir, exist_ok=True)

            all_files = list(split_dir.glob("*.*"))
            mask_files = [f for f in all_files
                         if "_mask" in f.stem.lower() or "mask" in f.stem.lower()]

            if mask_files:
                for mf in mask_files:
                    shutil.copy2(str(mf), str(out_mask_dir / mf.name))
                for img_f in image_files:
                    if img_f not in mask_files:
                        shutil.copy2(str(img_f), str(out_img_dir / img_f.name))

        # Count results
        out_img_dir = proc_path / "images" / out_split
        out_mask_dir = proc_path / "masks" / out_split
        if out_img_dir.exists() and out_mask_dir.exists():
            n_imgs = len(list(out_img_dir.glob("*.*")))
            n_masks = len(list(out_mask_dir.glob("*.*")))
            stats["splits"][out_split] = {
                "images": n_imgs, "masks": n_masks
            }

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare the Kvasir-SEG dataset from Roboflow"
    )
    parser.add_argument("--output-dir", default="data/raw",
                        help="Directory to download the raw dataset")
    parser.add_argument("--processed-dir", default="data/processed",
                        help="Directory for processed dataset")
    parser.add_argument("--workspace", default="teste-mhypc",
                        help="Roboflow workspace ID")
    parser.add_argument("--project", default="kvasir-seg-gouny",
                        help="Roboflow project ID")
    parser.add_argument("--version", type=int, default=4,
                        help="Dataset version on Roboflow")
    parser.add_argument("--format", default="png-mask",
                        choices=["png-mask", "coco"],
                        help="Export format to request from Roboflow")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download (use existing raw data)")

    args = parser.parse_args()

    if not args.skip_download:
        api_key = get_api_key()
        download_dataset(
            api_key=api_key,
            output_dir=args.output_dir,
            workspace=args.workspace,
            project_name=args.project,
            version=args.version,
            export_format=args.format,
        )

    stats = prepare_dataset(args.output_dir, args.processed_dir)
    print("\n" + "=" * 60)
    print("Dataset Preparation Summary")
    print("=" * 60)
    print(f"Format: {stats['format_detected']}")
    for split, info in stats["splits"].items():
        print(f"  {split}: {info['images']} images, {info['masks']} masks")
    print("=" * 60)


if __name__ == "__main__":
    main()
