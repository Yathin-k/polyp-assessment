"""
PyTorch Dataset class for polyp segmentation.

Loads image-mask pairs from the processed dataset structure:
    data/processed/images/{split}/
    data/processed/masks/{split}/

Applies synchronized augmentations (same spatial transforms to both
image and mask) using torchvision transforms or albumentations.

Mask interpolation always uses nearest-neighbor to preserve discrete labels.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class PolypDataset(Dataset):
    """
    PyTorch Dataset for polyp segmentation.

    Loads paired images and binary masks, applies synchronized augmentations.

    Args:
        images_dir: Path to directory containing images.
        masks_dir: Path to directory containing mask PNGs.
        img_size: Target (H, W) for resizing. Default: (256, 256).
        augment: Whether to apply training augmentations.
        normalize_mean: Per-channel mean for normalization.
        normalize_std: Per-channel std for normalization.
    """

    # ImageNet normalization (widely used default for pretrained backbones)
    DEFAULT_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        images_dir,
        masks_dir,
        img_size=(256, 256),
        augment=False,
        normalize_mean=None,
        normalize_std=None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.augment = augment
        self.mean = normalize_mean or self.DEFAULT_MEAN
        self.std = normalize_std or self.DEFAULT_STD

        # Collect paired files by matching stems
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in img_extensions
        ])

        mask_stems = set()
        if self.masks_dir.exists():
            mask_stems = {
                f.stem.lower()
                for f in self.masks_dir.iterdir()
                if f.suffix.lower() in img_extensions
            }

        self.pairs = []
        for img_file in image_files:
            if img_file.stem.lower() in mask_stems:
                # Find the actual mask file
                mask_file = self._find_mask(img_file.stem)
                if mask_file:
                    self.pairs.append((img_file, mask_file))

        if len(self.pairs) == 0:
            print(f"⚠ No image-mask pairs found in {images_dir}")

        # Normalization transform (applied only to image, not mask)
        self.normalize = T.Normalize(mean=self.mean, std=self.std)

    def _find_mask(self, stem):
        """Find mask file matching image stem."""
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = self.masks_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
            # Case-insensitive match
            for f in self.masks_dir.iterdir():
                if f.stem.lower() == stem.lower():
                    return f
        return None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (3, H, W), normalized.
            mask: Tensor of shape (1, H, W), binary float (0.0 or 1.0).
            meta: Dict with original filename and dimensions.
        """
        img_path, mask_path = self.pairs[idx]

        # Load image (RGB) and mask (grayscale)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        orig_size = image.size  # (W, H)

        # Synchronized augmentations
        if self.augment:
            image, mask = self._augment(image, mask)

        # Resize — bilinear for image, nearest for mask
        image = TF.resize(image, self.img_size,
                          interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.img_size,
                         interpolation=T.InterpolationMode.NEAREST)

        # Convert to tensors
        image = TF.to_tensor(image)  # [0, 1] float, shape (3, H, W)
        mask = TF.to_tensor(mask)    # [0, 1] float, shape (1, H, W)

        # Binarize mask (threshold at 0.5 to handle any interpolation artifacts)
        mask = (mask > 0.5).float()

        # Normalize image
        image = self.normalize(image)

        meta = {
            "filename": img_path.name,
            "original_size": orig_size,  # (W, H)
        }

        return image, mask, meta

    def _augment(self, image, mask):
        """
        Apply synchronized spatial augmentations.

        All spatial transforms are applied identically to both image and mask.
        Color transforms are applied only to the image.
        """
        # Random horizontal flip (p=0.5)
        if torch.rand(1).item() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip (p=0.5)
        if torch.rand(1).item() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation (±30 degrees)
        if torch.rand(1).item() > 0.5:
            angle = torch.randint(-30, 31, (1,)).item()
            image = TF.rotate(image, angle,
                              interpolation=T.InterpolationMode.BILINEAR,
                              fill=0)
            mask = TF.rotate(mask, angle,
                             interpolation=T.InterpolationMode.NEAREST,
                             fill=0)

        # Random affine (slight translation + scale)
        if torch.rand(1).item() > 0.5:
            angle = 0
            translate = [int(image.size[0] * 0.05), int(image.size[1] * 0.05)]
            scale = 0.9 + torch.rand(1).item() * 0.2  # [0.9, 1.1]
            shear = 0
            image = TF.affine(image, angle, translate, scale, shear,
                              interpolation=T.InterpolationMode.BILINEAR,
                              fill=0)
            mask = TF.affine(mask, angle, translate, scale, shear,
                             interpolation=T.InterpolationMode.NEAREST,
                             fill=0)

        # Color jitter (image only) — subtle for medical images
        if torch.rand(1).item() > 0.5:
            jitter = T.ColorJitter(
                brightness=0.15, contrast=0.15,
                saturation=0.1, hue=0.05,
            )
            image = jitter(image)

        return image, mask


def get_dataloaders(data_dir="data/processed",
                    img_size=(256, 256),
                    batch_size=8,
                    num_workers=2,
                    pin_memory=True,
                    normalize_mean=None,
                    normalize_std=None):
    """
    Create DataLoaders for train, val, and test splits.

    Args:
        data_dir: Root processed data directory.
        img_size: Target image size (H, W).
        batch_size: Batch size for DataLoaders.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory (for GPU transfer).

    Returns:
        Dictionary of DataLoaders: {"train": ..., "val": ..., "test": ...}
    """
    data_path = Path(data_dir)
    loaders = {}

    for split in ["train", "val", "test"]:
        img_dir = data_path / "images" / split
        mask_dir = data_path / "masks" / split

        if not img_dir.exists():
            print(f"⚠ Split '{split}' not found at {img_dir}")
            continue

        dataset = PolypDataset(
            images_dir=img_dir,
            masks_dir=mask_dir,
            img_size=img_size,
            augment=(split == "train"),
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

        loaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
        )

        print(f"  {split}: {len(dataset)} samples, "
              f"{len(loaders[split])} batches")

    return loaders
