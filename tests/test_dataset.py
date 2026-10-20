"""
Tests for dataset loading and validation.

Tests:
    - Dataset class initialization with synthetic data
    - Image-mask pair matching
    - Mask binarization
    - Augmentation consistency (spatial sync)
    - DataLoader batching
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import torch
from PIL import Image

from src.data.dataset import PolypDataset, get_dataloaders


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a minimal synthetic dataset for testing."""
    for split in ["train", "val", "test"]:
        img_dir = tmp_path / "images" / split
        mask_dir = tmp_path / "masks" / split
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        for i in range(5):
            # Create random RGB image
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(img).save(img_dir / f"sample_{i:03d}.png")

            # Create binary mask (some foreground)
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[20:40, 20:40] = 255  # Polyp region
            Image.fromarray(mask).save(mask_dir / f"sample_{i:03d}.png")

    return tmp_path


class TestPolypDataset:
    """Tests for the PolypDataset class."""

    def test_initialization(self, synthetic_dataset):
        """Dataset loads correct number of pairs."""
        ds = PolypDataset(
            images_dir=synthetic_dataset / "images" / "train",
            masks_dir=synthetic_dataset / "masks" / "train",
            img_size=(64, 64),
        )
        assert len(ds) == 5

    def test_getitem_shapes(self, synthetic_dataset):
        """Dataset returns correct tensor shapes."""
        ds = PolypDataset(
            images_dir=synthetic_dataset / "images" / "train",
            masks_dir=synthetic_dataset / "masks" / "train",
            img_size=(64, 64),
        )
        img, mask, meta = ds[0]

        assert img.shape == (3, 64, 64), f"Image shape: {img.shape}"
        assert mask.shape == (1, 64, 64), f"Mask shape: {mask.shape}"
        assert "filename" in meta

    def test_mask_binary(self, synthetic_dataset):
        """Mask values are binary (0.0 or 1.0)."""
        ds = PolypDataset(
            images_dir=synthetic_dataset / "images" / "train",
            masks_dir=synthetic_dataset / "masks" / "train",
            img_size=(64, 64),
        )
        _, mask, _ = ds[0]
        unique_values = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique_values), \
            f"Mask has non-binary values: {unique_values}"

    def test_mask_has_foreground(self, synthetic_dataset):
        """Synthetic masks contain foreground pixels."""
        ds = PolypDataset(
            images_dir=synthetic_dataset / "images" / "train",
            masks_dir=synthetic_dataset / "masks" / "train",
            img_size=(64, 64),
        )
        _, mask, _ = ds[0]
        assert mask.sum() > 0, "Mask should have foreground pixels"

    def test_augmentation_runs(self, synthetic_dataset):
        """Augmented dataset doesn't crash."""
        ds = PolypDataset(
            images_dir=synthetic_dataset / "images" / "train",
            masks_dir=synthetic_dataset / "masks" / "train",
            img_size=(64, 64),
            augment=True,
        )
        # Run multiple times to exercise different augmentation paths
        for _ in range(10):
            img, mask, meta = ds[0]
            assert img.shape == (3, 64, 64)
            assert mask.shape == (1, 64, 64)

    def test_augmented_mask_stays_binary(self, synthetic_dataset):
        """Mask remains binary after augmentation."""
        ds = PolypDataset(
            images_dir=synthetic_dataset / "images" / "train",
            masks_dir=synthetic_dataset / "masks" / "train",
            img_size=(64, 64),
            augment=True,
        )
        for _ in range(10):
            _, mask, _ = ds[0]
            unique = torch.unique(mask)
            assert all(v in [0.0, 1.0] for v in unique), \
                f"Augmented mask has non-binary values: {unique}"

    def test_empty_dataset(self, tmp_path):
        """Empty directory results in zero-length dataset."""
        img_dir = tmp_path / "empty_imgs"
        mask_dir = tmp_path / "empty_masks"
        img_dir.mkdir()
        mask_dir.mkdir()

        ds = PolypDataset(
            images_dir=img_dir, masks_dir=mask_dir, img_size=(64, 64),
        )
        assert len(ds) == 0


class TestDataLoaders:
    """Tests for DataLoader creation."""

    def test_get_dataloaders(self, synthetic_dataset):
        """DataLoaders are created for all splits."""
        loaders = get_dataloaders(
            data_dir=str(synthetic_dataset),
            img_size=(64, 64),
            batch_size=2,
            num_workers=0,
        )
        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders

    def test_batch_shapes(self, synthetic_dataset):
        """Batches have correct shapes."""
        loaders = get_dataloaders(
            data_dir=str(synthetic_dataset),
            img_size=(64, 64),
            batch_size=2,
            num_workers=0,
        )
        for images, masks, meta in loaders["train"]:
            assert images.shape[0] <= 2
            assert images.shape[1:] == (3, 64, 64)
            assert masks.shape[1:] == (1, 64, 64)
            break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
