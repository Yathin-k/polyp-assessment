"""
Tests for the inference pipeline.

Tests:
    - Predictor initialization with checkpoint
    - Preprocessing / postprocessing correctness
    - Prediction output structure
    - Missing checkpoint error handling
    - Invalid input handling
    - CPU inference
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import torch
from PIL import Image

from src.models.resunetpp import ResUNetPlusPlus
from src.inference.predict import PolypPredictor


@pytest.fixture
def dummy_checkpoint(tmp_path):
    """Create a dummy checkpoint for testing."""
    model = ResUNetPlusPlus(in_channels=3, out_channels=1)

    checkpoint = {
        "epoch": 1,
        "model_name": "resunetpp",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "val_dice": 0.0,
        "history": {},
        "config": {
            "model_name": "resunetpp",
            "in_channels": 3,
            "out_channels": 1,
            "img_size": [256, 256],
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "threshold": 0.5,
        },
    }

    ckpt_path = tmp_path / "test_checkpoint.pth"
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


@pytest.fixture
def sample_image():
    """Create a sample RGB image."""
    return Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    )


class TestPolypPredictor:
    """Tests for the inference predictor."""

    def test_initialization(self, dummy_checkpoint):
        """Predictor loads checkpoint successfully."""
        predictor = PolypPredictor(
            checkpoint_path=str(dummy_checkpoint),
            device="cpu",
        )
        assert predictor.model_name == "resunetpp"
        assert predictor.threshold == 0.5

    def test_missing_checkpoint(self):
        """Missing checkpoint raises clear error."""
        with pytest.raises(FileNotFoundError) as exc_info:
            PolypPredictor("nonexistent_checkpoint.pth")
        assert "not found" in str(exc_info.value).lower()
        assert "train" in str(exc_info.value).lower()

    def test_predict_pil_image(self, dummy_checkpoint, sample_image):
        """Prediction works with PIL Image input."""
        predictor = PolypPredictor(
            str(dummy_checkpoint), device="cpu",
        )
        result = predictor.predict(sample_image)

        # Verify result structure
        assert "original_image" in result
        assert "binary_mask" in result
        assert "overlay" in result
        assert "probability_map" in result
        assert "polyp_detected" in result
        assert "foreground_pixels" in result
        assert "foreground_ratio" in result
        assert "inference_time_ms" in result
        assert "threshold" in result
        assert "model_name" in result

    def test_predict_output_shapes(self, dummy_checkpoint, sample_image):
        """Prediction outputs have correct shapes."""
        predictor = PolypPredictor(
            str(dummy_checkpoint), device="cpu",
        )
        result = predictor.predict(sample_image)

        # Binary mask should match original image dimensions
        assert result["binary_mask"].shape == (256, 256), \
            f"Mask shape: {result['binary_mask'].shape}"
        # Overlay should be RGB
        assert result["overlay"].shape == (256, 256, 3)
        # Probability map should be 2D float
        assert result["probability_map"].shape == (256, 256)

    def test_predict_mask_binary(self, dummy_checkpoint, sample_image):
        """Predicted mask contains only 0 and 255."""
        predictor = PolypPredictor(
            str(dummy_checkpoint), device="cpu",
        )
        result = predictor.predict(sample_image)
        unique = np.unique(result["binary_mask"])
        assert all(v in [0, 255] for v in unique), \
            f"Mask has unexpected values: {unique}"

    def test_predict_numpy_input(self, dummy_checkpoint):
        """Prediction works with numpy array input."""
        predictor = PolypPredictor(
            str(dummy_checkpoint), device="cpu",
        )
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = predictor.predict(img)
        assert result["binary_mask"].shape == (128, 128)

    def test_predict_different_sizes(self, dummy_checkpoint):
        """Prediction handles various image sizes."""
        predictor = PolypPredictor(
            str(dummy_checkpoint), device="cpu",
        )
        for size in [(100, 100), (512, 384), (320, 240)]:
            img = Image.fromarray(
                np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            )
            result = predictor.predict(img)
            assert result["binary_mask"].shape == (size[1], size[0])

    def test_inference_time_positive(self, dummy_checkpoint, sample_image):
        """Inference time is recorded and positive."""
        predictor = PolypPredictor(
            str(dummy_checkpoint), device="cpu",
        )
        result = predictor.predict(sample_image)
        assert result["inference_time_ms"] > 0

    def test_foreground_ratio_range(self, dummy_checkpoint, sample_image):
        """Foreground ratio is between 0 and 1."""
        predictor = PolypPredictor(
            str(dummy_checkpoint), device="cpu",
        )
        result = predictor.predict(sample_image)
        assert 0.0 <= result["foreground_ratio"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
