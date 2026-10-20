"""
Inference module for polyp segmentation.

Provides a clean API for loading a trained model and running inference
on new colonoscopy images. Used by the Streamlit frontend.

Features:
    - Loads model + config from checkpoint
    - Applies identical preprocessing as training
    - Runs inference in eval mode without gradients
    - Resizes predictions back to original image dimensions
    - Returns binary mask, probability map, overlay, and statistics
    - Caches loaded model for repeated use
    - Handles missing checkpoints and invalid inputs gracefully
"""

import os
import sys
import time
from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.training.train import create_model


class PolypPredictor:
    """
    Inference engine for polyp segmentation.

    Usage:
        predictor = PolypPredictor("artifacts/checkpoints/resunetpp_best.pth")
        result = predictor.predict("path/to/colonoscopy.jpg")
    """

    def __init__(self, checkpoint_path, device=None, threshold=0.5):
        """
        Initialize the predictor by loading a trained checkpoint.

        Args:
            checkpoint_path: Path to the .pth checkpoint file.
            device: Target device (auto-detected if None).
            threshold: Binarization threshold for segmentation.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
            RuntimeError: If checkpoint is corrupted or incompatible.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = threshold

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Please train the model first using the Colab notebook, "
                f"then place the checkpoint at this path.\n"
                f"Expected location: artifacts/checkpoints/resunetpp_best.pth"
            )

        # Set device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Load checkpoint
        try:
            self.checkpoint = torch.load(
                str(self.checkpoint_path),
                map_location=self.device,
                weights_only=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint: {e}\n"
                f"The checkpoint file may be corrupted or incompatible."
            )

        # Extract config
        self.config = self.checkpoint.get("config", {})
        self.img_size = tuple(self.config.get("img_size", [256, 256]))
        self.normalize_mean = self.config.get(
            "normalize_mean", [0.485, 0.456, 0.406]
        )
        self.normalize_std = self.config.get(
            "normalize_std", [0.229, 0.224, 0.225]
        )
        # Use threshold from config if available, otherwise use constructor arg
        if "threshold" in self.config:
            self.threshold = self.config["threshold"]

        # Create and load model
        model_name = self.config.get("model", self.config.get("model_name", "resunetpp"))
        filters = self.config.get("filters", [32, 64, 128, 256, 512])
        self.model = create_model(
            model_name=model_name,
            in_channels=self.config.get("in_channels", 3),
            out_channels=self.config.get("out_channels", 1),
            filters=filters,
        )

        state_dict = self.checkpoint.get("model_state_dict", self.checkpoint)
        # Strip DataParallel prefix if present
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.model_name = model_name
        self._val_dice = self.checkpoint.get("val_dice", None)

        print(f"[OK] Model loaded: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Image size: {self.img_size}")
        print(f"  Threshold: {self.threshold}")
        if self._val_dice is not None:
            print(f"  Best val Dice: {self._val_dice:.4f}")

    def _preprocess(self, image):
        """
        Apply the exact same preprocessing used during training.

        Args:
            image: PIL Image (RGB).

        Returns:
            Preprocessed tensor (1, 3, H, W).
        """
        # Resize with bilinear interpolation
        img = TF.resize(
            image, self.img_size,
            interpolation=T.InterpolationMode.BILINEAR,
        )
        # To tensor [0, 1]
        img = TF.to_tensor(img)
        # Normalize
        img = TF.normalize(img, mean=self.normalize_mean,
                           std=self.normalize_std)
        # Add batch dimension
        return img.unsqueeze(0)

    def _postprocess(self, output, original_size):
        """
        Convert model output to binary mask at original resolution.

        Args:
            output: Model output tensor (1, 1, H, W) — logits or probabilities.
            original_size: Original image size as (W, H).

        Returns:
            prob_map: Probability map numpy array (H, W) in [0, 1].
            binary_mask: Binary mask numpy array (H, W) with values 0/255.
        """
        out_sq = output.detach().squeeze()
        # Only apply sigmoid if values are outside [0, 1] (i.e. logits)
        if out_sq.min() < 0.0 or out_sq.max() > 1.0:
            prob = torch.sigmoid(out_sq).cpu().numpy()
        else:
            prob = out_sq.cpu().numpy()

        # Resize to original dimensions
        prob_pil = Image.fromarray(prob.astype(np.float32))
        prob_pil = prob_pil.resize(original_size, Image.BILINEAR)
        prob_map = np.array(prob_pil)

        # Binarize
        binary_mask = (prob_map > self.threshold).astype(np.uint8) * 255

        return prob_map, binary_mask

    def _create_overlay(self, image_array, mask, color=(0, 255, 100),
                        alpha=0.4):
        """
        Create a colored overlay of the prediction on the original image.

        Args:
            image_array: Original image as numpy array (H, W, 3).
            mask: Binary mask (H, W) with values 0/255.
            color: Overlay color (R, G, B).
            alpha: Transparency.

        Returns:
            Overlay image as numpy array (H, W, 3) uint8.
        """
        overlay = image_array.copy().astype(np.float32)
        mask_bool = mask > 127

        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c],
            )

        return np.clip(overlay, 0, 255).astype(np.uint8)

    @torch.no_grad()
    def predict(self, image_input):
        """
        Run polyp segmentation on an input image.

        Args:
            image_input: File path (str/Path), PIL Image, or numpy array.

        Returns:
            dict with keys:
                - original_image: PIL Image
                - probability_map: numpy array (H, W) float [0, 1]
                - binary_mask: numpy array (H, W) uint8 (0 or 255)
                - overlay: numpy array (H, W, 3) uint8
                - polyp_detected: bool
                - foreground_pixels: int
                - foreground_ratio: float
                - inference_time_ms: float
                - threshold: float
                - model_name: str
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(
                f"Unsupported input type: {type(image_input)}. "
                f"Provide a file path, PIL Image, or numpy array."
            )

        original_size = image.size  # (W, H)
        image_array = np.array(image)

        # Preprocess
        input_tensor = self._preprocess(image).to(self.device)

        # Inference
        start = time.time()
        output = self.model(input_tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        inference_time = (time.time() - start) * 1000  # ms

        # Postprocess
        prob_map, binary_mask = self._postprocess(output, original_size)

        # Create overlay
        overlay = self._create_overlay(image_array, binary_mask)

        # Statistics
        fg_pixels = int((binary_mask > 127).sum())
        total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
        fg_ratio = fg_pixels / total_pixels if total_pixels > 0 else 0.0

        # Detection: require minimum foreground area (configurable)
        # Default: at least 0.1% of image area
        min_area_ratio = 0.001
        polyp_detected = fg_ratio > min_area_ratio

        return {
            "original_image": image,
            "probability_map": prob_map,
            "binary_mask": binary_mask,
            "overlay": overlay,
            "polyp_detected": polyp_detected,
            "foreground_pixels": fg_pixels,
            "foreground_ratio": fg_ratio,
            "inference_time_ms": inference_time,
            "threshold": self.threshold,
            "model_name": self.model_name,
        }


def get_predictor(checkpoint_path="artifacts/checkpoints/resunetpp_best.pth",
                  device=None, threshold=0.5):
    """
    Factory function to get or create a predictor instance.

    Caches the predictor to avoid reloading the model on every call.
    """
    if not hasattr(get_predictor, "_cache"):
        get_predictor._cache = {}

    cache_key = (str(checkpoint_path), str(device), threshold)
    if cache_key not in get_predictor._cache:
        get_predictor._cache[cache_key] = PolypPredictor(
            checkpoint_path, device, threshold,
        )

    return get_predictor._cache[cache_key]
