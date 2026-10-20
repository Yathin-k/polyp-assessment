import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predict import PolypPredictor

def test_live():
    checkpoint_path = "artifacts/checkpoints/resunetpp_best.pth"
    print(f"Testing live load of: {checkpoint_path}")
    predictor = PolypPredictor(checkpoint_path=checkpoint_path, device="cpu", threshold=0.5)
    
    # Dummy image
    img = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
    result = predictor.predict(img)
    
    print("\nLive Inference Succeeded!")
    print(f"Polyp Detected: {result['polyp_detected']}")
    print(f"Foreground Ratio: {result['foreground_ratio']*100:.2f}%")
    print(f"Inference Time: {result['inference_time_ms']:.1f} ms")
    print(f"Original Image Size: {result['original_image'].size}")
    print(f"Binary Mask Shape: {result['binary_mask'].shape}")
    print(f"Overlay Shape: {result['overlay'].shape}")
    print("ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_live()
