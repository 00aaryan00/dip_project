"""
Unit tests for src/preprocessing.py
"""
import numpy as np
import cv2
import os
from PIL import Image
from src.preprocessing import load_image, to_grayscale, resize_image, normalize_image, preprocess_pipeline

def test_preprocessing():
    print("Testing Preprocessing Module...")
    # 1. Create a dummy synthetic color image (1000x1200x3)
    dummy_rgb = np.zeros((1000, 1200, 3), dtype=np.uint8)
    dummy_rgb[:, :, 0] = 200  # Red
    dummy_rgb[:, :, 1] = 100  # Green
    dummy_rgb[:, :, 2] = 50   # Blue

    # 2. Test load from numpy
    loaded = load_image(dummy_rgb)
    assert loaded.shape == (1000, 1200, 3), f"Expected (1000, 1200, 3), got {loaded.shape}"

    # 3. Test grayscale conversion
    gray = to_grayscale(loaded)
    # ITU-R BT.601 formula: 0.299*200 + 0.587*100 + 0.114*50 = 59.8 + 58.7 + 5.7 = 124.2 -> 124
    expected_val = int(round(0.299 * 200 + 0.587 * 100 + 0.114 * 50))
    actual_val = gray[0, 0]
    print(f"Grayscale value check: Expected ~{expected_val}, Actual: {actual_val}")
    assert abs(actual_val - expected_val) <= 1, f"Grayscale mismatch: {actual_val} vs {expected_val}"

    # 4. Test aspect-ratio preserving resize (max_dimension = 600)
    resized = resize_image(loaded, max_dimension=600)
    # 1200 is max, so new width should be 600, height = 1000 * (600/1200) = 500
    assert resized.shape == (500, 600, 3), f"Expected (500, 600, 3), got {resized.shape}"

    # 5. Test normalization
    norm = normalize_image(gray)
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0 and norm.max() <= 1.0

    # 6. Test full pipeline with PIL Image
    pil_img = Image.fromarray(dummy_rgb)
    res = preprocess_pipeline(pil_img, max_dimension=400)
    assert "original_rgb" in res
    assert "grayscale" in res
    assert res["processed_shape"][1] == 400
    print("Preprocessing tests passed successfully!")

if __name__ == "__main__":
    test_preprocessing()

