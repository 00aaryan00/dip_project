"""
Unit tests for src/sharpness.py
"""
import numpy as np
import cv2
from src.sharpness import (
    compute_laplacian,
    calculate_sharpness_score,
    get_laplacian_display,
    analyze_sharpness
)

def test_sharpness():
    print("Testing Sharpness Module...")

    # 1. Create a synthetic sharp image: high-contrast checkerboard pattern
    sharp_img = np.zeros((200, 200), dtype=np.uint8)
    for i in range(0, 200, 20):
        for j in range(0, 200, 20):
            if ((i // 20) + (j // 20)) % 2 == 0:
                sharp_img[i:i+20, j:j+20] = 255

    score_sharp = calculate_sharpness_score(sharp_img)
    print(f"Sharp checkerboard score: {score_sharp:.2f}")
    assert score_sharp > 500, f"Expected high sharpness for checkerboard, got {score_sharp}"

    # 2. Artificially blur the sharp image using a heavy Gaussian Kernel
    blurred_img = cv2.GaussianBlur(sharp_img, (35, 35), sigmaX=12)
    score_blur = calculate_sharpness_score(blurred_img)
    print(f"Blurred checkerboard score: {score_blur:.2f}")
    assert score_blur < score_sharp * 0.01, f"Blurring should drastically reduce score ({score_blur} vs {score_sharp})"

    # 3. Test full analysis pipeline with calibrated threshold
    res_sharp = analyze_sharpness(sharp_img, threshold=100.0)
    assert res_sharp["sharpness_status"] == "SHARP"
    assert res_sharp["laplacian_display"].shape == (200, 200)

    res_blur = analyze_sharpness(blurred_img, threshold=100.0)
    assert res_blur["sharpness_status"] == "BLURRY"
    print("Sharpness tests passed successfully!")

if __name__ == "__main__":
    test_sharpness()
