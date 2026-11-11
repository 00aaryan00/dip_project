"""
Unit tests for src/quality_analysis.py
"""
import numpy as np
from src.quality_analysis import (
    compute_histogram,
    calculate_brightness,
    calculate_contrast,
    histogram_equalization,
    contrast_stretching,
    analyze_brightness_contrast
)

def test_quality_analysis():
    print("Testing Quality Analysis Module...")

    # 1. Dark test image (mean intensity ~ 30)
    dark_img = np.random.normal(loc=30, scale=10, size=(200, 200)).clip(0, 255).astype(np.uint8)
    b_dark = calculate_brightness(dark_img)
    c_dark = calculate_contrast(dark_img)
    print(f"Dark image: Brightness={b_dark:.2f}, Contrast={c_dark:.2f}")
    assert b_dark < 50, f"Expected dark image brightness < 50, got {b_dark}"

    # 2. Overexposed image (mean intensity ~ 230)
    bright_img = np.random.normal(loc=230, scale=10, size=(200, 200)).clip(0, 255).astype(np.uint8)
    b_bright = calculate_brightness(bright_img)
    print(f"Bright image: Brightness={b_bright:.2f}")
    assert b_bright > 210, f"Expected bright image brightness > 210, got {b_bright}"

    # 3. Low contrast image (narrow range between 120 and 130)
    low_contrast_img = np.random.randint(120, 131, size=(200, 200), dtype=np.uint8)
    c_low = calculate_contrast(low_contrast_img)
    print(f"Low contrast image: Contrast={c_low:.2f}")
    assert c_low < 10, f"Expected contrast < 10, got {c_low}"

    # 4. Contrast stretching test
    stretched = contrast_stretching(low_contrast_img)
    assert stretched.min() < low_contrast_img.min()
    assert stretched.max() > low_contrast_img.max()

    # 5. Full pipeline check
    res = analyze_brightness_contrast(dark_img)
    assert res["brightness_status"] == "LOW"
    assert len(res["original_histogram"]) == 256
    print("Quality analysis tests passed successfully!")

if __name__ == "__main__":
    test_quality_analysis()

