"""
Unit tests for src/morphology.py and src/segmentation.py
"""
import numpy as np
import cv2
from src.morphology import clean_segmentation_mask, apply_opening, apply_closing
from src.segmentation import segment_product, segment_otsu, assess_segmentation_reliability

def test_morphology_and_segmentation():
    print("Testing Morphology & Segmentation Modules...")

    # 1. Create synthetic binary mask (200x200):
    # A centered 80x80 box (foreground = 255) with a 6x6 hole inside (0) and 3 stray 2x2 specks in background (255)
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[60:140, 60:140] = 255
    # Interior hole
    mask[95:105, 95:105] = 0
    # Background specks (salt noise)
    mask[10:13, 10:13] = 255
    mask[180:183, 20:23] = 255

    # Test Opening removes specks
    opened = apply_opening(mask, ksize=5)
    assert opened[10:13, 10:13].sum() == 0, "Opening must eliminate background specks"

    # Test Closing fills hole
    closed = apply_closing(opened, ksize=13)
    assert closed[100, 100] == 255, "Closing must fill the interior hole"

    # 2. Test segmentation on synthetic product image
    # White background (240) with a dark grey product (50) centered in middle
    canvas = np.full((300, 300, 3), 245, dtype=np.uint8)
    canvas[75:225, 75:225] = [50, 60, 70]
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)

    seg_res = segment_product(canvas, gray, method="otsu")
    print(f"Segmentation reliability: {seg_res['reliability_status']}, Area ratio: {seg_res['area_ratio']:.4f}")
    assert seg_res["reliability_status"] == "RELIABLE"
    # Object is 150x150 = 22500 pixels out of 300x300 = 90000 pixels -> Exactly 25% (0.25)
    expected_ratio = (150 * 150) / (300 * 300)
    assert abs(seg_res["area_ratio"] - expected_ratio) < 0.02

    # Verify foreground extraction shape
    assert seg_res["foreground_rgb"].shape == (300, 300, 3)
    print("Morphology and Segmentation tests passed successfully!")

if __name__ == "__main__":
    test_morphology_and_segmentation()

