"""
Unit tests for src/object_analysis.py
"""
import numpy as np
from src.object_analysis import (
    find_primary_contour,
    calculate_centroid,
    calculate_alignment_score,
    check_boundary_proximity,
    calculate_background_uniformity,
    analyze_object_and_background
)

def test_object_analysis():
    print("Testing Object Analysis Module...")

    # 1. Canvas 400x400 with perfectly centered 100x100 product mask
    canvas_shape = (400, 400, 3)
    rgb_img = np.full(canvas_shape, 250, dtype=np.uint8)
    gray_img = np.full((400, 400), 250, dtype=np.uint8)
    mask = np.zeros((400, 400), dtype=np.uint8)

    # Box from y=150..250, x=150..250
    mask[150:250, 150:250] = 255
    rgb_img[150:250, 150:250] = [30, 40, 50]
    gray_img[150:250, 150:250] = 40

    res = analyze_object_and_background(rgb_img, gray_img, mask)
    print(f"Centroid: {res['centroid']}, Alignment score: {res['alignment_score']}")
    # Geometric center of 400x400 is (200, 200). Bounding box center is exactly (200, 200).
    assert abs(res["centroid"][0] - 200) <= 1
    assert abs(res["centroid"][1] - 200) <= 1
    assert res["alignment_score"] > 0.98, "Centered box must have high alignment score"
    assert not res["is_cropped_warning"], "Centered box should not trigger crop warning"
    assert res["background_uniformity_score"] > 0.90, "Solid background must have high uniformity"

    # 2. Test edge-touching product (cropping defect)
    mask_cropped = np.zeros((400, 400), dtype=np.uint8)
    mask_cropped[0:80, 0:80] = 255  # Touches top and left boundaries
    res_crop = analyze_object_and_background(rgb_img, gray_img, mask_cropped)
    print(f"Cropped product warning: {res_crop['is_cropped_warning']}, Margin: {res_crop['boundary_margin_ratio']}")
    assert res_crop["is_cropped_warning"], "Border-touching box must trigger crop warning"

    print("Object analysis tests passed successfully!")

if __name__ == "__main__":
    test_object_analysis()

