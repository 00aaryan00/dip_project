"""
Unit tests for src/feature_extraction.py
"""
import numpy as np
from src.feature_extraction import extract_features_from_image, FEATURE_COLUMNS

def test_feature_extraction():
    print("Testing Feature Extraction Module...")

    # Create synthetic RGB product image: white background with centered blue product
    canvas = np.full((300, 300, 3), 245, dtype=np.uint8)
    canvas[80:220, 80:220] = [30, 80, 190]  # Centered colored object

    features, intermediate = extract_features_from_image(canvas, max_dimension=300)

    print("Extracted features:")
    for col in FEATURE_COLUMNS:
        assert col in features, f"Missing feature: {col}"
        print(f"  - {col}: {features[col]}")

    assert "preprocessing" in intermediate
    assert "quality" in intermediate
    assert "sharpness" in intermediate
    assert "noise" in intermediate
    assert "segmentation" in intermediate
    assert "object_analysis" in intermediate

    print("Feature extraction test passed successfully!")

if __name__ == "__main__":
    test_feature_extraction()

