"""
Module: feature_extraction.py
Description: End-to-end extraction of numerical DIP feature vectors from raw images.

Digital Image Processing (DIP) Concepts:
Feature extraction bridges classical spatial image transformations with quantitative pattern recognition.
Each uploaded product image is transformed into an 8-dimensional numerical feature representation:
1. Brightness: Global mean intensity mu = (1/N) * sum(I).
2. Contrast: Intensity standard deviation sigma = sqrt( (1/N) * sum((I - mu)^2) ).
3. Sharpness: Variance of the discrete Laplacian operator Var(nabla^2 I).
4. Noise: High-frequency residual noise standard deviation sigma_noise (Immerkær estimator).
5. Object Area Ratio: Foreground mask area / total canvas area.
6. Alignment Score: Centroid closeness to geometric canvas center [0.0..1.0].
7. Background Uniformity: Uniformity index of pixels outside the product mask [0.0..1.0].
8. Boundary Margin Ratio: Minimum distance from product bounding box to image perimeter.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import os
import pandas as pd
import numpy as np
from PIL import Image

from src.preprocessing import preprocess_pipeline
from src.quality_analysis import analyze_brightness_contrast
from src.sharpness import analyze_sharpness
from src.noise import analyze_noise
from src.segmentation import segment_product
from src.object_analysis import analyze_object_and_background

FEATURE_COLUMNS = [
    "brightness",
    "contrast",
    "sharpness",
    "noise",
    "object_area_ratio",
    "alignment_score",
    "bg_uniformity",
    "boundary_margin"
]


def extract_features_from_image(
    source: Union[str, bytes, np.ndarray, Image.Image],
    max_dimension: int = 600
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Executes the entire DIP pipeline on an image and returns:
    1. A dictionary of 8 numerical features.
    2. A dictionary containing all intermediate visual artifacts for inspection and UI display.
    """
    # Stage 1: Preprocessing
    prep = preprocess_pipeline(source, max_dimension=max_dimension)
    rgb = prep["resized_rgb"]
    gray = prep["grayscale"]

    # Stage 2: Brightness & Contrast
    b_c = analyze_brightness_contrast(gray)

    # Stage 3: Sharpness & Blur
    sharp = analyze_sharpness(gray)

    # Stage 4: Noise Analysis
    noise_res = analyze_noise(gray)

    # Stage 5 & 6: Segmentation & Morphology
    seg = segment_product(rgb, gray, method="otsu")

    # Stage 7: Object Layout & Background Uniformity
    obj = analyze_object_and_background(rgb, gray, seg["binary_mask"])

    # Aggregate numerical features
    features = {
        "brightness": float(b_c["brightness_score"]),
        "contrast": float(b_c["contrast_score"]),
        "sharpness": float(sharp["sharpness_score"]),
        "noise": float(noise_res["noise_score"]),
        "object_area_ratio": float(obj["area_ratio"]),
        "alignment_score": float(obj["alignment_score"]),
        "bg_uniformity": float(obj["background_uniformity_score"]),
        "boundary_margin": float(obj["boundary_margin_ratio"])
    }

    # Aggregate intermediate visual artifacts
    intermediate = {
        "preprocessing": prep,
        "quality": b_c,
        "sharpness": sharp,
        "noise": noise_res,
        "segmentation": seg,
        "object_analysis": obj
    }

    return features, intermediate


def batch_extract_features(
    image_paths: List[str],
    labels: Optional[List[str]] = None,
    max_dimension: int = 600
) -> pd.DataFrame:
    """
    Extracts numerical features from a list of image paths and returns a Pandas DataFrame.
    """
    records = []
    for idx, path in enumerate(image_paths):
        try:
            feats, _ = extract_features_from_image(path, max_dimension=max_dimension)
            record = feats.copy()
            record["filename"] = os.path.basename(path)
            record["path"] = path
            if labels is not None:
                record["label"] = labels[idx]
            records.append(record)
        except Exception as e:
            print(f"Warning: Failed to process {path}: {e}")

    df = pd.DataFrame(records)
    return df
