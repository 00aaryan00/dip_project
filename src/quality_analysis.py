"""
Module: quality_analysis.py
Description: Brightness and Contrast computation, Histogram analysis, and Enhancement techniques.

Digital Image Processing (DIP) Concepts:
1. Pixel Intensity Histogram:
   h(r_k) = n_k, where r_k is the k-th intensity level and n_k is number of pixels having intensity r_k.
2. Mean Brightness:
   mu = (1 / N) * sum_{x,y} I(x, y)
   Indicates overall lighting level (underexposure when mu is low, overexposure when mu is high).
3. Contrast (Standard Deviation):
   sigma = sqrt( (1 / N) * sum_{x,y} (I(x, y) - mu)^2 )
   Reflects the dispersion of pixel intensities. Low contrast means a washed-out, narrow histogram.
4. Histogram Equalization:
   Transformation function s = T(r) = (L - 1) * sum_{j=0}^k p_r(r_j) (based on cumulative distribution function).
   Flattens the histogram and stretches dynamic range across full [0, 255] spectrum.
5. Contrast Stretching (Normalization):
   I_stretched = ((I - I_min) / (I_max - I_min)) * 255
   Linearly expands narrow intensity distributions to cover the entire available dynamic range.
   
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2


def compute_histogram(gray_img: np.ndarray) -> np.ndarray:
    """
    Computes a 256-bin 1D histogram of pixel intensities [0..255].
    """
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    return hist.flatten()


def calculate_brightness(gray_img: np.ndarray) -> float:
    """
    Calculates the mean pixel intensity mu in range [0, 255].
    Formula: mu = sum(I) / total_pixels
    """
    return float(np.mean(gray_img))


def calculate_contrast(gray_img: np.ndarray) -> float:
    """
    Calculates image contrast using standard deviation of grayscale intensities sigma.
    Higher standard deviation indicates higher tonal variation/contrast.
    """
    return float(np.std(gray_img))


def histogram_equalization(gray_img: np.ndarray) -> np.ndarray:
    """
    Enhances contrast using classical Cumulative Distribution Function (CDF) equalization.
    """
    return cv2.equalizeHist(gray_img)


def contrast_stretching(
    gray_img: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> np.ndarray:
    """
    Performs robust min-max linear contrast stretching.
    Percentiles are used instead of strict min/max to avoid noise outlier sensitivity.
    """
    p_low = np.percentile(gray_img, lower_percentile)
    p_high = np.percentile(gray_img, upper_percentile)

    if p_high <= p_low:
        return gray_img.copy()

    stretched = (gray_img.astype(np.float32) - p_low) / (p_high - p_low) * 255.0
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    return stretched


def interpret_brightness(
    brightness: float,
    low_threshold: float = 60.0,
    high_threshold: float = 210.0
) -> Tuple[str, str]:
    """
    Classifies brightness level and generates an explainable diagnostic message.
    """
    if brightness < low_threshold:
        return "LOW", f"Image is underexposed or too dark (Mean: {brightness:.1f} < {low_threshold})"
    elif brightness > high_threshold:
        return "HIGH", f"Image is overexposed or washed out (Mean: {brightness:.1f} > {high_threshold})"
    else:
        return "GOOD", f"Image has balanced lighting (Mean: {brightness:.1f})"


def interpret_contrast(
    contrast: float,
    low_threshold: float = 35.0
) -> Tuple[str, str]:
    """
    Classifies contrast level and generates an explainable diagnostic message.
    """
    if contrast < low_threshold:
        return "LOW", f"Image has poor tonal contrast (Std Dev: {contrast:.1f} < {low_threshold})"
    else:
        return "GOOD", f"Image has sufficient tonal distinction (Std Dev: {contrast:.1f})"


def analyze_brightness_contrast(
    gray_img: np.ndarray,
    brightness_thresholds: Tuple[float, float] = (60.0, 210.0),
    contrast_threshold: float = 35.0
) -> Dict[str, Any]:
    """
    Stage 2 Complete Quality Analysis Pipeline.
    
    Returns:
        dict:
            - 'brightness_score': Mean pixel intensity (0-255)
            - 'contrast_score': Standard deviation of pixel intensities
            - 'brightness_status': 'GOOD' / 'LOW' / 'HIGH'
            - 'contrast_status': 'GOOD' / 'LOW'
            - 'brightness_message': Explanation
            - 'contrast_message': Explanation
            - 'original_histogram': 256-element array
            - 'hist_equalized': Equalized image
            - 'hist_equalized_histogram': Equalized histogram
            - 'contrast_stretched': Stretched image
    """
    brightness = calculate_brightness(gray_img)
    contrast = calculate_contrast(gray_img)

    b_status, b_msg = interpret_brightness(
        brightness, low_threshold=brightness_thresholds[0], high_threshold=brightness_thresholds[1]
    )
    c_status, c_msg = interpret_contrast(
        contrast, low_threshold=contrast_threshold
    )

    orig_hist = compute_histogram(gray_img)
    equalized = histogram_equalization(gray_img)
    eq_hist = compute_histogram(equalized)
    stretched = contrast_stretching(gray_img)

    return {
        "brightness_score": round(brightness, 2),
        "contrast_score": round(contrast, 2),
        "brightness_status": b_status,
        "contrast_status": c_status,
        "brightness_message": b_msg,
        "contrast_message": c_msg,
        "original_histogram": orig_hist,
        "hist_equalized": equalized,
        "hist_equalized_histogram": eq_hist,
        "contrast_stretched": stretched
    }

