"""
Module: sharpness.py
Description: Sharpness estimation and blur detection using the Laplacian operator.

Digital Image Processing (DIP) Concepts:
1. 2nd-Order Spatial Derivative (Laplacian Operator):
   The Laplacian is an isotropic (rotation-invariant) 2nd-order differential operator:
       nabla^2 f(x, y) = (d^2 f / dx^2) + (d^2 f / dy^2)
   In discrete digital convolution, it is approximated with standard kernels such as:
       [ 0,  1,  0 ]
       [ 1, -4,  1 ]
       [ 0,  1,  0 ]
2. High-Frequency Detail & Edge Response:
   Edges in an image represent rapid intensity transitions. The second derivative produces
   a zero-crossing surrounded by strong positive and negative spikes at edges.
3. Variance of the Laplacian (Pech-Pacheco et al., 2000):
   Var(L) = (1 / N) * sum_{x,y} (L(x, y) - mu_L)^2
   - Sharp images have crisp transitions -> large spikes -> HIGH variance.
   - Blurry images have smoothed transitions -> shallow derivatives -> LOW variance.
4. Limitations:
   Low variance can also occur on inherently smooth, minimalist objects (e.g., solid monochrome silk fabric).
   Thresholds should therefore be configurable and considered alongside object area and texture.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2


def compute_laplacian(gray_img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Applies the Laplacian 2nd-derivative operator to a grayscale image.
    Uses 64-bit float (CV_64F) to preserve positive and negative edge responses without clipping.
    """
    return cv2.Laplacian(gray_img, cv2.CV_64F, ksize=ksize)


def calculate_sharpness_score(gray_img: np.ndarray, ksize: int = 3) -> float:
    """
    Calculates the sharpness score as the statistical variance of the Laplacian response.
    
    Formula:
        Var(nabla^2 I) = mean((L - mean(L))^2)
    """
    laplacian_resp = compute_laplacian(gray_img, ksize=ksize)
    variance = float(laplacian_resp.var())
    return variance


def get_laplacian_display(laplacian_64f: np.ndarray) -> np.ndarray:
    """
    Converts 64-bit float Laplacian output into an 8-bit displayable image.
    Uses absolute value mapping: |L(x, y)| scaled to uint8.
    """
    return cv2.convertScaleAbs(laplacian_64f)


def interpret_sharpness(
    sharpness_score: float,
    threshold: float = 100.0
) -> Tuple[str, str]:
    """
    Interprets the sharpness score against a calibrated threshold.
    """
    if sharpness_score < threshold:
        return (
            "BLURRY",
            f"Image lacks fine high-frequency edge detail (Sharpness: {sharpness_score:.1f} < {threshold:.1f})"
        )
    else:
        return (
            "SHARP",
            f"Image possesses crisp edge transitions (Sharpness: {sharpness_score:.1f} >= {threshold:.1f})"
        )


def analyze_sharpness(
    gray_img: np.ndarray,
    threshold: float = 100.0,
    ksize: int = 3
) -> Dict[str, Any]:
    """
    Stage 3 Complete Sharpness & Blur Analysis Pipeline.
    
    Returns:
        dict:
            - 'sharpness_score': Variance of Laplacian
            - 'sharpness_status': 'SHARP' or 'BLURRY'
            - 'sharpness_message': Explanatory message
            - 'laplacian_display': 8-bit edge visualization image
            - 'threshold_used': Threshold value applied
    """
    laplacian_64f = compute_laplacian(gray_img, ksize=ksize)
    score = float(laplacian_64f.var())
    status, msg = interpret_sharpness(score, threshold=threshold)
    display_img = get_laplacian_display(laplacian_64f)

    return {
        "sharpness_score": round(score, 2),
        "sharpness_status": status,
        "sharpness_message": msg,
        "laplacian_display": display_img,
        "threshold_used": threshold
    }

