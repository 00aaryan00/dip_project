"""
Module: noise.py
Description: Noise estimation, synthetic noise injection, and spatial filtering.

Digital Image Processing (DIP) Concepts:
1. Nature of Image Noise:
   - Gaussian Noise: Electronic sensor noise characterized by zero-mean additive normal distribution:
     n ~ N(0, sigma^2). Affects all pixels continuously.
   - Salt-and-Pepper (Impulse) Noise: Sharp, sudden disturbances caused by faulty transmission or sensor cells.
     Random pixels are saturated to minimum (0, Pepper) or maximum (255, Salt) values.
2. Noise Estimation (Immerkær, 1996):
   Single-image noise standard deviation estimation using a Laplacian-like mask:
       M = [  1, -2,  1 ]
           [ -2,  4, -2 ]
           [  1, -2,  1 ]
       sigma_noise = sqrt(pi / 2) * (1 / (6 * (W - 2) * (H - 2))) * sum_{x,y} |I * M|
   This mask acts as a high-pass filter that cancels out smooth image structures and constant/linear gradients,
   isolating zero-mean high-frequency noise variance.
3. Linear vs. Non-linear Filtering:
   - Gaussian Filter (Linear): Convolves image with a 2D Gaussian kernel. Weighted averaging smooths Gaussian noise
     effectively, but blurs sharp edges and smears impulse noise spikes across neighboring pixels.
   - Median Filter (Non-linear Order-Statistic): Replaces each pixel with the statistical median of its neighborhood:
     f_hat(x, y) = median_{(s, t) in S_{xy}} { g(s, t) }
     Because extreme values (0 or 255 from salt-and-pepper) are forced to the tails of the sorted list, the median
     completely discards them without blurring structural boundaries.
"""

from typing import Dict, Any, Tuple
import math
import numpy as np
import cv2


def estimate_noise(gray_img: np.ndarray) -> float:
    """
    Estimates standard deviation of additive zero-mean noise using the Immerkær (1996) formulation.
    
    Formula:
        sigma = sqrt(pi / 2) * (1 / (6 * (W - 2) * (H - 2))) * sum(|I * M|)
    """
    h, w = gray_img.shape[:2]
    if h < 3 or w < 3:
        return 0.0

    # Immerkær noise estimation mask
    kernel = np.array([
        [ 1, -2,  1],
        [-2,  4, -2],
        [ 1, -2,  1]
    ], dtype=np.float32)

    # Convolve with float representation
    filtered = cv2.filter2D(gray_img.astype(np.float32), -1, kernel)
    # Exclude boundary pixels where convolution is truncated
    valid_region = filtered[1:h-1, 1:w-1]
    abs_sum = np.sum(np.abs(valid_region))

    factor = math.sqrt(math.pi / 2.0) / (6.0 * (w - 2) * (h - 2))
    sigma_noise = float(factor * abs_sum)
    return sigma_noise


def add_gaussian_noise(img: np.ndarray, mean: float = 0.0, sigma: float = 25.0) -> np.ndarray:
    """
    Injects synthetic zero-mean Gaussian noise into an image.
    I_noisy = Clip(I + N(mean, sigma), 0, 255)
    """
    gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(img: np.ndarray, amount: float = 0.04, s_vs_p: float = 0.5) -> np.ndarray:
    """
    Injects synthetic impulse (salt and pepper) noise.
    
    Args:
        amount: Proportion of image pixels to alter.
        s_vs_p: Salt vs Pepper ratio (default 0.5 for equal distribution).
    """
    noisy = img.copy()
    num_salt = int(np.ceil(amount * img.size * s_vs_p))
    num_pepper = int(np.ceil(amount * img.size * (1.0 - s_vs_p)))

    # Salt (255)
    coords = [np.random.randint(0, i, num_salt) for i in img.shape[:2]]
    if img.ndim == 2:
        noisy[tuple(coords)] = 255
    else:
        noisy[coords[0], coords[1], :] = 255

    # Pepper (0)
    coords = [np.random.randint(0, i, num_pepper) for i in img.shape[:2]]
    if img.ndim == 2:
        noisy[tuple(coords)] = 0
    else:
        noisy[coords[0], coords[1], :] = 0

    return noisy


def apply_gaussian_filter(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Applies 2D Gaussian linear spatial smoothing.
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)


def apply_median_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Applies non-linear order-statistic median filtering.
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img, ksize)


def interpret_noise(noise_score: float, low_threshold: float = 4.0, high_threshold: float = 12.0) -> Tuple[str, str]:
    """
    Interprets estimated noise level.
    """
    if noise_score < low_threshold:
        return "LOW", f"Clean image with negligible sensor noise (Estimate: {noise_score:.2f} < {low_threshold})"
    elif noise_score <= high_threshold:
        return "MODERATE", f"Moderate noise present (Estimate: {noise_score:.2f})"
    else:
        return "HIGH", f"High noise detected (Estimate: {noise_score:.2f} > {high_threshold})"


def analyze_noise(
    gray_img: np.ndarray,
    noise_threshold: float = 12.0,
    filter_ksize: int = 5
) -> Dict[str, Any]:
    """
    Stage 4 Complete Noise Analysis and Filtering Pipeline.
    
    Returns:
        dict:
            - 'noise_score': Estimated noise standard deviation
            - 'noise_status': 'LOW' / 'MODERATE' / 'HIGH'
            - 'noise_message': Diagnostic explanation
            - 'gaussian_filtered': Gaussian smoothed output
            - 'median_filtered': Median filtered output
    """
    score = estimate_noise(gray_img)
    status, msg = interpret_noise(score, high_threshold=noise_threshold)
    gauss_img = apply_gaussian_filter(gray_img, ksize=filter_ksize)
    median_img = apply_median_filter(gray_img, ksize=filter_ksize)

    return {
        "noise_score": round(score, 2),
        "noise_status": status,
        "noise_message": msg,
        "gaussian_filtered": gauss_img,
        "median_filtered": median_img
    }

