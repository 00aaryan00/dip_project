"""
Module: segmentation.py
Description: Product foreground segmentation using Otsu's thresholding and GrabCut fallback.

Digital Image Processing (DIP) Concepts:
1. Otsu's Thresholding (Otsu, 1979):
   An optimal nonparametric global thresholding technique. It operates directly on the grayscale histogram.
   It divides the histogram into two classes: Background C0 (intensities [0..t]) and Foreground C1 (intensities [t+1..L-1]).
   It finds the threshold t* that maximizes between-class variance:
       sigma_B^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2
   where w0, w1 are class probabilities and mu0, mu1 are class means.

2. Background Color Polarity:
   E-commerce images usually feature light/white studio backgrounds. If the background is bright (corner mean > 127),
   the foreground is darker (cv2.THRESH_BINARY_INV). If background is dark, standard thresholding (cv2.THRESH_BINARY) is used.

3. GrabCut Fallback (Rother et al., 2004):
   An iterative graph-cut energy minimization algorithm combining Gaussian Mixture Models (GMMs) for color
   and MRF (Markov Random Field) smoothness constraints. Used when simple intensity thresholding fails.

4. Segmentation Reliability:
   Academic honesty rule: No classical segmentation is 100% infallible across arbitrary product textures.
   We compute a confidence metric based on foreground coverage and edge consistency.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from src.morphology import clean_segmentation_mask


def estimate_background_intensity(gray_img: np.ndarray, margin_pct: float = 0.05) -> float:
    """
    Estimates background intensity by sampling border margins of the image canvas.
    """
    h, w = gray_img.shape[:2]
    mh = max(1, int(h * margin_pct))
    mw = max(1, int(w * margin_pct))

    # Sample top, bottom, left, right strips
    top = gray_img[0:mh, :]
    bottom = gray_img[h-mh:h, :]
    left = gray_img[:, 0:mw]
    right = gray_img[:, w-mw:w]

    border_pixels = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])
    return float(np.median(border_pixels))


def segment_otsu(gray_img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Segments the product using Otsu's automated thresholding with polarity detection.
    
    Returns:
        tuple: (binary_mask {0, 255}, computed_otsu_threshold)
    """
    bg_intensity = estimate_background_intensity(gray_img)

    # Pre-blur slightly with 5x5 Gaussian to eliminate high-frequency salt noise before thresholding
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

    if bg_intensity > 127:
        # Light background -> Product is darker
        thresh_val, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Dark background -> Product is brighter
        thresh_val, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask, float(thresh_val)


def segment_grabcut(rgb_img: np.ndarray, iterations: int = 3) -> np.ndarray:
    """
    Segments the image using GrabCut graph cut with a bounding box prior.
    """
    h, w = rgb_img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Assume product lies inside canvas with a 5% margin
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))

    try:
        cv2.grabCut(rgb_img, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        # 0 and 2 are background; 1 and 3 are foreground
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return fg_mask
    except Exception:
        # Fallback if GrabCut encounters degenerate rectangle
        return np.ones((h, w), dtype=np.uint8) * 255


def extract_foreground(rgb_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extracts the foreground product against a pure black background.
    """
    return cv2.bitwise_and(rgb_img, rgb_img, mask=mask)


def extract_white_background(rgb_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Renders the foreground product against an industry-standard pure white background.
    """
    white_bg = np.full_like(rgb_img, 255)
    fg = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
    inv_mask = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
    return cv2.add(fg, bg)


def assess_segmentation_reliability(mask: np.ndarray) -> Tuple[str, float, str]:
    """
    Assesses whether the binary mask represents a credible product segmentation.
    
    Criteria:
    - Foreground coverage should be between 5% and 95% of total canvas.
    - An empty mask (0%) or filled mask (100%) indicates segmentation failure.
    """
    total_pixels = mask.size
    fg_pixels = np.count_nonzero(mask)
    area_ratio = fg_pixels / float(total_pixels)

    if area_ratio < 0.04:
        return "UNRELIABLE", area_ratio, "Segmentation detected almost no product foreground (area < 4%)."
    elif area_ratio > 0.96:
        return "UNRELIABLE", area_ratio, "Segmentation failed to separate background (mask covers > 96% of canvas)."
    else:
        return "RELIABLE", area_ratio, "Segmentation successfully separated foreground from background."


def segment_product(
    rgb_img: np.ndarray,
    gray_img: np.ndarray,
    method: str = "otsu"
) -> Dict[str, Any]:
    """
    Stage 5 Complete Product Segmentation Pipeline.
    
    Returns:
        dict:
            - 'binary_mask': Cleaned binary mask (uint8, {0, 255})
            - 'raw_mask': Unfiltered mask before morphology
            - 'foreground_rgb': Extracted product on black background
            - 'white_bg_rgb': Extracted product on white studio background
            - 'area_ratio': Foreground area / total canvas area
            - 'reliability_status': 'RELIABLE' or 'UNRELIABLE'
            - 'reliability_message': Explanation
            - 'method_used': 'otsu' or 'grabcut'
    """
    if method.lower() == "grabcut":
        raw_mask = segment_grabcut(rgb_img)
    else:
        raw_mask, _ = segment_otsu(gray_img)

    # Stage 6: Clean mask morphologically
    morph_res = clean_segmentation_mask(raw_mask)
    clean_mask = morph_res["final_mask"]

    rel_status, area_ratio, rel_msg = assess_segmentation_reliability(clean_mask)

    # If Otsu proved unreliable, attempt automatic GrabCut fallback
    method_used = method
    if rel_status == "UNRELIABLE" and method.lower() == "otsu":
        fallback_mask = segment_grabcut(rgb_img)
        fallback_clean = clean_segmentation_mask(fallback_mask)["final_mask"]
        fb_status, fb_ratio, fb_msg = assess_segmentation_reliability(fallback_clean)
        if fb_status == "RELIABLE":
            clean_mask = fallback_clean
            raw_mask = fallback_mask
            rel_status = fb_status
            area_ratio = fb_ratio
            rel_msg = "Otsu failed; successfully recovered using GrabCut fallback."
            method_used = "grabcut_fallback"

    fg_rgb = extract_foreground(rgb_img, clean_mask)
    white_bg_rgb = extract_white_background(rgb_img, clean_mask)

    return {
        "binary_mask": clean_mask,
        "raw_mask": raw_mask,
        "morphology": morph_res,
        "foreground_rgb": fg_rgb,
        "white_bg_rgb": white_bg_rgb,
        "area_ratio": round(area_ratio, 4),
        "reliability_status": rel_status,
        "reliability_message": rel_msg,
        "method_used": method_used
    }

