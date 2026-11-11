"""
Module: morphology.py
Description: Mathematical morphological operations for refining binary segmentation masks.

Digital Image Processing (DIP) Concepts:
Mathematical morphology investigates geometric structures using a small shape called a Structuring Element (B)
probing a binary image (A).

1. Erosion (A (-) B):
   Shrinks foreground objects by peeling away boundary pixels.
   Formula: A (-) B = { z | (B)_z is a subset of A }
   Removes thin protrusions and small isolated foreground specks.

2. Dilation (A (+) B):
   Expands foreground boundaries.
   Formula: A (+) B = { z | (B_hat)_z intersects A != empty }
   Fills small holes and bridges narrow fractures.

3. Morphological Opening (A o B):
   Erosion followed by Dilation:
   Formula: A o B = (A (-) B) (+) B
   Smoothes object contours, breaks narrow isthmuses, and eliminates small foreground noise spikes.

4. Morphological Closing (A . B):
   Dilation followed by Erosion:
   Formula: A . B = (A (+) B) (-) B
   Fills interior pinholes, closes narrow gaps, and joins nearby broken contours without significantly changing area.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2


def get_structuring_element(shape: str = "rect", ksize: int = 5) -> np.ndarray:
    """
    Creates a 2D morphological structuring element.
    
    Shapes:
        - 'rect': cv2.MORPH_RECT
        - 'ellipse': cv2.MORPH_ELLIPSE
        - 'cross': cv2.MORPH_CROSS
    """
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS
    }
    morph_shape = shape_map.get(shape.lower(), cv2.MORPH_RECT)
    return cv2.getStructuringElement(morph_shape, (ksize, ksize))


def apply_erosion(mask: np.ndarray, ksize: int = 5, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    """
    Applies morphological erosion to a binary mask.
    """
    kernel = get_structuring_element(shape, ksize)
    return cv2.erode(mask, kernel, iterations=iterations)


def apply_dilation(mask: np.ndarray, ksize: int = 5, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    """
    Applies morphological dilation to a binary mask.
    """
    kernel = get_structuring_element(shape, ksize)
    return cv2.dilate(mask, kernel, iterations=iterations)


def apply_opening(mask: np.ndarray, ksize: int = 5, shape: str = "ellipse") -> np.ndarray:
    """
    Applies morphological opening (Erosion -> Dilation).
    Eliminates small foreground specks.
    """
    kernel = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def apply_closing(mask: np.ndarray, ksize: int = 7, shape: str = "ellipse") -> np.ndarray:
    """
    Applies morphological closing (Dilation -> Erosion).
    Fills holes within the foreground product object.
    """
    kernel = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def clean_segmentation_mask(
    raw_mask: np.ndarray,
    opening_ksize: int = 5,
    closing_ksize: int = 9
) -> Dict[str, Any]:
    """
    Sequential morphological mask refinement:
    1. Opening: Eliminates background noise and isolated stray pixels.
    2. Closing: Fills internal gaps and product texture holes.
    
    Returns:
        dict:
            - 'raw_mask': Original input mask
            - 'opened_mask': After opening
            - 'closed_mask': After closing
            - 'final_mask': Cleaned binary mask (uint8, values in {0, 255})
    """
    opened = apply_opening(raw_mask, ksize=opening_ksize)
    closed = apply_closing(opened, ksize=closing_ksize)

    return {
        "raw_mask": raw_mask,
        "opened_mask": opened,
        "closed_mask": closed,
        "final_mask": closed
    }

