"""
Module: object_analysis.py
Description: Product contour detection, bounding box extraction, centroid calculation,
             canvas area ratio, boundary proximity (cropping check), and background uniformity.

Digital Image Processing (DIP) Concepts:
1. Contour Extraction (Suzuki & Abe, 1985):
   Topological structural analysis of binary images to detect object boundary curves.
2. Spatial Image Moments (Hu, 1962):
   m_{p,q} = sum_{x,y} x^p * y^q * I(x, y)
   - Zero-th moment m_{00} represents the total object area.
   - First moments m_{10} and m_{01} define the spatial center of gravity (Centroid):
     C_x = m_{10} / m_{00},  C_y = m_{01} / m_{00}
3. Alignment Score:
   Measures Euclidean deviation of the centroid from canvas geometric center (W/2, H/2),
   normalized by the half-diagonal of the image.
4. Boundary Proximity & Cropping Heuristic:
   Products touching the canvas perimeter (x <= margin or y <= margin) frequently suffer from
   accidental cropping. We compute minimum distance to borders as an interpretable warning indicator.
5. Background Uniformity:
   Calculates the standard deviation of pixel intensities in the non-product region (mask == 0).
   Pure studio white backgrounds yield near-zero variance; messy or textured backgrounds have high variance.
"""

from typing import Dict, Any, Tuple, Optional
import math
import numpy as np
import cv2


def find_primary_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Finds the largest external contour representing the primary product.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Sort contours by area descending
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:  # Ignore microscopic noise
        return None
    return largest


def calculate_centroid(contour: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Calculates spatial center of mass (Cx, Cy) from contour moments.
    Formula:
        Cx = m10 / m00,  Cy = m01 / m00
    """
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    cx = int(round(moments["m10"] / moments["m00"]))
    cy = int(round(moments["m01"] / moments["m00"]))
    return (cx, cy)


def calculate_alignment_score(centroid: Tuple[int, int], img_shape: Tuple[int, int]) -> Tuple[float, str]:
    """
    Computes normalized centering score in range [0.0, 1.0] (1.0 = perfectly centered).
    
    Formula:
        norm_dist = Euclidean_distance(Centroid, Canvas_Center) / Max_possible_distance
        score = max(0.0, 1.0 - norm_dist)
    """
    h, w = img_shape[:2]
    center_x = w / 2.0
    center_y = h / 2.0

    cx, cy = centroid
    dist = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
    max_dist = math.sqrt(center_x ** 2 + center_y ** 2)

    norm_dist = dist / max_dist if max_dist > 0 else 0.0
    score = max(0.0, 1.0 - norm_dist)

    if score > 0.85:
        msg = f"Well centered (Centering Score: {score:.2f})"
    elif score > 0.70:
        msg = f"Slightly off-center (Centering Score: {score:.2f})"
    else:
        msg = f"Poorly positioned / off-center (Centering Score: {score:.2f})"

    return round(score, 4), msg


def check_boundary_proximity(
    bbox: Tuple[int, int, int, int],
    img_shape: Tuple[int, int],
    margin_threshold: int = 5
) -> Tuple[float, bool, str]:
    """
    Evaluates whether the product touches canvas boundaries (potential cropping defect).
    
    Returns:
        tuple: (min_margin_ratio, is_at_border_flag, diagnostic_msg)
    """
    h, w = img_shape[:2]
    x, y, bw, bh = bbox

    # Distance to canvas edges
    dist_left = x
    dist_top = y
    dist_right = w - (x + bw)
    dist_bottom = h - (y + bh)

    min_dist_px = min(dist_left, dist_top, dist_right, dist_bottom)
    min_dim = min(h, w)
    min_margin_ratio = max(0.0, min_dist_px / float(min_dim))

    is_touching = min_dist_px <= margin_threshold

    if is_touching:
        msg = f"Warning: Product touches or is extremely close to image edge (Min margin: {min_dist_px}px). Potential accidental crop."
    else:
        msg = f"Adequate boundary margin preserved (Min margin: {min_dist_px}px)."

    return round(min_margin_ratio, 4), is_touching, msg


def calculate_background_uniformity(gray_img: np.ndarray, mask: np.ndarray) -> Tuple[float, str]:
    """
    Computes background uniformity score in range [0.0, 1.0] by analyzing variance outside the product.
    
    A clean studio background has low intensity standard deviation (sigma < 10).
    A cluttered or non-uniform background has high standard deviation.
    """
    bg_pixels = gray_img[mask == 0]
    if bg_pixels.size == 0:
        return 0.5, "Could not reliably evaluate background (empty mask)."

    bg_std = float(np.std(bg_pixels))
    # Map standard deviation into [0.0, 1.0] score: std >= 40 yields 0.0, std == 0 yields 1.0
    uniformity_score = max(0.0, min(1.0, 1.0 - (bg_std / 40.0)))

    if uniformity_score >= 0.80:
        msg = f"Clean, uniform background (Std Dev: {bg_std:.1f})"
    elif uniformity_score >= 0.50:
        msg = f"Moderate background variations detected (Std Dev: {bg_std:.1f})"
    else:
        msg = f"Cluttered or irregular background (Std Dev: {bg_std:.1f})"

    return round(uniformity_score, 4), msg


def draw_object_annotations(
    rgb_img: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    centroid: Optional[Tuple[int, int]]
) -> np.ndarray:
    """
    Draws bounding box and centroid marker on a copy of the RGB image for UI display.
    """
    annotated = rgb_img.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box

    if centroid is not None:
        cx, cy = centroid
        cv2.circle(annotated, (cx, cy), 6, (255, 0, 0), -1)  # Red circle center
        cv2.drawMarker(annotated, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 14, 2)

    return annotated


def analyze_object_and_background(
    rgb_img: np.ndarray,
    gray_img: np.ndarray,
    binary_mask: np.ndarray
) -> Dict[str, Any]:
    """
    Stage 7 Complete Object Layout and Background Analysis Pipeline.
    
    Returns:
        dict:
            - 'bounding_box': (x, y, w, h)
            - 'centroid': (cx, cy)
            - 'area_ratio': Foreground area / Image area
            - 'alignment_score': Centering score (0.0 to 1.0)
            - 'alignment_message': Position evaluation
            - 'boundary_margin_ratio': Proximity ratio
            - 'is_cropped_warning': Boolean flag
            - 'cropping_message': Explanation
            - 'background_uniformity_score': 0.0 to 1.0
            - 'background_message': Background evaluation
            - 'annotated_image': Visual overlay with bbox and centroid
    """
    h, w = rgb_img.shape[:2]
    total_area = h * w

    primary_contour = find_primary_contour(binary_mask)

    if primary_contour is None:
        return {
            "bounding_box": None,
            "centroid": None,
            "area_ratio": 0.0,
            "alignment_score": 0.0,
            "alignment_message": "No primary object detected.",
            "boundary_margin_ratio": 0.0,
            "is_cropped_warning": True,
            "cropping_message": "Object not detected.",
            "background_uniformity_score": 0.5,
            "background_message": "Background analysis inconclusive.",
            "annotated_image": rgb_img.copy()
        }

    # Bounding box
    bbox = cv2.boundingRect(primary_contour)
    # Centroid
    centroid = calculate_centroid(primary_contour)
    if centroid is None:
        centroid = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)

    # Object area ratio
    fg_area = float(cv2.contourArea(primary_contour))
    area_ratio = fg_area / total_area if total_area > 0 else 0.0

    # Alignment score
    alignment_score, align_msg = calculate_alignment_score(centroid, (h, w))

    # Boundary proximity
    margin_ratio, is_cropped, crop_msg = check_boundary_proximity(bbox, (h, w))

    # Background uniformity
    bg_score, bg_msg = calculate_background_uniformity(gray_img, binary_mask)

    # Visual overlay
    annotated = draw_object_annotations(rgb_img, bbox, centroid)

    return {
        "bounding_box": bbox,
        "centroid": centroid,
        "area_ratio": round(area_ratio, 4),
        "alignment_score": alignment_score,
        "alignment_message": align_msg,
        "boundary_margin_ratio": margin_ratio,
        "is_cropped_warning": is_cropped,
        "cropping_message": crop_msg,
        "background_uniformity_score": bg_score,
        "background_message": bg_msg,
        "annotated_image": annotated
    }

