"""
Module: preprocessing.py
Description: Image loading, color-space transformations, resizing, and normalization.

Digital Image Processing (DIP) Concepts:
- Matrix representation of digital images: f(x, y) where x, y are spatial coordinates.
- Color space conversion: RGB to Grayscale using ITU-R BT.601 perceptual luminance weights:
    Y = 0.299 * R + 0.587 * G + 0.114 * B
- Spatial Interpolation / Resizing: Preserving aspect ratio using bilinear/area interpolation.
- Intensity Normalization: Scaling pixel range from [0, 255] to [0.0, 1.0] for stable computation.
"""

from typing import Union, Tuple, Dict, Any, Optional
import os
import cv2
import numpy as np
from PIL import Image


def load_image(source: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Loads an image from various input types and returns it in standard RGB format (H, W, 3).
    
    Args:
        source: File path, raw bytes, numpy array, or PIL Image.
        
    Returns:
        np.ndarray: uint8 RGB image with shape (H, W, 3).
        
    Raises:
        ValueError: If image source cannot be read or is invalid.
    """
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image file does not exist: {source}")
        bgr = cv2.imread(source)
        if bgr is None:
            raise ValueError(f"Failed to read image from path: {source}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    elif isinstance(source, bytes):
        nparr = np.frombuffer(source, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image from bytes.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    elif isinstance(source, Image.Image):
        rgb_pil = source.convert("RGB")
        return np.array(rgb_pil, dtype=np.uint8)

    elif isinstance(source, np.ndarray):
        if source.ndim == 2:
            # Grayscale to RGB
            return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        elif source.ndim == 3:
            if source.shape[2] == 4:
                # RGBA to RGB
                return cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
            elif source.shape[2] == 3:
                return source.copy()
            else:
                raise ValueError(f"Unsupported channel count: {source.shape[2]}")
        else:
            raise ValueError(f"Invalid image array dimensions: {source.ndim}")

    else:
        raise TypeError(f"Unsupported image input type: {type(source)}")


def to_grayscale(rgb_img: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to single-channel 8-bit Grayscale.
    
    DIP Formula:
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        
    This formula accounts for human eye spectral sensitivity (highest to green, then red, then blue).
    """
    if rgb_img.ndim == 2:
        return rgb_img.copy()
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)


def resize_image(
    img: np.ndarray,
    max_dimension: Optional[int] = 800,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Resizes an image preserving aspect ratio (if max_dimension given) or to a fixed target (W, H).
    
    Uses cv2.INTER_AREA for downsampling (avoids Moiré artifacts) and cv2.INTER_LINEAR for upsampling.
    """
    h, w = img.shape[:2]

    if target_size is not None:
        target_w, target_h = target_size
        interpolation = cv2.INTER_AREA if (target_w < w and target_h < h) else cv2.INTER_LINEAR
        return cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    if max_dimension is not None and max(h, w) > max_dimension:
        scale = max_dimension / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img.copy()


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel intensities to [0.0, 1.0] float32.
    
    DIP Formula:
        I_norm(x, y) = I(x, y) / 255.0
    """
    return (img.astype(np.float32)) / 255.0


def preprocess_pipeline(
    source: Union[str, bytes, np.ndarray, Image.Image],
    max_dimension: int = 800
) -> Dict[str, Any]:
    """
    Full Stage 1 Preprocessing Pipeline.
    
    Returns:
        dict:
            - 'original_rgb': Untouched RGB image
            - 'resized_rgb': Resized RGB image (for consistent pipeline throughput)
            - 'grayscale': 8-bit Grayscale of resized image
            - 'normalized': Normalized [0.0, 1.0] float32 grayscale
            - 'original_shape': (H, W, C)
            - 'processed_shape': (H, W, C)
    """
    orig_rgb = load_image(source)
    resized_rgb = resize_image(orig_rgb, max_dimension=max_dimension)
    gray = to_grayscale(resized_rgb)
    norm = normalize_image(gray)

    return {
        "original_rgb": orig_rgb,
        "resized_rgb": resized_rgb,
        "grayscale": gray,
        "normalized": norm,
        "original_shape": orig_rgb.shape,
        "processed_shape": resized_rgb.shape
    }

