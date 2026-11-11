"""
Module: baseline.py
Description: Traditional Rule-Based Classifier using calibrated DIP thresholds.

Digital Image Processing (DIP) Concepts:
In traditional expert systems, image quality decisions are reached by evaluating physical/numerical image
metrics against fixed calibrated bounds.
- Deterministic and fully explainable: Every rejection reason maps directly to an explicit DIP failure.
- Zero-training requirement: Operates purely as an analytical heuristic rule engine.
- Serves as the quantitative benchmark against which the machine learning model is measured.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = {
    "min_sharpness": 80.0,          # Laplacian variance < 80 indicates blur
    "min_brightness": 55.0,         # Mean intensity < 55 indicates underexposure
    "max_brightness": 225.0,        # Mean intensity > 225 indicates washed-out overexposure
    "min_contrast": 28.0,           # Intensity std dev < 28 indicates low contrast
    "max_noise": 13.5,              # Immerkær noise estimate > 13.5 indicates excessive sensor noise
    "min_area_ratio": 0.05,         # Foreground object < 5% of canvas indicates object too small
    "max_area_ratio": 0.92,         # Foreground object > 92% of canvas indicates object fills frame or improper crop
    "min_bg_uniformity": 0.55,      # Background uniformity < 0.55 indicates noisy/textured background
    "min_boundary_margin": 0.015    # Margin ratio < 0.015 indicates object touches border (cropping warning)
}


def calibrate_thresholds_from_data(train_good_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calibrates baseline heuristic thresholds statistically from clean training product images.
    
    Uses percentiles to establish empirical bounds:
    - min_sharpness: 5th percentile of clean sharpness * 0.3 (allows naturally smooth objects)
    - min_brightness: 1st percentile of clean brightness * 0.8
    - max_brightness: 99th percentile of clean brightness * 1.02
    - min_contrast: 5th percentile of clean contrast * 0.7
    - max_noise: 95th percentile of clean noise * 2.5
    - min_area_ratio: 5th percentile of clean area ratio * 0.5
    - max_area_ratio: 95th percentile of clean area ratio * 1.2
    - min_bg_uniformity: 5th percentile of clean uniformity * 0.8
    """
    calibrated = DEFAULT_THRESHOLDS.copy()
    if len(train_good_df) == 0:
        return calibrated

    if "sharpness" in train_good_df:
        calibrated["min_sharpness"] = max(30.0, float(np.percentile(train_good_df["sharpness"], 5)) * 0.3)
    if "brightness" in train_good_df:
        calibrated["min_brightness"] = max(40.0, float(np.percentile(train_good_df["brightness"], 1)) * 0.8)
        calibrated["max_brightness"] = min(248.0, float(np.percentile(train_good_df["brightness"], 99)) * 1.02)
    if "contrast" in train_good_df:
        calibrated["min_contrast"] = max(20.0, float(np.percentile(train_good_df["contrast"], 5)) * 0.7)
    if "noise" in train_good_df:
        calibrated["max_noise"] = max(12.0, float(np.percentile(train_good_df["noise"], 95)) * 2.5)
    if "object_area_ratio" in train_good_df:
        calibrated["min_area_ratio"] = max(0.03, float(np.percentile(train_good_df["object_area_ratio"], 5)) * 0.5)
        calibrated["max_area_ratio"] = min(0.98, float(np.percentile(train_good_df["object_area_ratio"], 95)) * 1.2)
    if "bg_uniformity" in train_good_df:
        calibrated["min_bg_uniformity"] = max(0.40, float(np.percentile(train_good_df["bg_uniformity"], 5)) * 0.8)

    return {k: round(v, 2) for k, v in calibrated.items()}


def evaluate_rules(
    features: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Evaluates extracted numerical image features against deterministic rules.
    
    Returns:
        dict:
            - 'decision': 'ACCEPT' or 'REJECT'
            - 'is_accepted': bool
            - 'reasons': List[str] of explicit failure reasons (empty if ACCEPT)
            - 'feature_evaluations': Dict[str, Dict] detailed status for each metric
    """
    th = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        th.update(thresholds)

    reasons: List[str] = []
    evaluations: Dict[str, Dict[str, Any]] = {}

    # 1. Sharpness / Blur check
    sharpness = features.get("sharpness", 0.0)
    if sharpness < th["min_sharpness"]:
        reasons.append(f"Image is blurry (Sharpness: {sharpness:.1f} < {th['min_sharpness']:.1f})")
        evaluations["sharpness"] = {"status": "FAIL", "value": sharpness, "threshold": th["min_sharpness"]}
    else:
        evaluations["sharpness"] = {"status": "PASS", "value": sharpness, "threshold": th["min_sharpness"]}

    # 2. Brightness checks (underexposure / overexposure)
    brightness = features.get("brightness", 128.0)
    if brightness < th["min_brightness"]:
        reasons.append(f"Image is too dark / underexposed (Brightness: {brightness:.1f} < {th['min_brightness']:.1f})")
        evaluations["brightness"] = {"status": "FAIL", "value": brightness, "condition": "low"}
    elif brightness > th["max_brightness"]:
        reasons.append(f"Image is overexposed / washed out (Brightness: {brightness:.1f} > {th['max_brightness']:.1f})")
        evaluations["brightness"] = {"status": "FAIL", "value": brightness, "condition": "high"}
    else:
        evaluations["brightness"] = {"status": "PASS", "value": brightness, "condition": "optimal"}

    # 3. Contrast check
    contrast = features.get("contrast", 50.0)
    if contrast < th["min_contrast"]:
        reasons.append(f"Image has low contrast / flat dynamic range (Contrast: {contrast:.1f} < {th['min_contrast']:.1f})")
        evaluations["contrast"] = {"status": "FAIL", "value": contrast, "threshold": th["min_contrast"]}
    else:
        evaluations["contrast"] = {"status": "PASS", "value": contrast, "threshold": th["min_contrast"]}

    # 4. Noise check
    noise = features.get("noise", 0.0)
    if noise > th["max_noise"]:
        reasons.append(f"Excessive image noise detected (Noise: {noise:.2f} > {th['max_noise']:.2f})")
        evaluations["noise"] = {"status": "FAIL", "value": noise, "threshold": th["max_noise"]}
    else:
        evaluations["noise"] = {"status": "PASS", "value": noise, "threshold": th["max_noise"]}

    # 5. Object Area Ratio
    area_ratio = features.get("object_area_ratio", 0.3)
    if area_ratio < th["min_area_ratio"]:
        reasons.append(f"Product occupies too little canvas space (Area ratio: {area_ratio*100:.1f}% < {th['min_area_ratio']*100:.1f}%)")
        evaluations["area_ratio"] = {"status": "FAIL", "value": area_ratio, "condition": "too_small"}
    elif area_ratio > th["max_area_ratio"]:
        reasons.append(f"Product occupies too much canvas space (Area ratio: {area_ratio*100:.1f}% > {th['max_area_ratio']*100:.1f}%)")
        evaluations["area_ratio"] = {"status": "FAIL", "value": area_ratio, "condition": "too_large"}
    else:
        evaluations["area_ratio"] = {"status": "PASS", "value": area_ratio, "condition": "optimal"}

    # 6. Background Uniformity
    bg_uniformity = features.get("bg_uniformity", 0.8)
    if bg_uniformity < th["min_bg_uniformity"]:
        reasons.append(f"Non-uniform / cluttered background (Uniformity: {bg_uniformity:.2f} < {th['min_bg_uniformity']:.2f})")
        evaluations["bg_uniformity"] = {"status": "FAIL", "value": bg_uniformity, "threshold": th["min_bg_uniformity"]}
    else:
        evaluations["bg_uniformity"] = {"status": "PASS", "value": bg_uniformity, "threshold": th["min_bg_uniformity"]}

    # 7. Boundary Margin / Cropping
    boundary_margin = features.get("boundary_margin", 0.1)
    if boundary_margin < th["min_boundary_margin"]:
        reasons.append(f"Product touches image boundary, potential cropping defect (Margin: {boundary_margin:.3f})")
        evaluations["boundary_margin"] = {"status": "FAIL", "value": boundary_margin, "threshold": th["min_boundary_margin"]}
    else:
        evaluations["boundary_margin"] = {"status": "PASS", "value": boundary_margin, "threshold": th["min_boundary_margin"]}

    # Final Decision
    is_accepted = len(reasons) == 0
    decision = "ACCEPT" if is_accepted else "REJECT"

    return {
        "decision": decision,
        "is_accepted": is_accepted,
        "reasons": reasons,
        "feature_evaluations": evaluations
    }


def predict_baseline_dataframe(
    df: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None
) -> List[str]:
    """
    Vectorized/batched prediction across a DataFrame of extracted features.
    Returns a list of 'ACCEPT' or 'REJECT' strings.
    """
    decisions = []
    for _, row in df.iterrows():
        feat_dict = {col: float(row[col]) for col in DEFAULT_THRESHOLDS.keys() if col in row}
        # Map DataFrame column names to default threshold keys if needed
        full_feat = {
            "brightness": float(row.get("brightness", 128.0)),
            "contrast": float(row.get("contrast", 50.0)),
            "sharpness": float(row.get("sharpness", 100.0)),
            "noise": float(row.get("noise", 2.0)),
            "object_area_ratio": float(row.get("object_area_ratio", 0.3)),
            "alignment_score": float(row.get("alignment_score", 0.9)),
            "bg_uniformity": float(row.get("bg_uniformity", 0.8)),
            "boundary_margin": float(row.get("boundary_margin", 0.1))
        }
        res = evaluate_rules(full_feat, thresholds=thresholds)
        decisions.append(res["decision"])
    return decisions

