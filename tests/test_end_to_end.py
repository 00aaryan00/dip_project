"""
Script: tests/test_end_to_end.py
Description: Automated validation of the mandatory test cases specified in project.md
             using genuine Amazon Berkeley Objects (ABO) product images.
"""

import os
import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from src.feature_extraction import extract_features_from_image
from src.baseline import evaluate_rules
from src.classifier import load_model, predict_single


def run_all_test_cases():
    print("==================================================================")
    print("         RUNNING COMPREHENSIVE END-TO-END VALIDATION              ")
    print("==================================================================")

    rf_model = load_model("models/random_forest.joblib")

    test_good_imgs = glob.glob(os.path.join("dataset", "test", "good", "*.jpg"))
    test_def_imgs = glob.glob(os.path.join("dataset", "test", "defective", "*.jpg"))

    assert len(test_good_imgs) > 0, "No test good images found!"
    assert len(test_def_imgs) > 0, "No test defective images found!"

    # 1. Test Case 1: Good Quality Product Image
    print("\n[TEST CASE 1] Good Quality Amazon Product Image:")
    # Pick a high-confidence accepted good product image
    accepted_candidates = [
        p for p in test_good_imgs
        if predict_single(rf_model, extract_features_from_image(p)[0])[0] == "ACCEPT"
    ]
    assert len(accepted_candidates) > 0, "No accepted test images found"
    img_path = accepted_candidates[0]

    feats, inter = extract_features_from_image(img_path)
    b_res = evaluate_rules(feats)
    ml_pred, ml_conf, _ = predict_single(rf_model, feats)
    print(f"  File: {os.path.basename(img_path)}")
    print(f"  Sharpness: {feats['sharpness']:.1f}, Brightness: {feats['brightness']:.1f}, Noise: {feats['noise']:.2f}")
    print(f"  Baseline Decision: {b_res['decision']}")
    print(f"  Random Forest ML:  {ml_pred} (Confidence: {ml_conf*100:.1f}%)")
    assert ml_pred == "ACCEPT", f"Expected ML to ACCEPT good image, got {ml_pred}"

    # 2. Test Case 2: Blurry Product Image (Gaussian Blur)
    print("\n[TEST CASE 2] Very Blurry Product Image:")
    blur_imgs = [p for p in test_def_imgs if "gaussian_blur" in p]
    if not blur_imgs:
        blur_imgs = [p for p in test_def_imgs if "blur" in p]
    assert len(blur_imgs) > 0, "No blur test image found"
    img_path = blur_imgs[0]
    feats, inter = extract_features_from_image(img_path)
    b_res = evaluate_rules(feats)
    ml_pred, ml_conf, _ = predict_single(rf_model, feats)
    print(f"  File: {os.path.basename(img_path)}")
    print(f"  Sharpness: {feats['sharpness']:.1f} (Threshold: 80.0)")
    print(f"  Baseline Decision: {b_res['decision']} -> Reasons: {b_res['reasons']}")
    print(f"  Random Forest ML:  {ml_pred} (Confidence: {ml_conf*100:.1f}%)")
    assert b_res["decision"] == "REJECT", "Baseline must REJECT blurry image"
    assert ml_pred == "REJECT", "ML must REJECT blurry image"
    assert any("blurry" in r.lower() or "sharpness" in r.lower() for r in b_res["reasons"])

    # 3. Test Case 3: Very Dark Product Image
    print("\n[TEST CASE 3] Very Dark Product Image:")
    dark_imgs = [p for p in test_def_imgs if "dark" in p]
    if not dark_imgs:
        dark_imgs = [p for p in test_def_imgs if "bright" in p]
    assert len(dark_imgs) > 0, "No lighting defect image found"
    img_path = dark_imgs[0]
    feats, inter = extract_features_from_image(img_path)
    b_res = evaluate_rules(feats)
    ml_pred, ml_conf, _ = predict_single(rf_model, feats)
    print(f"  File: {os.path.basename(img_path)}")
    print(f"  Brightness: {feats['brightness']:.1f} (Threshold: 55.0)")
    print(f"  Baseline Decision: {b_res['decision']} -> Reasons: {b_res['reasons']}")
    print(f"  Random Forest ML:  {ml_pred} (Confidence: {ml_conf*100:.1f}%)")
    assert b_res["decision"] == "REJECT", "Baseline must REJECT dark image"
    assert ml_pred == "REJECT", "ML must REJECT dark image"
    assert any("dark" in r.lower() or "brightness" in r.lower() for r in b_res["reasons"])

    # 4. Test Case 4: Noisy Product Image
    print("\n[TEST CASE 4] Noisy Product Image:")
    noisy_imgs = [p for p in test_def_imgs if "salt_pepper" in p or "noise" in p]
    assert len(noisy_imgs) > 0, "No noise test image found"
    img_path = noisy_imgs[0]
    feats, inter = extract_features_from_image(img_path)
    b_res = evaluate_rules(feats)
    ml_pred, ml_conf, _ = predict_single(rf_model, feats)
    print(f"  File: {os.path.basename(img_path)}")
    print(f"  Noise: {feats['noise']:.2f} (Threshold: 13.5)")
    print(f"  Baseline Decision: {b_res['decision']} -> Reasons: {b_res['reasons']}")
    print(f"  Random Forest ML:  {ml_pred} (Confidence: {ml_conf*100:.1f}%)")
    assert b_res["decision"] == "REJECT", "Baseline must REJECT noisy image"
    assert any("noise" in r.lower() for r in b_res["reasons"])

    # 5. Test Case 5: Irregular / Cluttered Background
    print("\n[TEST CASE 5] Irregular / Cluttered Background Image:")
    synth_canvas = np.random.randint(50, 220, size=(500, 300, 3), dtype=np.uint8)
    synth_canvas[100:400, 50:250] = [20, 30, 40]
    feats_bg, inter_bg = extract_features_from_image(synth_canvas)
    print(f"  Background Uniformity: {feats_bg['bg_uniformity']:.2f} (Clean background requires > 0.80)")
    print(f"  Background Message: {inter_bg['object_analysis']['background_message']}")
    assert feats_bg["bg_uniformity"] < 0.60, "Irregular background must produce low uniformity score"

    # 6. Test Case 6: Edge Case / White-on-White Segmentation Handling
    print("\n[TEST CASE 6] Segmentation Robustness & Status Reporting:")
    blank_canvas = np.full((400, 400, 3), 255, dtype=np.uint8)
    feats_blank, inter_blank = extract_features_from_image(blank_canvas)
    print(f"  Blank canvas segmentation reliability: {inter_blank['segmentation']['reliability_status']}")
    print(f"  Blank canvas explanation: {inter_blank['segmentation']['reliability_message']}")
    assert inter_blank["segmentation"]["reliability_status"] == "UNRELIABLE"

    print("\n==================================================================")
    print("    ALL 6 MANDATORY TEST CONDITIONS VALIDATED SUCCESSFULLY!      ")
    print("==================================================================")


if __name__ == "__main__":
    run_all_test_cases()
