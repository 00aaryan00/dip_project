"""
Unit tests for src/baseline.py
"""
from src.baseline import evaluate_rules

def test_baseline():
    print("Testing Baseline Rule-Based Classifier...")

    # Case 1: Ideal good quality image
    good_features = {
        "brightness": 130.0,
        "contrast": 55.0,
        "sharpness": 250.0,
        "noise": 2.5,
        "object_area_ratio": 0.45,
        "alignment_score": 0.95,
        "bg_uniformity": 0.92,
        "boundary_margin": 0.15
    }
    res_good = evaluate_rules(good_features)
    print(f"Good features decision: {res_good['decision']}")
    assert res_good["decision"] == "ACCEPT"
    assert len(res_good["reasons"]) == 0

    # Case 2: Blurry and dark image
    bad_features = {
        "brightness": 35.0,        # Underexposed
        "contrast": 20.0,        # Low contrast
        "sharpness": 22.0,       # Blurry
        "noise": 18.0,           # High noise
        "object_area_ratio": 0.45,
        "alignment_score": 0.95,
        "bg_uniformity": 0.90,
        "boundary_margin": 0.15
    }
    res_bad = evaluate_rules(bad_features)
    print(f"Bad features decision: {res_bad['decision']}, Reasons: {res_bad['reasons']}")
    assert res_bad["decision"] == "REJECT"
    assert len(res_bad["reasons"]) >= 3

    print("Baseline tests passed successfully!")

if __name__ == "__main__":
    test_baseline()

