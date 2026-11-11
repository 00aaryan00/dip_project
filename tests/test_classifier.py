"""
Unit tests for src/classifier.py
"""
import numpy as np
import pandas as pd
from src.classifier import (
    train_random_forest,
    evaluate_classifier,
    save_model,
    load_model,
    predict_single
)
from src.feature_extraction import FEATURE_COLUMNS

def test_classifier():
    print("Testing Classifier Module...")

    # Generate synthetic training and testing feature data
    np.random.seed(42)
    n_samples = 100

    # Good: high sharpness, moderate brightness, high contrast, low noise
    good_data = {
        "brightness": np.random.normal(140, 20, n_samples // 2),
        "contrast": np.random.normal(60, 10, n_samples // 2),
        "sharpness": np.random.normal(300, 50, n_samples // 2),
        "noise": np.random.normal(2, 0.5, n_samples // 2),
        "object_area_ratio": np.random.uniform(0.3, 0.7, n_samples // 2),
        "alignment_score": np.random.uniform(0.85, 1.0, n_samples // 2),
        "bg_uniformity": np.random.uniform(0.8, 1.0, n_samples // 2),
        "boundary_margin": np.random.uniform(0.05, 0.2, n_samples // 2),
        "label": ["ACCEPT"] * (n_samples // 2)
    }

    # Defective: blurry OR dark OR high noise
    bad_data = {
        "brightness": np.random.normal(40, 15, n_samples // 2),
        "contrast": np.random.normal(20, 5, n_samples // 2),
        "sharpness": np.random.normal(30, 15, n_samples // 2),
        "noise": np.random.normal(20, 5, n_samples // 2),
        "object_area_ratio": np.random.uniform(0.1, 0.9, n_samples // 2),
        "alignment_score": np.random.uniform(0.4, 0.8, n_samples // 2),
        "bg_uniformity": np.random.uniform(0.3, 0.7, n_samples // 2),
        "boundary_margin": np.random.uniform(0.0, 0.05, n_samples // 2),
        "label": ["REJECT"] * (n_samples // 2)
    }

    df_good = pd.DataFrame(good_data)
    df_bad = pd.DataFrame(bad_data)
    df = pd.concat([df_good, df_bad], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    # Split 80/20
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    clf = train_random_forest(train_df, train_df["label"], n_estimators=50)
    metrics = evaluate_classifier(clf, test_df, test_df["label"])

    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Feature Importances: {metrics['feature_importances']}")

    assert metrics["accuracy"] >= 0.90, "Classifier should easily separate clean vs heavily defective synthetic features"

    # Test save and load
    save_model(clf, "models/test_rf.joblib")
    loaded = load_model("models/test_rf.joblib")

    test_single = {col: float(test_df[col].iloc[0]) for col in FEATURE_COLUMNS}
    pred, conf, probs = predict_single(loaded, test_single)
    print(f"Single inference: Prediction={pred}, Confidence={conf:.4f}, Probs={probs}")
    assert pred in ["ACCEPT", "REJECT"]

    print("Classifier tests passed successfully!")

if __name__ == "__main__":
    test_classifier()

