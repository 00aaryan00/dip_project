"""
Module: classifier.py
Description: Random Forest Machine Learning classifier trained on extracted DIP features.

Digital Image Processing & Machine Learning Concepts:
1. Moving from Hard Heuristics to Statistical Learning:
   Fixed baseline thresholds evaluate features independently in axis-aligned 1D slices (e.g. Sharpness < T).
   Random Forest constructs an ensemble of decorrelated decision trees that learn non-linear, multi-dimensional
   interactions between features (e.g., a slightly lower sharpness may be acceptable if contrast is high and noise is low).

2. Class Balancing:
   In real e-commerce catalogs, defective images may be a minority or vice-versa. Setting `class_weight='balanced'`
   adjusts tree split weights inversely proportional to class frequencies:
       w_j = n_samples / (n_classes * n_samples_j)
   This prevents the model from trivially predicting the majority class.

3. Evaluation Metrics:
   - Accuracy: Overall correctness (TP + TN) / Total.
   - Precision: When predicted REJECT, how many were genuinely defective? TP / (TP + FP).
   - Recall: Out of all defective images, how many did the system catch? TP / (TP + FN).
   - F1 Score: Harmonic mean of precision and recall.
   - False Reject Rate (FRR): Proportion of genuine GOOD images mistakenly rejected. In e-commerce, high FRR
     frustrates sellers and hurts platform revenue.
"""

from typing import Dict, Any, Tuple, Optional, List
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.feature_extraction import FEATURE_COLUMNS


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = 8,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Trains a Random Forest Classifier on numerical DIP feature vectors.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train[FEATURE_COLUMNS], y_train)
    return clf


def evaluate_classifier(
    clf: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Calculates quantitative performance metrics on held-out test data.
    Assumes binary labels: 'ACCEPT' (or 'Good') and 'REJECT' (or 'Defective').
    """
    y_pred = clf.predict(X_test[FEATURE_COLUMNS])

    # Standardize label mapping: positive class = 'REJECT' (detecting defect)
    pos_label = "REJECT" if "REJECT" in np.unique(y_test) else "Defective"

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0))

    labels_order = ["ACCEPT", "REJECT"] if "ACCEPT" in np.unique(y_test) else ["Good", "Defective"]
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)

    # False Reject Rate (FRR): Proportion of ACCEPT (Good) labeled as REJECT
    # cm[0, 0] = True Accept, cm[0, 1] = False Reject
    good_total = cm[0, 0] + cm[0, 1]
    frr = float(cm[0, 1] / good_total) if good_total > 0 else 0.0

    # Feature importances
    importances = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLUMNS, clf.feature_importances_)
    }
    # Sort descending
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_reject_rate": round(frr, 4),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels_order,
        "feature_importances": sorted_importances,
        "classification_report": report,
        "predictions": y_pred.tolist()
    }


def save_model(clf: RandomForestClassifier, filepath: str = "models/random_forest.joblib") -> None:
    """
    Serializes trained Random Forest model using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(clf, filepath)


def load_model(filepath: str = "models/random_forest.joblib") -> RandomForestClassifier:
    """
    Loads serialized model.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return joblib.load(filepath)


def predict_single(
    clf: RandomForestClassifier,
    features: Dict[str, float]
) -> Tuple[str, float, Dict[str, float]]:
    """
    Runs model inference on a single feature dictionary.
    
    Returns:
        tuple: (predicted_label, confidence, class_probabilities_dict)
    """
    df_row = pd.DataFrame([features])[FEATURE_COLUMNS]
    pred = clf.predict(df_row)[0]
    probs = clf.predict_proba(df_row)[0]

    prob_dict = {
        str(cls_name): round(float(p), 4)
        for cls_name, p in zip(clf.classes_, probs)
    }
    confidence = float(np.max(probs))

    return str(pred), round(confidence, 4), prob_dict

