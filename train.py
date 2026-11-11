"""
Script: train.py
Description: Feature extraction, Random Forest training, Baseline vs. ML evaluation,
             and quantitative comparison report generation.

Digital Image Processing & Machine Learning Course Requirement:
Both the Baseline (Rule-Based DIP) and Improved (Random Forest) models are evaluated
on the EXACT same held-out test images with zero data leakage. All metrics are calculated
from actual predictions.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.feature_extraction import extract_features_from_image, FEATURE_COLUMNS
from src.baseline import evaluate_rules, DEFAULT_THRESHOLDS
from src.classifier import train_random_forest, evaluate_classifier, save_model


def run_pipeline():
    manifest_path = os.path.join("dataset", "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Please run data_generator.py first.")

    manifest_df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(manifest_df)} records.")

    # 1. Feature Extraction
    features_csv = os.path.join("dataset", "extracted_features.csv")
    feature_records = []

    print("\n--- STAGE: DIP FEATURE EXTRACTION ---")
    for idx, row in manifest_df.iterrows():
        img_path = row["filepath"]
        try:
            feats, _ = extract_features_from_image(img_path, max_dimension=500)
            rec = feats.copy()
            rec["image_id"] = row["image_id"]
            rec["original_product_id"] = row["original_product_id"]
            rec["split"] = row["split"]
            rec["category"] = row["category"]
            rec["defect_type"] = row["defect_type"]
            rec["ground_truth"] = row["ground_truth"]
            rec["filepath"] = img_path
            feature_records.append(rec)
            if (idx + 1) % 20 == 0 or (idx + 1) == len(manifest_df):
                print(f"Extracted features for {idx + 1}/{len(manifest_df)} images...")
        except Exception as e:
            print(f"Error extracting features from {img_path}: {e}")

    df_features = pd.DataFrame(feature_records)
    df_features.to_csv(features_csv, index=False)
    print(f"Saved extracted features to: {features_csv}")

    # 2. Strict Train / Test Split
    train_df = df_features[df_features["split"] == "train"].copy()
    test_df = df_features[df_features["split"] == "test"].copy()

    print(f"\nTrain set: {len(train_df)} images ({train_df['ground_truth'].value_counts().to_dict()})")
    print(f"Test set:  {len(test_df)} images ({test_df['ground_truth'].value_counts().to_dict()})")

    # 3. Train Machine Learning Model (Random Forest)
    print("\n--- STAGE: TRAINING RANDOM FOREST CLASSIFIER ---")
    rf_model = train_random_forest(
        X_train=train_df,
        y_train=train_df["ground_truth"],
        n_estimators=100,
        random_state=42
    )
    save_model(rf_model, os.path.join("models", "random_forest.joblib"))
    print("Saved trained model to: models/random_forest.joblib")

    # 3b. 5-Fold Stratified Cross-Validation on Training Data
    print("\n--- STAGE: 5-FOLD STRATIFIED CROSS-VALIDATION (TRAIN SET) ---")
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc_scores = cross_val_score(rf_model, train_df[FEATURE_COLUMNS], train_df["ground_truth"], cv=cv_kfold, scoring="accuracy")
    cv_f1_scores = cross_val_score(rf_model, train_df[FEATURE_COLUMNS], train_df["ground_truth"], cv=cv_kfold, scoring="f1_macro")
    print(f"5-Fold CV Accuracy: {cv_acc_scores.mean():.4f} (+/- {cv_acc_scores.std():.4f})")
    print(f"5-Fold CV F1 Score: {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std():.4f})")
    cv_summary = {
        "mean_accuracy": round(float(cv_acc_scores.mean()), 4),
        "std_accuracy": round(float(cv_acc_scores.std()), 4),
        "mean_f1": round(float(cv_f1_scores.mean()), 4),
        "std_f1": round(float(cv_f1_scores.std()), 4),
        "fold_accuracies": [round(float(s), 4) for s in cv_acc_scores]
    }

    # 4. Evaluate Improved Approach (Random Forest) on Test Set
    rf_eval = evaluate_classifier(rf_model, test_df, test_df["ground_truth"])
    rf_preds = rf_eval["predictions"]

    # 5. Evaluate Baseline Approach (Rule-Based DIP) on Same Test Set
    print("\n--- STAGE: EVALUATING BASELINE (RULE-BASED) ---")
    baseline_preds = []
    baseline_reasons = []

    for _, row in test_df.iterrows():
        feat_dict = {col: float(row[col]) for col in FEATURE_COLUMNS}
        res = evaluate_rules(feat_dict)
        baseline_preds.append(res["decision"])
        baseline_reasons.append(res["reasons"])

    test_df["baseline_pred"] = baseline_preds
    test_df["rf_pred"] = rf_preds

    # Compute Baseline Metrics
    y_true = test_df["ground_truth"].tolist()
    labels_order = ["ACCEPT", "REJECT"]

    b_acc = float(accuracy_score(y_true, baseline_preds))
    b_prec = float(precision_score(y_true, baseline_preds, pos_label="REJECT", zero_division=0))
    b_rec = float(recall_score(y_true, baseline_preds, pos_label="REJECT", zero_division=0))
    b_f1 = float(f1_score(y_true, baseline_preds, pos_label="REJECT", zero_division=0))
    b_cm = confusion_matrix(y_true, baseline_preds, labels=labels_order)

    # Baseline False Reject Rate (ACCEPT misclassified as REJECT)
    good_total_b = b_cm[0, 0] + b_cm[0, 1]
    b_frr = float(b_cm[0, 1] / good_total_b) if good_total_b > 0 else 0.0

    # 6. Quantitative Comparison Table
    comparison_data = {
        "Metric": [
            "Accuracy",
            "Precision (Defect Detection)",
            "Recall (Defect Detection)",
            "F1 Score",
            "False Reject Rate (FRR)"
        ],
        "Baseline (Rule-Based DIP)": [
            f"{b_acc:.4f}",
            f"{b_prec:.4f}",
            f"{b_rec:.4f}",
            f"{b_f1:.4f}",
            f"{b_frr:.4f}"
        ],
        "Improved (Random Forest)": [
            f"{rf_eval['accuracy']:.4f}",
            f"{rf_eval['precision']:.4f}",
            f"{rf_eval['recall']:.4f}",
            f"{rf_eval['f1_score']:.4f}",
            f"{rf_eval['false_reject_rate']:.4f}"
        ]
    }
    comparison_df = pd.DataFrame(comparison_data)

    comp_csv = os.path.join("results", "comparisons", "baseline_vs_random_forest.csv")
    os.makedirs(os.path.dirname(comp_csv), exist_ok=True)
    comparison_df.to_csv(comp_csv, index=False)

    print("\n==================================================================")
    print("           QUANTITATIVE PERFORMANCE COMPARISON REPORT             ")
    print("==================================================================")
    print(comparison_df.to_string(index=False))
    print("==================================================================")
    print("\nConfusion Matrix (Baseline - Rule-Based):")
    print(f"                Predicted ACCEPT  Predicted REJECT")
    print(f"Actual ACCEPT:        {b_cm[0, 0]:<17} {b_cm[0, 1]}")
    print(f"Actual REJECT:        {b_cm[1, 0]:<17} {b_cm[1, 1]}")

    rf_cm = np.array(rf_eval["confusion_matrix"])
    print("\nConfusion Matrix (Improved - Random Forest):")
    print(f"                Predicted ACCEPT  Predicted REJECT")
    print(f"Actual ACCEPT:        {rf_cm[0, 0]:<17} {rf_cm[0, 1]}")
    print(f"Actual REJECT:        {rf_cm[1, 0]:<17} {rf_cm[1, 1]}")

    print("\nRandom Forest Feature Importances:")
    for feat, imp in rf_eval["feature_importances"].items():
        print(f"  • {feat:<20}: {imp:.4f}")

    # 7. Generate Visual Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Baseline CM
    im0 = axes[0].imshow(b_cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title("Baseline (Rule-Based DIP)\nConfusion Matrix")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(labels_order)
    axes[0].set_yticklabels(labels_order)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(b_cm[i, j]), ha="center", va="center", color="red" if i != j else "black", fontsize=14, weight="bold")

    # RF CM
    im1 = axes[1].imshow(rf_cm, interpolation='nearest', cmap=plt.cm.Greens)
    axes[1].set_title("Improved (Random Forest)\nConfusion Matrix")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(labels_order)
    axes[1].set_yticklabels(labels_order)
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(rf_cm[i, j]), ha="center", va="center", color="red" if i != j else "black", fontsize=14, weight="bold")

    plt.tight_layout()
    cm_plot_path = os.path.join("results", "plots", "confusion_matrices.png")
    os.makedirs(os.path.dirname(cm_plot_path), exist_ok=True)
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"\nSaved confusion matrix plot to: {cm_plot_path}")

    # Feature Importances Plot
    feat_imp = rf_eval["feature_importances"]
    fig_imp, ax_imp = plt.subplots(figsize=(8, 4.5))
    feats_sorted = list(feat_imp.keys())[::-1]
    vals_sorted = [feat_imp[k] for k in feats_sorted]
    ax_imp.barh([f.replace("_", " ").title() for f in feats_sorted], vals_sorted, color="#2563EB", edgecolor="black")
    ax_imp.set_xlabel("Gini Feature Importance", fontsize=10)
    ax_imp.set_title("Random Forest Feature Importance on DIP Metrics", fontsize=12, fontweight="bold")
    ax_imp.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    imp_plot_path = os.path.join("results", "plots", "feature_importances.png")
    plt.savefig(imp_plot_path, dpi=200)
    plt.close()
    print(f"Saved feature importance plot to: {imp_plot_path}")

    # Save full metrics JSON
    metrics_summary = {
        "baseline": {
            "accuracy": b_acc,
            "precision": b_prec,
            "recall": b_rec,
            "f1_score": b_f1,
            "false_reject_rate": b_frr,
            "confusion_matrix": b_cm.tolist()
        },
        "random_forest": rf_eval,
        "cross_validation": cv_summary
    }
    with open(os.path.join("results", "metrics", "evaluation_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print("Saved evaluation summary to: results/metrics/evaluation_summary.json")

    return comparison_df


if __name__ == "__main__":
    run_pipeline()

