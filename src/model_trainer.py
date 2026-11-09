"""
src/model_trainer.py - Stratified Cross-Validation & Model Evaluation
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data_loader import load_and_preprocess_data

# Ensure assets directory exists
os.makedirs("assets", exist_ok=True)


def train_and_evaluate():
    print("\n--- Training Models with 5-Fold Stratified Cross-Validation ---")

    # Load merged feature dataset
    df = load_and_preprocess_data()

    # Target & Feature Definitions (matching columns in fused_df)
    target = "conversion_status"

    imaging_features = ["norm_hippocampus_ratio"]
    genomic_features = ["apoe_e4_count", "age", "sex_encoded"]
    fused_features = imaging_features + genomic_features

    X_img = df[imaging_features]
    X_fused = df[fused_features]
    y = df[target]

    # Setup 5-Fold Stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_img_folds = []
    auc_fused_folds = []

    rf_img = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_fused = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )

    for train_idx, val_idx in skf.split(X_fused, y):
        # Imaging-only split
        X_train_img, X_val_img = X_img.iloc[train_idx], X_img.iloc[val_idx]
        # Fused split
        X_train_fused, X_val_fused = (
            X_fused.iloc[train_idx],
            X_fused.iloc[val_idx],
        )
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train & Score Imaging Model
        rf_img.fit(X_train_img, y_train)
        preds_img = rf_img.predict_proba(X_val_img)[:, 1]
        auc_img_folds.append(roc_auc_score(y_val, preds_img))

        # Train & Score Fused Model
        rf_fused.fit(X_train_fused, y_train)
        preds_fused = rf_fused.predict_proba(X_val_fused)[:, 1]
        auc_fused_folds.append(roc_auc_score(y_val, preds_fused))

    mean_auc_img = np.mean(auc_img_folds)
    mean_auc_fused = np.mean(auc_fused_folds)
    auc_lift = mean_auc_fused - mean_auc_img

    print("\n" + "=" * 55)
    print(" 5-FOLD CV MODEL PERFORMANCE COMPARISON (ROC-AUC)")
    print("=" * 55)
    print(f" Neuroimaging-Only Model Mean ROC-AUC : {mean_auc_img:.4f}")
    print(f" Fused (Genomic + Imaging) Mean ROC-AUC: {mean_auc_fused:.4f}")
    print(f" Mean Performance Improvement         : +{auc_lift * 100:.2f}%")
    print("=" * 55)

    # Generate Feature Importance Bar Plot for Fused Model
    feature_importances = pd.Series(
        rf_fused.feature_importances_, index=fused_features
    ).sort_values(ascending=True)

    plt.figure(figsize=(8, 4))
    feature_importances.plot(kind="barh", color="steelblue")
    plt.title("Fused Model Feature Importances (Random Forest)")
    plt.xlabel("Relative Gini Importance")
    plt.tight_layout()

    save_path = "assets/model_performance_comparison.png"
    plt.savefig(save_path)
    plt.close()

    print(f"\nPerformance visualization saved to {save_path}")


if __name__ == "__main__":
    train_and_evaluate()