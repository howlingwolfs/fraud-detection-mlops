# src/train.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py
========

A lightweight credit‑card‑fraud training pipeline.

* All models are stored in ``models/`` as ``*.joblib``.
* All classification reports and feature‑importance dumps are stored in ``reports/``.
* Plots are **not** generated – the script is fully head‑less.
* Hyper‑parameters, train/test split and other knobs are kept at the top of the file
  so they can be tweaked easily.
* Progress bars (`tqdm`) and timestamps (`time`) give you real‑time feedback.
* At the end of the run the script prints accuracy, AUC and the full classification
  report for every model and shows a summary table.

Author: howlingwolfs
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import os
import time
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import joblib

# --------------------------------------------------------------------------- #
# Configuration – change these to tweak the pipeline
# --------------------------------------------------------------------------- #
# File names / directories
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"
MODEL_DIR        = Path("models")
REPORT_DIR       = Path("reports")
FINAL_RESULTS_FILE = REPORT_DIR / "final_results.csv"

# Train / test split
TEST_SIZE        = 0.30          # 30 % test set
RANDOM_STATE     = 42

# Scaling – only Logistic Regression and KNN need it
SCALE_MODELS     = ["LogisticRegression", "KNeighborsClassifier"]

# Hyper‑parameters
HYPER_PARAMS = {
    "LogisticRegression": {
        "solver": "lbfgs",
        "max_iter": 300,
        "class_weight": "balanced",
    },
    "KNeighborsClassifier": {"n_neighbors": 6},
    "DecisionTreeClassifier": {"criterion": "gini", "max_depth": 3},
    "RandomForestClassifier": {"n_estimators": 200, "max_depth": 4, "random_state": RANDOM_STATE},
    "XGBClassifier": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "seed": RANDOM_STATE,
        "nthread": 1,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    },
}

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_report(text: str, filename: Path) -> None:
    """Write text to a file."""
    with open(filename, "w") as f:
        f.write(text)


def feature_importance_to_csv(model, feature_names, filename: Path) -> None:
    """Export feature importance / coefficient to CSV."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        return  # nothing to export

    df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance}
    ).sort_values("Importance", ascending=False)

    df.to_csv(filename, index=False)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    # 1. Ensure output folders exist
    ensure_dir(MODEL_DIR)
    ensure_dir(REPORT_DIR)

    # 2. Load & clean data
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 3. Train / test split
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 4. Prepare scaler (used only for models that require it)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Define model instances
    models = {
        "LogisticRegression": LogisticRegression(**HYPER_PARAMS["LogisticRegression"]),
        "KNeighborsClassifier": KNeighborsClassifier(**HYPER_PARAMS["KNeighborsClassifier"]),
        "DecisionTreeClassifier": DecisionTreeClassifier(**HYPER_PARAMS["DecisionTreeClassifier"]),
        "RandomForestClassifier": RandomForestClassifier(**HYPER_PARAMS["RandomForestClassifier"]),
        "XGBClassifier": xgb.XGBClassifier(**HYPER_PARAMS["XGBClassifier"]),
    }

    # 6. Containers for results
    results = {
        "Algorithm": [],
        "Accuracy": [],
        "AUC": [],
    }

    # 7. Iterate over models with tqdm progress bar
    for name, model in tqdm(models.items(), desc="Training models", unit="model"):
        start = time.time()
        print(f"\n{datetime.now():%Y-%m-%d %H:%M:%S} | Training {name}")

        # Decide which data to use
        X_tr = X_train_scaled if name in SCALE_MODELS else X_train
        X_te = X_test_scaled if name in SCALE_MODELS else X_test

        # Fit
        model.fit(X_tr, y_train)

        # Predict
        y_pred = model.predict(X_te)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_te)[:, 1]
        else:
            # Some models only return decision function
            y_proba = model.decision_function(X_te)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())  # rescale to [0,1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cr = classification_report(y_test, y_pred, output_dict=True)
        cr_text = classification_report(y_test, y_pred)

        # Store results
        results["Algorithm"].append(name)
        results["Accuracy"].append(acc)
        results["AUC"].append(auc)

        # Print metrics
        print(f"{name} – Accuracy: {acc:.4f} | AUC: {auc:.4f}")
        print(cr_text)

        # 8. Save model
        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # 9. Save classification report
        report_path = REPORT_DIR / f"{name}_report.txt"
        save_report(cr_text, report_path)
        print(f"Report saved to {report_path}")

        # 10. Save feature importance if available
        feature_path = REPORT_DIR / f"{name}_features.csv"
        feature_importance_to_csv(model, X.columns, feature_path)
        if feature_path.exists():
            print(f"Feature importance saved to {feature_path}")

        duration = time.time() - start
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {name} finished in {duration:.2f}s")

    # 11. Final results table
    final_df = pd.DataFrame(results)
    final_df.sort_values("AUC", ascending=False, inplace=True)
    print("\nFinal results:")
    print(final_df.to_string(index=False))

    # Save summary CSV
    final_df.to_csv(FINAL_RESULTS_FILE, index=False)
    print(f"\nSummary CSV written to {FINAL_RESULTS_FILE}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Suppress deprecation warnings from XGBoost
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
