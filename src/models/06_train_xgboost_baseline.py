"""
DATA 298A — Graph-Based Fraud Intelligence Platform
Team 2 | San José State University

06_train_xgboost_baseline.py
-----------------------------
Closes Issues #1 and #2.

Trains an XGBoost classifier on the tabular features only (no graph features).
This is the Layer 1 baseline model. Its metrics establish the floor that the
graph-enhanced model (07) must beat.

Key design decisions:
- scale_pos_weight = 906 to handle 906:1 class imbalance without resampling
- Primary metric: PR-AUC (precision-recall), not ROC-AUC or accuracy
- Time-aware splits (train/val/test already sorted chronologically by 02_data_cleaning.py)
- SHAP values computed on val set for interpretability
- Model artifact + metrics + SHAP summary saved to data/models/

Output
------
data/models/
    xgboost_baseline.json           — trained XGBoost model
    xgboost_baseline_metrics.json   — full evaluation metrics dict
    xgboost_baseline_shap.parquet   — SHAP values for val set (top features)
    xgboost_baseline_cm.png         — confusion matrix plot
    xgboost_baseline_pr_curve.png   — precision-recall curve

Usage
-----
    python3 src/models/06_train_xgboost_baseline.py
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJ       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR   = os.path.join(PROJ, "data", "processed")
MODEL_DIR  = os.path.join(PROJ, "data", "models")

TRAIN_FILE = os.path.join(PROC_DIR, "split_train.parquet")
VAL_FILE   = os.path.join(PROC_DIR, "split_val.parquet")
TEST_FILE  = os.path.join(PROC_DIR, "split_test.parquet")

# Tabular features — exactly the columns produced by 02_data_cleaning.py
FEATURE_COLS = [
    "log_amount", "is_ACH", "is_Cheque", "is_CC",
    "is_Wire", "is_Bitcoin", "hour", "dow", "is_weekend",
    "is_cross_currency", "amount_bucket",
]
TARGET_COL = "Is Laundering"

# XGBoost hyperparameters
CLASS_RATIO      = 904          # 904 legitimate transactions per 1 fraud
XGBOOST_PARAMS   = {
    "n_estimators":       500,
    "max_depth":          6,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "scale_pos_weight":   CLASS_RATIO,
    "eval_metric":        "aucpr",
    "early_stopping_rounds": 20,
    "tree_method":        "hist",
    "random_state":       42,
    "n_jobs":             -1,
}

# Decision threshold — tuned for precision/recall balance at deployment
DEFAULT_THRESHOLD = 0.30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(path: str, feature_cols: list, target_col: str, split_name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Split not found: {path}\n"
            "Run notebooks/02_data_cleaning.py first."
        )
    df = pd.read_parquet(path)

    # Validate columns
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{split_name}] Missing columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    df = df.copy()
    if "amount_bucket" in df.columns:
        df["amount_bucket"] = df["amount_bucket"].astype("category").cat.codes
    X = df[feature_cols].astype("float32")
    y = df[target_col].astype("int8")
    log.info("[%s] %d rows | fraud: %d (%.4f%%)", split_name, len(df), y.sum(), 100 * y.mean())
    return X, y


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X, y, split_name: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc  = average_precision_score(y, y_prob)
    roc_auc = roc_auc_score(y, y_prob)
    f1      = f1_score(y, y_pred, zero_division=0)
    report  = classification_report(y, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "split":     split_name,
        "threshold": threshold,
        "pr_auc":    round(pr_auc, 6),
        "roc_auc":   round(roc_auc, 6),
        "f1_fraud":  round(f1, 6),
        "precision": round(report["1"]["precision"], 6),
        "recall":    round(report["1"]["recall"], 6),
        "support_fraud": int(y.sum()),
        "support_total": int(len(y)),
    }

    log.info(
        "[%s] PR-AUC=%.4f | ROC-AUC=%.4f | F1=%.4f | P=%.4f | R=%.4f",
        split_name, pr_auc, roc_auc, f1,
        report["1"]["precision"], report["1"]["recall"],
    )
    return metrics, y_prob


def plot_confusion_matrix(model, X, y, threshold, out_path):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"XGBoost Baseline — Confusion Matrix (threshold={threshold})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Confusion matrix saved: %s", out_path)


def plot_pr_curve(model, X_val, y_val, out_path):
    y_prob = model.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    auc_score = average_precision_score(y_val, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, color="#1e88e5", label=f"PR-AUC = {auc_score:.4f}")
    ax.axhline(y=y_val.mean(), color="gray", linestyle="--", lw=1, label="Baseline (random)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("XGBoost Baseline — Precision-Recall Curve (Validation Set)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("PR curve saved: %s", out_path)


def compute_shap(model, X_val, out_path):
    """Compute SHAP values using XGBoost's built-in method (no shap package required)."""
    try:
        import shap
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_val)
        df_shap    = pd.DataFrame(shap_vals, columns=X_val.columns)
        df_shap.to_parquet(out_path, index=False)
        log.info("SHAP values saved: %s", out_path)

        # Print mean absolute SHAP (feature importance)
        mean_shap = df_shap.abs().mean().sort_values(ascending=False)
        print("\n  Top Features by Mean |SHAP|:")
        for feat, val in mean_shap.items():
            print(f"    {feat:<30} {val:.6f}")

    except ImportError:
        log.warning("shap package not installed — skipping SHAP computation. pip install shap")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  STAGE 6: XGBOOST BASELINE TRAINING")
    print("  Tabular features only | PR-AUC primary metric")
    print("  DATA 298A | Team 2 | Issues #1 & #2")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Load splits ---
    X_train, y_train = load_split(TRAIN_FILE, FEATURE_COLS, TARGET_COL, "train")
    X_val,   y_val   = load_split(VAL_FILE,   FEATURE_COLS, TARGET_COL, "val")
    X_test,  y_test  = load_split(TEST_FILE,  FEATURE_COLS, TARGET_COL, "test")

    # --- Train ---
    log.info("Training XGBoost baseline (scale_pos_weight=%d)...", CLASS_RATIO)
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    log.info("Best iteration: %d", model.best_iteration)

    # --- Evaluate on all three splits ---
    all_metrics = {}
    for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        m, _ = evaluate(model, X, y, split_name)
        all_metrics[split_name] = m

    # --- Plots ---
    plot_confusion_matrix(
        model, X_test, y_test,
        DEFAULT_THRESHOLD,
        os.path.join(MODEL_DIR, "xgboost_baseline_cm.png"),
    )
    plot_pr_curve(
        model, X_val, y_val,
        os.path.join(MODEL_DIR, "xgboost_baseline_pr_curve.png"),
    )

    # --- SHAP ---
    compute_shap(
        model, X_val,
        os.path.join(MODEL_DIR, "xgboost_baseline_shap.parquet"),
    )

    # --- Save model ---
    model_path = os.path.join(MODEL_DIR, "xgboost_baseline.json")
    model.save_model(model_path)
    log.info("Model saved: %s", model_path)

    # --- Save metrics ---
    metrics_path = os.path.join(MODEL_DIR, "xgboost_baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Metrics saved: %s", metrics_path)

    # --- Final summary ---
    test_m = all_metrics["test"]
    print("\n" + "=" * 70)
    print("  XGBOOST BASELINE — TEST SET RESULTS")
    print(f"  PR-AUC    : {test_m['pr_auc']:.4f}")
    print(f"  ROC-AUC   : {test_m['roc_auc']:.4f}")
    print(f"  F1 (fraud): {test_m['f1_fraud']:.4f}")
    print(f"  Precision : {test_m['precision']:.4f}")
    print(f"  Recall    : {test_m['recall']:.4f}")
    print(f"  Threshold : {DEFAULT_THRESHOLD}")
    print("=" * 70)
    print(f"\n  Artifacts saved to: {MODEL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()