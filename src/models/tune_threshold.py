"""
tune_threshold.py
------------------
Finds the optimal decision threshold for the trained XGBoost baseline.
Plots precision, recall, and F1 across all thresholds so you can pick
the right operating point for your use case.

Usage
-----
    python3 src/models/tune_threshold.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

PROJ       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR   = os.path.join(PROJ, "data", "processed")
MODEL_DIR  = os.path.join(PROJ, "data", "models")

VAL_FILE   = os.path.join(PROC_DIR, "split_val.parquet")
MODEL_FILE = os.path.join(MODEL_DIR, "xgboost_baseline.json")

FEATURE_COLS = [
    "log_amount", "is_ACH", "is_Cheque", "is_CC",
    "is_Wire", "is_Bitcoin", "hour", "dow", "is_weekend",
    "is_cross_currency", "amount_bucket",
]
TARGET_COL = "Is Laundering"

# ---------------------------------------------------------------------------

def main():
    # Load val set
    df = pd.read_parquet(VAL_FILE)
    df["amount_bucket"] = df["amount_bucket"].astype("category").cat.codes
    X_val = df[FEATURE_COLS].astype("float32")
    y_val = df[TARGET_COL].astype("int8")

    # Load model
    model = XGBClassifier()
    model.load_model(MODEL_FILE)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Compute precision/recall/F1 across thresholds
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)

    f1_scores = []
    for p, r in zip(precision[:-1], recall[:-1]):
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1_scores.append(f1)
    f1_scores = np.array(f1_scores)

    best_idx       = f1_scores.argmax()
    best_threshold = thresholds[best_idx]
    best_f1        = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall    = recall[best_idx]

    print("=" * 60)
    print("  THRESHOLD TUNING RESULTS (Validation Set)")
    print("=" * 60)
    print(f"\n  Current threshold : 0.30")
    print(f"  Optimal threshold : {best_threshold:.4f}")
    print(f"  At optimal threshold:")
    print(f"    F1        : {best_f1:.4f}")
    print(f"    Precision : {best_precision:.4f}")
    print(f"    Recall    : {best_recall:.4f}")

    # Also show a few candidate thresholds
    print(f"\n  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 44)
    candidates = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, best_threshold]
    seen = set()
    for t in sorted(set(round(c, 2) for c in candidates)):
        idx = np.searchsorted(thresholds, t)
        if idx >= len(thresholds):
            continue
        p = precision[idx]
        r = recall[idx]
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        marker = " ← optimal" if abs(t - round(best_threshold, 2)) < 0.01 else ""
        print(f"  {t:>10.2f} {p:>10.4f} {r:>10.4f} {f:>10.4f}{marker}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: P/R/F1 vs threshold
    ax = axes[0]
    ax.plot(thresholds, precision[:-1], label="Precision", color="#1e88e5")
    ax.plot(thresholds, recall[:-1],    label="Recall",    color="#e53935")
    ax.plot(thresholds, f1_scores,      label="F1",        color="#43a047", lw=2)
    ax.axvline(best_threshold, color="gray", linestyle="--", label=f"Optimal ({best_threshold:.2f})")
    ax.axvline(0.30, color="orange", linestyle=":", label="Current (0.30)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: PR curve
    ax = axes[1]
    pr_auc = average_precision_score(y_val, y_prob)
    ax.plot(recall, precision, color="#1e88e5", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.scatter([best_recall], [best_precision], color="#43a047", s=100, zorder=5, label=f"Optimal threshold ({best_threshold:.2f})")
    ax.axhline(y_val.mean(), color="gray", linestyle="--", lw=1, label="Random baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(MODEL_DIR, "threshold_tuning.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {out_path}")

    # Save optimal threshold back to metrics
    metrics_path = os.path.join(MODEL_DIR, "xgboost_baseline_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["optimal_threshold"] = round(float(best_threshold), 4)
        metrics["optimal_f1"]        = round(float(best_f1), 4)
        metrics["optimal_precision"] = round(float(best_precision), 4)
        metrics["optimal_recall"]    = round(float(best_recall), 4)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Optimal threshold saved to: {metrics_path}")

    print("\n  → Update DEFAULT_THRESHOLD in 06 and 07 to the optimal value above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
