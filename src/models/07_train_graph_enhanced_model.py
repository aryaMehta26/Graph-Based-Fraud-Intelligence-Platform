"""
DATA 298A — Graph-Based Fraud Intelligence Platform
Team 2 | San José State University

07_train_graph_enhanced_model.py
---------------------------------
Closes Issue #6.

Trains an XGBoost classifier on the graph-enriched feature store produced by
05_build_feature_store.py. Every transaction row now includes structural
features from Neo4j/parquets (degree centrality + Leiden community features)
for both the sender and receiver accounts. The model uses
community_size/community_fraud_rate; community_id is joined upstream but
intentionally excluded from training.

This is the model that proves the core thesis: adding graph structure
improves fraud detection beyond what tabular features alone can achieve.

At the end the script loads the saved baseline metrics from
xgboost_baseline_metrics.json and prints a head-to-head comparison table.

Output
------
data/models/
    xgboost_graph_enhanced.json
    xgboost_graph_enhanced_metrics.json
    xgboost_graph_enhanced_shap.parquet
    xgboost_graph_enhanced_cm.png
    xgboost_graph_enhanced_pr_curve.png
    model_comparison.json               ← baseline vs graph-enhanced delta

Usage
-----
    python3 src/models/07_train_graph_enhanced_model.py

Prerequisites
-------------
    - 04_extract_graph_features.py completed
    - 05_build_feature_store.py completed (graph-enriched parquets exist)
    - 06_train_xgboost_baseline.py completed (baseline metrics exist for comparison)
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

TRAIN_FILE = os.path.join(PROC_DIR, "train_graph_enriched.parquet")
VAL_FILE   = os.path.join(PROC_DIR, "val_graph_enriched.parquet")
TEST_FILE  = os.path.join(PROC_DIR, "test_graph_enriched.parquet")

BASELINE_METRICS_FILE = os.path.join(MODEL_DIR, "xgboost_baseline_metrics.json")

TARGET_COL = "Is Laundering"

# Original tabular features — identical to 06_train_xgboost_baseline.py
TABULAR_FEATURES = [
    "log_amount",
    "is_ACH",
    "is_Cheque",
    "is_CC",
    "is_Wire",
    "is_Bitcoin",
    "hour",
    "dow",
    "is_weekend",
    "is_cross_currency",
    "amount_bucket",
]

# Graph features added by 05_build_feature_store.py
# Community features populated by 04b_louvain_communities.py (Issue #4).
# Note: community_id is intentionally excluded — it is a high-cardinality
# integer label (up to 66k values) with no ordinal meaning. XGBoost would
# treat it as a numeric feature, producing arbitrary splits. The meaningful
# community signal is captured by community_size and community_fraud_rate.
GRAPH_FEATURES = [
    "src_out_degree",
    "src_in_degree",
    "src_total_degree",
    "src_degree_centrality",
    "dst_out_degree",
    "dst_in_degree",
    "dst_total_degree",
    "dst_degree_centrality",
    "src_community_size",
    "src_community_fraud_rate",
    "dst_community_size",
    "dst_community_fraud_rate",
]

ALL_FEATURES = TABULAR_FEATURES + GRAPH_FEATURES

CLASS_RATIO    = 904
XGBOOST_PARAMS = {
    "n_estimators":          500,
    "max_depth":             6,
    "learning_rate":         0.05,
    "subsample":             0.8,
    "colsample_bytree":      0.8,
    "scale_pos_weight":      CLASS_RATIO,
    "eval_metric":           "aucpr",
    "early_stopping_rounds": 20,
    "tree_method":           "hist",
    "random_state":          42,
    "n_jobs":                -1,
}

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
            f"Graph-enriched split not found: {path}\n"
            "Run 04_extract_graph_features.py and 05_build_feature_store.py first."
        )
    df = pd.read_parquet(path)

    # Only use features that actually exist (graceful if graph features are
    # absent e.g. during demo with partial data)
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.warning("[%s] %d features not found, skipping: %s", split_name, len(missing), missing)

    df = df.copy()
    if "amount_bucket" in df.columns:
        df["amount_bucket"] = df["amount_bucket"].astype("category").cat.codes
    X = df[available].astype("float32")
    y = df[target_col].astype("int8")
    log.info(
        "[%s] %d rows | %d features | fraud: %d (%.4f%%)",
        split_name, len(df), len(available), y.sum(), 100 * y.mean(),
    )
    return X, y, available


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
        "split":          split_name,
        "threshold":      threshold,
        "pr_auc":         round(pr_auc,  6),
        "roc_auc":        round(roc_auc, 6),
        "f1_fraud":       round(f1, 6),
        "precision":      round(report["1"]["precision"], 6),
        "recall":         round(report["1"]["recall"],    6),
        "support_fraud":  int(y.sum()),
        "support_total":  int(len(y)),
    }

    log.info(
        "[%s] PR-AUC=%.4f | ROC-AUC=%.4f | F1=%.4f | P=%.4f | R=%.4f",
        split_name, pr_auc, roc_auc, f1,
        report["1"]["precision"], report["1"]["recall"],
    )
    return metrics, y_prob


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(model, X, y, threshold, out_path):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Greens")
    ax.set_title(f"Graph-Enhanced XGBoost — Confusion Matrix (threshold={threshold})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Confusion matrix saved: %s", out_path)


def plot_pr_curve(model, X_val, y_val, out_path):
    y_prob = model.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    auc_score = average_precision_score(y_val, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, color="#43a047", label=f"PR-AUC = {auc_score:.4f} (graph-enhanced)")
    ax.axhline(y=y_val.mean(), color="gray", linestyle="--", lw=1, label="Baseline (random)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Graph-Enhanced XGBoost — Precision-Recall Curve (Validation Set)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("PR curve saved: %s", out_path)


def compute_shap(model, X_val, out_path):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_val)
        df_shap   = pd.DataFrame(shap_vals, columns=X_val.columns)
        df_shap.to_parquet(out_path, index=False)
        log.info("SHAP values saved: %s", out_path)

        mean_shap = df_shap.abs().mean().sort_values(ascending=False)
        print("\n  Top Features by Mean |SHAP| (Graph-Enhanced Model):")
        for feat, val in mean_shap.head(15).items():
            marker = " ◄ graph" if any(feat.startswith(p) for p in ["src_", "dst_"]) else ""
            print(f"    {feat:<35} {val:.6f}{marker}")

    except ImportError:
        log.warning("shap package not installed — skipping SHAP. pip install shap")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def build_comparison(baseline_path: str, graph_metrics: dict) -> dict:
    if not os.path.exists(baseline_path):
        log.warning("Baseline metrics not found — skipping comparison.")
        return {}

    with open(baseline_path) as f:
        baseline = json.load(f)

    comparison = {}
    for split in ["train", "val", "test"]:
        if split not in baseline or split not in graph_metrics:
            continue
        b = baseline[split]
        g = graph_metrics[split]
        comparison[split] = {
            "baseline_pr_auc":      b["pr_auc"],
            "graph_pr_auc":         g["pr_auc"],
            "pr_auc_delta":         round(g["pr_auc"]  - b["pr_auc"],  6),
            "baseline_f1":          b["f1_fraud"],
            "graph_f1":             g["f1_fraud"],
            "f1_delta":             round(g["f1_fraud"] - b["f1_fraud"], 6),
            "baseline_recall":      b["recall"],
            "graph_recall":         g["recall"],
            "recall_delta":         round(g["recall"]   - b["recall"],   6),
        }
    return comparison


def print_comparison_table(comparison: dict):
    if not comparison:
        return
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON: TABULAR BASELINE vs GRAPH-ENHANCED")
    print("=" * 70)
    print(f"  {'Metric':<20} {'Split':<8} {'Baseline':>10} {'Graph':>10} {'Delta':>10}")
    print("  " + "-" * 60)
    for split, vals in comparison.items():
        for metric_key, label in [("pr_auc", "PR-AUC"), ("f1", "F1"), ("recall", "Recall")]:
            b_key = f"baseline_{metric_key}"
            g_key = f"graph_{metric_key}"
            d_key = f"{metric_key}_delta"
            if b_key not in vals:
                continue
            delta = vals[d_key]
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            print(
                f"  {label:<20} {split:<8} {vals[b_key]:>10.4f} {vals[g_key]:>10.4f} {delta_str:>10}"
            )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  STAGE 7: GRAPH-ENHANCED XGBOOST TRAINING")
    print("  Tabular + Graph features | PR-AUC primary metric")
    print("  DATA 298A | Team 2 | Issue #6")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Load graph-enriched splits ---
    X_train, y_train, train_feats = load_split(TRAIN_FILE, ALL_FEATURES, TARGET_COL, "train")
    X_val,   y_val,   _           = load_split(VAL_FILE,   ALL_FEATURES, TARGET_COL, "val")
    X_test,  y_test,  _           = load_split(TEST_FILE,  ALL_FEATURES, TARGET_COL, "test")

    graph_feats_used = [f for f in train_feats if f in GRAPH_FEATURES]
    log.info("Graph features in use: %d — %s", len(graph_feats_used), graph_feats_used)

    # --- Train ---
    log.info("Training graph-enhanced XGBoost (scale_pos_weight=%d)...", CLASS_RATIO)
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    log.info("Best iteration: %d", model.best_iteration)

    # --- Evaluate ---
    all_metrics = {}
    for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        m, _ = evaluate(model, X, y, split_name)
        all_metrics[split_name] = m

    # --- Save model + metrics (before SHAP so artifacts are never lost) ---
    model_path = os.path.join(MODEL_DIR, "xgboost_graph_enhanced.json")
    model.save_model(model_path)
    log.info("Model saved: %s", model_path)

    metrics_path = os.path.join(MODEL_DIR, "xgboost_graph_enhanced_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Metrics saved: %s", metrics_path)

    # --- Compare to baseline ---
    comparison = build_comparison(BASELINE_METRICS_FILE, all_metrics)
    if comparison:
        comparison_path = os.path.join(MODEL_DIR, "model_comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        log.info("Comparison saved: %s", comparison_path)
        print_comparison_table(comparison)

    # --- Plots ---
    plot_confusion_matrix(
        model, X_test, y_test,
        DEFAULT_THRESHOLD,
        os.path.join(MODEL_DIR, "xgboost_graph_enhanced_cm.png"),
    )
    plot_pr_curve(
        model, X_val, y_val,
        os.path.join(MODEL_DIR, "xgboost_graph_enhanced_pr_curve.png"),
    )

    # --- SHAP (slow — runs last so metrics are already saved above) ---
    compute_shap(
        model, X_val,
        os.path.join(MODEL_DIR, "xgboost_graph_enhanced_shap.parquet"),
    )

    # --- Final summary ---
    test_m = all_metrics["test"]
    print("\n" + "=" * 70)
    print("  GRAPH-ENHANCED XGBOOST — TEST SET RESULTS")
    print(f"  PR-AUC    : {test_m['pr_auc']:.4f}")
    print(f"  ROC-AUC   : {test_m['roc_auc']:.4f}")
    print(f"  F1 (fraud): {test_m['f1_fraud']:.4f}")
    print(f"  Precision : {test_m['precision']:.4f}")
    print(f"  Recall    : {test_m['recall']:.4f}")
    print(f"\n  Artifacts saved to: {MODEL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
