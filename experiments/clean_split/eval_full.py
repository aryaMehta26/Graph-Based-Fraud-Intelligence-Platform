"""
Full evaluation of the clean-split XGBoost models (baseline and graph-enhanced).

Answers the Demo 1 feedback: precision, recall, F1, false-positive counts and
confusion matrices, not recall alone. Thresholds are chosen on the validation
split, never on test:
  - 0.30            team default used in DATA 298A
  - F1-optimal      maximises F1 on validation
  - precision >= 20% lowest validation threshold reaching 20% precision
  - alert budget    validation threshold that flags 1,000 alerts per day
Also reports test precision/recall at fixed top-k alert counts, and scores the
held-out laundering-only tail separately as a diagnostic.

Writes results/eval_full.json and prints a summary.
"""

import importlib.util
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "data", "processed")
MODELS = os.path.join(HERE, "data", "models")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
TARGET = "Is Laundering"
ALERTS_PER_DAY = 1000


def load_module(rel):
    spec = importlib.util.spec_from_file_location(os.path.basename(rel)[:-3].lstrip("0123456789_"), os.path.join(HERE, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def features(df, cols):
    df = df.copy()
    if "amount_bucket" in df.columns:
        df["amount_bucket"] = df["amount_bucket"].astype("category").cat.codes
    return df[cols].astype("float32")


def days_spanned(ts):
    ts = pd.to_datetime(ts)
    return max((ts.max() - ts.min()).total_seconds() / 86400.0, 1e-9)


def at_threshold(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": float(t), "alerts": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


def choose_thresholds(y_val, p_val, val_days):
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    prec, rec = prec[:-1], rec[:-1]
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
    out = {"default_0.30": 0.30, "f1_optimal_on_val": float(thr[np.argmax(f1)])}
    ok = np.where(prec >= 0.20)[0]
    out["precision_20pct_on_val"] = float(thr[ok[0]]) if len(ok) else None
    k = int(round(ALERTS_PER_DAY * val_days))
    out[f"budget_{ALERTS_PER_DAY}_per_day_on_val"] = float(np.sort(p_val)[::-1][min(k, len(p_val)) - 1])
    return out


def evaluate(name, model_path, val_df, test_df, tail_df, cols):
    m = XGBClassifier(); m.load_model(model_path)
    cols = [c for c in m.get_booster().feature_names]
    p_val = m.predict_proba(features(val_df, cols))[:, 1]
    p_test = m.predict_proba(features(test_df, cols))[:, 1]
    p_tail = m.predict_proba(features(tail_df, cols))[:, 1]
    y_val = val_df[TARGET].to_numpy(); y_test = test_df[TARGET].to_numpy(); y_tail = tail_df[TARGET].to_numpy()
    test_days = days_spanned(test_df["Timestamp"])
    thr = choose_thresholds(y_val, p_val, days_spanned(val_df["Timestamp"]))
    order = np.argsort(-p_test)
    topk = {}
    for k in [500, 1000, 2000, 5000, 10000, 50000]:
        tp = int(y_test[order[:k]].sum())
        topk[str(k)] = {"alerts": k, "tp": tp, "precision": tp / k, "recall": tp / int(y_test.sum())}
    return {
        "model": name,
        "test_rows": int(len(y_test)), "test_laundering": int(y_test.sum()), "test_days": test_days,
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "thresholds_chosen_on_val": thr,
        "test_at_thresholds": {k: (at_threshold(y_test, p_test, t) if t is not None else None) for k, t in thr.items()},
        "test_top_k": topk,
        "tail_diagnostic": {"rows": int(len(y_tail)), "laundering": int(y_tail.sum()),
                            "pr_auc": float(average_precision_score(y_tail, p_tail)),
                            "at_0.30": at_threshold(y_tail, p_tail, 0.30)},
    }


fs = load_module("src/models/05_build_feature_store.py")
graph_feats = fs.load_graph_features()
tail = pd.read_parquet(os.path.join(PROC, "tail_holdout.parquet"))

results = {
    "baseline": evaluate("Baseline XGBoost (11 tabular)", os.path.join(MODELS, "xgboost_baseline.json"),
                         pd.read_parquet(os.path.join(PROC, "split_val.parquet")),
                         pd.read_parquet(os.path.join(PROC, "split_test.parquet")), tail, None),
    "graph": evaluate("Graph-enhanced XGBoost (23)", os.path.join(MODELS, "xgboost_graph_enhanced.json"),
                      pd.read_parquet(os.path.join(PROC, "val_graph_enriched.parquet")),
                      pd.read_parquet(os.path.join(PROC, "test_graph_enriched.parquet")),
                      fs.enrich_split(tail, graph_feats, "tail"), None),
}
with open(os.path.join(OUT, "eval_full.json"), "w") as f:
    json.dump(results, f, indent=2)

for key, r in results.items():
    print(f"\n=== {r['model']}  |  test {r['test_rows']:,} rows, {r['test_laundering']:,} laundering, {r['test_days']:.2f} days")
    print(f"PR-AUC {r['pr_auc']:.4f}   ROC-AUC {r['roc_auc']:.4f}")
    for k, v in r["test_at_thresholds"].items():
        if v is None:
            print(f"  {k:32} not reachable on validation"); continue
        print(f"  {k:32} thr {v['threshold']:.4f}  alerts {v['alerts']:>9,}  TP {v['tp']:>5,}  FP {v['fp']:>9,}  FN {v['fn']:>5,}  "
              f"precision {v['precision']:6.2%}  recall {v['recall']:6.2%}  F1 {v['f1']:.4f}")
    for k, v in r["test_top_k"].items():
        print(f"  top {int(k):>6,}: TP {v['tp']:>5,}  precision {v['precision']:6.2%}  recall {v['recall']:6.2%}")
    t = r["tail_diagnostic"]
    print(f"  tail diagnostic: {t['rows']:,} rows, {t['laundering']:,} laundering, PR-AUC {t['pr_auc']:.3f}, "
          f"precision@0.30 {t['at_0.30']['precision']:.2%}, recall@0.30 {t['at_0.30']['recall']:.2%}")
