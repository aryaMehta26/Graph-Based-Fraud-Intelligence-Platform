"""
Train and evaluate XGBoost with the leak-free pattern features on the clean splits.

Models (same hyperparameters as the team's 06/07 scripts):
  pattern        11 tabular + 28 pattern features        fully time-respecting
  pattern_graph  pattern + the 12 existing degree/Leiden features (transductive, as in 07)

Evaluation matches experiments/clean_split/eval_full.py: thresholds chosen on validation,
precision / recall / F1 / false positives on test, top-k alert budgets.
Writes data/models/*.json, results/results.json.
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "..", "clean_split", "data", "processed")
FEATS = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "data", "models")
OUT = os.path.join(HERE, "results")
os.makedirs(MODELS, exist_ok=True); os.makedirs(OUT, exist_ok=True)

TARGET = "Is Laundering"
ALERTS_PER_DAY = 1000
TABULAR = ["log_amount", "is_ACH", "is_Cheque", "is_CC", "is_Wire", "is_Bitcoin",
           "hour", "dow", "is_weekend", "is_cross_currency", "amount_bucket"]
GRAPH = ["src_out_degree", "src_in_degree", "src_total_degree", "src_degree_centrality",
         "dst_out_degree", "dst_in_degree", "dst_total_degree", "dst_degree_centrality",
         "src_community_size", "src_community_fraud_rate", "dst_community_size", "dst_community_fraud_rate"]
PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              scale_pos_weight=904, eval_metric="aucpr", early_stopping_rounds=20,
              tree_method="hist", random_state=42, n_jobs=-1)

t0 = time.time()
def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def load(split):
    base = pd.read_parquet(os.path.join(SPLITS, f"{split}_graph_enriched.parquet"),
                           columns=TABULAR + GRAPH + [TARGET, "Timestamp"])
    ref = pd.read_parquet(os.path.join(SPLITS, f"split_{split}.parquet"), columns=[TARGET, "Timestamp"])
    assert (base[TARGET].to_numpy() == ref[TARGET].to_numpy()).all() and \
           (base["Timestamp"].to_numpy() == ref["Timestamp"].to_numpy()).all(), f"{split}: row order differs"
    base["amount_bucket"] = base["amount_bucket"].astype("category").cat.codes
    pat = pd.read_parquet(os.path.join(FEATS, f"pattern_{split}.parquet"))
    assert len(pat) == len(base)
    df = pd.concat([base.reset_index(drop=True), pat], axis=1)
    return df


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
            "precision": prec, "recall": rec, "f1": f1, "accuracy": (tp + tn) / len(y)}


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


train, val, test = load("train"), load("val"), load("test")
PATTERN = [c for c in pd.read_parquet(os.path.join(FEATS, "pattern_val.parquet")).columns]
log(f"loaded train {len(train):,} val {len(val):,} test {len(test):,}; {len(PATTERN)} pattern features")

results = {}
for name, cols in [("pattern", TABULAR + PATTERN), ("pattern_graph", TABULAR + PATTERN + GRAPH)]:
    Xtr, Xva, Xte = (d[cols].astype("float32") for d in (train, val, test))
    ytr, yva, yte = (d[TARGET].to_numpy() for d in (train, val, test))
    log(f"[{name}] training on {len(cols)} features")
    m = XGBClassifier(**PARAMS)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=100)
    m.save_model(os.path.join(MODELS, f"xgboost_{name}.json"))
    p_val = m.predict_proba(Xva)[:, 1]; p_test = m.predict_proba(Xte)[:, 1]
    thr = choose_thresholds(yva, p_val, days_spanned(val["Timestamp"]))
    order = np.argsort(-p_test)
    topk = {str(k): {"alerts": k, "tp": int(yte[order[:k]].sum()), "precision": int(yte[order[:k]].sum()) / k,
                     "recall": int(yte[order[:k]].sum()) / int(yte.sum())}
            for k in [500, 1000, 2000, 5000, 10000, 50000]}
    imp = pd.Series(m.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    results[name] = {
        "features": cols, "best_iteration": int(m.best_iteration),
        "val_pr_auc": float(average_precision_score(yva, p_val)),
        "pr_auc": float(average_precision_score(yte, p_test)),
        "roc_auc": float(roc_auc_score(yte, p_test)),
        "thresholds_chosen_on_val": thr,
        "test_at_thresholds": {k: (at_threshold(yte, p_test, t) if t is not None else None) for k, t in thr.items()},
        "test_top_k": topk,
        "top_gain_features": {k: float(v) for k, v in imp.head(15).items()},
    }
    r = results[name]
    log(f"[{name}] best iter {r['best_iteration']}  val PR-AUC {r['val_pr_auc']:.4f}  "
        f"test PR-AUC {r['pr_auc']:.4f}  ROC-AUC {r['roc_auc']:.4f}")
    for k, v in r["test_at_thresholds"].items():
        if v is None:
            print(f"  {k:28} not reachable on validation"); continue
        print(f"  {k:28} thr {v['threshold']:.4f}  alerts {v['alerts']:>9,}  TP {v['tp']:>5,}  FP {v['fp']:>9,}  "
              f"precision {v['precision']:6.2%}  recall {v['recall']:6.2%}  F1 {v['f1']:.4f}")
    for k, v in topk.items():
        print(f"  top {int(k):>6,}: TP {v['tp']:>5,}  precision {v['precision']:6.2%}  recall {v['recall']:6.2%}")
    print("  top gain:", ", ".join(f"{k} {v:.0f}" for k, v in list(r["top_gain_features"].items())[:10]), flush=True)
    del Xtr, Xva, Xte, m

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
log("done")
