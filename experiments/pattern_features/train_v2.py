"""
Round 2 of XGBoost training on the clean splits.

  1. sweep scale_pos_weight in {1, 10, 50, 904} on the leak-free feature set
     (lr 0.1, early stopping on validation PR-AUC), keep the best
  2. final models with that weight, lr 0.05, up to 4,000 trees (no 500 cap):
       v2_leakfree   11 tabular + 28 pattern + 13 extra
       v2_graph      v2_leakfree + 12 degree/Leiden (transductive, as in 07)
Saves models, validation/test scores (for the account-level stage) and results/results_v2.json.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import evaluate_scores  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "..", "clean_split", "data", "processed")
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(DATA, "models")
SCORES = os.path.join(DATA, "scores")
OUT = os.path.join(HERE, "results")
for d in (MODELS, SCORES, OUT):
    os.makedirs(d, exist_ok=True)

TARGET = "Is Laundering"
TABULAR = ["log_amount", "is_ACH", "is_Cheque", "is_CC", "is_Wire", "is_Bitcoin",
           "hour", "dow", "is_weekend", "is_cross_currency", "amount_bucket"]
GRAPH = ["src_out_degree", "src_in_degree", "src_total_degree", "src_degree_centrality",
         "dst_out_degree", "dst_in_degree", "dst_total_degree", "dst_degree_centrality",
         "src_community_size", "src_community_fraud_rate", "dst_community_size", "dst_community_fraud_rate"]
BASE = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", max_depth=6,
            subsample=0.8, colsample_bytree=0.8, seed=42, nthread=-1)

t0 = time.time()
def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


PATTERN = list(pd.read_parquet(os.path.join(DATA, "pattern_val.parquet")).columns)
EXTRA = list(pd.read_parquet(os.path.join(DATA, "extra_val.parquet")).columns)
LEAKFREE = TABULAR + PATTERN + EXTRA


def load(split, cols):
    """float32 feature matrix with only the requested columns (keeps memory low)."""
    sources = [(os.path.join(SPLITS, f"{split}_graph_enriched.parquet"), [c for c in cols if c in TABULAR + GRAPH]),
               (os.path.join(DATA, f"pattern_{split}.parquet"), [c for c in cols if c in PATTERN]),
               (os.path.join(DATA, f"extra_{split}.parquet"), [c for c in cols if c in EXTRA])]
    blocks = []
    for path, want in sources:
        if not want:
            continue
        df = pd.read_parquet(path, columns=want)
        if "amount_bucket" in df:
            df["amount_bucket"] = df["amount_bucket"].astype("category").cat.codes
        blocks.append(df)
    df = pd.concat(blocks, axis=1)[cols]
    X = df.to_numpy(np.float32)
    del df, blocks
    return X


meta = {s: pd.read_parquet(os.path.join(SPLITS, f"split_{s}.parquet"), columns=[TARGET, "Timestamp"]) for s in ("train", "val", "test")}
y_tr, y_va, y_te = (meta[s][TARGET].to_numpy() for s in ("train", "val", "test"))
val_ts = meta["val"]["Timestamp"]
log(f"leak-free features {len(LEAKFREE)}")


def matrices(cols):
    X = load("train", cols); dtr = xgb.QuantileDMatrix(X, label=y_tr, feature_names=cols); del X
    X = load("val", cols); dva = xgb.QuantileDMatrix(X, label=y_va, feature_names=cols, ref=dtr); del X
    dte = xgb.DMatrix(load("test", cols), feature_names=cols)
    log(f"matrices built for {len(cols)} features")
    return dtr, dva, dte


def fit(dtr, dva, spw, lr, rounds, patience):
    params = dict(BASE, scale_pos_weight=spw, eta=lr)
    return xgb.train(params, dtr, rounds, evals=[(dva, "val")], early_stopping_rounds=patience, verbose_eval=250)


results = {"sweep": {}}

# 1. class-weight sweep on the leak-free set
dtr, dva, dte = matrices(LEAKFREE)
for spw in [1, 10, 50, 904]:
    b = fit(dtr, dva, spw, 0.1, 1500, 50)
    p_va = b.predict(dva, iteration_range=(0, b.best_iteration + 1))
    results["sweep"][str(spw)] = {"val_pr_auc": float(average_precision_score(y_va, p_va)), "best_iteration": b.best_iteration}
    log(f"sweep spw={spw}: val PR-AUC {results['sweep'][str(spw)]['val_pr_auc']:.4f} at iter {b.best_iteration}")
best_spw = int(max(results["sweep"], key=lambda k: results["sweep"][k]["val_pr_auc"]))
results["best_scale_pos_weight"] = best_spw
log(f"best scale_pos_weight = {best_spw}")

# 2. final models
for name, cols in [("v2_leakfree", LEAKFREE), ("v2_graph", LEAKFREE + GRAPH)]:
    if name != "v2_leakfree":
        del dtr, dva, dte
        dtr, dva, dte = matrices(cols)
    b = fit(dtr, dva, best_spw, 0.05, 4000, 100)
    b.save_model(os.path.join(MODELS, f"xgboost_{name}.json"))
    rng = (0, b.best_iteration + 1)
    p_va, p_te = b.predict(dva, iteration_range=rng), b.predict(dte, iteration_range=rng)
    np.save(os.path.join(SCORES, f"{name}_val.npy"), p_va); np.save(os.path.join(SCORES, f"{name}_test.npy"), p_te)
    r = evaluate_scores(y_va, p_va, y_te, p_te, val_ts)
    gain = pd.Series(b.get_score(importance_type="gain")).sort_values(ascending=False)
    r.update(features=cols, best_iteration=b.best_iteration, scale_pos_weight=best_spw,
             top_gain_features={k: float(x) for k, x in gain.head(15).items()})
    results[name] = r
    log(f"[{name}] iter {b.best_iteration}  val PR-AUC {r['val_pr_auc']:.4f}  test PR-AUC {r['pr_auc']:.4f}  ROC {r['roc_auc']:.4f}")
    for k, x in r["test_at_thresholds"].items():
        if x:
            print(f"  {k:28} alerts {x['alerts']:>8,}  TP {x['tp']:>5,}  FP {x['fp']:>8,}  "
                  f"precision {x['precision']:6.2%}  recall {x['recall']:6.2%}  F1 {x['f1']:.4f}")
    print("  top-k:", ", ".join(f"{k}: {x['precision']:.1%}/{x['recall']:.1%}" for k, x in r["test_top_k"].items()))
    print("  top gain:", ", ".join(list(r["top_gain_features"])[:10]), flush=True)
    json.dump(results, open(os.path.join(OUT, "results_v2.json"), "w"), indent=2)
log("done")
