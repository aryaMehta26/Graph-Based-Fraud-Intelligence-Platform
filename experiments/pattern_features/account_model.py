"""
Step 5: account-level model (batch review of accounts over a period, as banks run case review).

Stage 1 = a transaction model's scores (data/scores/<name>_{val,test}.npy; the model was trained on the
train period, so validation and test scores are out-of-sample).
Stage 2 = XGBoost over accounts. One row per account active in the period:
  own activity   payments sent / received, max / mean / top-3 stage-1 score on each side,
                 payments above several score levels
  neighbourhood  max and mean of counterparties' max score (1-hop propagation)
Label: the account sent or received at least one laundering payment in that period.
Trained on validation-period accounts, evaluated on test-period accounts. Thresholds for the account
model come from 5-fold out-of-fold scores on validation.

Reports account precision / recall / F1 and the laundering transactions an analyst reaches by opening
every flagged account (a laundering payment is reached if its sender or receiver is flagged).
Usage: python account_model.py v2_graph
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import at_threshold  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "..", "clean_split", "data", "processed")
NAME = sys.argv[1] if len(sys.argv) > 1 else "v2_graph"
TARGET = "Is Laundering"
PARAMS = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", max_depth=6, eta=0.05,
              subsample=0.8, colsample_bytree=0.8, seed=42, nthread=-1)
ROUNDS = 600


def account_table(split):
    tx = pd.read_parquet(os.path.join(SPLITS, f"split_{split}.parquet"), columns=["src_acct", "dst_acct", TARGET])
    tx["p"] = np.load(os.path.join(HERE, "data", "scores", f"{NAME}_{split}.npy"))
    for q in (0.5, 0.9, 0.99):
        tx[f"a{q}"] = (tx["p"] >= q).astype(np.int32)
    sides = []
    for side, acct in [("out", "src_acct"), ("in", "dst_acct")]:
        g = tx.groupby(acct)
        top3 = tx.sort_values("p", ascending=False).groupby(acct).head(3).groupby(acct)["p"].mean()
        f = pd.DataFrame({
            f"{side}_n": g.size(),
            f"{side}_max": g["p"].max(),
            f"{side}_mean": g["p"].mean(),
            f"{side}_top3": top3,
            **{f"{side}_n_above_{q}": g[f"a{q}"].sum() for q in (0.5, 0.9, 0.99)},
            f"{side}_label": g[TARGET].max(),
        })
        f.index.name = "acct"
        sides.append(f)
    acc = sides[0].join(sides[1], how="outer").fillna(0)
    acc["label"] = np.maximum(acc.pop("out_label"), acc.pop("in_label")).astype(int)
    acc["max_any"] = acc[["out_max", "in_max"]].max(1)
    # 1-hop: counterparties' max score
    cp = pd.concat([tx[["src_acct", "dst_acct"]].rename(columns={"src_acct": "a", "dst_acct": "b"}),
                    tx[["dst_acct", "src_acct"]].rename(columns={"dst_acct": "a", "src_acct": "b"})])
    cp = cp[cp["a"] != cp["b"]]
    cp["b_max"] = acc["max_any"].reindex(cp["b"]).to_numpy()
    g = cp.groupby("a")["b_max"]
    acc["nb_max"] = g.max().reindex(acc.index).fillna(0)
    acc["nb_mean"] = g.mean().reindex(acc.index).fillna(0)
    cp["b_hi"] = (cp["b_max"] >= 0.9).astype(np.int32)
    acc["nb_n_above_0.9"] = cp.groupby("a")["b_hi"].sum().reindex(acc.index).fillna(0)
    return acc, tx


print(f"stage-1 scores: {NAME}", flush=True)
acc_va, tx_va = account_table("val")
acc_te, tx_te = account_table("test")
FEATS = [c for c in acc_va.columns if c != "label"]
print(f"val accounts {len(acc_va):,} ({acc_va['label'].sum():,} laundering), "
      f"test accounts {len(acc_te):,} ({acc_te['label'].sum():,} laundering)", flush=True)

Xva, yva = acc_va[FEATS].to_numpy(np.float32), acc_va["label"].to_numpy()
oof = np.zeros(len(yva))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(Xva, yva):
    b = xgb.train(PARAMS, xgb.DMatrix(Xva[tr], label=yva[tr]), ROUNDS)
    oof[te] = b.predict(xgb.DMatrix(Xva[te]))
booster = xgb.train(PARAMS, xgb.DMatrix(Xva, label=yva, feature_names=FEATS), ROUNDS)
p_te = booster.predict(xgb.DMatrix(acc_te[FEATS].to_numpy(np.float32), feature_names=FEATS))
yte = acc_te["label"].to_numpy()

# simple baseline: rank accounts by their single highest transaction score
base_te = acc_te["max_any"].to_numpy()


def thresholds_from(y, p):
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    out = {"f1_optimal": float(thr[np.argmax(f1)])}
    for target in (0.5, 0.2):
        ok = np.where(prec >= target)[0]
        out[f"precision_{int(target * 100)}pct"] = float(thr[ok[0]]) if len(ok) else None
    return out


def reach(flagged_accts, tx):
    lab = tx[tx[TARGET] == 1]
    hit = lab["src_acct"].isin(flagged_accts) | lab["dst_acct"].isin(flagged_accts)
    return float(hit.mean())


res = {"stage1": NAME, "features": FEATS, "test_accounts": int(len(yte)), "test_laundering_accounts": int(yte.sum()),
       "val_oof_pr_auc": float(average_precision_score(yva, oof)),
       "test_pr_auc": float(average_precision_score(yte, p_te)),
       "test_pr_auc_max_score_baseline": float(average_precision_score(yte, base_te)),
       "operating_points": {}, "top_k_accounts": {}}
print(f"account PR-AUC: stage-2 {res['test_pr_auc']:.4f} (val OOF {res['val_oof_pr_auc']:.4f}), "
      f"max-score baseline {res['test_pr_auc_max_score_baseline']:.4f}")
for k, t in thresholds_from(yva, oof).items():
    if t is None:
        continue
    m = at_threshold(yte, p_te, t)
    m["laundering_txns_reached"] = reach(set(acc_te.index[p_te >= t]), tx_te)
    res["operating_points"][k] = m
    print(f"  {k:16} accounts flagged {m['alerts']:>6,}  TP {m['tp']:>5,}  FP {m['fp']:>6,}  precision {m['precision']:6.2%}  "
          f"recall {m['recall']:6.2%}  F1 {m['f1']:.4f}  laundering txns reached {m['laundering_txns_reached']:.1%}")
order = np.argsort(-p_te)
for k in (500, 1000, 2000, 5000, 10000):
    top = order[:k]
    tp = int(yte[top].sum())
    res["top_k_accounts"][str(k)] = {"precision": tp / k, "recall": tp / int(yte.sum()),
                                     "laundering_txns_reached": reach(set(acc_te.index[top]), tx_te)}
    r = res["top_k_accounts"][str(k)]
    print(f"  top {k:>6,} accounts: precision {r['precision']:6.2%}  account recall {r['recall']:6.2%}  "
          f"laundering txns reached {r['laundering_txns_reached']:.1%}")
json.dump(res, open(os.path.join(HERE, "results", f"account_{NAME}.json"), "w"), indent=2)
