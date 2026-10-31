"""Shared evaluation: thresholds chosen on validation, applied to test (same rules as clean_split/eval_full.py)."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ALERTS_PER_DAY = 1000


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


def evaluate_scores(y_val, p_val, y_test, p_test, val_timestamps):
    thr = choose_thresholds(y_val, p_val, days_spanned(val_timestamps))
    order = np.argsort(-p_test)
    pos = int(y_test.sum())
    topk = {}
    for k in [500, 1000, 2000, 5000, 10000, 50000]:
        tp = int(y_test[order[:k]].sum())
        topk[str(k)] = {"alerts": k, "tp": tp, "precision": tp / k, "recall": tp / pos}
    return {
        "val_pr_auc": float(average_precision_score(y_val, p_val)),
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "thresholds_chosen_on_val": thr,
        "test_at_thresholds": {k: (at_threshold(y_test, p_test, t) if t is not None else None) for k, t in thr.items()},
        "test_top_k": topk,
    }
