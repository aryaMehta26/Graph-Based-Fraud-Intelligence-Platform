"""
Case-level evaluation: does an alert reach each laundering account / laundering attempt?

Banks investigate accounts and schemes, not single payments: one alert on a scheme leads the
analyst to the rest of it. So besides transaction recall we report
  account recall  share of accounts sending laundering in test with at least one alerted laundering payment
  attempt recall  share of labelled laundering attempts (HI-Medium_Patterns.txt) with at least one
                  alerted payment in test, overall and per typology
Thresholds are the validation-chosen ones from results/results.json.
"""

import json
import os
import re

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "..", "clean_split", "data", "processed")
PATTERNS = os.path.expanduser("~/.cache/kagglehub/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/"
                              "versions/8/HI-Medium_Patterns.txt")
TARGET = "Is Laundering"
results = json.load(open(os.path.join(HERE, "results", "results.json")))

# --- test rows with features and identifiers
ids = pd.read_parquet(os.path.join(SPLITS, "split_test.parquet"),
                      columns=["Timestamp", "src_acct", "dst_acct", "Amount Paid", TARGET])
base = pd.read_parquet(os.path.join(SPLITS, "test_graph_enriched.parquet"))
base["amount_bucket"] = base["amount_bucket"].astype("category").cat.codes
X = pd.concat([base.reset_index(drop=True), pd.read_parquet(os.path.join(HERE, "data", "pattern_test.parquet"))], axis=1)
y = ids[TARGET].to_numpy()

# --- attempts from the patterns file, matched to test rows
rows, attempt, typ = [], -1, None
for line in open(PATTERNS):
    line = line.strip()
    if line.startswith("BEGIN"):
        attempt += 1; typ = re.sub(r"BEGIN LAUNDERING ATTEMPT - ", "", line).split(":")[0].strip()
    elif line and not line.startswith("END"):
        f = line.split(",")
        rows.append((pd.Timestamp(f[0].replace("/", "-")), f[2], f[4], round(float(f[7]), 2), attempt, typ))
pat = pd.DataFrame(rows, columns=["Timestamp", "src_acct", "dst_acct", "amt", "attempt", "typology"])
ids["amt"] = ids["Amount Paid"].round(2)
ids["row"] = np.arange(len(ids))
m = ids[y == 1].merge(pat, on=["Timestamp", "src_acct", "dst_acct", "amt"], how="left").drop_duplicates("row")
print(f"test laundering rows {int(y.sum()):,}; matched to a labelled attempt {m['attempt'].notna().sum():,}; "
      f"attempts touching test {m['attempt'].nunique():,}")

out = {}
for name, r in results.items():
    model = XGBClassifier(); model.load_model(os.path.join(HERE, "data", "models", f"xgboost_{name}.json"))
    p = model.predict_proba(X[r["features"]].astype("float32"))[:, 1]
    out[name] = {}
    print(f"\n=== {name}")
    for key in ["f1_optimal_on_val", "precision_20pct_on_val", "budget_1000_per_day_on_val"]:
        t = r["thresholds_chosen_on_val"][key]
        flag = p >= t
        laund = ids[y == 1].assign(flag=flag[y == 1])
        acc = laund.groupby("src_acct")["flag"].any()
        mm = m.assign(flag=flag[m["row"].to_numpy()]).dropna(subset=["attempt"])
        att = mm.groupby("attempt").agg(flag=("flag", "any"), typology=("typology", "first"))
        by_typ = att.groupby("typology")["flag"].agg(["mean", "size"])
        res = {"threshold": t, "alerts": int(flag.sum()),
               "txn_recall": float(flag[y == 1].mean()),
               "account_recall": float(acc.mean()), "accounts": int(len(acc)),
               "attempt_recall": float(att["flag"].mean()), "attempts": int(len(att)),
               "attempt_recall_by_typology": {k: {"recall": float(v["mean"]), "n": int(v["size"])} for k, v in by_typ.iterrows()}}
        out[name][key] = res
        print(f"  {key:28} alerts {res['alerts']:>6,}  txn recall {res['txn_recall']:.1%}  "
              f"account recall {res['account_recall']:.1%} of {res['accounts']:,}  "
              f"attempt recall {res['attempt_recall']:.1%} of {res['attempts']:,}")
        print("    by typology: " + ", ".join(f"{k} {v['mean']:.0%} (n={int(v['size'])})" for k, v in by_typ.iterrows()))

json.dump(out, open(os.path.join(HERE, "results", "case_level.json"), "w"), indent=2)
