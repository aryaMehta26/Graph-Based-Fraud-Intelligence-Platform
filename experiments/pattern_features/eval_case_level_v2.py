"""Scheme-level (attempt) recall for round-2 models from saved scores; same matching as eval_case_level.py."""

import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "..", "clean_split", "data", "processed")
PATTERNS = os.path.expanduser("~/.cache/kagglehub/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/"
                              "versions/8/HI-Medium_Patterns.txt")
TARGET = "Is Laundering"
SOURCES = {"v2_leakfree": ("results/results_v2.json", "v2_leakfree", "data/scores/v2_leakfree_test.npy"),
           "v2_graph": ("results/results_v2.json", "v2_graph", "data/scores/v2_graph_test.npy"),
           "gnn": ("../gnn/results/gnn_results.json", None, "../gnn/scores/gnn_test.npy")}

ids = pd.read_parquet(os.path.join(SPLITS, "split_test.parquet"), columns=["Timestamp", "src_acct", "dst_acct", "Amount Paid", TARGET])
y = ids[TARGET].to_numpy()
rows, attempt, typ = [], -1, None
for line in open(PATTERNS):
    line = line.strip()
    if line.startswith("BEGIN"):
        attempt += 1; typ = re.sub(r"BEGIN LAUNDERING ATTEMPT - ", "", line).split(":")[0].strip()
    elif line and not line.startswith("END"):
        f = line.split(",")
        rows.append((pd.Timestamp(f[0].replace("/", "-")), f[2], f[4], round(float(f[7]), 2), attempt, typ))
pat = pd.DataFrame(rows, columns=["Timestamp", "src_acct", "dst_acct", "amt", "attempt", "typology"])
ids["amt"] = ids["Amount Paid"].round(2); ids["row"] = np.arange(len(ids))
m = ids[y == 1].merge(pat, on=["Timestamp", "src_acct", "dst_acct", "amt"], how="left").drop_duplicates("row").dropna(subset=["attempt"])

out = {}
for name, (rpath, key, spath) in SOURCES.items():
    r = json.load(open(os.path.join(HERE, rpath))); r = r[key] if key else r
    p = np.load(os.path.join(HERE, spath))
    out[name] = {}
    print(f"=== {name}")
    for k in ["f1_optimal_on_val", "precision_20pct_on_val", "budget_1000_per_day_on_val"]:
        t = r["thresholds_chosen_on_val"][k]
        flag = p >= t
        att = m.assign(flag=flag[m["row"].to_numpy()]).groupby("attempt").agg(flag=("flag", "any"), typology=("typology", "first"))
        bt = att.groupby("typology")["flag"].mean()
        out[name][k] = {"alerts": int(flag.sum()), "txn_recall": float(flag[y == 1].mean()),
                        "attempt_recall": float(att["flag"].mean()), "by_typology": bt.to_dict()}
        print(f"  {k:28} alerts {flag.sum():>6,}  txn recall {flag[y == 1].mean():.1%}  scheme recall {att['flag'].mean():.1%}  "
              + ", ".join(f"{a} {b:.0%}" for a, b in bt.items()))
json.dump(out, open(os.path.join(HERE, "results", "case_level_v2.json"), "w"), indent=2)
