"""
Rebuild the train/val/test splits without the laundering-only tail.

The IBM HI-Medium simulator stops generating normal transactions after
2022-09-16 while laundering chains continue to 2022-09-28 (6,987 rows, 58%
laundering). The original 70/15/15 split put that whole tail in test.

This script reproduces the split of notebooks/02_data_cleaning.py (sort by
Timestamp, cut at 70% and 85% of rows) on the normal-activity window only,
keeps the tail as a separate diagnostic file, and recomputes the degree
features that src/models/04_extract_graph_features.py reads from Neo4j, using
the same definitions on the same graph rows (self-loops removed).

Reads:  ../../data/processed/transactions_clean.parquet   (unchanged)
Writes: data/processed/split_{train,val,test}.parquet
        data/processed/tail_holdout.parquet
        data/processed/graph_features_accounts.csv          (degree columns)
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "data", "processed", "transactions_clean.parquet")
OUT = os.path.join(HERE, "data", "processed")
CUTOFF = pd.Timestamp("2022-09-17")

os.makedirs(OUT, exist_ok=True)
tx = pd.read_parquet(SRC)
tx = tx.sort_values("Timestamp", kind="stable").reset_index(drop=True)

tail = tx[tx["Timestamp"] >= CUTOFF]
tx = tx[tx["Timestamp"] < CUTOFF].reset_index(drop=True)
tail.to_parquet(os.path.join(OUT, "tail_holdout.parquet"), index=False)
print(f"tail held out: {len(tail):,} rows, {int(tail['Is Laundering'].sum()):,} laundering")

n = len(tx)
i70, i85 = int(n * 0.70), int(n * 0.85)
splits = {"train": tx.iloc[:i70], "val": tx.iloc[i70:i85], "test": tx.iloc[i85:]}
for name, df in splits.items():
    df.to_parquet(os.path.join(OUT, f"split_{name}.parquet"), index=False)
    f = int(df["Is Laundering"].sum())
    print(f"{name:5} {df['Timestamp'].min()} -> {df['Timestamp'].max()}  rows {len(df):,}  laundering {f:,} ({f / len(df):.4%})")

# Degree features: same definitions as the Neo4j extraction (SENT / RECEIVED_BY
# counts over the self-loop-free graph, centrality = total / (N - 1)).
g = tx.loc[tx["src_acct"] != tx["dst_acct"], ["src_acct", "dst_acct"]]
out_deg = g.groupby("src_acct").size().rename("out_degree")
in_deg = g.groupby("dst_acct").size().rename("in_degree")
deg = pd.concat([out_deg, in_deg], axis=1).fillna(0).astype("int64")
deg.index.name = "account_id"
deg["total_degree"] = deg["out_degree"] + deg["in_degree"]
n_accounts = len(deg)
deg["degree_centrality"] = deg["total_degree"] / (n_accounts - 1)
deg.reset_index().to_csv(os.path.join(OUT, "graph_features_accounts.csv"), index=False)
print(f"degree features: {n_accounts:,} accounts from {len(g):,} graph edges")
