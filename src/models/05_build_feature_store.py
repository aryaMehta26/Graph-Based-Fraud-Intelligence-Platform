"""
DATA 298A — Graph-Based Fraud Intelligence Platform
Team 2 | San José State University

05_build_feature_store.py
--------------------------
Closes Issue #5.

Merges the graph-derived account features (degree centrality, Louvain
community membership) produced by 04_extract_graph_features.py back into
the three transaction parquet splits (train / val / test).

The join is performed twice per split — once on src_acct (sender) and once
on dst_acct (receiver) — so each transaction row picks up graph context for
both parties involved.

Output
------
data/processed/
    train_graph_enriched.parquet
    val_graph_enriched.parquet
    test_graph_enriched.parquet

New columns added (prefixed src_ / dst_):
    src_out_degree, src_in_degree, src_total_degree, src_degree_centrality,
    src_community_id, src_community_size, src_community_fraud_rate,
    dst_out_degree, dst_in_degree, dst_total_degree, dst_degree_centrality,
    dst_community_id, dst_community_size, dst_community_fraud_rate

Usage
-----
    python3 src/models/05_build_feature_store.py

Prerequisites
-------------
    - 02_data_cleaning.py has run (train/val/test parquet splits exist)
    - 04_extract_graph_features.py has run (graph_features_accounts.csv exists)

Transductive Learning Note
--------------------------
Graph features joined here were computed on the full 27-day graph. Train rows
therefore receive degree information derived from future val/test edges. This
is standard transductive graph inference but is not strictly split-safe.
See 04_extract_graph_features.py for details and a proposed future fix.
"""

import csv
import os
import logging
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJ      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR  = os.path.join(PROJ, "data", "processed")

GRAPH_FEATURES_FILE = os.path.join(PROC_DIR, "graph_features_accounts.csv")

SPLITS = {
    "train": os.path.join(PROC_DIR, "split_train.parquet"),
    "val":   os.path.join(PROC_DIR, "split_val.parquet"),
    "test":  os.path.join(PROC_DIR, "split_test.parquet"),
}

OUTPUT_FILES = {
    "train": os.path.join(PROC_DIR, "train_graph_enriched.parquet"),
    "val":   os.path.join(PROC_DIR, "val_graph_enriched.parquet"),
    "test":  os.path.join(PROC_DIR, "test_graph_enriched.parquet"),
}

# Graph feature columns produced by 04_extract_graph_features.py and
# 04b_louvain_communities.py (community columns).
GRAPH_COLS = [
    "account_id",
    "out_degree",
    "in_degree",
    "total_degree",
    "degree_centrality",
    "community_id",
    "community_size",
    "community_fraud_rate",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_paths():
    missing = []
    if not os.path.exists(GRAPH_FEATURES_FILE):
        missing.append(GRAPH_FEATURES_FILE)
    for split, path in SPLITS.items():
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Required input files not found. Run prerequisite scripts first.\n"
            + "\n".join(f"  MISSING: {p}" for p in missing)
        )


def load_graph_features() -> pd.DataFrame:
    # Read only the columns that exist in the CSV (community columns present
    # only after 04b_louvain_communities.py has run)
    with open(GRAPH_FEATURES_FILE, "r") as f:
        header = next(csv.reader(f))
    DEGREE_COLS    = ["out_degree", "in_degree", "total_degree", "degree_centrality"]
    COMMUNITY_COLS = ["community_id", "community_size", "community_fraud_rate"]

    available_cols    = [c for c in GRAPH_COLS if c in header]
    missing_community = [c for c in COMMUNITY_COLS if c not in header]
    missing_degree    = [c for c in DEGREE_COLS if c not in header]

    if missing_degree:
        raise ValueError(
            f"Required degree columns missing from graph_features_accounts.csv: {missing_degree}\n"
            f"Detected columns: {header}\n"
            "Run 04_extract_graph_features.py to regenerate the file."
        )

    df = pd.read_csv(GRAPH_FEATURES_FILE, usecols=available_cols)
    log.info("Graph features loaded: %d accounts | columns: %s", len(df), available_cols)

    df["out_degree"]        = df["out_degree"].astype("int32")
    df["in_degree"]         = df["in_degree"].astype("int32")
    df["total_degree"]      = df["total_degree"].astype("int32")
    df["degree_centrality"] = df["degree_centrality"].astype("float32")

    if missing_community:
        log.warning("Community columns not found — stubbing to 0. Run 04b_louvain_communities.py to populate: %s", missing_community)
        for col in missing_community:
            df[col] = 0

    df["community_id"]         = df["community_id"].astype("int32")
    df["community_size"]       = df["community_size"].astype("int32")
    df["community_fraud_rate"] = df["community_fraud_rate"].astype("float32")
    return df


def enrich_split(df_txn: pd.DataFrame, df_graph: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Join graph features onto a transaction split twice:
      - src_acct → all columns prefixed with src_
      - dst_acct → all columns prefixed with dst_
    """
    n_before = len(df_txn)

    all_feature_cols = [c for c in df_graph.columns if c != "account_id"]

    # Sender-side join
    src_cols = {c: f"src_{c}" for c in all_feature_cols}
    df_src = df_graph.rename(columns={"account_id": "src_acct", **src_cols})
    df_txn = df_txn.merge(df_src, on="src_acct", how="left")

    # Receiver-side join
    dst_cols = {c: f"dst_{c}" for c in all_feature_cols}
    df_dst = df_graph.rename(columns={"account_id": "dst_acct", **dst_cols})
    df_txn = df_txn.merge(df_dst, on="dst_acct", how="left")

    # Fill accounts that had no graph entry (shouldn't happen on full graph,
    # but safe for partial/demo loads)
    fill_int   = [c for c in df_txn.columns if any(c.endswith(s) for s in ["_degree", "_community_id", "_community_size"])]
    fill_float = [c for c in df_txn.columns if c.endswith("_centrality") or c.endswith("_fraud_rate")]

    df_txn[fill_int]   = df_txn[fill_int].fillna(0).astype("int32")
    df_txn[fill_float] = df_txn[fill_float].fillna(0.0).astype("float32")

    if len(df_txn) != n_before:
        raise ValueError(
            f"[{split_name}] Row count changed after join ({n_before} → {len(df_txn)}) — "
            "check for duplicate account_ids in graph_features_accounts.csv."
        )

    log.info(
        "[%s] Enriched %d rows | New graph columns added: %d",
        split_name, len(df_txn), len(all_feature_cols) * 2,
    )
    return df_txn


def print_feature_summary(df: pd.DataFrame, split_name: str):
    fraud_df = df[df["Is Laundering"] == 1]
    legit_df = df[df["Is Laundering"] == 0]

    print(f"\n  [{split_name.upper()}] Graph Feature Summary")
    print(f"  {'Feature':<30} {'Fraud Mean':>12} {'Legit Mean':>12}")
    print("  " + "-" * 56)
    graph_feature_cols = [
        "src_total_degree", "dst_total_degree",
        "src_community_fraud_rate", "dst_community_fraud_rate",
        "src_community_size", "dst_community_size",
    ]
    for col in graph_feature_cols:
        if col in df.columns:
            fraud_mean = fraud_df[col].mean()
            legit_mean = legit_df[col].mean()
            print(f"  {col:<30} {fraud_mean:>12.4f} {legit_mean:>12.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  STAGE 5: FEATURE STORE — GRAPH ENRICHMENT JOIN")
    print("  Merging graph features into transaction splits")
    print("  DATA 298A | Team 2 | Issue #5")
    print("=" * 70)

    validate_paths()
    df_graph = load_graph_features()

    for split_name, in_path in SPLITS.items():
        log.info("Processing split: %s", split_name)
        df_txn = pd.read_parquet(in_path)
        log.info("  Loaded %d rows", len(df_txn))

        df_enriched = enrich_split(df_txn, df_graph, split_name)
        print_feature_summary(df_enriched, split_name)

        out_path = OUTPUT_FILES[split_name]
        df_enriched.to_parquet(out_path, index=False)
        log.info("  Saved: %s (%.1f MB)", out_path, os.path.getsize(out_path) / 1e6)

    print("\n" + "=" * 70)
    print("  FEATURE STORE BUILD COMPLETE")
    print("  Outputs:")
    for split_name, out_path in OUTPUT_FILES.items():
        if os.path.exists(out_path):
            size_mb = os.path.getsize(out_path) / 1e6
            print(f"    {split_name:<6} → {out_path}  ({size_mb:.1f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
