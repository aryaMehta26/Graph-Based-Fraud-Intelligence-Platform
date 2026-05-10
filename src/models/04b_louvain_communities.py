"""
DATA 298A — Graph-Based Fraud Intelligence Platform
Team 2 | San José State University

04b_louvain_communities.py
---------------------------
Closes Issue #4.

Runs Leiden community detection (an improved Louvain variant) on the full
account-to-account transaction graph and computes per-account community
features that replace the zero stubs in graph_features_accounts.csv.

Why Leiden over Louvain
-----------------------
Leiden fixes a connectivity flaw in Louvain where communities can become
internally disconnected. It produces better-quality partitions at the same
computational cost and is the current standard in graph ML.

Algorithm
---------
1. Load all three splits (train/val/test) for graph topology only — no labels
   read here. This is the standard transductive setting for community detection.
   Load train split separately with labels to compute community_fraud_rate.
2. Build an undirected weighted account-to-account graph:
   - Nodes  : unique accounts
   - Edges  : one per unique (src_acct, dst_acct) pair, weighted by
              transaction count between those two accounts.
3. Run Leiden community detection (RBConfigurationVertexPartition).
4. For each community compute:
   - community_id         : integer community label
   - community_size       : number of member accounts
   - community_fraud_rate : fraction of TRAINING transactions (where src OR
                            dst is a member) flagged as laundering.
                            Train-only to avoid leaking val/test labels.
5. Merge these three columns into graph_features_accounts.csv, replacing
   the zero stubs written by 04_extract_graph_features.py.

Output
------
data/processed/graph_features_accounts.csv  (updated in-place)
    Adds / overwrites columns:
        community_id, community_size, community_fraud_rate

Note: Issue #4 originally specified separate communities.csv and
community_fraud_scores.csv outputs. These were consolidated into
graph_features_accounts.csv so that script 05 has a single join source
for all per-account graph features (degree + community).

Usage
-----
    python3 src/models/04b_louvain_communities.py

Prerequisites
-------------
    - 04_extract_graph_features.py completed (graph_features_accounts.csv exists)
    - 02_data_cleaning.py completed (split parquets exist)
    - pip install igraph leidenalg

After running
-------------
    Re-run 05_build_feature_store.py to rebuild the enriched parquets with
    real community features, then retrain 07_train_graph_enhanced_model.py.
"""

import os
import time
import logging
from collections import Counter
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJ     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR = os.path.join(PROJ, "data", "processed")

# All splits — used for graph topology only (no labels read)
GRAPH_SPLITS = [
    os.path.join(PROC_DIR, "split_train.parquet"),
    os.path.join(PROC_DIR, "split_val.parquet"),
    os.path.join(PROC_DIR, "split_test.parquet"),
]

# Train only — used to compute community_fraud_rate without leaking val/test labels
TRAIN_SPLIT = os.path.join(PROC_DIR, "split_train.parquet")

GRAPH_FEATURES_CSV = os.path.join(PROC_DIR, "graph_features_accounts.csv")

RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Load all transactions
# ---------------------------------------------------------------------------

def load_transactions():
    """Return two DataFrames:
    - txns_graph : all splits, columns src_acct/dst_acct only (no labels)
    - txns_train : train split only, includes Is Laundering for fraud-rate computation
    """
    # Graph topology — load all splits without labels
    graph_frames = []
    for path in GRAPH_SPLITS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Split not found: {path}\nRun 02_data_cleaning.py first.")
        df = pd.read_parquet(path, columns=["src_acct", "dst_acct"])
        graph_frames.append(df)
        log.info("  Loaded %s  (%d rows)", os.path.basename(path), len(df))

    txns_graph = pd.concat(graph_frames, ignore_index=True)
    log.info("Graph transactions (all splits, no labels): %d", len(txns_graph))

    # Train labels — only for community_fraud_rate, never val/test
    if not os.path.exists(TRAIN_SPLIT):
        raise FileNotFoundError(f"Train split not found: {TRAIN_SPLIT}\nRun 02_data_cleaning.py first.")
    txns_train = pd.read_parquet(TRAIN_SPLIT, columns=["src_acct", "dst_acct", "Is Laundering"])
    log.info("Train transactions (for fraud rate): %d", len(txns_train))

    return txns_graph, txns_train


# ---------------------------------------------------------------------------
# Step 2: Build account-to-account graph
# ---------------------------------------------------------------------------

def build_graph(txns_graph: pd.DataFrame):
    log.info("Building account-to-account edge list ...")
    t0 = time.time()

    # Remove self-loops
    txns = txns_graph[txns_graph["src_acct"] != txns_graph["dst_acct"]].copy()

    # Vectorized undirected edge key — elementwise min/max avoids Python-level loop
    a, b = txns["src_acct"].values, txns["dst_acct"].values
    txns["ea"] = np.where(a <= b, a, b)
    txns["eb"] = np.where(a <= b, b, a)
    edge_weights = txns.groupby(["ea", "eb"]).size().reset_index(name="weight")
    log.info("  Unique account pairs (edges): %d  (%.1fs)", len(edge_weights), time.time() - t0)

    # np.unique on concatenated arrays avoids materializing large Python sets
    all_accounts = np.unique(np.concatenate([txns["src_acct"].values, txns["dst_acct"].values]))
    acct_to_idx  = {a: i for i, a in enumerate(all_accounts)}
    log.info("  Unique accounts (nodes): %d", len(all_accounts))

    # Build igraph
    log.info("Building igraph object ...")
    src_idx = edge_weights["ea"].map(acct_to_idx).tolist()
    dst_idx = edge_weights["eb"].map(acct_to_idx).tolist()
    weights = edge_weights["weight"].tolist()

    G = ig.Graph(
        n=len(all_accounts),
        edges=list(zip(src_idx, dst_idx)),
        directed=False,
    )
    G.es["weight"] = weights
    G.vs["name"]   = all_accounts
    log.info("  Graph ready: %d nodes, %d edges  (%.1fs)", G.vcount(), G.ecount(), time.time() - t0)

    return G, all_accounts


# ---------------------------------------------------------------------------
# Step 3: Leiden community detection
# ---------------------------------------------------------------------------

def run_leiden(G: ig.Graph) -> list[int]:
    log.info("Running Leiden community detection ...")
    t0 = time.time()

    partition = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        seed=RANDOM_SEED,
    )

    membership = partition.membership
    n_communities = len(set(membership))
    log.info(
        "  Leiden done: %d communities  (%.1fs)",
        n_communities, time.time() - t0,
    )
    return membership


# ---------------------------------------------------------------------------
# Step 4: Compute community-level features
# ---------------------------------------------------------------------------

def compute_community_features(
    txns_train: pd.DataFrame,
    all_accounts: list,
    membership: list[int],
) -> pd.DataFrame:
    log.info("Computing community features ...")

    # account → community_id
    acct_community = dict(zip(all_accounts, membership))

    # community_size via Counter
    community_size_map = dict(Counter(membership))

    # community_fraud_rate — TRAIN LABELS ONLY, never val/test
    # Deduplicate so each (transaction, community) pair counts exactly once.
    # Without dedup, within-community transactions (src and dst in same community)
    # would be counted twice, skewing the denominator and fraud rate.
    train = txns_train[txns_train["src_acct"] != txns_train["dst_acct"]].copy()
    train = train.reset_index(drop=True)
    train["src_community"] = train["src_acct"].map(acct_community)
    train["dst_community"] = train["dst_acct"].map(acct_community)

    src_side = train[["src_community", "Is Laundering"]].copy()
    src_side.columns = ["community_id", "Is Laundering"]
    src_side["txn_idx"] = train.index

    dst_side = train[["dst_community", "Is Laundering"]].copy()
    dst_side.columns = ["community_id", "Is Laundering"]
    dst_side["txn_idx"] = train.index

    combined = pd.concat([src_side, dst_side], ignore_index=True).dropna(subset=["community_id"])
    combined["community_id"] = combined["community_id"].astype(int)
    # Drop duplicates: if src and dst are in same community, count the transaction once
    combined = combined.drop_duplicates(subset=["txn_idx", "community_id"]).drop(columns="txn_idx")

    community_stats = combined.groupby("community_id")["Is Laundering"].agg(["sum", "count"])
    community_stats["fraud_rate"] = community_stats["sum"] / community_stats["count"]
    community_fraud_map = community_stats["fraud_rate"].to_dict()

    # Build per-account DataFrame vectorized via Series.map (avoids 1.7M-row Python loop)
    df_comm = pd.DataFrame({
        "account_id":   list(acct_community.keys()),
        "community_id": list(acct_community.values()),
    })
    df_comm["community_size"]       = df_comm["community_id"].map(community_size_map).fillna(0)
    df_comm["community_fraud_rate"] = df_comm["community_id"].map(community_fraud_map).fillna(0.0)

    df_comm["community_id"]         = df_comm["community_id"].astype("int32")
    df_comm["community_size"]       = df_comm["community_size"].astype("int32")
    df_comm["community_fraud_rate"] = df_comm["community_fraud_rate"].astype("float32")

    log.info(
        "  community_fraud_rate stats: mean=%.4f  max=%.4f  >0.5: %d accounts",
        df_comm["community_fraud_rate"].mean(),
        df_comm["community_fraud_rate"].max(),
        (df_comm["community_fraud_rate"] > 0.5).sum(),
    )
    return df_comm


# ---------------------------------------------------------------------------
# Step 5: Merge into graph_features_accounts.csv
# ---------------------------------------------------------------------------

def update_graph_features(df_comm: pd.DataFrame):
    if not os.path.exists(GRAPH_FEATURES_CSV):
        raise FileNotFoundError(
            f"graph_features_accounts.csv not found: {GRAPH_FEATURES_CSV}\n"
            "Run 04_extract_graph_features.py first."
        )

    df_existing = pd.read_csv(GRAPH_FEATURES_CSV)
    log.info("Existing graph features: %d accounts", len(df_existing))

    # Align account_id dtype so the merge doesn't silently produce all-NaN rows
    df_existing["account_id"] = df_existing["account_id"].astype(str)
    df_comm["account_id"]     = df_comm["account_id"].astype(str)

    # Drop old stub community columns if present
    drop_cols = [c for c in ["community_id", "community_size", "community_fraud_rate"] if c in df_existing.columns]
    if drop_cols:
        df_existing = df_existing.drop(columns=drop_cols)

    df_merged = df_existing.merge(df_comm, on="account_id", how="left")

    # Fill any accounts with no community assignment (shouldn't happen on full graph)
    df_merged["community_id"]         = df_merged["community_id"].fillna(-1).astype("int32")
    df_merged["community_size"]       = df_merged["community_size"].fillna(0).astype("int32")
    df_merged["community_fraud_rate"] = df_merged["community_fraud_rate"].fillna(0.0).astype("float32")

    if len(df_merged) != len(df_existing):
        raise ValueError(
            f"Row count changed after merge ({len(df_existing)} → {len(df_merged)}) — check for duplicate account_ids in df_comm."
        )

    df_merged.to_csv(GRAPH_FEATURES_CSV, index=False)
    log.info(
        "Updated: %s  (%.1f MB)",
        GRAPH_FEATURES_CSV, os.path.getsize(GRAPH_FEATURES_CSV) / 1e6,
    )
    return df_merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  STAGE 4b: LEIDEN COMMUNITY DETECTION")
    print("  Account-to-account graph | community_fraud_rate feature")
    print("  DATA 298A | Team 2 | Issue #4")
    print("=" * 70)

    t_start = time.time()

    # Step 1
    log.info("=== Step 1: Load transactions ===")
    txns_graph, txns_train = load_transactions()

    # Step 2
    log.info("=== Step 2: Build graph ===")
    G, all_accounts = build_graph(txns_graph)

    # Step 3
    log.info("=== Step 3: Leiden community detection ===")
    membership = run_leiden(G)

    # Step 4
    log.info("=== Step 4: Compute community features ===")
    df_comm = compute_community_features(txns_train, all_accounts, membership)

    # Step 5
    log.info("=== Step 5: Update graph_features_accounts.csv ===")
    df_merged = update_graph_features(df_comm)

    elapsed = time.time() - t_start
    n_communities = df_merged["community_id"].nunique()
    median_size   = int(df_merged["community_size"].median())

    print("\n" + "=" * 70)
    print("  LEIDEN COMMUNITY DETECTION COMPLETE")
    print(f"  Accounts processed  : {len(df_merged):,}")
    print(f"  Communities found   : {n_communities:,}")
    print(f"  Median community sz : {median_size:,}")
    print(f"  Mean fraud rate     : {df_merged['community_fraud_rate'].mean():.4f}")
    print(f"  Max fraud rate      : {df_merged['community_fraud_rate'].max():.4f}")
    print(f"  Elapsed             : {elapsed:.1f}s")
    print("\n  Next steps:")
    print("    python3 src/models/05_build_feature_store.py")
    print("    python3 src/models/07_train_graph_enhanced_model.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
