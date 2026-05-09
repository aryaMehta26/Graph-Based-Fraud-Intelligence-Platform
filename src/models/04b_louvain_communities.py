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
1. Load all three transaction splits (train/val/test) — same transductive
   approach as degree features in 04_extract_graph_features.py.
2. Build an undirected weighted account-to-account graph:
   - Nodes  : unique accounts
   - Edges  : one per unique (src_acct, dst_acct) pair, weighted by
              transaction count between those two accounts.
3. Run Leiden community detection (RBConfigurationVertexPartition).
4. For each community compute:
   - community_id         : integer community label
   - community_size       : number of member accounts
   - community_fraud_rate : fraction of transactions (where src OR dst is a
                            member) flagged as laundering
5. Merge these three columns into graph_features_accounts.csv, replacing
   the zero stubs written by 04_extract_graph_features.py.

Output
------
data/processed/graph_features_accounts.csv  (updated in-place)
    Adds / overwrites columns:
        community_id, community_size, community_fraud_rate

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
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJ     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR = os.path.join(PROJ, "data", "processed")

SPLITS = [
    os.path.join(PROC_DIR, "split_train.parquet"),
    os.path.join(PROC_DIR, "split_val.parquet"),
    os.path.join(PROC_DIR, "split_test.parquet"),
]

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

def load_transactions() -> pd.DataFrame:
    frames = []
    for path in SPLITS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Split not found: {path}\nRun 02_data_cleaning.py first.")
        df = pd.read_parquet(path, columns=["src_acct", "dst_acct", "Is Laundering"])
        frames.append(df)
        log.info("  Loaded %s  (%d rows)", os.path.basename(path), len(df))

    txns = pd.concat(frames, ignore_index=True)
    log.info("Total transactions: %d", len(txns))
    return txns


# ---------------------------------------------------------------------------
# Step 2: Build account-to-account graph
# ---------------------------------------------------------------------------

def build_graph(txns: pd.DataFrame):
    log.info("Building account-to-account edge list ...")
    t0 = time.time()

    # Remove self-loops (same account sends to itself — not useful for communities)
    txns = txns[txns["src_acct"] != txns["dst_acct"]].copy()

    # Aggregate: count transactions between each unique account pair (undirected)
    txns["pair"] = txns.apply(
        lambda r: tuple(sorted([r["src_acct"], r["dst_acct"]])), axis=1
    )
    edge_weights = txns.groupby("pair").size().reset_index(name="weight")
    log.info("  Unique account pairs (edges): %d  (%.1fs)", len(edge_weights), time.time() - t0)

    # Map account IDs to integer indices
    all_accounts = sorted(set(txns["src_acct"]) | set(txns["dst_acct"]))
    acct_to_idx  = {a: i for i, a in enumerate(all_accounts)}
    log.info("  Unique accounts (nodes): %d", len(all_accounts))

    # Build igraph
    log.info("Building igraph object ...")
    src_idx = edge_weights["pair"].apply(lambda p: acct_to_idx[p[0]]).tolist()
    dst_idx = edge_weights["pair"].apply(lambda p: acct_to_idx[p[1]]).tolist()
    weights = edge_weights["weight"].tolist()

    G = ig.Graph(
        n=len(all_accounts),
        edges=list(zip(src_idx, dst_idx)),
        directed=False,
    )
    G.es["weight"] = weights
    G.vs["name"]   = all_accounts
    log.info("  Graph ready: %d nodes, %d edges  (%.1fs)", G.vcount(), G.ecount(), time.time() - t0)

    return G, acct_to_idx, all_accounts


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
    txns: pd.DataFrame,
    all_accounts: list,
    membership: list[int],
) -> pd.DataFrame:
    log.info("Computing community features ...")

    # account → community_id
    acct_community = {acct: cid for acct, cid in zip(all_accounts, membership)}

    # community_size
    community_size_map = {}
    from collections import Counter
    size_counts = Counter(membership)
    community_size_map = dict(size_counts)

    # community_fraud_rate
    # For each transaction, tag community of src and dst
    # A transaction contributes to its community's fraud count if is_laundering=1
    txns = txns[txns["src_acct"] != txns["dst_acct"]].copy()
    txns["src_community"] = txns["src_acct"].map(acct_community)
    txns["dst_community"] = txns["dst_acct"].map(acct_community)

    # Melt so each transaction appears once per unique community it touches
    src_side = txns[["src_community", "Is Laundering"]].rename(columns={"src_community": "community_id"})
    dst_side = txns[["dst_community", "Is Laundering"]].rename(columns={"dst_community": "community_id"})
    combined = pd.concat([src_side, dst_side], ignore_index=True).dropna(subset=["community_id"])
    combined["community_id"] = combined["community_id"].astype(int)

    community_stats = combined.groupby("community_id")["Is Laundering"].agg(["sum", "count"])
    community_stats["fraud_rate"] = community_stats["sum"] / community_stats["count"]
    community_fraud_map = community_stats["fraud_rate"].to_dict()

    # Build per-account feature DataFrame
    rows = []
    for acct, cid in acct_community.items():
        rows.append({
            "account_id":            acct,
            "community_id":          cid,
            "community_size":        community_size_map.get(cid, 0),
            "community_fraud_rate":  community_fraud_map.get(cid, 0.0),
        })

    df_comm = pd.DataFrame(rows)
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

    # Drop old stub community columns if present
    drop_cols = [c for c in ["community_id", "community_size", "community_fraud_rate"] if c in df_existing.columns]
    if drop_cols:
        df_existing = df_existing.drop(columns=drop_cols)

    df_merged = df_existing.merge(df_comm, on="account_id", how="left")

    # Fill any accounts with no community assignment (shouldn't happen on full graph)
    df_merged["community_id"]         = df_merged["community_id"].fillna(-1).astype("int32")
    df_merged["community_size"]       = df_merged["community_size"].fillna(0).astype("int32")
    df_merged["community_fraud_rate"] = df_merged["community_fraud_rate"].fillna(0.0).astype("float32")

    assert len(df_merged) == len(df_existing), "Row count changed after merge — check for duplicates."

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
    txns = load_transactions()

    # Step 2
    log.info("=== Step 2: Build graph ===")
    G, acct_to_idx, all_accounts = build_graph(txns)

    # Step 3
    log.info("=== Step 3: Leiden community detection ===")
    membership = run_leiden(G)

    # Step 4
    log.info("=== Step 4: Compute community features ===")
    df_comm = compute_community_features(txns, all_accounts, membership)

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
