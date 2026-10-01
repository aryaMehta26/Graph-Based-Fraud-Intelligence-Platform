"""
============================================================
EXPORT PARQUET → CSV FOR NEO4J BULK IMPORT
============================================================
Reads transactions_graph.parquet (29.3M rows, no self-loops)
and exports two CSVs into data/neo4j_import/:
  - accounts.csv  (unique nodes)
  - transactions.csv (edges)

Usage:
  python3 src/graph/export_csv_for_neo4j.py           # sample 100K
  python3 src/graph/export_csv_for_neo4j.py --full     # full 29.3M
============================================================
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARQUET = os.path.join(PROJ, "data", "processed", "transactions_graph.parquet")
OUT_DIR = os.path.join(PROJ, "data", "neo4j_import")
os.makedirs(OUT_DIR, exist_ok=True)

DIVIDER = "=" * 60

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Load full 29.3M rows (slow)")
    parser.add_argument("--sample", type=int, default=100_000, help="Sample size for dev mode")
    args = parser.parse_args()

    print(DIVIDER)
    print("EXPORT PARQUET → CSV FOR NEO4J")
    print(DIVIDER)

    # ── Load ─────────────────────────────────────────────
    print(f"\n[1/4] Reading {PARQUET}...")
    tx = pd.read_parquet(PARQUET)
    print(f"  Total rows: {len(tx):,}")

    if not args.full:
        n = min(args.sample, len(tx))
        print(f"\n  ⚡ SAMPLE MODE: using {n:,} rows (use --full for all)")
        tx = tx.sample(n=n, random_state=42).reset_index(drop=True)
    else:
        print(f"\n  🔥 FULL MODE: exporting all {len(tx):,} rows")

    # ── Extract unique accounts ──────────────────────────
    print(f"\n[2/4] Extracting unique accounts...")
    src = tx[["src_acct", "From Bank"]].rename(columns={"src_acct": "account_id", "From Bank": "bank_id"})
    dst = tx[["dst_acct", "To Bank"]].rename(columns={"dst_acct": "account_id", "To Bank": "bank_id"})
    accounts = pd.concat([src, dst]).drop_duplicates(subset="account_id")
    print(f"  Unique accounts: {len(accounts):,}")

    # ── Save accounts CSV ────────────────────────────────
    print(f"\n[3/4] Saving accounts.csv...")
    acc_path = os.path.join(OUT_DIR, "accounts.csv")
    accounts.to_csv(acc_path, index=False)
    size_mb = os.path.getsize(acc_path) / (1024 * 1024)
    print(f"  ✅ {acc_path}  ({size_mb:.1f} MB)")

    # ── Save transactions CSV ────────────────────────────
    print(f"\n[4/4] Saving transactions.csv...")
    edges = tx[["src_acct", "dst_acct", "Amount Paid", "Payment Format",
                "Timestamp", "Is Laundering", "Payment Currency"]].copy()
    edges.columns = ["src_acct", "dst_acct", "amount", "payment_format",
                     "timestamp", "is_laundering", "currency"]
    edges["timestamp"] = pd.to_datetime(edges["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

    tx_path = os.path.join(OUT_DIR, "transactions.csv")
    edges.to_csv(tx_path, index=False)
    size_mb2 = os.path.getsize(tx_path) / (1024 * 1024)
    print(f"  ✅ {tx_path}  ({size_mb2:.1f} MB)")

    print(f"\n{DIVIDER}")
    print("EXPORT COMPLETE ✅")
    print(f"  Accounts  : {len(accounts):,}")
    print(f"  Edges     : {len(edges):,}")
    print(f"  Output    : {OUT_DIR}")
    print(f"  Next step : docker compose up -d && python3 src/graph/load_neo4j.py")
    print(DIVIDER)

if __name__ == "__main__":
    main()
