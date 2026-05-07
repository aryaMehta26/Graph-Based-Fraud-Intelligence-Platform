"""
============================================================
EXPORT PARQUET → NEO4J-ADMIN IMPORT CSV FORMAT
============================================================
Creates CSVs explicitly formatted for the neo4j-admin bulk 
import tool. Upgrades the ontology to:
(Account)-[:SENT]->(Transaction)-[:RECEIVED_BY]->(Account)
============================================================
"""

import os
import argparse
import pandas as pd
import numpy as np
import time

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARQUET = os.path.join(PROJ, "data", "processed", "transactions_graph.parquet")
OUT_DIR = os.path.join(PROJ, "data", "neo4j_bulk_import")
os.makedirs(OUT_DIR, exist_ok=True)

DIVIDER = "=" * 60

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Load full 29.3M rows")
    parser.add_argument("--sample", type=int, default=100_000, help="Sample size")
    args = parser.parse_args()

    print(DIVIDER)
    print("🚀 EXPORTING FOR NEO4J BULK IMPORT (NEW ONTOLOGY)")
    print(DIVIDER)
    start_time = time.time()

    # 1. Read Parquet
    print(f"[1/5] Loading Parquet file...")
    tx = pd.read_parquet(PARQUET)
    
    if not args.full:
        print(f"      ⚡ SAMPLE MODE: {args.sample:,} rows")
        tx = tx.sample(n=min(args.sample, len(tx)), random_state=42).reset_index(drop=True)
    else:
        print(f"      🔥 FULL MODE: {len(tx):,} rows")

    # Add a unique txn_id since the dataset doesn't have one
    print("      Generating unique Transaction IDs...")
    tx['txn_id'] = 'TXN_' + tx.index.astype(str)

    # 2. Extract Account Nodes
    print(f"[2/5] Creating Account nodes...")
    src = tx[["src_acct", "From Bank"]].rename(columns={"src_acct": "account_id", "From Bank": "bank_id"})
    dst = tx[["dst_acct", "To Bank"]].rename(columns={"dst_acct": "account_id", "To Bank": "bank_id"})
    accounts = pd.concat([src, dst]).drop_duplicates(subset="account_id")
    
    # Format for neo4j-admin
    acc_nodes = pd.DataFrame({
        'account_id:ID(Account)': accounts['account_id'],
        'bank_id:int': accounts['bank_id'],
        ':LABEL': 'Account'
    })
    
    acc_path = os.path.join(OUT_DIR, "nodes_accounts.csv")
    acc_nodes.to_csv(acc_path, index=False)
    print(f"      ✅ Saved {len(acc_nodes):,} Accounts")

    # 3. Extract Transaction Nodes
    print(f"[3/5] Creating Transaction nodes...")
    txn_nodes = pd.DataFrame({
        'txn_id:ID(Transaction)': tx['txn_id'],
        'amount:float': tx['Amount Paid'],
        'timestamp:datetime': pd.to_datetime(tx['Timestamp']).dt.strftime("%Y-%m-%dT%H:%M:%S"),
        'payment_format': tx['Payment Format'],
        'currency': tx['Payment Currency'],
        'is_laundering:int': tx['Is Laundering'],
        ':LABEL': 'Transaction'
    })
    
    txn_path = os.path.join(OUT_DIR, "nodes_transactions.csv")
    txn_nodes.to_csv(txn_path, index=False)
    print(f"      ✅ Saved {len(txn_nodes):,} Transactions")

    # 4. Extract SENT Relationships (Account -> Transaction)
    print(f"[4/5] Creating SENT edges...")
    sent_edges = pd.DataFrame({
        ':START_ID(Account)': tx['src_acct'],
        ':END_ID(Transaction)': tx['txn_id'],
        ':TYPE': 'SENT'
    })
    
    sent_path = os.path.join(OUT_DIR, "edges_sent.csv")
    sent_edges.to_csv(sent_path, index=False)
    print(f"      ✅ Saved {len(sent_edges):,} SENT edges")

    # 5. Extract RECEIVED_BY Relationships (Transaction -> Account)
    print(f"[5/5] Creating RECEIVED_BY edges...")
    recv_edges = pd.DataFrame({
        ':START_ID(Transaction)': tx['txn_id'],
        ':END_ID(Account)': tx['dst_acct'],
        ':TYPE': 'RECEIVED_BY'
    })
    
    recv_path = os.path.join(OUT_DIR, "edges_received.csv")
    recv_edges.to_csv(recv_path, index=False)
    print(f"      ✅ Saved {len(recv_edges):,} RECEIVED_BY edges")

    elapsed = time.time() - start_time
    print(DIVIDER)
    print(f"DONE in {elapsed:.1f} seconds! Everything saved to: {OUT_DIR}")
    print(DIVIDER)

if __name__ == "__main__":
    main()
