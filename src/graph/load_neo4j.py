"""
============================================================
LOAD CSV DATA INTO NEO4J
============================================================
Reads accounts.csv and transactions.csv from data/neo4j_import/
and batch-loads them into Neo4j using the Python driver.

Usage:
  python3 src/graph/load_neo4j.py              # load whatever CSVs exist
  python3 src/graph/load_neo4j.py --clear      # wipe DB first, then load
============================================================
"""

import os
import sys
import csv
import time
import argparse
from neo4j import GraphDatabase

PROJ  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ACC_CSV = os.path.join(PROJ, "data", "neo4j_import", "accounts.csv")
TX_CSV  = os.path.join(PROJ, "data", "neo4j_import", "transactions.csv")

NEO4J_URI  = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Aryamehta@26"

BATCH_SIZE = 5_000
DIVIDER = "=" * 60


def run_query(driver, query, params=None):
    with driver.session() as s:
        result = s.run(query, params or {})
        return [r.data() for r in result]


def count_csv_rows(path):
    with open(path) as f:
        return sum(1 for _ in f) - 1  # minus header


def load_accounts(driver, csv_path):
    total = count_csv_rows(csv_path)
    print(f"\n[ACCOUNTS] Loading {total:,} nodes in batches of {BATCH_SIZE:,}...")

    batch = []
    loaded = 0

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.append({
                "account_id": row["account_id"],
                "bank_id": int(row["bank_id"]),
            })
            if len(batch) >= BATCH_SIZE:
                _flush_accounts(driver, batch)
                loaded += len(batch)
                pct = loaded / total * 100
                print(f"  {loaded:>10,} / {total:,}  ({pct:.1f}%)", end="\r")
                batch = []

    if batch:
        _flush_accounts(driver, batch)
        loaded += len(batch)

    print(f"\n  ✅ Loaded {loaded:,} Account nodes")
    return loaded


def _flush_accounts(driver, batch):
    query = """
    UNWIND $rows AS row
    MERGE (a:Account {account_id: row.account_id})
    ON CREATE SET a.bank_id = row.bank_id
    """
    with driver.session() as s:
        s.run(query, {"rows": batch})


def load_transactions(driver, csv_path):
    total = count_csv_rows(csv_path)
    print(f"\n[TRANSACTIONS] Loading {total:,} edges in batches of {BATCH_SIZE:,}...")

    batch = []
    loaded = 0

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.append({
                "src":       row["src_acct"],
                "dst":       row["dst_acct"],
                "amount":    float(row["amount"]),
                "format":    row["payment_format"],
                "ts":        row["timestamp"],
                "fraud":     int(row["is_laundering"]),
                "currency":  row["currency"],
            })
            if len(batch) >= BATCH_SIZE:
                _flush_transactions(driver, batch)
                loaded += len(batch)
                pct = loaded / total * 100
                print(f"  {loaded:>10,} / {total:,}  ({pct:.1f}%)", end="\r")
                batch = []

    if batch:
        _flush_transactions(driver, batch)
        loaded += len(batch)

    print(f"\n  ✅ Loaded {loaded:,} TRANSFERRED_TO edges")
    return loaded


def _flush_transactions(driver, batch):
    query = """
    UNWIND $rows AS row
    MATCH (a:Account {account_id: row.src})
    MATCH (b:Account {account_id: row.dst})
    CREATE (a)-[:TRANSFERRED_TO {
        amount: row.amount,
        payment_format: row.format,
        timestamp: datetime(row.ts),
        is_laundering: row.fraud,
        currency: row.currency
    }]->(b)
    """
    with driver.session() as s:
        s.run(query, {"rows": batch})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Wipe database before loading")
    args = parser.parse_args()

    print(DIVIDER)
    print("LOAD DATA INTO NEO4J")
    print(DIVIDER)

    # ── Connect ──────────────────────────────────────────
    print(f"\n[CONNECT] {NEO4J_URI} as {NEO4J_USER}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    driver.verify_connectivity()
    print("  ✅ Connected to Neo4j")

    # ── Optionally clear ─────────────────────────────────
    if args.clear:
        print("\n[CLEAR] Wiping all nodes and relationships...")
        run_query(driver, "MATCH (n) DETACH DELETE n")
        print("  ✅ Database cleared")

    # ── Create constraints ───────────────────────────────
    print("\n[SCHEMA] Creating constraints and indexes...")
    try:
        run_query(driver, """
            CREATE CONSTRAINT account_id_unique IF NOT EXISTS
            FOR (a:Account) REQUIRE a.account_id IS UNIQUE
        """)
        print("  ✅ Unique constraint on Account.account_id")
    except Exception as e:
        print(f"  ⚠️  Constraint may already exist: {e}")

    try:
        run_query(driver, """
            CREATE INDEX fraud_idx IF NOT EXISTS
            FOR ()-[r:TRANSFERRED_TO]-()
            ON (r.is_laundering)
        """)
        print("  ✅ Index on TRANSFERRED_TO.is_laundering")
    except Exception as e:
        print(f"  ⚠️  Index may already exist: {e}")

    # ── Load ─────────────────────────────────────────────
    t0 = time.time()

    if not os.path.exists(ACC_CSV):
        print(f"\n❌ {ACC_CSV} not found. Run export_csv_for_neo4j.py first.")
        sys.exit(1)
    if not os.path.exists(TX_CSV):
        print(f"\n❌ {TX_CSV} not found. Run export_csv_for_neo4j.py first.")
        sys.exit(1)

    n_accounts = load_accounts(driver, ACC_CSV)
    n_edges    = load_transactions(driver, TX_CSV)

    elapsed = time.time() - t0

    # ── Verify ───────────────────────────────────────────
    print(f"\n[VERIFY] Checking counts...")
    node_count = run_query(driver, "MATCH (n:Account) RETURN count(n) AS cnt")[0]["cnt"]
    edge_count = run_query(driver, "MATCH ()-[r:TRANSFERRED_TO]->() RETURN count(r) AS cnt")[0]["cnt"]
    fraud_count = run_query(driver, """
        MATCH ()-[r:TRANSFERRED_TO]->()
        WHERE r.is_laundering = 1
        RETURN count(r) AS cnt
    """)[0]["cnt"]

    print(f"  Account nodes     : {node_count:,}")
    print(f"  Transaction edges : {edge_count:,}")
    print(f"  Fraud edges       : {fraud_count:,}")

    driver.close()

    print(f"\n{DIVIDER}")
    print("NEO4J LOAD COMPLETE ✅")
    print(f"  Nodes       : {node_count:,}")
    print(f"  Edges       : {edge_count:,}")
    print(f"  Fraud edges : {fraud_count:,}")
    print(f"  Time        : {elapsed:.1f} seconds")
    print(f"  Browser     : http://localhost:7474")
    print(f"  Next step   : python3 src/graph/visualize_graph.py")
    print(DIVIDER)


if __name__ == "__main__":
    main()
