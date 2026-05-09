"""
DATA 298A — Graph-Based Fraud Intelligence Platform
Team 2 | San José State University

04_extract_graph_features.py
-----------------------------
Owner : Person 2 (Graph Features — Neo4j + GDS)
Closes: GitHub Issue #3

Connects to the Neo4j knowledge graph and computes per-account degree features:
    - in_degree  : number of incoming transactions received
    - out_degree : number of outgoing transactions sent
    - total_degree : sum of in + out
    - degree_centrality : total_degree / (n_accounts - 1)  — normalised

Degree counts are derived directly via Cypher on the SENT / RECEIVED_BY
relationships so no GDS in-memory projection is required.

Output
------
data/processed/graph_features_accounts.csv
    Columns: account_id, in_degree, out_degree, total_degree, degree_centrality

Usage
-----
    python3 src/models/04_extract_graph_features.py

Prerequisites
-------------
- Neo4j running with full graph loaded (bulk_load_cypher.py complete)
- .env at project root: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

import os
import time
import logging
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Aryamehta@26")
NEO4J_DB       = os.getenv("NEO4J_DB",       "neo4j")

PROJ     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR  = os.path.join(PROJ, "data", "processed")
OUT_CSV  = os.path.join(OUT_DIR, "graph_features_accounts.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def get_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    log.info("Connected: %s", NEO4J_URI)
    return driver


# ---------------------------------------------------------------------------
# Feature extraction  (pure Cypher — no GDS projection needed)
# ---------------------------------------------------------------------------

def fetch_degree_features(session) -> pd.DataFrame:
    log.info("Fetching out-degree (SENT edges) ...")
    t0 = time.time()
    result = session.run("""
        MATCH (a:Account)-[:SENT]->()
        RETURN a.account_id AS account_id, count(*) AS out_degree
    """)
    out_map = {r["account_id"]: r["out_degree"] for r in result}
    log.info("  out_degree done in %.1fs — %d accounts", time.time() - t0, len(out_map))

    log.info("Fetching in-degree (RECEIVED_BY edges) ...")
    t0 = time.time()
    result = session.run("""
        MATCH ()-[:RECEIVED_BY]->(a:Account)
        RETURN a.account_id AS account_id, count(*) AS in_degree
    """)
    in_map = {r["account_id"]: r["in_degree"] for r in result}
    log.info("  in_degree done in %.1fs — %d accounts", time.time() - t0, len(in_map))

    log.info("Fetching total account count ...")
    n_accounts = session.run("MATCH (a:Account) RETURN count(a) AS n").single()["n"]
    log.info("  Total accounts: %d", n_accounts)

    all_accounts = set(out_map) | set(in_map)
    rows = []
    for acct in all_accounts:
        out  = out_map.get(acct, 0)
        inp  = in_map.get(acct, 0)
        tot  = out + inp
        rows.append({
            "account_id":        acct,
            "out_degree":        out,
            "in_degree":         inp,
            "total_degree":      tot,
            "degree_centrality": tot / (n_accounts - 1) if n_accounts > 1 else 0.0,
        })

    df = pd.DataFrame(rows)
    log.info("Degree features built: %d accounts", len(df))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  STAGE 4: GRAPH FEATURE EXTRACTION")
    print("  Degree features via Cypher (SENT / RECEIVED_BY)")
    print("  DATA 298A | Team 2 | Person 2 | Issue #3")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    driver = get_driver()

    with driver.session(database=NEO4J_DB) as session:
        df_features = fetch_degree_features(session)

    driver.close()

    df_features.to_csv(OUT_CSV, index=False)
    log.info("[SUCCESS] Saved: %s  (%.1f MB)", OUT_CSV, os.path.getsize(OUT_CSV) / 1e6)

    print("\n" + "=" * 70)
    print("  GRAPH FEATURE EXTRACTION COMPLETE")
    print(f"  Accounts exported : {len(df_features):,}")
    print(f"  Output CSV        : {OUT_CSV}")
    print(f"  Columns           : {', '.join(df_features.columns)}")
    print("  → Next: python3 src/models/05_build_feature_store.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
