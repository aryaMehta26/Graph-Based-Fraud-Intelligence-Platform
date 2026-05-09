"""
DATA 298A — Graph-Based Fraud Intelligence Platform
Team 2 | San José State University

04_extract_graph_features.py
-----------------------------
Owner : Person 2 (Graph Features — Neo4j + GDS)
Closes: GitHub Issue #3

Connects to the Neo4j knowledge graph and runs two GDS algorithms:
    1. Degree Centrality   — in/out/total transaction count per account
    2. Betweenness Centrality (optional, flag below) — bridge accounts

Exports a flat CSV so Person 3 can join directly onto the transaction parquet.

Output
------
data/processed/graph_features_accounts.csv
    Columns: account_id, in_degree, out_degree, total_degree,
             degree_centrality [, betweenness_centrality]

Usage
-----
    python3 src/models/04_extract_graph_features.py

    # Skip betweenness (faster, smaller graph):
    COMPUTE_BETWEENNESS=false python3 src/models/04_extract_graph_features.py

Prerequisites
-------------
- Neo4j running with full graph loaded (bulk_load_cypher.py complete)
- GDS plugin active
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
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "fraud2026")
NEO4J_DB       = os.getenv("NEO4J_DB",       "neo4j")

# Set to "false" to skip betweenness (much faster on large graphs)
COMPUTE_BETWEENNESS = os.getenv("COMPUTE_BETWEENNESS", "true").lower() != "false"

PROJ     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR  = os.path.join(PROJ, "data", "processed")
OUT_CSV  = os.path.join(OUT_DIR, "graph_features_accounts.csv")

GDS_GRAPH_NAME = "fraud_graph_p2"

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


def drop_projection_if_exists(session):
    result = session.run(
        "CALL gds.graph.exists($name) YIELD exists", name=GDS_GRAPH_NAME
    ).single()
    if result and result["exists"]:
        session.run("CALL gds.graph.drop($name)", name=GDS_GRAPH_NAME)
        log.info("Dropped existing GDS projection: %s", GDS_GRAPH_NAME)


def project_graph(session):
    log.info("Projecting GDS in-memory graph (Account nodes, UNDIRECTED)...")
    t0 = time.time()
    session.run("""
        CALL gds.graph.project(
            $name,
            'Account',
            {
                TRANSACTS_WITH: {
                    type: '*',
                    orientation: 'UNDIRECTED',
                    aggregation: 'COUNT'
                }
            }
        )
    """, name=GDS_GRAPH_NAME)
    log.info("Projection ready in %.1fs", time.time() - t0)


def write_degree_centrality(session):
    log.info("Running degree centrality (write mode → degree_centrality property)...")
    t0 = time.time()
    session.run("""
        CALL gds.degree.write($name, {
            writeProperty: 'degree_centrality',
            orientation: 'UNDIRECTED'
        })
        YIELD nodePropertiesWritten
    """, name=GDS_GRAPH_NAME)
    log.info("Degree centrality written in %.1fs", time.time() - t0)


def write_betweenness(session):
    log.info("Running betweenness centrality (write mode → betweenness_centrality property)...")
    log.info("Note: this can take 5-30 min on the full 31M graph. Set COMPUTE_BETWEENNESS=false to skip.")
    t0 = time.time()
    session.run("""
        CALL gds.betweenness.write($name, {
            writeProperty: 'betweenness_centrality'
        })
        YIELD nodePropertiesWritten
    """, name=GDS_GRAPH_NAME)
    log.info("Betweenness centrality written in %.1fs", time.time() - t0)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def fetch_degree_features(session) -> pd.DataFrame:
    log.info("Fetching in/out degree counts from Transaction edges...")

    result = session.run("""
        MATCH (a:Account)-[:SENT]->()
        RETURN a.account_id AS account_id, count(*) AS out_degree
    """)
    out_map = {r["account_id"]: r["out_degree"] for r in result}

    result = session.run("""
        MATCH ()-[:RECEIVED_BY]->(a:Account)
        RETURN a.account_id AS account_id, count(*) AS in_degree
    """)
    in_map = {r["account_id"]: r["in_degree"] for r in result}

    result = session.run("""
        MATCH (a:Account)
        RETURN a.account_id AS account_id,
               coalesce(a.degree_centrality, 0.0) AS degree_centrality
    """)
    centrality_map = {r["account_id"]: r["degree_centrality"] for r in result}

    all_accounts = set(out_map) | set(in_map) | set(centrality_map)
    rows = []
    for acct in all_accounts:
        rows.append({
            "account_id":        acct,
            "out_degree":        out_map.get(acct, 0),
            "in_degree":         in_map.get(acct, 0),
            "total_degree":      out_map.get(acct, 0) + in_map.get(acct, 0),
            "degree_centrality": centrality_map.get(acct, 0.0),
        })

    df = pd.DataFrame(rows)
    log.info("Degree features: %d accounts", len(df))
    return df


def fetch_betweenness(session) -> pd.DataFrame:
    log.info("Fetching betweenness centrality scores...")
    result = session.run("""
        MATCH (a:Account)
        WHERE a.betweenness_centrality IS NOT NULL
        RETURN a.account_id AS account_id,
               a.betweenness_centrality AS betweenness_centrality
    """)
    rows = [{"account_id": r["account_id"], "betweenness_centrality": r["betweenness_centrality"]}
            for r in result]
    df = pd.DataFrame(rows)
    log.info("Betweenness scores: %d accounts", len(df))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  STAGE 4: GRAPH FEATURE EXTRACTION")
    print("  Degree Centrality" + (" + Betweenness" if COMPUTE_BETWEENNESS else "") + " → CSV")
    print("  DATA 298A | Team 2 | Person 2 | Issue #3")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    driver = get_driver()

    with driver.session(database=NEO4J_DB) as session:
        drop_projection_if_exists(session)
        project_graph(session)
        write_degree_centrality(session)

        if COMPUTE_BETWEENNESS:
            write_betweenness(session)

        df_degree = fetch_degree_features(session)

        if COMPUTE_BETWEENNESS:
            df_between = fetch_betweenness(session)
            df_features = df_degree.merge(df_between, on="account_id", how="left")
            df_features["betweenness_centrality"] = df_features["betweenness_centrality"].fillna(0.0)
        else:
            df_features = df_degree.copy()

    driver.close()

    # Save as CSV (Person 3 reads this directly)
    df_features.to_csv(OUT_CSV, index=False)

    log.info("[SUCCESS] Saved: %s", OUT_CSV)
    log.info("Shape: %s | Columns: %s", df_features.shape, list(df_features.columns))

    print("\n" + "=" * 70)
    print("  GRAPH FEATURE EXTRACTION COMPLETE")
    print(f"  Accounts exported : {len(df_features):,}")
    print(f"  Output CSV        : {OUT_CSV}")
    print(f"  Columns           : {', '.join(df_features.columns)}")
    print("=" * 70)
    print("\n  → Hand off graph_features_accounts.csv to Person 3 for the join.")
    print("=" * 70)


if __name__ == "__main__":
    main()
