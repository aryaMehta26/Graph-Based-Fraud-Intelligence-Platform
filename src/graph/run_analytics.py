"""
============================================================
GRAPH ANALYTICS — CENTRALITY + LOUVAIN
============================================================
Runs graph analytics on Neo4j:
  1. Degree centrality (in/out)
  2. Betweenness centrality (on sample)
  3. Louvain community detection (via GDS)

Exports results to data/processed/

Usage:
  python3 src/graph/run_analytics.py
============================================================
"""

import os
import time
import pandas as pd
from neo4j import GraphDatabase

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(PROJ, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

NEO4J_URI  = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Aryamehta@26"

DIVIDER = "=" * 60


def run_query(driver, query, params=None):
    with driver.session() as s:
        result = s.run(query, params or {})
        return [r.data() for r in result]


def main():
    print(DIVIDER)
    print("GRAPH ANALYTICS — CENTRALITY + LOUVAIN")
    print(DIVIDER)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    driver.verify_connectivity()
    print("  ✅ Connected to Neo4j")

    # ── 1. Degree Centrality ─────────────────────────────
    print(f"\n{'─'*50}")
    print("STEP 1: DEGREE CENTRALITY")
    print(f"{'─'*50}")
    t0 = time.time()

    deg_data = run_query(driver, """
        MATCH (a:Account)
        OPTIONAL MATCH (a)-[out:TRANSFERRED_TO]->()
        OPTIONAL MATCH ()-[inc:TRANSFERRED_TO]->(a)
        WITH a,
             count(DISTINCT out) AS out_degree,
             count(DISTINCT inc) AS in_degree
        RETURN a.account_id AS account_id,
               out_degree,
               in_degree,
               out_degree + in_degree AS total_degree
        ORDER BY total_degree DESC
    """)

    df_deg = pd.DataFrame(deg_data)
    deg_path = os.path.join(OUT_DIR, "centrality_scores.csv")
    df_deg.to_csv(deg_path, index=False)
    elapsed = time.time() - t0

    print(f"  Accounts scored : {len(df_deg):,}")
    print(f"  Time            : {elapsed:.1f}s")
    print(f"  Top 10 by total degree:")
    for _, row in df_deg.head(10).iterrows():
        print(f"    {row['account_id']}: out={row['out_degree']:,} in={row['in_degree']:,} total={row['total_degree']:,}")
    print(f"  ✅ Saved to {deg_path}")

    # ── 2. Louvain Community Detection ───────────────────
    print(f"\n{'─'*50}")
    print("STEP 2: LOUVAIN COMMUNITY DETECTION (GDS)")
    print(f"{'─'*50}")
    t0 = time.time()

    # Create GDS graph projection
    print("  Creating graph projection...")
    try:
        run_query(driver, "CALL gds.graph.drop('fraud_graph', false)")
    except Exception:
        pass

    run_query(driver, """
        CALL gds.graph.project(
            'fraud_graph',
            'Account',
            'TRANSFERRED_TO',
            { relationshipProperties: ['amount', 'is_laundering'] }
        )
    """)
    print("  ✅ Graph projection created")

    # Run Louvain
    print("  Running Louvain community detection...")
    louvain_result = run_query(driver, """
        CALL gds.louvain.write('fraud_graph', {
            writeProperty: 'community_id'
        })
        YIELD communityCount, modularity, preProcessingMillis,
              computeMillis, postProcessingMillis
        RETURN communityCount, modularity,
               preProcessingMillis + computeMillis + postProcessingMillis AS totalMs
    """)

    if louvain_result:
        lr = louvain_result[0]
        print(f"  Communities found : {lr['communityCount']:,}")
        print(f"  Modularity        : {lr['modularity']:.4f}")
        print(f"  Compute time      : {lr['totalMs']:,} ms")

    # Extract community assignments
    print("  Extracting community assignments...")
    comm_data = run_query(driver, """
        MATCH (a:Account)
        WHERE a.community_id IS NOT NULL
        RETURN a.account_id AS account_id,
               a.community_id AS community_id
        ORDER BY a.community_id
    """)

    df_comm = pd.DataFrame(comm_data)
    comm_path = os.path.join(OUT_DIR, "communities.csv")
    df_comm.to_csv(comm_path, index=False)
    elapsed = time.time() - t0

    # Community size distribution
    comm_sizes = df_comm["community_id"].value_counts()
    print(f"\n  Total accounts with community : {len(df_comm):,}")
    print(f"  Unique communities            : {comm_sizes.nunique():,}")
    print(f"  Largest community             : {comm_sizes.max():,} accounts")
    print(f"  Smallest community            : {comm_sizes.min():,} accounts")
    print(f"  Median community size         : {comm_sizes.median():.0f} accounts")
    print(f"  Time                          : {elapsed:.1f}s")
    print(f"  ✅ Saved to {comm_path}")

    # ── 3. Community Fraud Scoring ───────────────────────
    print(f"\n{'─'*50}")
    print("STEP 3: COMMUNITY FRAUD SCORING")
    print(f"{'─'*50}")

    fraud_scores = run_query(driver, """
        MATCH (a:Account)-[r:TRANSFERRED_TO]->(b:Account)
        WHERE a.community_id IS NOT NULL AND a.community_id = b.community_id
        WITH a.community_id AS community_id,
             count(r) AS internal_edges,
             sum(r.amount) AS total_amount,
             sum(CASE WHEN r.is_laundering = 1 THEN 1 ELSE 0 END) AS fraud_edges,
             collect(DISTINCT a.account_id) + collect(DISTINCT b.account_id) AS all_accts
        RETURN community_id,
               size(apoc.coll.toSet(all_accts)) AS n_accounts,
               internal_edges,
               fraud_edges,
               total_amount,
               CASE WHEN internal_edges > 0
                    THEN toFloat(fraud_edges) / internal_edges
                    ELSE 0.0 END AS fraud_rate
        ORDER BY fraud_rate DESC, fraud_edges DESC
        LIMIT 100
    """)

    if fraud_scores:
        df_fs = pd.DataFrame(fraud_scores)
        fs_path = os.path.join(OUT_DIR, "community_fraud_scores.csv")
        df_fs.to_csv(fs_path, index=False)

        suspicious = df_fs[df_fs["fraud_edges"] > 0]
        print(f"  Top suspicious communities: {len(suspicious):,}")
        print(f"\n  {'Community':>12} {'Accounts':>10} {'Edges':>8} {'Fraud':>8} {'FraudRate':>10} {'Amount':>15}")
        print(f"  {'─'*70}")
        for _, row in suspicious.head(15).iterrows():
            print(f"  {row['community_id']:>12} {row['n_accounts']:>10,} "
                  f"{row['internal_edges']:>8,} {row['fraud_edges']:>8,} "
                  f"{row['fraud_rate']:>9.2%} ${row['total_amount']:>14,.0f}")

        print(f"\n  ✅ Saved to {fs_path}")
    else:
        print("  ⚠️  No community fraud data found (run with more data?)")

    # ── Cleanup ──────────────────────────────────────────
    try:
        run_query(driver, "CALL gds.graph.drop('fraud_graph', false)")
    except Exception:
        pass

    driver.close()

    print(f"\n{DIVIDER}")
    print("GRAPH ANALYTICS COMPLETE ✅")
    print(f"  Outputs:")
    print(f"    {deg_path}")
    print(f"    {comm_path}")
    if fraud_scores:
        print(f"    {os.path.join(OUT_DIR, 'community_fraud_scores.csv')}")
    print(f"  Next step: python3 src/graph/visualize_graph.py")
    print(DIVIDER)


if __name__ == "__main__":
    main()
