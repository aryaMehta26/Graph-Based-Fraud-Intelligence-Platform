"""
============================================================
INTERACTIVE GRAPH VISUALIZATION (PYVIS)
============================================================
Queries Neo4j for a sample subgraph (top suspicious accounts
+ their neighbors) and generates an interactive HTML file.

Usage:
  python3 src/graph/visualize_graph.py
  python3 src/graph/visualize_graph.py --top 30  # top 30 accounts
============================================================
"""

import os
import argparse
from neo4j import GraphDatabase
from pyvis.network import Network

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_HTML = os.path.join(PROJ, "notebooks", "graph_viz.html")

NEO4J_URI  = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Aryamehta@26"

# Colors
FRAUD_EDGE = "#E05C5C"
LEGIT_EDGE = "#3A7BD5"
FRAUD_NODE = "#FF4444"
NORMAL_NODE = "#4C9BE8"
HUB_NODE = "#F39C12"

DIVIDER = "=" * 60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20, help="Top N hub accounts to visualize")
    parser.add_argument("--hops", type=int, default=1, help="Neighbor hops from hubs")
    args = parser.parse_args()

    print(DIVIDER)
    print("INTERACTIVE GRAPH VISUALIZATION")
    print(DIVIDER)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    driver.verify_connectivity()
    print("  ✅ Connected to Neo4j")

    # ── Strategy 1: Top hub accounts + their fraud neighbors ──
    print(f"\n[1/3] Finding top {args.top} hub accounts by out-degree...")

    with driver.session() as s:
        # Get top hub accounts
        hubs = s.run(f"""
            MATCH (a:Account)-[r:TRANSFERRED_TO]->()
            RETURN a.account_id AS id, count(r) AS deg
            ORDER BY deg DESC
            LIMIT {args.top}
        """)
        hub_ids = [r["id"] for r in hubs]
        print(f"  Found {len(hub_ids)} hubs (max degree accounts)")

    # ── Strategy 2: Get fraud subgraph around hubs ───────────
    print(f"\n[2/3] Extracting subgraph (fraud + hub neighborhoods)...")

    with driver.session() as s:
        # Get edges from/to hubs (limit to keep viz manageable)
        result = s.run("""
            MATCH (a:Account)-[r:TRANSFERRED_TO]->(b:Account)
            WHERE a.account_id IN $hubs OR b.account_id IN $hubs
                  OR r.is_laundering = 1
            RETURN a.account_id AS src,
                   b.account_id AS dst,
                   r.amount AS amount,
                   r.is_laundering AS fraud,
                   r.payment_format AS fmt
            LIMIT 2000
        """, {"hubs": hub_ids})

        edges = [r.data() for r in result]
        print(f"  Edges in subgraph: {len(edges):,}")

    if not edges:
        print("  ❌ No edges found — is the database loaded?")
        driver.close()
        return

    # ── Build pyvis network ──────────────────────────────
    print(f"\n[3/3] Building interactive visualization...")

    net = Network(
        height="900px", width="100%",
        bgcolor="#0F1117", font_color="white",
        directed=True, notebook=False
    )
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.001,
        damping=0.09
    )

    # Collect unique nodes
    nodes_seen = set()
    fraud_nodes = set()

    for e in edges:
        if e["fraud"] == 1:
            fraud_nodes.add(e["src"])
            fraud_nodes.add(e["dst"])

    for e in edges:
        for nid in [e["src"], e["dst"]]:
            if nid not in nodes_seen:
                nodes_seen.add(nid)
                if nid in hub_ids:
                    color = HUB_NODE
                    size = 25
                    title = f"🔶 HUB: {nid}"
                elif nid in fraud_nodes:
                    color = FRAUD_NODE
                    size = 15
                    title = f"🔴 FRAUD-LINKED: {nid}"
                else:
                    color = NORMAL_NODE
                    size = 8
                    title = f"Account: {nid}"

                net.add_node(nid, label=nid[:8], color=color,
                             size=size, title=title)

    # Add edges
    fraud_count = 0
    for e in edges:
        is_fraud = e["fraud"] == 1
        if is_fraud:
            fraud_count += 1
        color = FRAUD_EDGE if is_fraud else LEGIT_EDGE
        width = 3 if is_fraud else 1
        title = f"{'🔴 FRAUD' if is_fraud else 'Legit'} | ${e['amount']:,.2f} | {e['fmt']}"

        net.add_edge(
            e["src"], e["dst"],
            color=color, width=width,
            title=title, arrows="to"
        )

    # Add legend as HTML
    net.set_options("""
    {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -3000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.001,
                "damping": 0.09
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true
        }
    }
    """)

    # Save
    net.save_graph(OUT_HTML)

    # Inject legend HTML at top of file
    with open(OUT_HTML, "r") as f:
        html = f.read()

    legend = """
    <div style="position:fixed; top:10px; left:10px; z-index:9999;
                background:rgba(15,17,23,0.9); padding:15px; border-radius:8px;
                border:1px solid #3A3D4D; font-family:monospace; color:white;">
        <b>Graph Fraud Intelligence Platform</b><br/><br/>
        <span style="color:#F39C12;">⬤</span> Hub Account (high degree)<br/>
        <span style="color:#FF4444;">⬤</span> Fraud-linked Account<br/>
        <span style="color:#4C9BE8;">⬤</span> Normal Account<br/><br/>
        <span style="color:#E05C5C;">━━</span> Fraud Transaction<br/>
        <span style="color:#3A7BD5;">━━</span> Legitimate Transaction<br/><br/>
        <small>Hover nodes/edges for details</small>
    </div>
    """
    html = html.replace("<body>", f"<body>{legend}")
    with open(OUT_HTML, "w") as f:
        f.write(html)

    driver.close()

    print(f"\n  Nodes in viz    : {len(nodes_seen):,}")
    print(f"  Edges in viz    : {len(edges):,}")
    print(f"  Fraud edges     : {fraud_count:,}")
    print(f"\n{DIVIDER}")
    print("VISUALIZATION COMPLETE ✅")
    print(f"  Output: {OUT_HTML}")
    print(f"  Open in browser: open {OUT_HTML}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
