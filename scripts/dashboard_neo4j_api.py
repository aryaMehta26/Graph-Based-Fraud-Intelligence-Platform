"""Read-only local API bridge for the React dashboard and Neo4j.

Run from the repository root:
    python3 scripts/dashboard_neo4j_api.py

This service never writes to Neo4j. It exposes only health, database counts,
and a bounded account-neighborhood query for the local demo.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
def read_env_file(path: Path) -> dict[str, str]:
    """Read the small local .env file without requiring python-dotenv."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


ENV = read_env_file(ROOT / ".env")
NEO4J_URI = os.getenv("NEO4J_URI", ENV.get("NEO4J_URI", "bolt://127.0.0.1:7687"))
NEO4J_USER = os.getenv("NEO4J_USER", ENV.get("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", ENV.get("NEO4J_PASSWORD", ""))
NEO4J_DATABASE = os.getenv("NEO4J_DB", ENV.get("NEO4J_DB", "neo4j"))
PORT = int(os.getenv("DASHBOARD_API_PORT", "8765"))
ACCOUNT_NODE_COUNT = int(os.getenv("NEO4J_ACCOUNT_COUNT", "1758573"))


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


class Handler(BaseHTTPRequestHandler):
    driver = None

    def send_json(self, payload, status=200):
        body = json.dumps(jsonable(payload), default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        try:
            with self.driver.session(database=NEO4J_DATABASE) as session:
                if parsed.path == "/api/health":
                    row = session.run("RETURN 1 AS ok").single()
                    self.send_json({"status": "online", "database": NEO4J_DATABASE, "ok": row["ok"]})
                    return
                if parsed.path == "/api/neo4j/stats":
                    row = session.run(
                        "MATCH (n) WITH count(n) AS nodes "
                        "OPTIONAL MATCH ()-[r]->() RETURN nodes, count(r) AS relationships"
                    ).single()
                    self.send_json({"database": NEO4J_DATABASE, "nodes": row["nodes"], "relationships": row["relationships"]})
                    return
                if parsed.path.startswith("/api/graph/"):
                    account_id = parsed.path.rsplit("/", 1)[-1]
                    limit = min(int(parse_qs(parsed.query).get("limit", ["100"])[0]), 200)
                    account = session.run(
                        "MATCH (a:Account {account_id: $account_id}) RETURN a LIMIT 1",
                        account_id=account_id,
                    ).single()
                    if account is None:
                        self.send_json({"error": "account_not_found", "account_id": account_id}, 404)
                        return
                    stats = session.run(
                        "MATCH (a:Account {account_id: $account_id}) "
                        "CALL (a) { MATCH (a)-[:SENT]->(out_tx:Transaction)-[:RECEIVED_BY]->(:Account) "
                        "RETURN count(DISTINCT out_tx) AS out_degree } "
                        "CALL (a) { MATCH (a)<-[:RECEIVED_BY]-(in_tx:Transaction)<-[:SENT]-(:Account) "
                        "RETURN count(DISTINCT in_tx) AS in_degree } "
                        "RETURN a, out_degree, in_degree",
                        account_id=account_id,
                    ).single()
                    account_payload = dict(stats["a"])
                    out_degree = int(stats["out_degree"])
                    in_degree = int(stats["in_degree"])
                    account_payload.update({
                        "out_degree": out_degree,
                        "in_degree": in_degree,
                        "total_degree": out_degree + in_degree,
                        "degree_centrality": (out_degree + in_degree) / max(ACCOUNT_NODE_COUNT - 1, 1),
                    })
                    edges = []
                    outgoing = session.run(
                        "MATCH (src:Account {account_id: $account_id})-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(dst:Account) "
                        "RETURN src.account_id AS src_acct, dst.account_id AS dst_acct, t AS transaction LIMIT $limit",
                        account_id=account_id, limit=limit,
                    )
                    incoming = session.run(
                        "MATCH (src:Account)-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(dst:Account {account_id: $account_id}) "
                        "RETURN src.account_id AS src_acct, dst.account_id AS dst_acct, t AS transaction LIMIT $limit",
                        account_id=account_id, limit=limit,
                    )
                    for row in list(outgoing) + list(incoming):
                        transaction = dict(row["transaction"])
                        edges.append({"src_acct": row["src_acct"], "dst_acct": row["dst_acct"], **transaction})
                    self.send_json({"account": account_payload, "edges": edges})
                    return
            self.send_json({"error": "not_found"}, 404)
        except Exception as exc:  # Keep the demo API from exposing credentials.
            self.send_json({"status": "offline", "error": type(exc).__name__}, 503)

    def log_message(self, format, *args):  # noqa: A002
        print(f"dashboard-api: {format % args}")


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), connection_timeout=5)
    driver.verify_connectivity()
    Handler.driver = driver
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Neo4j dashboard API: http://127.0.0.1:{PORT}")
    print(f"Database: {NEO4J_DATABASE}; read-only endpoints enabled")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        driver.close()


if __name__ == "__main__":
    main()
