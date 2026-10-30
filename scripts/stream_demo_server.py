"""Local SSE stream for replayed real pipeline transactions.

The default source replays rows from the processed graph-enriched test split
and scores them with the saved graph-enhanced XGBoost model. Synthetic events
are used only when the processed split is unavailable.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data" / "models" / "xgboost_graph_enhanced.json"
SNAPSHOT_PATH = ROOT / "frontend" / "src" / "data" / "dashboard.json"
HOST = os.getenv("LIVE_STREAM_HOST", "127.0.0.1")
PORT = int(os.getenv("LIVE_STREAM_PORT", "8765"))
FEATURES = [
    "log_amount", "is_ACH", "is_Cheque", "is_CC", "is_Wire", "is_Bitcoin",
    "hour", "dow", "is_weekend", "is_cross_currency", "amount_bucket",
    "src_out_degree", "src_in_degree", "src_total_degree", "src_degree_centrality",
    "dst_out_degree", "dst_in_degree", "dst_total_degree", "dst_degree_centrality",
    "src_community_size", "src_community_fraud_rate",
    "dst_community_size", "dst_community_fraud_rate",
]
PAYMENT_FORMATS = ("ACH", "Wire", "Credit Card", "Cheque", "Bitcoin")
REAL_SOURCE_PATH = ROOT / "data" / "processed" / "test_graph_enriched.parquet"
if not REAL_SOURCE_PATH.exists():
    REAL_SOURCE_PATH = ROOT / "data" / "processed" / "split_test.parquet"


def read_snapshot() -> dict:
    try:
        return json.loads(SNAPSHOT_PATH.read_text())
    except Exception:
        return {"accounts": [], "graphEdges": []}


SNAPSHOT = read_snapshot()
ACCOUNT_FEATURES = {str(row.get("account_id")): row for row in SNAPSHOT.get("accounts", [])}
EDGE_ROWS = SNAPSHOT.get("graphEdges", [])
ACCOUNT_IDS = sorted({str(value) for row in EDGE_ROWS for value in (row.get("src_acct"), row.get("dst_acct")) if value} | set(ACCOUNT_FEATURES))


def account_features(account_id: str, rng: random.Random) -> dict[str, float]:
    row = ACCOUNT_FEATURES.get(account_id, {})
    if row:
        return {
            "out_degree": float(row.get("out_degree", 0)),
            "in_degree": float(row.get("in_degree", 0)),
            "total_degree": float(row.get("total_degree", 0)),
            "degree_centrality": float(row.get("degree_centrality", 0)),
            "community_size": float(row.get("community_size", 0)),
            "community_fraud_rate": float(row.get("community_fraud_rate", 0)),
        }
    out_degree = float(rng.randint(1, 12))
    in_degree = float(rng.randint(1, 12))
    return {
        "out_degree": out_degree,
        "in_degree": in_degree,
        "total_degree": out_degree + in_degree,
        "degree_centrality": (out_degree + in_degree) / 1_758_572,
        "community_size": float(rng.randint(2, 80)),
        "community_fraud_rate": rng.uniform(0, 0.02),
    }


def build_features(event: dict, rng: random.Random) -> pd.DataFrame:
    stamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
    payment = event["payment_format"]
    source = account_features(event["src_acct"], rng)
    destination = account_features(event["dst_acct"], rng)
    amount = float(event["amount"])
    values = {
        "log_amount": math.log1p(amount),
        "is_ACH": int(payment == "ACH"), "is_Cheque": int(payment == "Cheque"),
        "is_CC": int(payment == "Credit Card"), "is_Wire": int(payment == "Wire"),
        "is_Bitcoin": int(payment == "Bitcoin"), "hour": stamp.hour,
        "dow": stamp.weekday(), "is_weekend": int(stamp.weekday() >= 5),
        "is_cross_currency": int(event["cross_currency"]),
        "amount_bucket": min(4, max(0, int(math.log10(max(amount, 1))))),
        "src_out_degree": source["out_degree"], "src_in_degree": source["in_degree"],
        "src_total_degree": source["total_degree"], "src_degree_centrality": source["degree_centrality"],
        "dst_out_degree": destination["out_degree"], "dst_in_degree": destination["in_degree"],
        "dst_total_degree": destination["total_degree"], "dst_degree_centrality": destination["degree_centrality"],
        "src_community_size": source["community_size"], "src_community_fraud_rate": source["community_fraud_rate"],
        "dst_community_size": destination["community_size"], "dst_community_fraud_rate": destination["community_fraud_rate"],
    }
    return pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)


def normalize_model_row(row: dict) -> pd.DataFrame:
    """Convert parquet feature values to the numeric schema expected by XGBoost."""
    bucket = row.get("amount_bucket", 0)
    bucket_codes = {"<100": 0, "100-500": 1, "500-1K": 2, "1K-5K": 3, "5K-10K": 4, ">=10K": 5}
    values = {name: row.get(name, 0) for name in FEATURES}
    values["amount_bucket"] = bucket_codes.get(str(bucket), float(bucket) if str(bucket).replace('.', '', 1).isdigit() else 0)
    return pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)


def load_model():
    if XGBClassifier is None or not MODEL_PATH.exists():
        return None
    try:
        model = XGBClassifier()
        model.load_model(str(MODEL_PATH))
        return model
    except Exception as exc:  # pragma: no cover
        print(f"Warning: model unavailable, using deterministic demo scores: {exc}")
        return None


MODEL = load_model()
REAL_ROWS: list[dict] = []
try:
    if REAL_SOURCE_PATH.exists():
        columns = ["Timestamp", "src_acct", "dst_acct", "Amount Paid", "Payment Format", "Receiving Currency", "Is Laundering"] + FEATURES
        available = [name for name in columns if name]
        REAL_ROWS = pd.read_parquet(REAL_SOURCE_PATH, columns=available).head(2000).to_dict("records")
except Exception as exc:  # pragma: no cover
    print(f"Warning: real replay unavailable, using fallback events: {exc}")


def make_real_event(sequence: int, row: dict) -> dict:
    stamp = row.get("Timestamp")
    timestamp = stamp.isoformat() if hasattr(stamp, "isoformat") else str(stamp)
    if not timestamp.endswith("Z"):
        timestamp = timestamp.replace("+00:00", "Z")
    return {
        "event_id": f"LIVE-{sequence:06d}", "timestamp": timestamp,
        "src_acct": str(row.get("src_acct", "")), "dst_acct": str(row.get("dst_acct", "")),
        "amount": float(row.get("Amount Paid", 0) or 0),
        "payment_format": str(row.get("Payment Format", "Unknown")),
        "cross_currency": str(row.get("Receiving Currency", "")) != "",
        "source": "pipeline_test_split_replay", "_features": row,
        "ground_truth": "Fraud" if int(row.get("Is Laundering", 0) or 0) == 1 else "Legitimate",
    }


def make_event(sequence: int, rng: random.Random) -> dict:
    if REAL_ROWS:
        event = make_real_event(sequence, REAL_ROWS[(sequence - 1) % len(REAL_ROWS)])
        if MODEL is not None:
            try:
                score = float(MODEL.predict_proba(normalize_model_row(event["_features"]))[0, 1])
            except Exception as exc:  # pragma: no cover
                print(f"Warning: replay prediction failed: {exc}")
                score = 0.0
        else:
            score = 0.0
        event.pop("_features", None)
        event["risk_score"] = round(max(0.0, min(1.0, score)), 4)
        event["prediction"] = "Suspicious" if score >= 0.30 else "Legitimate"
        event["risk_level"] = "High" if score >= 0.75 else "Medium" if score >= 0.30 else "Low"
        return event
    source = rng.choice(ACCOUNT_IDS or ["DEMO-SOURCE"])
    destination = rng.choice(ACCOUNT_IDS or ["DEMO-DESTINATION"])
    while destination == source and len(ACCOUNT_IDS) > 1:
        destination = rng.choice(ACCOUNT_IDS)
    amount = round(rng.choice([rng.uniform(25, 4000), rng.uniform(9000, 9999), rng.uniform(10000, 250000)]), 2)
    event = {
        "event_id": f"LIVE-{sequence:06d}",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "src_acct": source, "dst_acct": destination, "amount": amount,
        "payment_format": rng.choice(PAYMENT_FORMATS), "cross_currency": rng.random() < 0.12,
        "source": "synthetic_demo_stream",
    }
    if MODEL is not None:
        try:
            score = float(MODEL.predict_proba(build_features(event, rng))[0, 1])
        except Exception as exc:  # pragma: no cover
            print(f"Warning: prediction failed, using demo score: {exc}")
            score = 0.85 if 9800 <= amount < 10000 else rng.uniform(0.01, 0.35)
    else:
        score = 0.88 if 9800 <= amount < 10000 else rng.uniform(0.01, 0.35)
    event["risk_score"] = round(max(0.0, min(1.0, score)), 4)
    event["prediction"] = "Suspicious" if score >= 0.30 else "Legitimate"
    event["risk_level"] = "High" if score >= 0.75 else "Medium" if score >= 0.30 else "Low"
    return event


def attach_live_case(event: dict, groups: list[dict], next_case: list[int]) -> dict:
    """Attach an explainable rolling case to a replayed transaction."""
    accounts = {event["src_acct"], event["dst_acct"]}
    group = next((item for item in reversed(groups) if accounts & item["accounts"]), None)
    if group is None:
        group = {"case_id": f"LIVE-CASE-{next_case[0]:04d}", "accounts": set(), "transactions": []}
        next_case[0] += 1
        groups.append(group)
    group["transactions"].append(event.copy())
    group["transactions"] = group["transactions"][-8:]
    group["accounts"] = {value for tx in group["transactions"] for value in (tx["src_acct"], tx["dst_acct"])}
    sources = {tx["src_acct"] for tx in group["transactions"]}
    destinations = {tx["dst_acct"] for tx in group["transactions"]}
    if len(destinations) == 1 and len(sources) > 1:
        pattern = "FAN_IN"
    elif len(sources) == 1 and len(destinations) > 1:
        pattern = "FAN_OUT"
    elif any(tx["src_acct"] == event["dst_acct"] and tx["dst_acct"] == event["src_acct"] for tx in group["transactions"][:-1]):
        pattern = "CYCLE"
    else:
        pattern = "CONNECTED_ACTIVITY"
    case_transactions = [{key: tx[key] for key in ("event_id", "timestamp", "src_acct", "dst_acct", "amount", "payment_format", "risk_score", "prediction", "risk_level")} for tx in group["transactions"]]
    event.update({"case_id": group["case_id"], "case_pattern": pattern, "case_transaction_ids": [tx["event_id"] for tx in group["transactions"]], "case_account_ids": sorted(group["accounts"]), "case_transactions": case_transactions})
    del groups[:-20]
    return event


class StreamHandler(BaseHTTPRequestHandler):
    server_version = "FraudLiveDemo/1.0"

    def send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            mode = "pipeline_test_split_replay" if REAL_ROWS else "synthetic_fallback"
            body = json.dumps({"ok": True, "model_loaded": MODEL is not None, "mode": mode, "source": str(REAL_SOURCE_PATH)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors()
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path != "/api/stream":
            self.send_response(404)
            self.end_headers()
            return
        query = parse_qs(parsed.query)
        interval = max(0.5, min(float(query.get("interval", ["2"])[0]), 10.0))
        count = int(query.get("count", ["0"])[0])
        rng = random.Random()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "keep-alive" if count <= 0 else "close")
        self.send_cors()
        self.end_headers()
        sequence = 1
        case_groups: list[dict] = []
        next_case = [1]
        try:
            while count <= 0 or sequence <= count:
                event = attach_live_case(make_event(sequence, rng), case_groups, next_case)
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
                sequence += 1
                time.sleep(interval)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args) -> None:
        print(f"[live-stream] {self.address_string()} - {format % args}")


if __name__ == "__main__":
    print(f"Live prediction stream: http://{HOST}:{PORT}")
    print(f"Model loaded: {MODEL is not None} ({MODEL_PATH})")
    print(f"Replay source: {REAL_SOURCE_PATH} ({len(REAL_ROWS)} buffered rows)")
    print("Stream endpoint: /api/stream?interval=2")
    ThreadingHTTPServer((HOST, PORT), StreamHandler).serve_forever()
