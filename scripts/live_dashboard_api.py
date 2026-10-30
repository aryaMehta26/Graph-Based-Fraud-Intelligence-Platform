"""Small read-only/live bridge for the React DATA 298B dashboard.

The prediction endpoint replays a bounded prefix of the existing chronological
test Parquet and scores it with the saved graph-enhanced XGBoost model. The
investigation endpoint loads the existing Gemma adapter lazily and is only
enabled when DATA298B_LIVE_GEMMA=1.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_PATH = Path(os.getenv("DATA298B_LIVE_DATA", str(ROOT / "data/processed/test_graph_enriched.parquet")))
MODEL_PATH = Path(os.getenv("DATA298B_LIVE_MODEL", str(ROOT / "data/models/xgboost_graph_enhanced.json")))
PORT = int(os.getenv("DATA298B_LIVE_API_PORT", "8765"))
MODEL_ID = os.getenv("DATA298B_GEMMA_MODEL_ID", "google/gemma-4-31B")
ADAPTER_DIR = os.getenv("DATA298B_GEMMA_ADAPTER_DIR", "")
LIVE_GEMMA = os.getenv("DATA298B_LIVE_GEMMA", "0").lower() in {"1", "true", "yes"}

BASELINE_FEATURES = [
    "log_amount", "is_ACH", "is_Cheque", "is_CC", "is_Wire", "is_Bitcoin",
    "hour", "dow", "is_weekend", "is_cross_currency", "amount_bucket",
]
GRAPH_FEATURES = [
    "src_out_degree", "src_in_degree", "src_total_degree", "src_degree_centrality",
    "dst_out_degree", "dst_in_degree", "dst_total_degree", "dst_degree_centrality",
    "src_community_size", "src_community_fraud_rate", "dst_community_size",
    "dst_community_fraud_rate",
]
FEATURES = BASELINE_FEATURES + GRAPH_FEATURES
DISPLAY_COLUMNS = ["Timestamp", "src_acct", "dst_acct", "Amount Paid", "Payment Format"]

_lock = threading.Lock()
_cursor = 0
_rows: list[dict] = []
_graph_cache: dict[str, dict] = {}
_model = None
_gemma_backend = None


def _jsonable(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _load_replay_rows(limit: int = 2000) -> None:
    global _rows
    if _rows or not DATA_PATH.exists():
        return
    parquet = pq.ParquetFile(DATA_PATH)
    columns = list(dict.fromkeys(DISPLAY_COLUMNS + FEATURES))
    batch = next(parquet.iter_batches(batch_size=limit, columns=columns)).to_pandas()
    if batch.empty:
        return
    _load_model()
    features = batch[FEATURES].copy()
    if "amount_bucket" in features.columns and not pd.api.types.is_numeric_dtype(features["amount_bucket"]):
        features["amount_bucket"] = features["amount_bucket"].astype("category").cat.codes
    import xgboost as xgb

    probabilities = _model.predict(xgb.DMatrix(features.astype("float32")))
    for index, (_, row) in enumerate(batch.iterrows()):
        probability = float(probabilities[index])
        payload = {key: _jsonable(row[key]) for key in DISPLAY_COLUMNS}
        payload.update({
            "case_id": f"live-{index}-{str(row['Timestamp']).replace(':', '').replace(' ', 'T')}",
            "xgboost_score": probability,
            "fraud_probability": probability,
            "risk_level": "HIGH" if probability >= 0.7 else "MEDIUM" if probability >= 0.3 else "LOW",
            "prediction": "SUSPICIOUS" if probability >= 0.3 else "LEGITIMATE",
            "source": "replayed test transaction",
            "investigatable": probability >= 0.3,
        })
        _rows.append(payload)


def _load_model():
    global _model
    if _model is not None:
        return
    from xgboost import Booster

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"XGBoost model not found: {MODEL_PATH}")
    _model = Booster()
    _model.load_model(MODEL_PATH)


def _next_predictions(limit: int) -> list[dict]:
    global _cursor
    with _lock:
        _load_replay_rows()
        if not _rows:
            return []
        limit = max(1, min(limit, 50))
        result = [_rows[(_cursor + offset) % len(_rows)] for offset in range(limit)]
        _cursor = (_cursor + limit) % len(_rows)
        return result


def _graph_neighborhood(account_id: str, limit: int = 100) -> dict:
    """Return a bounded read-only neighborhood from the replay Parquet.

    This keeps the React Network page on the same live API port. It uses only
    transaction rows from the configured replay source and never reads labels.
    """
    account_id = str(account_id)
    if account_id in _graph_cache:
        return _graph_cache[account_id]
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Replay source not found: {DATA_PATH}")
    columns = ["Timestamp", "src_acct", "dst_acct", "Amount Paid", "Payment Format"]
    edges: list[dict] = []
    parquet = pq.ParquetFile(DATA_PATH)
    for batch in parquet.iter_batches(batch_size=20000, columns=columns):
        frame = batch.to_pandas()
        matched = frame[(frame["src_acct"].astype(str) == account_id) | (frame["dst_acct"].astype(str) == account_id)]
        for _, row in matched.head(max(1, min(limit, 200))).iterrows():
            edges.append({
                "src_acct": str(row["src_acct"]),
                "dst_acct": str(row["dst_acct"]),
                "timestamp": _jsonable(row["Timestamp"]),
                "amount": _jsonable(row["Amount Paid"]),
                "payment_format": _jsonable(row["Payment Format"]),
            })
        if len(edges) >= limit:
            break
    out_degree = sum(edge["src_acct"] == account_id for edge in edges)
    in_degree = sum(edge["dst_acct"] == account_id for edge in edges)
    account = {
        "account_id": account_id,
        "out_degree": out_degree,
        "in_degree": in_degree,
        "total_degree": out_degree + in_degree,
        "degree_centrality": (out_degree + in_degree) / max(1758573 - 1, 1),
        "community_id": "Unavailable",
        "community_size": 0,
    }
    payload = {"account": account, "edges": edges, "source": "live replay Parquet"}
    _graph_cache[account_id] = payload
    return payload


def _load_gemma_backend():
    global _gemma_backend
    if not LIVE_GEMMA:
        raise RuntimeError("Live Gemma is disabled. Set DATA298B_LIVE_GEMMA=1 before starting the API.")
    if not ADAPTER_DIR or not Path(ADAPTER_DIR).exists():
        raise RuntimeError(f"Gemma adapter path is unavailable: {ADAPTER_DIR or '<empty>'}")
    from src.aml.backend import TransformersBackend

    if _gemma_backend is None:
        _gemma_backend = TransformersBackend(
            MODEL_ID,
            adapter_path=ADAPTER_DIR,
            load_in_4bit=True,
            device_map="auto",
        )
    return _gemma_backend


def _run_gemma(payload: dict) -> dict:
    from src.aml.agent import AMLAgent
    from src.aml.schemas import InvestigationCase

    backend = _load_gemma_backend()
    case_id = str(payload.get("case_id") or f"live_case_{int(time.time())}")
    context = payload.get("context") if isinstance(payload.get("context"), dict) else payload
    case = InvestigationCase(case_id=case_id, context=context, ground_truth={}, split="live")
    trace_dir = Path(os.getenv("DATA298B_GEMMA_TRACE_DIR", str(ROOT / "artifacts/live_gemma_traces")))
    trace = AMLAgent(backend, max_tool_calls=1, artifact_dir=str(trace_dir)).investigate(
        case,
        model_family="gemma",
        variant="finetuned_live",
        seed=42,
        inference_config={"max_tokens": 700, "temperature": 0.0},
    )
    return {"report": trace.get("report", {}), "latency_seconds": trace.get("latency_seconds", 0), "case_id": case_id}


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload: dict, status: int = 200):
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # noqa: N802
        self._send({"ok": True})

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/live/status":
            self._send({
                "status": "online",
                "data_available": DATA_PATH.exists(),
                "model_available": MODEL_PATH.exists(),
                "replay_rows_loaded": len(_rows),
                "gemma_enabled": LIVE_GEMMA,
                "gemma_loaded": _gemma_backend is not None,
            })
            return
        if parsed.path == "/api/live/warmup":
            try:
                _load_gemma_backend()
                self._send({"status": "ready", "gemma_loaded": True})
            except Exception as exc:
                self._send({"error": str(exc), "gemma_loaded": False}, 500)
            return
        if parsed.path == "/api/live/predictions":
            try:
                limit = int(parse_qs(parsed.query).get("limit", ["8"])[0])
                self._send({"items": _next_predictions(limit)})
            except Exception as exc:
                self._send({"error": str(exc)}, 500)
            return
        if parsed.path.startswith("/api/graph/"):
            try:
                account_id = parsed.path.rsplit("/", 1)[-1]
                limit = int(parse_qs(parsed.query).get("limit", ["100"])[0])
                self._send(_graph_neighborhood(account_id, limit))
            except Exception as exc:
                self._send({"error": str(exc)}, 500)
            return
        self._send({"error": "not_found"}, 404)

    def do_POST(self):  # noqa: N802
        if self.path != "/api/live/investigate":
            self._send({"error": "not_found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or "{}")
            self._send(_run_gemma(payload))
        except Exception as exc:
            self._send({"error": str(exc)}, 500)


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Live dashboard API listening on 0.0.0.0:{PORT}", flush=True)
    print(f"Replay source: {DATA_PATH}", flush=True)
    print(f"Live Gemma enabled: {LIVE_GEMMA}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
