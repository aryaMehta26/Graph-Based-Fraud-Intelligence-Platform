"""Small read-only/live bridge for the React DATA 298B dashboard.

The prediction endpoint replays a bounded prefix of the existing chronological
test Parquet and scores it with the saved graph-enhanced XGBoost model. The
investigation endpoint loads the existing Gemma adapter lazily and is only
enabled when DATA298B_LIVE_GEMMA=1.
"""

from __future__ import annotations

import json
import os
import re
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


def _build_live_case_context(context: dict) -> dict:
    """Turn one replay alert into a bounded, point-in-time investigation case."""
    transactions = context.get("transactions") if isinstance(context.get("transactions"), list) else []
    if not transactions:
        return context
    selected = transactions[0]
    account_ids = {
        str(account.get("account_id"))
        for account in (context.get("accounts") or [])
        if isinstance(account, dict) and account.get("account_id") is not None
    }
    for key in ("src_acct", "dst_acct"):
        if selected.get(key) is not None:
            account_ids.add(str(selected[key]))
    cutoff = pd.to_datetime(selected.get("Timestamp"), errors="coerce")
    candidates = []
    for row in _rows:
        row_src = str(row.get("src_acct"))
        row_dst = str(row.get("dst_acct"))
        row_time = pd.to_datetime(row.get("Timestamp"), errors="coerce")
        if pd.notna(cutoff) and pd.notna(row_time) and row_time > cutoff:
            continue
        if row_src not in account_ids and row_dst not in account_ids:
            continue
        candidates.append({key: row.get(key) for key in DISPLAY_COLUMNS})
    # Keep the prompt close to the trained case format while still giving Gemma
    # historical context beyond the selected alert.
    candidates = candidates[-7:]
    selected_key = tuple(selected.get(key) for key in DISPLAY_COLUMNS)
    case_transactions = [selected]
    for row in candidates:
        row_key = tuple(row.get(key) for key in DISPLAY_COLUMNS)
        if row_key != selected_key:
            case_transactions.append(row)
    case_transactions = case_transactions[:25]
    case_accounts = set(account_ids)
    for row in case_transactions:
        case_accounts.update({str(row.get("src_acct")), str(row.get("dst_acct"))})
    enriched = dict(context)
    enriched["transactions"] = case_transactions
    enriched["accounts"] = [{"account_id": account_id} for account_id in sorted(case_accounts)]
    graph_stats = dict(context.get("graph_stats") or {})
    graph_stats.update({
        "case_window_transactions": len(case_transactions),
        "case_window_accounts": len(case_accounts),
        "case_window_cutoff": _jsonable(selected.get("Timestamp")),
        "case_window_source": "prior replay transactions involving selected accounts",
    })
    enriched["graph_stats"] = graph_stats
    evidence = list(enriched.get("retrieved_evidence") or [])
    evidence.append(
        f"Case window contains {len(case_transactions)} transaction(s) and {len(case_accounts)} account(s) at or before the selected alert timestamp."
    )
    enriched["retrieved_evidence"] = evidence
    return enriched


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


def _recover_report_from_raw_response(trace: dict) -> tuple[dict, str]:
    """Recover valid JSON when an older AMLAgent parser returns defaults."""
    report = trace.get("report") or {}
    raw = str(trace.get("raw_response") or "")
    if not raw:
        return report, "agent_parser"
    try:
        # Decode the first complete JSON object. Some older model/parser
        # combinations append repeated or malformed text after the valid
        # object; a greedy `{.*}` regex incorrectly rejects that response.
        start = raw.find("{")
        parsed, _ = json.JSONDecoder().raw_decode(raw[start:]) if start >= 0 else (None, 0)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    required = {"decision", "pattern", "risk_level", "confidence", "evidence", "recommended_actions", "summary"}
    if isinstance(parsed, dict) and required.issubset(parsed):
        return parsed, "raw_model_json"

    # A truncated response can still contain reliable leading fields. Recover
    # only values explicitly emitted by the model and mark the result partial;
    # never manufacture confidence or evidence.
    def field(pattern: str, default=None):
        match = re.search(pattern, raw, re.S)
        return match.group(1) if match else default

    decision = field(r'"decision"\s*:\s*"([^"]+)"')
    pattern = field(r'"pattern"\s*:\s*"([^"]+)"')
    risk = field(r'"risk_level"\s*:\s*"([^"]+)"')
    confidence_text = field(r'"confidence"\s*:\s*([-+]?\d*\.?\d+)')
    evidence_match = re.search(r'"evidence"\s*:\s*\[(.*?)(?:\]|$)', raw, re.S)
    evidence = []
    if evidence_match:
        for item in re.finditer(r'"((?:\\.|[^"\\])*)"', evidence_match.group(1)):
            try:
                value = json.loads('"' + item.group(1) + '"')
            except json.JSONDecodeError:
                continue
            if value and value not in evidence:
                evidence.append(value)
            if len(evidence) >= 5:
                break
    if decision and pattern and risk and confidence_text and evidence:
        confidence = float(confidence_text)
        return {
            "decision": decision,
            "pattern": pattern,
            "risk_level": risk,
            "confidence": confidence,
            "evidence": evidence,
            "recommended_actions": [],
            "summary": f"Gemma identified a {pattern} pattern with {risk} risk; corroborate the partial report in the network view.",
        }, "raw_model_partial"
    return report, "agent_parser"


def _run_gemma(payload: dict) -> dict:
    import inspect
    from src.aml.agent import AMLAgent
    from src.aml.schemas import InvestigationCase

    backend = _load_gemma_backend()
    case_id = str(payload.get("case_id") or f"live_case_{int(time.time())}")
    context = payload.get("context") if isinstance(payload.get("context"), dict) else payload
    _load_replay_rows()
    context = _build_live_case_context(context)
    case = InvestigationCase(case_id=case_id, context=context, ground_truth={}, split="live")
    trace_dir = Path(os.getenv("DATA298B_GEMMA_TRACE_DIR", str(ROOT / "artifacts/live_gemma_traces")))
    agent = AMLAgent(backend, max_tool_calls=3, artifact_dir=str(trace_dir))
    investigate_kwargs = {
        "model_family": "gemma",
        "variant": "finetuned_live",
        "seed": 42,
        "inference_config": {"max_tokens": 500, "temperature": 0.0},
    }
    # The live branch must work with both the older checked-in agent and the
    # newer agent that supports controlled evidence prefetching.
    if "prefetch_tools" in inspect.signature(agent.investigate).parameters:
        investigate_kwargs["prefetch_tools"] = True
    trace = agent.investigate(case, **investigate_kwargs)
    report, report_source = _recover_report_from_raw_response(trace)
    graph = context.get("graph_stats") if isinstance(context.get("graph_stats"), dict) else {}
    score = float(context.get("xgboost_score", context.get("fraud_probability", graph.get("xgboost_score", 0))) or 0)
    transactions = context.get("transactions") if isinstance(context.get("transactions"), list) else []
    accounts = context.get("accounts") if isinstance(context.get("accounts"), list) else []
    report_is_valid = report_source in {"raw_model_json", "raw_model_partial"}
    pattern = str(report.get("pattern") or "EMERGING_UNKNOWN")
    risk = str(report.get("risk_level") or "MEDIUM")
    if report_is_valid:
        rationale = (
            f"The transaction was escalated because the frozen XGBoost fraud score was {score:.3f}, "
            "above the live review threshold of 0.300. "
            f"Gemma classified the grounded case as {pattern} with {risk} risk "
            f"using {len(transactions)} observed transaction(s), {len(accounts)} account(s), and the retrieved graph context. "
            "The pattern is an investigation hypothesis that should be corroborated in the linked network view; "
            "a small replay window is not, by itself, proof of coordinated fraud."
        )
    else:
        rationale = (
            f"The transaction was escalated because the frozen XGBoost fraud score was {score:.3f}, "
            "above the live review threshold of 0.300. Gemma did not return a valid structured report "
            "for this request, so no Gemma pattern, confidence, or narrative is being asserted. "
            "Review the transaction window and network evidence before making a disposition."
        )
    report["rationale"] = rationale
    out_degree = int(graph.get("out_degree", 0) or 0)
    in_degree = int(graph.get("in_degree", 0) or 0)
    if pattern == "FAN_OUT" and out_degree >= 3 and out_degree > in_degree:
        structural_reason = (
            f"The selected account has {out_degree} outgoing versus {in_degree} incoming connections, "
            "so funds are moving outward to multiple counterparties; this supports Gemma's FAN_OUT hypothesis."
        )
    elif pattern == "FAN_IN" and in_degree >= 3 and in_degree > out_degree:
        structural_reason = (
            f"The selected account has {in_degree} incoming versus {out_degree} outgoing connections, "
            "so funds are concentrating into one account; this supports Gemma's FAN_IN hypothesis."
        )
    elif pattern == "FAN_OUT" and in_degree > out_degree:
        structural_reason = (
            f"The selected account has {in_degree} incoming versus {out_degree} outgoing connections, "
            "which is more consistent with inbound concentration than FAN_OUT; the graph evidence does not support Gemma's FAN_OUT prediction."
        )
    elif pattern == "FAN_IN" and out_degree > in_degree:
        structural_reason = (
            f"The selected account has {out_degree} outgoing versus {in_degree} incoming connections, "
            "which is more consistent with outward distribution than FAN_IN; the graph evidence does not support Gemma's FAN_IN prediction."
        )
    else:
        structural_reason = (
            f"The selected account has {out_degree} outgoing and {in_degree} incoming connections; "
            f"the available topology is not sufficient to confirm Gemma's {pattern} prediction."
        )
    if report_is_valid:
        rationale_points = [
            f"XGBoost escalated the alert with a fraud score of {score:.3f}, above the 0.300 review threshold.",
            structural_reason,
            f"Gemma classified the grounded case as {pattern} with {risk} risk using {len(transactions)} transaction(s) across {len(accounts)} account(s).",
            "This is an investigation hypothesis, not proof of fraud; corroborate it in the network view and transaction history.",
        ]
    else:
        rationale_points = [
            f"XGBoost escalated the alert with a fraud score of {score:.3f}, above the 0.300 review threshold.",
            structural_reason.replace(f"Gemma's {pattern} prediction", "a specific Gemma pattern prediction"),
            f"The grounded live window contains {len(transactions)} transaction(s) across {len(accounts)} account(s); inspect the linked network and transaction history.",
            "Gemma did not return a valid structured report for this request, so its pattern and confidence are unavailable.",
        ]
    return {
        "report": report,
        "latency_seconds": trace.get("latency_seconds", 0),
        "case_id": case_id,
        "report_source": report_source,
    }


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
