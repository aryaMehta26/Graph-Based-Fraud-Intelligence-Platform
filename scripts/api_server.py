"""Additive local API for the DATA 298B dashboard.

This service reads existing artifacts, optionally queries Neo4j, replays real
processed transactions, and exposes one investigator contract for local
Transformers, Ollama, Claude, or saved reports. It never retrains or mutates
the fraud pipeline.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SNAPSHOT = ROOT / "frontend" / "src" / "data" / "dashboard.json"
REPORTS = ROOT / "frontend" / "src" / "data" / "agentic_reports.json"
REGISTRY = ROOT / "configs" / "models" / "registry.yaml"
ADAPTER_ROOT = ROOT / "artifacts" / "adapters"
HOST = os.getenv("DASHBOARD_API_HOST", "127.0.0.1")
PORT = int(os.getenv("DASHBOARD_API_PORT", "8766"))


def read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return fallback


def snapshot() -> dict[str, Any]:
    return read_json(SNAPSHOT, {})


def model_registry() -> dict[str, Any]:
    try:
        import yaml
        return yaml.safe_load(REGISTRY.read_text()) or {}
    except Exception:
        # Keep the API usable in the lightweight dashboard environment where
        # PyYAML is not installed. This deliberately parses only this flat
        # registry's scalar fields; the YAML file remains the source of truth.
        result: dict[str, dict[str, Any]] = {}
        current: dict[str, Any] | None = None
        for raw in REGISTRY.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if not raw.startswith(' ') and line.endswith(':'):
                current = result.setdefault(line[:-1], {})
                continue
            if current is None or ':' not in line:
                continue
            key, value = [part.strip() for part in line.split(':', 1)]
            value = value.split('#', 1)[0].strip()
            if value.lower() in {'true', 'false'}:
                current[key] = value.lower() == 'true'
            else:
                try:
                    current[key] = float(value) if '.' in value else int(value)
                except ValueError:
                    current[key] = value.strip('"\'')
        return result


def adapter_metadata() -> list[dict[str, Any]]:
    result = []
    for path in sorted((ROOT / "artifacts" / "adapters").glob("*/training_run.json")):
        item = read_json(path, {})
        result.append({"model": path.parent.name, "path": str(path.parent), **item})
    return result


def adapter_for_model(model: str) -> dict[str, Any]:
    aliases = {"gemma": "gemma_full_grounded", "granite": "granite_full_grounded", "qwen": "qwen_full_grounded", "phi": "phi_full_grounded", "ministral": "ministral_full_grounded"}
    key = aliases.get(model.lower().strip(), model)
    path = ADAPTER_ROOT / key
    metadata = read_json(path / "training_run.json", {})
    if not (path / "adapter_config.json").exists() or not (path / "adapter_model.safetensors").exists():
        raise LookupError(f"No fine-tuned adapter is installed for {model}")
    base_model = metadata.get("model", {}).get("model_id")
    if not base_model:
        raise LookupError(f"Adapter metadata is missing the base model for {model}")
    return {"name": key, "path": str(path), "base_model": base_model, "metadata": metadata}


def neo4j_driver():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        if not uri or not user or not password:
            return None
        return GraphDatabase.driver(uri, auth=(user, password))
    except Exception:
        return None


def fallback_account(account_id: str) -> dict[str, Any]:
    data = snapshot()
    account = next((item for item in data.get("accounts", []) if str(item.get("account_id")) == account_id), None)
    edges = [edge for edge in data.get("graphEdges", []) if account_id in {str(edge.get("src_acct")), str(edge.get("dst_acct"))}]
    account = account or {"account_id": account_id, "out_degree": 0, "in_degree": 0, "total_degree": len(edges), "degree_centrality": 0, "community_id": None, "community_size": 0, "community_fraud_rate": 0}
    return {"account": account, "edges": edges, "source": "pipeline_snapshot"}


def live_account(account_id: str) -> dict[str, Any]:
    driver = neo4j_driver()
    if driver is None:
        return fallback_account(account_id)
    try:
        with driver.session(database=os.getenv("NEO4J_DB", "neo4j")) as session:
            account_result = session.run("""
                MATCH (a:Account {account_id: $account_id})
                RETURN a.account_id AS account_id, a.out_degree AS out_degree,
                       a.in_degree AS in_degree, a.total_degree AS total_degree,
                       a.degree_centrality AS degree_centrality,
                       a.community_id AS community_id, a.community_size AS community_size,
                       a.community_fraud_rate AS community_fraud_rate
            """, account_id=account_id).single()
            edge_rows = session.run("""
                MATCH (a:Account {account_id: $account_id})-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(b:Account)
                RETURN t.txn_id AS txn_id, a.account_id AS src_acct, b.account_id AS dst_acct,
                       t.amount AS amount, t.timestamp AS timestamp
                LIMIT 500
            """, account_id=account_id).data()
            if account_result is None:
                return fallback_account(account_id)
            return {"account": dict(account_result), "edges": edge_rows, "source": "neo4j"}
    finally:
        driver.close()


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").removeprefix("json").removesuffix("```").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Some hosted models occasionally return prose or malformed JSON.
            # Preserve that response as a visible, non-failing investigation.
            pass
    return {
        "decision": "INVESTIGATE",
        "pattern": "UNPARSED_MODEL_RESPONSE",
        "risk_level": "MEDIUM",
        "confidence": 0.0,
        "evidence": [],
        "recommended_actions": ["Review the raw model response before taking action"],
        "summary": cleaned[:4000] or "The model returned no usable content.",
        "raw_output": text,
    }


def investigator_prompt(context: dict[str, Any]) -> str:
    return "Return only JSON with keys decision, pattern, risk_level, confidence, evidence, recommended_actions, summary. Ground every claim in this case context.\n\n" + json.dumps(context, default=str)


def run_ollama(model: str, context: dict[str, Any]) -> dict[str, Any]:
    url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/chat"
    payload = json.dumps({"model": model, "stream": False, "format": "json", "messages": [{"role": "system", "content": "You are an AML investigator. Return a grounded JSON investigation report."}, {"role": "user", "content": investigator_prompt(context)}]}).encode()
    request = urllib.request.Request(url, payload, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read())
    report = parse_json_response(data.get("message", {}).get("content", "{}"))
    report["provider"] = "ollama"
    report["model"] = model
    return report


def run_claude(model: str, context: dict[str, Any]) -> dict[str, Any]:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("your-"):
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(model=model, max_tokens=1800, system="You are an AML investigator. Return only one complete valid JSON object with exactly these keys: decision, pattern, risk_level, confidence, evidence, recommended_actions, summary. Keep evidence to at most 3 concise strings, recommended_actions to at most 3 concise strings, and summary under 120 words. Do not use Markdown, do not truncate, and do not include any text outside the JSON object.", messages=[{"role": "user", "content": json.dumps(context, default=str)}])
    report = parse_json_response(message.content[0].text)
    report["evidence"] = [item if isinstance(item, str) else item.get("description", json.dumps(item, default=str)) for item in report.get("evidence", [])]
    report["recommended_actions"] = [item if isinstance(item, str) else item.get("description", item.get("action", json.dumps(item, default=str))) for item in report.get("recommended_actions", [])]
    report["provider"] = "claude"
    report["model"] = model
    return report


_TRANSFORMER_AGENTS: dict[str, Any] = {}


def run_transformers(model: str, context: dict[str, Any]) -> dict[str, Any]:
    adapter = adapter_for_model(model)
    if adapter["name"].startswith("ministral_pilot"):
        raise RuntimeError("Select Ministral Full for live adapter inference; pilot adapters are retained for evaluation comparison")
    try:
        from src.aml.agent import AMLAgent
        from src.aml.backend import TransformersBackend
        from src.aml.schemas import InvestigationCase
    except ImportError as exc:
        raise RuntimeError("Install the local Transformers/PEFT inference dependencies before using fine-tuned adapters") from exc
    cache_key = adapter["name"]
    if cache_key not in _TRANSFORMER_AGENTS:
        inference_kwargs: dict[str, Any] = {"device_map": "auto"}
        if platform.system() == "Linux":
            inference_kwargs["load_in_4bit"] = True
        else:
            import torch
            inference_kwargs["torch_dtype"] = torch.float16
        backend = TransformersBackend(adapter["base_model"], adapter_path=adapter["path"], **inference_kwargs)
        _TRANSFORMER_AGENTS[cache_key] = AMLAgent(backend)
    case = InvestigationCase(case_id=str(context.get("case_id", "dashboard-case")), context=context)
    trace = _TRANSFORMER_AGENTS[cache_key].investigate(case, model_family=model.lower(), variant="finetuned", inference_config={"max_tokens": 700, "temperature": 0.0})
    report = dict(trace.get("report", {}))
    report.update({"provider": "transformers", "model": model, "adapter": adapter["name"], "adapter_path": adapter["path"], "latency_seconds": trace.get("latency_seconds", 0)})
    return report


def saved_investigation(model: str | None, case_id: str | None) -> dict[str, Any]:
    reports = read_json(REPORTS, [])
    matches = [item for item in reports if (not model or item.get("model") == model) and (not case_id or item.get("caseId") == case_id)]
    if not matches:
        raise LookupError("No saved investigation matched the requested model and case")
    return {**matches[0], "provider": "saved_artifact"}


def live_case_triage(context: dict[str, Any]) -> dict[str, Any]:
    """Provide an offline case result for the live demo button."""
    transactions = context.get("transactions") or []
    suspicious = [item for item in transactions if item.get("prediction") == "Suspicious"]
    max_risk = max((float(item.get("risk_score", 0)) for item in transactions), default=0.0)
    risk_level = "HIGH" if max_risk >= 0.75 or len(suspicious) >= 3 else "MEDIUM" if suspicious else "LOW"
    return {
        "case_id": context.get("case_id"),
        "decision": "INVESTIGATE" if suspicious else "MONITOR",
        "pattern": context.get("pattern", "CONNECTED_ACTIVITY"),
        "risk_level": risk_level,
        "confidence": round(max_risk, 3),
        "evidence": [
            f"Case contains {len(transactions)} connected replay transactions across {len(context.get('account_ids') or [])} accounts.",
            f"{len(suspicious)} transaction(s) currently exceed the suspicious threshold.",
            f"Highest pipeline risk score is {max_risk:.3f}.",
        ],
        "recommended_actions": ["Review the linked transactions and account network", "Run the case through Ollama or Claude for narrative analysis"],
        "summary": "Offline live-case triage generated from the streamed pipeline predictions. No external LLM was called.",
        "provider": "live_case_triage",
        "model": "Pipeline triage",
    }


def normalize_items(value: Any) -> list[str]:
    if not value:
        return []
    values = value if isinstance(value, list) else [value]
    normalized = []
    for item in values:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict):
            description = item.get("description") or item.get("action")
            if description:
                normalized.append(str(description))
            elif item.get("event_id"):
                normalized.append(
                    f"{item.get('event_id')}: {item.get('src_acct')} sent ${float(item.get('amount', 0)):,.2f} to {item.get('dst_acct')} with risk {float(item.get('risk_score', 0)):.3f} ({item.get('prediction', 'Unknown')})."
                )
            else:
                normalized.append(json.dumps(item, default=str))
        else:
            normalized.append(str(item))
    return normalized


def add_live_case_evidence(result: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Guarantee traceable transaction evidence for every live case result."""
    if not str(context.get("case_id", "")).startswith("LIVE-CASE-"):
        return result
    transactions = context.get("transactions") or []
    transaction_evidence = []
    for item in transactions[:3]:
        transaction_evidence.extend([
            f"Transaction {item.get('event_id', 'unknown')} moved ${float(item.get('amount', 0)):,.2f} from account {item.get('src_acct')} to account {item.get('dst_acct')} via {item.get('payment_format', 'unknown')}.",
            f"The pipeline assigned transaction {item.get('event_id', 'unknown')} a risk score of {float(item.get('risk_score', 0)):.3f} and prediction {item.get('prediction', 'Unknown')} ({item.get('risk_level', 'Unknown')} risk).",
        ])
    model_evidence = normalize_items(result.get("evidence"))
    result["evidence"] = list(dict.fromkeys(transaction_evidence + model_evidence))[:6]
    result["case_id"] = context.get("case_id")
    result["transaction_ids"] = context.get("transaction_ids") or [item.get("event_id") for item in transactions]
    result["account_ids"] = context.get("account_ids") or []
    result["recommended_actions"] = normalize_items(result.get("recommended_actions")) or ["Review the linked transactions and account network"]
    return result


def run_investigation(body: dict[str, Any]) -> dict[str, Any]:
    provider = body.get("provider", "saved")
    model = body.get("model")
    if provider == "saved":
        context = body.get("context") or {}
        if str(context.get("case_id", "")).startswith("LIVE-CASE-"):
            return live_case_triage(context)
        return saved_investigation(model, body.get("case_id"))
    context = body.get("context") or {}
    started = time.perf_counter()
    if provider == "ollama":
        aliases = {"Gemma": "gemma3", "Qwen": "qwen3", "Ministral": "ministral", "Phi": "phi4", "Llama 3.1 8B": "llama3.1"}
        result = run_ollama(os.getenv("OLLAMA_MODEL", aliases.get(model or "", model or "gemma3")), context)
    elif provider == "claude":
        # The UI model selector names the case/model family being compared;
        # Anthropic requires an actual Claude model identifier.
        claude_model = model if model and model.startswith("claude-") else os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
        result = run_claude(claude_model, context)
    elif provider == "transformers":
        result = run_transformers(model or "Gemma", context)
    else:
        raise RuntimeError(f"Unknown investigation provider: {provider}")
    result = add_live_case_evidence(result, context)
    result["latency_seconds"] = round(time.perf_counter() - started, 3)
    return result


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "FraudDashboardAPI/1.0"

    def send_api_headers(self, content_type="application/json"):
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache")

    def write_json(self, payload: Any, status=200):
        data = json.dumps(payload, default=str).encode()
        self.send_response(status); self.send_api_headers(); self.end_headers(); self.wfile.write(data)

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204); self.send_api_headers(); self.end_headers()

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health": return self.write_json({"ok": True, "neo4j": neo4j_driver() is not None, "providers": ["saved", "ollama", "claude", "transformers"]})
        if path == "/api/dashboard/snapshot": return self.write_json(snapshot())
        if path == "/api/models/comparison": return self.write_json(snapshot().get("llmComparison", []))
        if path == "/api/reports": return self.write_json(read_json(REPORTS, []))
        if path == "/api/model-registry": return self.write_json({"models": model_registry(), "adapters": adapter_metadata()})
        if path.startswith("/api/accounts/"):
            return self.write_json(live_account(path.rsplit("/", 1)[-1]))
        if path.startswith("/api/graph/"):
            return self.write_json(live_account(path.rsplit("/", 1)[-1]))
        if path == "/api/transactions":
            query = parse_qs(urlparse(self.path).query); limit = min(int(query.get("limit", [100])[0]), 1000)
            return self.write_json(snapshot().get("transactions", [])[:limit])
        return self.write_json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path != "/api/investigations": return self.write_json({"error": "not found"}, 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            return self.write_json(run_investigation(body))
        except (LookupError, RuntimeError, ValueError, urllib.error.URLError) as exc:
            return self.write_json({"error": str(exc)}, 503)
        except Exception as exc:
            return self.write_json({"error": f"investigation failed: {exc}"}, 500)

    def log_message(self, format, *args):
        print(f"[dashboard-api] {self.address_string()} - {format % args}")


if __name__ == "__main__":
    print(f"Dashboard API: http://{HOST}:{PORT}")
    print("Providers: saved artifacts, Ollama (optional), Claude (optional)")
    ThreadingHTTPServer((HOST, PORT), ApiHandler).serve_forever()
