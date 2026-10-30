"""Normalize the supplied agentic LLM traces for the local dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "agentic_models"
OUTPUT = ROOT / "frontend" / "src" / "data" / "agentic_reports.json"

MODEL_LABELS = {
    "gemma": "Gemma",
    "qwen": "Qwen",
    "ministral": "Ministral",
    "phi": "Phi",
    "llama31_8b": "Llama 3.1 8B",
}


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or not str(value).strip():
        return []
    return [str(value)]


def normalize_trace(path: Path, model: str) -> dict[str, Any] | None:
    try:
        trace = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    report = trace.get("report") if isinstance(trace, dict) else None
    if not isinstance(report, dict):
        return None
    return {
        "id": f"{model}-{trace.get('case_id', path.stem)}-{path.stem[:8]}",
        "model": MODEL_LABELS[model],
        "modelKey": model,
        "caseId": str(trace.get("case_id", path.stem)),
        "riskLevel": str(report.get("risk_level", "UNKNOWN")).upper(),
        "decision": str(report.get("decision", "UNKNOWN")).upper(),
        "pattern": str(report.get("pattern", "UNKNOWN")),
        "confidence": float(report.get("confidence", 0) or 0),
        "summary": str(report.get("summary", "")),
        "evidence": as_list(report.get("evidence")),
        "actions": as_list(report.get("recommended_actions")),
        "latencySeconds": float(trace.get("latency_seconds", 0) or 0),
        "timestamp": trace.get("timestamp"),
    }


def main() -> None:
    reports: list[dict[str, Any]] = []
    for model in MODEL_LABELS:
        for path in sorted((SOURCE / f"agentic_{model}" / "traces").glob("*.json")):
            item = normalize_trace(path, model)
            if item:
                reports.append(item)
    OUTPUT.write_text(json.dumps(reports, indent=2) + "\n")
    print(f"Wrote {OUTPUT} ({len(reports)} normalized reports)")


if __name__ == "__main__":
    main()
