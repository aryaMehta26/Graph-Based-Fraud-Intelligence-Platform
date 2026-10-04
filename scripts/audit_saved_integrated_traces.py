"""Audit saved integrated traces without loading models or rerunning inference.

This is intentionally a post-hoc quality audit.  It checks claims against the
full saved case context, rather than the old exact-string faithfulness metric.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


MODEL_FILES = {
    "ministral": "agentic_traces.jsonl",
    "phi": "agentic_traces (2).jsonl",
    "qwen": "agentic_traces (3).jsonl",
    "gemma": "agentic_traces (1).jsonl",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flatten(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False).lower()


def numeric_tokens(text: str) -> set[str]:
    # Keep decimal values and integer IDs, but ignore ordinary one-digit prose.
    return set(re.findall(r"(?<![\w])\d{2,}(?:,\d{3})*(?:\.\d+)?|(?<![\w])\d+\.\d+", text))


def identifier_tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"\b[A-Z0-9][A-Z0-9_-]{5,}\b", text.upper())
        if any(char.isdigit() for char in token)
    }


def source_facts(trace: dict[str, Any]) -> dict[str, set[str]]:
    context = trace.get("context", {})
    transactions = context.get("transactions", [])
    accounts = context.get("accounts", [])
    full = flatten(context)
    return {
        "text": {full},
        "numbers": numeric_tokens(full),
        "identifiers": identifier_tokens(full),
        "amounts": {str(t.get("amount")) for t in transactions},
        "txn_ids": {str(t.get("txn_id", "")) for t in transactions},
        "account_ids": {str(a.get("account_id", "")) for a in accounts}
        | {str(t.get("src_acct", "")) for t in transactions}
        | {str(t.get("dst_acct", "")) for t in transactions},
        "transactions": transactions,
    }


def audit_claim(claim: Any, facts: dict[str, set[str]]) -> str:
    text = str(claim)
    lower = text.lower()
    # Exact sentence support remains the strongest outcome.
    if lower in next(iter(facts["text"])):
        return "SUPPORTED"

    ids = identifier_tokens(text)
    if ids and not ids.issubset(facts["identifiers"]):
        return "UNSUPPORTED"

    # Any stated amount/date/degree number must occur in the source context.
    nums = numeric_tokens(text)
    source_nums = facts["numbers"]
    if nums and not nums.issubset(source_nums):
        # Permit a derived amount only when it matches the saved transactions.
        # This catches incorrect totals while allowing correctly computed ones.
        amounts = [
            float(t["amount"])
            for t in facts.get("transactions", [])
            if t.get("amount") is not None
        ]
        derived = ([sum(amounts), min(amounts), max(amounts)] if amounts else [])
        stated = [float(value.replace(",", "")) for value in nums if "." in value or "," in value]
        if stated and all(
            any(abs(value - candidate) <= max(0.01, abs(candidate) * 1e-6) for candidate in derived)
            for value in stated
        ):
            return "SUPPORTED"
        return "UNSUPPORTED"

    # A claim with source identifiers and source numbers is grounded enough for
    # this audit; prose-only qualitative claims are marked partial.
    if ids or nums:
        return "SUPPORTED"
    words = set(re.findall(r"[a-z]{4,}", lower))
    context_words = set(re.findall(r"[a-z]{4,}", next(iter(facts["text"]))))
    overlap = len(words & context_words) / max(len(words), 1)
    return "PARTIALLY_SUPPORTED" if overlap >= 0.35 else "UNSUPPORTED"


def topology_flags(trace: dict[str, Any]) -> list[str]:
    txns = trace.get("context", {}).get("transactions", [])
    sources = [str(t.get("src_acct")) for t in txns]
    destinations = [str(t.get("dst_acct")) for t in txns]
    flags = []
    if len(set(destinations)) == 1 and len(set(sources)) > 1:
        flags.append("obvious_fan_in")
    if len(set(sources)) == 1 and len(set(destinations)) > 1:
        flags.append("obvious_fan_out")
    edges = {(s, d) for s, d in zip(sources, destinations)}
    if any((d, s) in edges for s, d in edges):
        flags.append("direct_two_node_cycle")
    if len(set(sources)) == len(sources) and len(set(destinations)) == len(destinations):
        flags.append("mostly_disjoint_edges")
    return flags


def audit_model(model: str, traces: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    claim_counts = Counter()
    for trace in traces:
        report = trace.get("report", {})
        facts = source_facts(trace)
        evidence = report.get("evidence", [])
        statuses = [audit_claim(item, facts) for item in evidence]
        claim_counts.update(statuses)
        unsupported = sum(status == "UNSUPPORTED" for status in statuses)
        decision = report.get("decision")
        rows.append({
            "model": model,
            "case_id": trace.get("case_id"),
            "ground_truth_decision": trace.get("ground_truth", {}).get("decision"),
            "predicted_decision": decision,
            "ground_truth_pattern": trace.get("ground_truth", {}).get("pattern"),
            "predicted_pattern": report.get("pattern"),
            "evidence_items": len(evidence),
            "supported_claims": statuses.count("SUPPORTED"),
            "partially_supported_claims": statuses.count("PARTIALLY_SUPPORTED"),
            "unsupported_claims": unsupported,
            "evidence_supported_rate": statuses.count("SUPPORTED") / max(len(statuses), 1),
            "suspicious_without_supported_evidence": int(decision == "SUSPICIOUS" and statuses.count("SUPPORTED") == 0),
            "binary_correct": int(decision == trace.get("ground_truth", {}).get("decision")),
            "pattern_correct": int(report.get("pattern") == trace.get("ground_truth", {}).get("pattern")),
            "topology_flags": ";".join(topology_flags(trace)),
        })
    summary = {
        "model": model,
        "cases": len(traces),
        "binary_accuracy": sum(r["binary_correct"] for r in rows) / max(len(rows), 1),
        "pattern_accuracy": sum(r["pattern_correct"] for r in rows) / max(len(rows), 1),
        "evidence_supported_rate": claim_counts["SUPPORTED"] / max(sum(claim_counts.values()), 1),
        "unsupported_claim_rate": claim_counts["UNSUPPORTED"] / max(sum(claim_counts.values()), 1),
        "cases_with_unsupported_claim": sum(r["unsupported_claims"] > 0 for r in rows),
        "suspicious_without_supported_evidence": sum(r["suspicious_without_supported_evidence"] for r in rows),
        "claim_counts": dict(claim_counts),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_rows = []
    for model, filename in MODEL_FILES.items():
        path = args.traces_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        summary, rows = audit_model(model, load_jsonl(path))
        summaries.append(summary)
        all_rows.extend(rows)
        (args.output_dir / f"{model}_audit.json").write_text(json.dumps({"summary": summary, "cases": rows}, indent=2))

    (args.output_dir / "audit_summary.json").write_text(json.dumps(summaries, indent=2))
    with (args.output_dir / "case_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    print(json.dumps(summaries, indent=2))
    print("saved:", args.output_dir)


if __name__ == "__main__":
    main()
