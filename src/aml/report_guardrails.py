"""Deterministic checks applied after LLM report generation.

These checks do not use ground-truth labels and do not modify model weights.
They prevent obvious graph/prediction contradictions from being accepted as a
final report.  Reports remain reviewable through the emitted guardrail flags.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from typing import Any, Dict, List


def _transactions(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(context.get("transactions", []))


def topology_summary(context: Dict[str, Any]) -> Dict[str, Any]:
    txns = _transactions(context)
    sources = [str(t.get("src_acct")) for t in txns]
    destinations = [str(t.get("dst_acct")) for t in txns]
    edges = [(s, d) for s, d in zip(sources, destinations)]
    self_loop = any(source == destination for source, destination in edges)
    adjacency: Dict[str, List[str]] = {}
    for source, destination in edges:
        adjacency.setdefault(source, []).append(destination)
    repeated_source = max(Counter(sources).values(), default=0)
    repeated_destination = max(Counter(destinations).values(), default=0)

    # Detect a directed cycle of any length in the observed subgraph.
    visiting, visited = set(), set()

    def has_cycle(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(child != node and has_cycle(child) for child in adjacency.get(node, [])):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    # A single self-loop is recorded separately; it is not enough by itself to
    # establish a multi-account laundering cycle.
    cycle = any(has_cycle(node) for node in adjacency if any(child != node for child in adjacency.get(node, [])))
    max_score = max((float(t.get("xgboost_score", 0.0) or 0.0) for t in txns), default=0.0)
    return {
        "transaction_count": len(txns),
        "unique_sources": len(set(sources)),
        "unique_destinations": len(set(destinations)),
        "max_source_multiplicity": repeated_source,
        "max_destination_multiplicity": repeated_destination,
        "has_cycle": cycle,
        "has_self_loop": self_loop,
        "max_xgboost_score": max_score,
    }


def apply_report_guardrails(report: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Return a guarded copy of an LLM report and explicit audit metadata."""
    guarded = deepcopy(report)
    summary = topology_summary(context)
    flags: List[str] = []
    pattern = guarded.get("pattern", "EMERGING_UNKNOWN")

    fan_in_valid = summary["max_destination_multiplicity"] >= 2 and summary["unique_sources"] >= 2
    fan_out_valid = summary["max_source_multiplicity"] >= 2 and summary["unique_destinations"] >= 2
    if pattern == "FAN_IN" and not fan_in_valid:
        guarded["pattern"] = "NONE"
        flags.append("fan_in_conflicts_with_observed_topology")
    elif pattern == "FAN_OUT" and not fan_out_valid:
        guarded["pattern"] = "NONE"
        flags.append("fan_out_conflicts_with_observed_topology")

    structural_signal = (
        fan_in_valid
        or fan_out_valid
        or summary["has_cycle"]
        or summary["max_source_multiplicity"] >= 2
        or summary["max_destination_multiplicity"] >= 2
    )
    model_signal = summary["max_xgboost_score"] >= 0.30
    supported_signal = structural_signal or model_signal
    evidence = guarded.get("evidence") or []

    if guarded.get("decision") == "SUSPICIOUS" and not supported_signal:
        guarded["decision"] = "LEGITIMATE"
        guarded["risk_level"] = "LOW"
        guarded["confidence"] = min(float(guarded.get("confidence", 0.0) or 0.0), 0.50)
        flags.append("suspicious_decision_without_structural_or_xgboost_signal")
    if guarded.get("decision") == "SUSPICIOUS" and not evidence:
        flags.append("suspicious_decision_has_no_evidence")

    guarded["guardrail_flags"] = flags
    guarded["guardrail_topology"] = summary
    return guarded
