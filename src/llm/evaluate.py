"""
DATA 298A — Graph-Based Fraud Intelligence Platform
Team 2 | San José State University

src/llm/evaluate.py
--------------------
Closes Issue #14.

Evaluation framework for the LLM Investigator layer.
Runs against saved outputs from investigator.py — does NOT call the API again.

Metrics computed per variant:
    1. Consistency    — Jaccard similarity across 3 runs (target ≥ 0.8)
    2. Faithfulness   — % of evidence items grounded in input subgraph
    3. Schema compliance — all required fields present and valid
    4. Cross-model comparison table across all 4 variants

Output
------
artifacts/metrics/llm_eval.json     — full evaluation results
artifacts/metrics/llm_eval_table.txt — human-readable comparison table

Usage
-----
    python3 src/llm/evaluate.py --subgraph sample_subgraph

Prerequisites
-------------
    - investigator.py has been run and outputs exist in artifacts/llm_outputs/
"""

import os
import json
import argparse
import logging
from pathlib import Path
from itertools import combinations

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJ        = Path(__file__).resolve().parents[2]
OUTPUT_DIR  = PROJ / "artifacts" / "llm_outputs"
METRICS_DIR = PROJ / "artifacts" / "metrics"

CONSISTENCY_TARGET = 0.80   # Jaccard similarity target per issue #14 spec
VALID_RISK_LEVELS  = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
VALID_PATTERNS     = {
    "Layering Ring", "Smurfing", "Fan-Out", "Fan-In",
    "Rapid Movement", "Unknown"
}
REQUIRED_FIELDS = ["pattern", "evidence", "risk_level", "actions"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric 1: Schema Compliance
# ---------------------------------------------------------------------------

def check_schema_compliance(report: dict) -> dict:
    """Check that all required fields are present and valid."""
    result = {
        "has_all_fields":     True,
        "risk_level_valid":   True,
        "evidence_non_empty": True,
        "actions_non_empty":  True,
        "pattern_valid":      True,
        "score":              1.0,
        "issues":             [],
    }

    # Required fields
    for field in REQUIRED_FIELDS:
        if field not in report:
            result["has_all_fields"] = False
            result["issues"].append(f"Missing field: {field}")

    # Risk level validity
    if report.get("risk_level") not in VALID_RISK_LEVELS:
        result["risk_level_valid"] = False
        result["issues"].append(f"Invalid risk_level: {report.get('risk_level')}")

    # Evidence non-empty
    if not isinstance(report.get("evidence"), list) or len(report.get("evidence", [])) == 0:
        result["evidence_non_empty"] = False
        result["issues"].append("Evidence list is empty")

    # Actions non-empty
    if not isinstance(report.get("actions"), list) or len(report.get("actions", [])) == 0:
        result["actions_non_empty"] = False
        result["issues"].append("Actions list is empty")

    # Pattern validity
    if report.get("pattern") not in VALID_PATTERNS:
        result["pattern_valid"] = False
        result["issues"].append(f"Unrecognized pattern: {report.get('pattern')}")

    # Score = fraction of checks passed
    checks = [
        result["has_all_fields"],
        result["risk_level_valid"],
        result["evidence_non_empty"],
        result["actions_non_empty"],
        result["pattern_valid"],
    ]
    result["score"] = sum(checks) / len(checks)
    return result


# ---------------------------------------------------------------------------
# Metric 2: Faithfulness
# ---------------------------------------------------------------------------

def extract_subgraph_values(subgraph: dict) -> dict:
    """
    Extract all verifiable values from the subgraph.
    Returns a dict of value_type -> set of values for richer matching.
    """
    account_ids   = set()
    amounts       = set()
    payment_fmts  = set()
    numeric_vals  = set()
    degree_vals   = set()

    for acct in subgraph.get("accounts", []):
        if "account_id" in acct:
            aid = str(acct["account_id"])
            account_ids.add(aid.lower())
            account_ids.add(aid)           # original case too

        for field in ["out_degree", "in_degree", "total_degree"]:
            if field in acct:
                numeric_vals.add(str(acct[field]))

        if "degree_centrality" in acct:
            degree_vals.add(str(round(float(acct["degree_centrality"]), 3)))
            degree_vals.add(str(acct["degree_centrality"]))

        if "community_id" in acct:
            numeric_vals.add(str(acct["community_id"]))

        if "community_fraud_rate" in acct:
            rate = float(acct["community_fraud_rate"])
            numeric_vals.add(str(rate))
            numeric_vals.add(f"{rate:.1%}")
            numeric_vals.add(f"{int(rate * 100)}%")

    for txn in subgraph.get("transactions", []):
        if "txn_id" in txn:
            account_ids.add(str(txn["txn_id"]).lower())

        if "amount" in txn:
            amt = float(txn["amount"])
            amounts.add(str(amt))
            amounts.add(str(int(amt)))
            amounts.add(f"{amt:,.2f}")
            amounts.add(f"{int(amt):,}")
            amounts.add(f"{amt:.2f}")
            amounts.add(f"{amt:.0f}")
            # Also add partial matches for large numbers (e.g. "9800" matches "$9,800.00")
            amounts.add(str(int(amt)).replace(",", ""))

        if "payment_format" in txn:
            payment_fmts.add(txn["payment_format"].lower())

        if "src_acct" in txn:
            account_ids.add(str(txn["src_acct"]).lower())
            account_ids.add(str(txn["src_acct"]))
        if "dst_acct" in txn:
            account_ids.add(str(txn["dst_acct"]).lower())
            account_ids.add(str(txn["dst_acct"]))

    # Top-level graph stats
    stats = subgraph.get("graph_stats", {})
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            numeric_vals.add(str(v))
            numeric_vals.add(str(int(v)))

    if "community_fraud_rate" in subgraph:
        rate = float(subgraph["community_fraud_rate"])
        numeric_vals.add(f"{rate:.1%}")
        numeric_vals.add(f"{int(rate * 100)}%")

    return {
        "account_ids":  account_ids,
        "amounts":      amounts,
        "payment_fmts": payment_fmts,
        "numeric_vals": numeric_vals,
        "degree_vals":  degree_vals,
    }


def check_faithfulness(report: dict, subgraph: dict) -> dict:
    """
    Check what fraction of evidence items are grounded in the subgraph.

    An evidence item is considered faithful if it contains:
    - An exact account ID from the input, OR
    - An exact dollar amount from the input, OR
    - A payment format from the input, OR
    - A numeric value (degree, community size, fraud rate) from the input

    This is more lenient than exact string match to avoid penalising
    valid paraphrasing of numbers (e.g. "$9,800" vs "9800.0").
    """
    import re
    sv = extract_subgraph_values(subgraph)
    evidence_items = report.get("evidence", [])

    if not evidence_items:
        return {"score": 0.0, "grounded": 0, "total": 0, "ungrounded_items": []}

    grounded   = 0
    ungrounded = []

    for item in evidence_items:
        item_lower = item.lower()
        item_clean = re.sub(r'[$,]', '', item)   # strip $ and commas for amount matching

        is_grounded = False

        # Check account IDs (case-insensitive)
        if any(aid in item_lower for aid in sv["account_ids"] if len(aid) >= 4):
            is_grounded = True

        # Check amounts — strip formatting before comparing
        if not is_grounded:
            item_nums = set(re.findall(r'\d+\.?\d*', item_clean))
            if item_nums & sv["amounts"]:
                is_grounded = True

        # Check payment formats
        if not is_grounded:
            if any(fmt in item_lower for fmt in sv["payment_fmts"]):
                is_grounded = True

        # Check numeric values (degrees, community stats)
        if not is_grounded:
            item_nums = set(re.findall(r'\d+\.?\d*', item))
            if item_nums & sv["numeric_vals"]:
                is_grounded = True

        # Check degree centrality values
        if not is_grounded:
            if any(dv in item for dv in sv["degree_vals"]):
                is_grounded = True

        if is_grounded:
            grounded += 1
        else:
            ungrounded.append(item)

    score = grounded / len(evidence_items)
    return {
        "score":            round(score, 4),
        "grounded":         grounded,
        "total":            len(evidence_items),
        "ungrounded_items": ungrounded,
    }


# ---------------------------------------------------------------------------
# Metric 3: Consistency (Jaccard similarity across runs)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> set:
    """Simple word tokenizer for Jaccard similarity."""
    import re
    return set(re.findall(r'\b\w+\b', text.lower()))


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def check_consistency(reports: list) -> dict:
    """
    Compute pairwise Jaccard similarity across all runs for a variant.
    Considers: pattern agreement, risk_level agreement, evidence token overlap.
    """
    valid_reports = [r for r in reports if "_error" not in r]

    if len(valid_reports) < 2:
        return {
            "score": None,
            "pattern_agreement": None,
            "risk_agreement": None,
            "evidence_jaccard": None,
            "meets_target": False,
            "note": f"Only {len(valid_reports)} valid run(s) — need ≥2 for consistency",
        }

    # Pattern agreement
    patterns = [r.get("pattern", "") for r in valid_reports]
    pattern_agreement = len(set(patterns)) == 1

    # Risk level agreement
    risks = [r.get("risk_level", "") for r in valid_reports]
    risk_agreement = len(set(risks)) == 1

    # Evidence Jaccard — average pairwise
    evidence_texts = [" ".join(r.get("evidence", [])) for r in valid_reports]
    evidence_token_sets = [tokenize(t) for t in evidence_texts]

    pairs = list(combinations(range(len(evidence_token_sets)), 2))
    if pairs:
        jaccard_scores = [
            jaccard_similarity(evidence_token_sets[i], evidence_token_sets[j])
            for i, j in pairs
        ]
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
    else:
        avg_jaccard = 1.0

    # Overall consistency score = weighted average
    pattern_score = 1.0 if pattern_agreement else 0.0
    risk_score    = 1.0 if risk_agreement    else 0.0
    overall       = (pattern_score * 0.4) + (risk_score * 0.3) + (avg_jaccard * 0.3)

    return {
        "score":              round(overall, 4),
        "pattern_agreement":  pattern_agreement,
        "risk_agreement":     risk_agreement,
        "evidence_jaccard":   round(avg_jaccard, 4),
        "meets_target":       overall >= CONSISTENCY_TARGET,
        "n_runs":             len(valid_reports),
    }


# ---------------------------------------------------------------------------
# Per-variant evaluation
# ---------------------------------------------------------------------------

def evaluate_variant(variant_key: str, subgraph_name: str, subgraph: dict) -> dict:
    """Load saved outputs for a variant and compute all metrics."""
    output_file = OUTPUT_DIR / f"{subgraph_name}_{variant_key}.json"

    if not output_file.exists():
        log.warning("Output file not found for %s: %s", variant_key, output_file)
        return {"variant": variant_key, "error": f"No output file: {output_file}"}

    with open(output_file) as f:
        reports = json.load(f)

    valid_reports = [r for r in reports if "_error" not in r]
    log.info("Evaluating %s: %d/%d runs successful", variant_key, len(valid_reports), len(reports))

    if not valid_reports:
        return {"variant": variant_key, "error": "All runs failed"}

    # Use first valid run for schema + faithfulness
    first_report = valid_reports[0]

    schema      = check_schema_compliance(first_report)
    faithfulness = check_faithfulness(first_report, subgraph)
    consistency = check_consistency(reports)

    # Token usage stats
    token_stats = {}
    input_tokens  = [r["_meta"].get("input_tokens")  for r in valid_reports if "_meta" in r and r["_meta"].get("input_tokens") is not None]
    output_tokens = [r["_meta"].get("output_tokens") for r in valid_reports if "_meta" in r and r["_meta"].get("output_tokens") is not None]
    if input_tokens:
        token_stats = {
            "avg_input_tokens":  round(sum(input_tokens)  / len(input_tokens)),
            "avg_output_tokens": round(sum(output_tokens) / len(output_tokens)),
        }

    return {
        "variant":        variant_key,
        "model":          valid_reports[0].get("_meta", {}).get("model", "unknown"),
        "n_runs":         len(reports),
        "n_successful":   len(valid_reports),
        "schema":         schema,
        "faithfulness":   faithfulness,
        "consistency":    consistency,
        "token_stats":    token_stats,
        "sample_output":  {
            "pattern":    first_report.get("pattern"),
            "risk_level": first_report.get("risk_level"),
            "evidence":   first_report.get("evidence", [])[:2],
            "actions":    first_report.get("actions", [])[:2],
        },
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(results: dict) -> str:
    lines = []
    lines.append("=" * 90)
    lines.append("  LLM INVESTIGATOR — MODEL COMPARISON TABLE")
    lines.append("  DATA 298A | Team 2 | Issue #14")
    lines.append("=" * 90)
    lines.append(
        f"  {'Variant':<8} {'Model':<30} {'Schema':>8} {'Faithful':>10} {'Consist':>10} {'Target':>8}"
    )
    lines.append("  " + "-" * 76)

    for vk, result in results.items():
        if "error" in result:
            lines.append(f"  {vk:<8} ERROR: {result['error']}")
            continue

        schema_score  = result["schema"]["score"]
        faith_score   = result["faithfulness"]["score"]
        consist       = result["consistency"]
        consist_score = consist.get("score") if consist.get("score") is not None else "N/A"
        meets_target  = "✓" if consist.get("meets_target") else "✗"

        model_short = result.get("model", "unknown")
        if len(model_short) > 28:
            model_short = model_short[:25] + "..."

        consist_str = f"{consist_score:.4f}" if isinstance(consist_score, float) else str(consist_score)

        lines.append(
            f"  {vk:<8} {model_short:<30} {schema_score:>8.4f} {faith_score:>10.4f} "
            f"{consist_str:>10} {meets_target:>8}"
        )

    lines.append("=" * 90)
    lines.append(f"  Consistency target: ≥ {CONSISTENCY_TARGET}  (✓ = meets target, ✗ = below target)")
    lines.append("=" * 90)

    # Recommended variant
    scored = {
        vk: (
            result["schema"]["score"] * 0.3 +
            result["faithfulness"]["score"] * 0.4 +
            (result["consistency"].get("score") or 0) * 0.3
        )
        for vk, result in results.items()
        if "error" not in result
    }
    if scored:
        best = max(scored, key=scored.get)
        lines.append(f"\n  Recommended variant: {best} (composite score: {scored[best]:.4f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM Investigator Evaluation Framework")
    parser.add_argument(
        "--subgraph",
        default="sample_subgraph",
        help="Subgraph name (stem of the file in artifacts/llm_outputs/)"
    )
    parser.add_argument(
        "--input",
        default=str(PROJ / "artifacts" / "sample_subgraph.json"),
        help="Path to original subgraph JSON (for faithfulness checking)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  LAYER 4: LLM EVALUATION FRAMEWORK")
    print("  DATA 298A | Team 2 | Issue #14")
    print("=" * 70)

    # Load subgraph for faithfulness checking
    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Subgraph file not found: %s", input_path)
        return

    with open(input_path) as f:
        subgraph = json.load(f)

    # Evaluate each variant
    variants = ["v1", "v2", "v3", "v4"]
    all_results = {}

    for vk in variants:
        log.info("Evaluating variant %s...", vk)
        result = evaluate_variant(vk, args.subgraph, subgraph)
        all_results[vk] = result

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    # Save full results
    eval_path = METRICS_DIR / "llm_eval.json"
    with open(eval_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Full evaluation saved: %s", eval_path)

    # Build and print comparison table
    table = build_comparison_table(all_results)
    print("\n" + table)

    # Save table
    table_path = METRICS_DIR / "llm_eval_table.txt"
    with open(table_path, "w") as f:
        f.write(table)
    log.info("Comparison table saved: %s", table_path)

    print(f"\n  Artifacts saved to: {METRICS_DIR}")


if __name__ == "__main__":
    main()
