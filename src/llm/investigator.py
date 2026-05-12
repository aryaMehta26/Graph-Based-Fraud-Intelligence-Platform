"""
DATA 298A — Graph-Based Fraud Intelligence Platform
Team 2 | San José State University

src/llm/investigator.py
------------------------
Closes Issue #13.

Implements the Layer 4 LLM Investigator — accepts a pre-computed subgraph
JSON summary and returns a structured fraud investigation report.

4 model variants for comparison (all using Anthropic Claude):
    v1 — claude-haiku-4-5          | speed + cost baseline
    v2 — claude-sonnet-4-6         | balanced performance
    v3 — claude-sonnet-4-6 + v2 system prompt  | prompt engineering impact
    v4 — claude-sonnet-4-6 + chain-of-thought  | reasoning depth impact

All variants:
    - temperature=0 for reproducibility
    - Accept subgraph JSON as input (no raw transaction data sent to LLM)
    - Return structured schema: pattern, evidence, risk_level, actions
    - Save outputs to artifacts/llm_outputs/

Usage
-----
    # Run a specific variant against a subgraph file:
    python3 src/llm/investigator.py --variant v2 --input artifacts/sample_subgraph.json

    # Run all 4 variants:
    python3 src/llm/investigator.py --variant all --input artifacts/sample_subgraph.json

Prerequisites
-------------
    - ANTHROPIC_API_KEY in .env
    - pip install anthropic
"""

import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJ         = Path(__file__).resolve().parents[2]
OUTPUT_DIR   = PROJ / "artifacts" / "llm_outputs"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
OUTPUT_SCHEMA = {
    "pattern":    str,   # Suspected laundering pattern (e.g. "Layering Ring", "Smurfing")
    "evidence":   list,  # List of specific evidence strings grounded in the subgraph
    "risk_level": str,   # One of: "LOW", "MEDIUM", "HIGH", "CRITICAL"
    "actions":    list,  # List of recommended investigative or remediation actions
    "reasoning":  str,   # Internal reasoning trace (populated in v4 only)
}

VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V1_V2 = """You are a financial crimes investigator specializing in anti-money laundering (AML).

You will be given a JSON summary of a suspicious transaction subgraph extracted from a banking knowledge graph. The subgraph contains account nodes, transaction edges, and graph analytics features including degree centrality and Louvain community membership.

Your job is to analyze the subgraph and produce a structured investigation report.

You MUST respond with valid JSON only. No preamble, no explanation outside the JSON. The JSON must contain exactly these fields:
{
  "pattern": "<one of: Layering Ring, Smurfing, Fan-Out, Fan-In, Rapid Movement, Unknown>",
  "evidence": ["<specific fact from the subgraph>", "<another specific fact>", ...],
  "risk_level": "<one of: LOW, MEDIUM, HIGH, CRITICAL>",
  "actions": ["<recommended action>", ...],
  "reasoning": ""
}

Rules:
- "pattern" must be EXACTLY one value from this list: Layering Ring, Smurfing, Fan-Out, Fan-In, Rapid Movement, Unknown. No combining, no variations.
- Layering Ring covers ALL cyclic transfer patterns (A→B→C→A, A→B→A, or any cycle). Do NOT use "Cyclic Transfer" — use "Layering Ring" instead.
- Smurfing = many accounts feeding one hub. Fan-In = same shape but not necessarily structured below thresholds. When in doubt between the two, use Smurfing if amounts are just below round numbers, Fan-In otherwise.
- Every item in "evidence" MUST contain the exact account ID or exact dollar amount from the input JSON. Example: "Account 800A3F2 sent 47 transactions" not "the hub account sent many transactions". If a value is not in the input, do not include it.
- "risk_level" must be exactly one of: LOW, MEDIUM, HIGH, CRITICAL
- "actions" should be specific and actionable, referencing exact account IDs (e.g. "Freeze account 800A3F2 pending investigation")
- "reasoning" should be an empty string for this variant"""

SYSTEM_PROMPT_V3 = """You are a senior financial crimes investigator at a Tier-1 bank, specializing in detecting sophisticated money laundering operations.

You have deep expertise in:
- Layering schemes (A→B→C→A cycles)
- Smurfing / structuring (many small accounts feeding one)  
- Fan-out patterns (one source distributing to many destinations)
- Rapid velocity transfers designed to obscure origin

You will receive a structured subgraph JSON from our graph analytics engine. This subgraph has already been flagged by our XGBoost model (PR-AUC: 0.56) and community detection layer as high-risk.

Respond ONLY with valid JSON in this exact schema:
{
  "pattern": "<Layering Ring | Smurfing | Fan-Out | Fan-In | Rapid Movement | Unknown>",
  "evidence": ["<verbatim fact from input>", ...],
  "risk_level": "<LOW | MEDIUM | HIGH | CRITICAL>",
  "actions": ["<specific, actionable step>", ...],
  "reasoning": ""
}

Critical requirements:
- "pattern" must be EXACTLY one value from the list above. Do not combine patterns or add qualifiers.
- Layering Ring covers ALL cyclic patterns (A→B→C→A or any cycle). Do NOT say "Cyclic Transfer".
- Smurfing = many small accounts structuring funds into one hub. Fan-In = same topology but general aggregation.
- Every evidence item MUST contain the exact account ID (e.g. 800A3F2) or exact dollar amount (e.g. $9,800.00) from the input JSON. Quote the actual value — do not paraphrase or generalize.
- Actions must reference specific account IDs from the input."""

SYSTEM_PROMPT_V4_COT = """You are a financial crimes investigator with expertise in AML pattern detection.

You will analyze a suspicious transaction subgraph. Before producing your final answer, think through the problem step by step inside a <reasoning> block. Then output your structured JSON report.

Your response format:
<reasoning>
Step 1: List every account ID present and their exact degree centrality scores from the input.
Step 2: What is the community_id and community_fraud_rate from the input?
Step 3: List each transaction with its exact amount, direction, and timestamp from the input.
Step 4: Which single AML pattern best fits from this list ONLY: Layering Ring, Smurfing, Fan-Out, Fan-In, Rapid Movement, Unknown? Choose ONE. Note: Layering Ring covers ALL cyclic patterns including A→B→C→A cycles. Do NOT say "Cyclic Transfer".
Step 5: What is the appropriate risk level: LOW, MEDIUM, HIGH, or CRITICAL?
</reasoning>
{
  "pattern": "<EXACTLY one of: Layering Ring, Smurfing, Fan-Out, Fan-In, Rapid Movement, Unknown>",
  "evidence": ["<must contain exact account ID or dollar amount from input>", ...],
  "risk_level": "<LOW | MEDIUM | HIGH | CRITICAL>",
  "actions": ["<specific action referencing exact account IDs>", ...],
  "reasoning": "<your step-by-step reasoning from above>"
}

Critical requirements:
- "pattern" must be EXACTLY one value — no combining (e.g. NOT "Smurfing + Fan-In"), no variations
- Every evidence item MUST quote the exact account ID or exact dollar amount from the input
- reasoning field should contain your full thinking process

Rules:
- Evidence must only reference values that appear in the input JSON
- risk_level must be exactly: LOW, MEDIUM, HIGH, or CRITICAL
- reasoning field in JSON should contain your full thinking process"""

# ---------------------------------------------------------------------------
# Model variants config
# ---------------------------------------------------------------------------
VARIANTS = {
    "v1": {
        "model":         "claude-haiku-4-5-20251001",
        "system_prompt": SYSTEM_PROMPT_V1_V2,
        "description":   "Haiku — speed + cost baseline",
        "max_tokens":    2000,
    },
    "v2": {
        "model":         "claude-sonnet-4-6",
        "system_prompt": SYSTEM_PROMPT_V1_V2,
        "description":   "Sonnet — balanced performance",
        "max_tokens":    2000,
    },
    "v3": {
        "model":         "claude-sonnet-4-6",
        "system_prompt": SYSTEM_PROMPT_V3,
        "description":   "Sonnet + enhanced system prompt — prompt engineering impact",
        "max_tokens":    2000,
    },
    "v4": {
        "model":         "claude-sonnet-4-6",
        "system_prompt": SYSTEM_PROMPT_V4_COT,
        "description":   "Sonnet + chain-of-thought — reasoning depth impact",
        "max_tokens":    4000,
    },
}

# ---------------------------------------------------------------------------
# Core investigator
# ---------------------------------------------------------------------------

def build_user_prompt(subgraph: dict) -> str:
    return f"""Analyze this suspicious transaction subgraph and produce your investigation report.

Subgraph JSON:
{json.dumps(subgraph, indent=2)}"""


def call_llm(variant_key: str, subgraph: dict, run_id: int = 1) -> dict:
    """Call one model variant and return the parsed structured report."""
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to your .env file."
        )

    variant  = VARIANTS[variant_key]
    client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    log.info("Calling %s (run %d) — %s", variant_key, run_id, variant["description"])

    message = client.messages.create(
        model=variant["model"],
        max_tokens=variant["max_tokens"],
        temperature=0,
        system=variant["system_prompt"],
        messages=[
            {"role": "user", "content": build_user_prompt(subgraph)}
        ],
    )

    raw_text = message.content[0].text

    # Parse JSON — attach raw_text to any exception so the caller can preserve it
    try:
        report = parse_response(raw_text, variant_key)
    except Exception as e:
        e.raw_response = raw_text
        raise

    # Attach metadata
    report["_meta"] = {
        "variant":      variant_key,
        "model":        variant["model"],
        "run_id":       run_id,
        "timestamp":    datetime.utcnow().isoformat(),
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "raw_response": raw_text,
    }

    return report


def parse_response(raw_text: str, variant_key: str) -> dict:
    """Extract and parse JSON from model response."""
    text = raw_text.strip()

    # v4 has <reasoning>...</reasoning> before the JSON
    if "<reasoning>" in text:
        reasoning_start = text.find("<reasoning>") + len("<reasoning>")
        reasoning_end   = text.find("</reasoning>")
        reasoning_text  = text[reasoning_start:reasoning_end].strip()
        text            = text[reasoning_end + len("</reasoning>"):].strip()
    else:
        reasoning_text = ""

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    # Try direct parse; fall back to extracting the first {...} block
    try:
        report = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            report = json.loads(text[start:end])
        else:
            raise

    # Populate reasoning from CoT if available and not already in JSON
    if reasoning_text and not report.get("reasoning"):
        report["reasoning"] = reasoning_text

    validate_schema(report)
    return report


def validate_schema(report: dict):
    """Raise ValueError if required fields are missing or invalid."""
    required = ["pattern", "evidence", "risk_level", "actions"]
    missing  = [f for f in required if f not in report]
    if missing:
        raise ValueError(f"Report missing required fields: {missing}")

    if report["risk_level"] not in VALID_RISK_LEVELS:
        raise ValueError(
            f"Invalid risk_level '{report['risk_level']}'. "
            f"Must be one of {VALID_RISK_LEVELS}"
        )

    if not isinstance(report["evidence"], list) or len(report["evidence"]) == 0:
        raise ValueError("'evidence' must be a non-empty list")

    if not isinstance(report["actions"], list) or len(report["actions"]) == 0:
        raise ValueError("'actions' must be a non-empty list")

    valid_patterns = {"Layering Ring", "Smurfing", "Fan-Out", "Fan-In", "Rapid Movement", "Unknown"}
    if report.get("pattern") not in valid_patterns:
        raise ValueError(
            f"Invalid pattern '{report.get('pattern')}'. Must be one of {valid_patterns}"
        )


def run_variant(variant_key: str, subgraph: dict, n_runs: int = 3) -> list:
    """Run one variant n_runs times (for consistency evaluation in evaluate.py)."""
    results = []
    for run_id in range(1, n_runs + 1):
        try:
            report = call_llm(variant_key, subgraph, run_id=run_id)
            results.append(report)
            log.info(
                "  %s run %d — pattern: %s | risk: %s",
                variant_key, run_id,
                report.get("pattern"), report.get("risk_level")
            )
        except Exception as e:
            log.error("  %s run %d failed: %s", variant_key, run_id, e)
            results.append({
                "_error": str(e),
                "_meta": {
                    "variant":      variant_key,
                    "run_id":       run_id,
                    "raw_response": getattr(e, "raw_response", None),
                },
            })
    return results


def save_outputs(variant_key: str, results: list, subgraph_name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    """Save all runs for a variant to artifacts/llm_outputs/."""
    filename = OUTPUT_DIR / f"{subgraph_name}_{variant_key}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", filename)
    return filename


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM Fraud Investigator — Layer 4")
    parser.add_argument(
        "--variant",
        choices=["v1", "v2", "v3", "v4", "all"],
        default="v2",
        help="Which model variant to run (default: v2)"
    )
    parser.add_argument(
        "--input",
        default=str(PROJ / "artifacts" / "sample_subgraph.json"),
        help="Path to subgraph JSON file"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per variant for consistency scoring (default: 3)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  LAYER 4: LLM FRAUD INVESTIGATOR")
    print("  DATA 298A | Team 2 | Issue #13")
    print("=" * 70)

    # Load subgraph
    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Subgraph file not found: %s", input_path)
        log.error("Run python3 src/llm/build_subgraph.py to generate sample subgraphs")
        return

    with open(input_path) as f:
        subgraph = json.load(f)

    subgraph_name = input_path.stem
    log.info("Loaded subgraph: %s (%d accounts, %d transactions)",
             subgraph_name,
             len(subgraph.get("accounts", [])),
             len(subgraph.get("transactions", [])))

    # Determine variants to run
    variants_to_run = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]

    all_outputs = {}
    for vk in variants_to_run:
        log.info("Running variant %s (%d runs)...", vk, args.runs)
        results = run_variant(vk, subgraph, n_runs=args.runs)
        save_outputs(vk, results, subgraph_name)
        all_outputs[vk] = results

    # Print summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    for vk, results in all_outputs.items():
        variant = VARIANTS[vk]
        successful = [r for r in results if "_error" not in r]
        print(f"\n  {vk} — {variant['description']}")
        for r in successful:
            meta = r.get("_meta", {})
            print(f"    Run {meta.get('run_id', '?')} | "
                  f"pattern: {r.get('pattern', '?'):<20} | "
                  f"risk: {r.get('risk_level', '?'):<8} | "
                  f"tokens: {meta.get('output_tokens', '?')}")

    print("\n" + "=" * 70)
    print(f"  Outputs saved to: {OUTPUT_DIR}")
    print("  Next step: python3 src/llm/evaluate.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
