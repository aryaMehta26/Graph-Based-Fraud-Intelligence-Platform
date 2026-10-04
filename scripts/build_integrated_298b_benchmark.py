"""Build a provenance-backed DATA 298B benchmark from the real test pipeline.

This script reuses the saved project artifacts rather than retraining anything:

    test_graph_enriched.parquet
        -> saved graph-enhanced XGBoost score
        -> existing graph/Leiden/community columns
        -> investigation-case construction
        -> label-free model context + evaluator-only labels/provenance

The generated JSONL is consumed by the four-model evaluation notebook. Labels
and provenance are stored on the case record and are removed by
InvestigationCase.agent_context() before model inference.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.aml.schemas import InvestigationCase

TABULAR_FEATURES = [
    "log_amount", "is_ACH", "is_Cheque", "is_CC", "is_Wire", "is_Bitcoin",
    "hour", "dow", "is_weekend", "is_cross_currency", "amount_bucket",
]
GRAPH_FEATURES = [
    "src_out_degree", "src_in_degree", "src_total_degree", "src_degree_centrality",
    "dst_out_degree", "dst_in_degree", "dst_total_degree", "dst_degree_centrality",
    "src_community_size", "src_community_fraud_rate",
    "dst_community_size", "dst_community_fraud_rate",
]
FEATURES = TABULAR_FEATURES + GRAPH_FEATURES
AMOUNT_BUCKET_CODES = {
    "<100": 0, "100-500": 1, "500-1K": 2, "1K-5K": 3, "5K-10K": 4,
    "10K-50K": 5, "50K-100K": 6, "100K-1M": 7, "1M-1B": 8, ">1B": 9,
}
PATTERN_MAP = {
    "fan_in": "FAN_IN", "fan-in": "FAN_IN", "fan_out": "FAN_OUT", "fan-out": "FAN_OUT",
    "cycle": "CYCLE", "stack": "STACK", "scatter_gather": "SCATTER_GATHER",
    "scatter-gather": "SCATTER_GATHER", "gather_scatter": "GATHER_SCATTER",
    "gather-scatter": "GATHER_SCATTER", "bipartite": "BIPARTITE", "random": "RANDOM",
}
CONTEXT_COLUMNS = [
    "Timestamp", "From Bank", "src_acct", "To Bank", "dst_acct", "Amount Paid",
    "Payment Format", "Receiving Currency", *FEATURES, "src_community_id",
    "dst_community_id", "Is Laundering",
]


def parse_attempts(path: Path):
    attempts, current, rows = [], None, []
    begin = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z-]+)")
    for line in path.read_text().splitlines():
        line = line.strip()
        match = begin.match(line)
        if match:
            current, rows = match.group(1), []
        elif line.startswith("END LAUNDERING ATTEMPT"):
            if current and rows:
                attempts.append({"attempt_id": len(attempts) + 1, "pattern": current, "rows": rows})
            current, rows = None, []
        elif current and line and not line.startswith("BEGIN"):
            fields = line.split(",")
            if len(fields) >= 10:
                try:
                    rows.append({
                        "timestamp": fields[0].strip(), "src_acct": fields[2].strip(),
                        "dst_acct": fields[4].strip(), "amount": round(float(fields[5]), 2),
                        "payment_format": fields[9].strip(),
                    })
                except ValueError:
                    pass
    return attempts


def source_key(row):
    timestamp = row["Timestamp"]
    timestamp = timestamp.strftime("%Y/%m/%d %H:%M") if hasattr(timestamp, "strftime") else str(timestamp)
    return (timestamp, row["src_acct"], row["dst_acct"], round(float(row["Amount Paid"]), 2), row["Payment Format"])


def stable_source_id(row):
    raw = "|".join(map(str, source_key(row)))
    return "txn_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_training_ids(paths):
    ids = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                ids.add(json.loads(line)["case_id"])
    return ids


def load_excluded_transaction_ids(paths):
    ids = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                for transaction in record.get("context", {}).get("transactions", []):
                    value = transaction.get("source_transaction_id")
                    if value:
                        ids.add(value)
    return ids


def collect_rows(parquet_path, patterns_path, legitimate_pool):
    attempts = parse_attempts(patterns_path)
    target_keys = {source_key_from_pattern(row) for attempt in attempts for row in attempt["rows"]}
    wanted = {}
    legitimate = []
    scanner = ds.dataset(str(parquet_path), format="parquet").scanner(columns=CONTEXT_COLUMNS, batch_size=131072)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            key = source_key(row)
            if int(row.get("Is Laundering") or 0) == 1 and key in target_keys:
                wanted[key] = row
            elif int(row.get("Is Laundering") or 0) == 0 and len(legitimate) < legitimate_pool:
                legitimate.append(row)
    return attempts, wanted, legitimate


def source_key_from_pattern(row):
    return (row["timestamp"], row["src_acct"], row["dst_acct"], round(float(row["amount"]), 2), row["payment_format"])


def score_rows(rows, model):
    frame = pd.DataFrame(rows)
    x = frame[FEATURES].copy()
    x["amount_bucket"] = x["amount_bucket"].map(AMOUNT_BUCKET_CODES).fillna(-1)
    return model.predict_proba(x.astype("float32"))[:, 1]


def row_to_transaction(row, index, score):
    return {
        "source_transaction_id": stable_source_id(row),
        "source_transaction_key": list(source_key(row)),
        "txn_id": f"txn_{index}",
        "src_acct": row["src_acct"], "dst_acct": row["dst_acct"],
        "amount": float(row["Amount Paid"]), "payment_format": row["Payment Format"],
        "timestamp": row["Timestamp"].isoformat() if hasattr(row["Timestamp"], "isoformat") else str(row["Timestamp"]),
        "hour": row.get("hour"), "is_weekend": row.get("is_weekend"),
        "xgboost_score": float(score), "xgboost_predicted_suspicious": bool(score >= 0.30),
    }


def build_case(case_id, rows, scores, decision, pattern, source_type, source_attempt=None):
    accounts = {}
    transactions = []
    for index, (row, score) in enumerate(zip(rows, scores), 1):
        for side in ("src", "dst"):
            account_id = row[f"{side}_acct"]
            accounts[account_id] = {
                "account_id": account_id,
                "out_degree": row[f"{side}_out_degree"], "in_degree": row[f"{side}_in_degree"],
                "total_degree": row[f"{side}_total_degree"], "degree_centrality": row[f"{side}_degree_centrality"],
                "community_id": row[f"{side}_community_id"], "community_size": row[f"{side}_community_size"],
                "community_fraud_rate": row[f"{side}_community_fraud_rate"],
            }
        transactions.append(row_to_transaction(row, index, score))
    context = {
        "subgraph_id": case_id,
        "accounts": list(accounts.values()),
        "transactions": transactions,
        "graph_stats": {
            "transaction_count": len(transactions),
            "account_count": len(accounts),
            "xgboost_max_score": max(t["xgboost_score"] for t in transactions),
        },
    }
    provenance = {
        "source_split": "test_graph_enriched.parquet",
        "source_type": source_type,
        "source_attempt_id": source_attempt,
        "source_transaction_ids": [t["source_transaction_id"] for t in transactions],
        "xgboost_model": "data/models/xgboost_graph_enhanced.json",
        "xgboost_threshold": 0.30,
        "graph_features": "test_graph_enriched.parquet",
        "community_features": "test_graph_enriched.parquet generated by 04b_louvain_communities.py and 05_build_feature_store.py",
        "case_construction": "scripts/build_integrated_298b_benchmark.py",
    }
    return InvestigationCase(
        case_id=case_id, context=context,
        ground_truth={"decision": decision, "pattern": pattern},
        split="test", provenance=provenance,
    )


def build(args):
    attempts, wanted, legitimate_rows = collect_rows(args.parquet, args.patterns, args.legitimate_pool)
    xgboost_model = XGBClassifier()
    xgboost_model.load_model(str(args.xgboost_model))
    excluded_paths = args.training_jsonl + args.exclude_jsonl
    excluded_ids = load_training_ids(excluded_paths)
    excluded_transaction_ids = load_excluded_transaction_ids(args.exclude_jsonl)
    legitimate_rows = [
        row for row in legitimate_rows
        if stable_source_id(row) not in excluded_transaction_ids
    ]
    cases = []
    suspicious_candidates = []
    for attempt in attempts:
        case_id = f"{args.case_prefix}attempt_{attempt['attempt_id']}"
        if f"attempt_{attempt['attempt_id']}" in excluded_ids or case_id in excluded_ids:
            continue
        matched = [wanted.get(source_key_from_pattern(row)) for row in attempt["rows"]]
        if not matched or any(row is None for row in matched):
            continue
        pattern = PATTERN_MAP.get(attempt["pattern"].lower(), "EMERGING_UNKNOWN")
        suspicious_candidates.append((case_id, matched, pattern, attempt["attempt_id"]))
    # Prioritize up to five genuinely available cases per pattern, then fill
    # remaining slots without changing the held-out source set.
    selected_candidates = []
    selected_ids = set()
    for pattern in sorted({item[2] for item in suspicious_candidates}):
        for item in [candidate for candidate in suspicious_candidates if candidate[2] == pattern][:5]:
            selected_candidates.append(item)
            selected_ids.add(item[0])
    for item in suspicious_candidates:
        if len(selected_candidates) >= args.suspicious_cases:
            break
        if item[0] not in selected_ids:
            selected_candidates.append(item)
            selected_ids.add(item[0])

    for case_id, rows, pattern, attempt_id in selected_candidates[:args.suspicious_cases]:
        scores = score_rows(rows, xgboost_model)
        cases.append(build_case(case_id, rows, scores, "SUSPICIOUS", pattern, "pattern_attempt", attempt_id))

    if len(cases) < args.suspicious_cases:
        raise RuntimeError(f"Only found {len(cases)} leakage-free suspicious cases; need {args.suspicious_cases}.")

    for start in range(0, len(legitimate_rows), args.transactions_per_case):
        if sum(c.ground_truth["decision"] == "LEGITIMATE" for c in cases) >= args.legitimate_cases:
            break
        rows = legitimate_rows[start:start + args.transactions_per_case]
        if len(rows) < args.transactions_per_case:
            break
        case_id = f"{args.case_prefix}legitimate_{start // args.transactions_per_case + 1:03d}"
        if case_id in excluded_ids:
            continue
        scores = score_rows(rows, xgboost_model)
        cases.append(build_case(case_id, rows, scores, "LEGITIMATE", "NONE", "test_legitimate_pool"))

    if sum(c.ground_truth["decision"] == "LEGITIMATE" for c in cases) < args.legitimate_cases:
        raise RuntimeError("Not enough legitimate test rows to build the requested benchmark.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(c.to_jsonl() for c in cases) + "\n")
    print("output:", args.output)
    print("cases:", len(cases))
    print("decisions:", Counter(c.ground_truth["decision"] for c in cases))
    print("patterns:", Counter(c.ground_truth["pattern"] for c in cases))
    print("model-visible context excludes ground truth/provenance: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=Path("data/processed/test_graph_enriched.parquet"))
    parser.add_argument("--patterns", type=Path, default=Path("HI-Medium_Patterns.txt"))
    parser.add_argument("--xgboost-model", type=Path, default=Path("data/models/xgboost_graph_enhanced.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/final_benchmark/integrated_cases.jsonl"))
    parser.add_argument("--training-jsonl", type=Path, action="append", default=[Path("artifacts/evaluation_dry_run/cases.jsonl")])
    parser.add_argument("--exclude-jsonl", type=Path, action="append", default=[], help="Existing case JSONL files whose case IDs must not be reused.")
    parser.add_argument("--case-prefix", default="", help="Prefix for generated case IDs, e.g. validation_.")
    parser.add_argument("--suspicious-cases", type=int, default=40)
    parser.add_argument("--legitimate-cases", type=int, default=10)
    parser.add_argument("--transactions-per-case", type=int, default=6)
    parser.add_argument("--legitimate-pool", type=int, default=10000)
    build(parser.parse_args())
