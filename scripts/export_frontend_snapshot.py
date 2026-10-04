"""Export read-only pipeline artifacts for the React dashboard.

This does not run or mutate the fraud pipeline. It creates a small browser-safe
snapshot from the existing Parquet, CSV, JSON, model-metric, and LLM artifacts.
Run it whenever the underlying artifacts are refreshed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "data" / "models"
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "frontend" / "src" / "data" / "dashboard.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def first_successful_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        if isinstance(item, dict) and not item.get("_error"):
            return item
    return {}


def main() -> None:
    clean = PROCESSED / "transactions_clean.parquet"
    split_test = PROCESSED / "split_test.parquet"
    accounts_path = PROCESSED / "graph_features_accounts.csv"
    communities_path = PROCESSED / "communities.csv"

    total_transactions = pq.ParquetFile(clean).metadata.num_rows if clean.exists() else 0
    fraud_transactions = 0
    if clean.exists():
        parquet_file = pq.ParquetFile(clean)
        for batch in parquet_file.iter_batches(columns=["Is Laundering"], batch_size=500_000):
            fraud_transactions += int(batch.to_pandas()["Is Laundering"].sum())

    accounts = pd.read_csv(accounts_path) if accounts_path.exists() else pd.DataFrame()
    communities = pd.read_csv(communities_path, usecols=["community_id"]) if communities_path.exists() else pd.DataFrame()
    if communities.empty and "community_id" in accounts.columns:
        communities = accounts[["community_id"]]
    top_accounts = accounts.sort_values("total_degree", ascending=False).head(12).fillna(0).to_dict("records") if not accounts.empty else []

    transaction_columns = ["Timestamp", "src_acct", "dst_acct", "Amount Paid", "Payment Format", "Is Laundering"]
    transactions = pd.DataFrame()
    if split_test.exists():
        transactions = next(pq.ParquetFile(split_test).iter_batches(batch_size=30, columns=transaction_columns)).to_pandas()
    transaction_rows = transactions.fillna("").to_dict("records") if not transactions.empty else []

    # Keep a small, real edge sample for the interactive network view. The full
    # graph remains in Neo4j; this browser snapshot only needs enough edges to
    # make account selection visibly change the local neighborhood.
    edge_rows: list[dict[str, Any]] = []
    focal_ids = set(str(row.get("account_id", "")) for row in top_accounts)
    focal_ids.update(str(row.get("src_acct", "")) for row in transaction_rows)
    focal_ids.update(str(row.get("dst_acct", "")) for row in transaction_rows)
    if split_test.exists() and focal_ids:
        for batch in pq.ParquetFile(split_test).iter_batches(columns=["src_acct", "dst_acct", "Amount Paid", "Is Laundering"], batch_size=500_000):
            frame = batch.to_pandas()
            related = frame[frame["src_acct"].astype(str).isin(focal_ids) | frame["dst_acct"].astype(str).isin(focal_ids)]
            edge_rows.extend(related.head(2_000).fillna("").to_dict("records"))
            if len(edge_rows) >= 2_000:
                break

    # Prefer current DATA 298B Gemma validation traces. The older llm_outputs
    # directory contains historical/demo artifacts and must not be presented as
    # the current investigator report set.
    reports: list[dict[str, Any]] = []
    current_traces = ARTIFACTS / "final_benchmark" / "posthoc_audit" / "gemma_validation_traces_guarded_v2.jsonl"
    if current_traces.exists():
        for line in current_traces.read_text().splitlines():
            trace = json.loads(line)
            report = trace.get("report", {})
            context = trace.get("context", {})
            accounts_in_case = context.get("accounts", []) if isinstance(context, dict) else []
            reports.append(
                {
                    "reportId": f"GEMMA-{len(reports) + 1:03d}",
                    "case": trace.get("case_id", "unknown"),
                    "account": accounts_in_case[0].get("account_id", "") if accounts_in_case else "",
                    "pattern": report.get("pattern", "UNKNOWN"),
                    "decision": report.get("decision", "UNKNOWN"),
                    "riskLevel": str(report.get("risk_level", "UNKNOWN")).upper(),
                    "confidence": report.get("confidence", 0),
                    "model": "Gemma",
                    "date": trace.get("variant", "final unseen validation"),
                    "status": "Validated artifact",
                    "evidence": report.get("evidence", []),
                    "actions": report.get("recommended_actions", []),
                    "summary": report.get("summary", ""),
                }
            )

    llm_model_counts: dict[str, int] = {"Gemma": 0, "Qwen": 0, "Ministral": 0, "Phi-4": 0}
    for report in reports:
        model_name = str(report.get("model", "")).lower()
        key = next((name for name in ("Gemma", "Qwen", "Ministral", "Phi-4") if name.lower() in model_name), None)
        if key:
            llm_model_counts[key] += 1

    snapshot = {
        "meta": {
            "source": "existing pipeline artifacts",
            "generatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
            "mockFields": ["liveTransactionSearch", "liveGraphQuery", "liveLLMRun"],
        },
        "summary": {
            "totalTransactions": total_transactions,
            "totalAccounts": int(len(accounts)),
            "fraudTransactions": fraud_transactions,
            "communities": int(communities["community_id"].nunique()) if not communities.empty else 0,
        },
        "models": {
            "baseline": read_json(MODELS / "xgboost_baseline_metrics.json"),
            "graphEnhanced": read_json(MODELS / "xgboost_graph_enhanced_metrics.json"),
            "comparison": read_json(MODELS / "model_comparison.json"),
        },
        "transactions": transaction_rows,
        "accounts": top_accounts,
        "graphEdges": edge_rows,
        "llmModelCounts": llm_model_counts,
        "reports": reports,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(snapshot, indent=2, default=str) + "\n")
    print(f"Wrote {OUTPUT}")
    print(f"Transactions: {total_transactions:,}; accounts: {len(accounts):,}; reports: {len(reports)}")


if __name__ == "__main__":
    main()
