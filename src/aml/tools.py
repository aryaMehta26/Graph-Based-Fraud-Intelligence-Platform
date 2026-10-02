"""Deterministic AML tools. They only expose retrieved data, never labels."""
from collections import Counter, defaultdict
from typing import Any, Dict, List

def _txns(context): return context.get("transactions", [])

def transaction_tool(context: Dict[str, Any], account_id: str = "", limit: int = 100) -> Dict[str, Any]:
    rows = [t for t in _txns(context) if not account_id or t.get("src_acct") == account_id or t.get("dst_acct") == account_id]
    return {"tool": "transaction_lookup", "count": len(rows), "transactions": rows[:limit]}

def graph_tool(context: Dict[str, Any], account_id: str = "") -> Dict[str, Any]:
    accounts = [a for a in context.get("accounts", []) if not account_id or a.get("account_id") == account_id]
    edges = [{k: t.get(k) for k in ("txn_id", "src_acct", "dst_acct", "amount", "timestamp", "payment_format")} for t in _txns(context)]
    return {"tool": "graph_lookup", "accounts": accounts, "edges": edges, "graph_stats": context.get("graph_stats", {})}

def community_tool(context: Dict[str, Any], community_id: Any = None) -> Dict[str, Any]:
    accounts = context.get("accounts", [])
    cid = community_id if community_id is not None else context.get("community_id")
    members = [a for a in accounts if cid is None or a.get("community_id") == cid]
    return {"tool": "community_lookup", "community_id": cid, "community_size": context.get("community_size", len(members)), "community_fraud_rate": context.get("community_fraud_rate"), "members": members}

TOOLS = {"transaction_lookup": transaction_tool, "graph_lookup": graph_tool, "community_lookup": community_tool}

def execute_tool(name: str, context: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name not in TOOLS: raise KeyError(f"Unknown tool: {name}")
    return TOOLS[name](context, **arguments)
