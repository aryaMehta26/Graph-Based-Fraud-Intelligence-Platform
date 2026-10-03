"""One common tool-using agent loop shared by all four model families."""
import json, re, time, uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict
from .schemas import InvestigationCase, InvestigationReport
from .tools import execute_tool, TOOLS

SYSTEM_PROMPT = """You are an AML investigator. Inspect the supplied transaction and graph context and return one compact JSON object only.

Your response MUST begin with { and end with }. Do not write analysis, reasoning, steps, headings, Markdown, or explanatory text before or after the JSON. Do not mention the prompt or describe how you reasoned.

Use exactly these keys: decision, pattern, risk_level, confidence, evidence, recommended_actions, summary.
decision must be SUSPICIOUS or LEGITIMATE. risk_level must be LOW, MEDIUM, HIGH, or CRITICAL. confidence must be a number from 0 to 1. evidence must be a list of concise factual strings grounded in the supplied context. recommended_actions must be a list of concise actions. summary must be one concise sentence.

Do not invent IDs, amounts, timestamps, or metrics. Pattern must be one of FAN_IN, FAN_OUT, CYCLE, STACK, SCATTER_GATHER, GATHER_SCATTER, BIPARTITE, RANDOM, EMERGING_UNKNOWN, NONE."""

def _json(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match: raise ValueError("model response did not contain JSON")
    return json.loads(match.group(0))

class AMLAgent:
    def __init__(self, backend, max_tool_calls=6, artifact_dir="artifacts/traces"):
        self.backend = backend; self.max_tool_calls = max_tool_calls; self.artifact_dir = artifact_dir

    def investigate(self, case: InvestigationCase, *, model_family: str, variant: str = "base", seed: int = 42, inference_config: Dict[str, Any] = None) -> Dict[str, Any]:
        started = time.perf_counter(); trace_id = uuid.uuid4().hex
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(case.agent_context(), default=str)}]
        trace = {"trace_id": trace_id, "case_id": case.case_id, "model_family": model_family, "variant": variant, "seed": seed, "tool_budget": self.max_tool_calls, "tool_calls": [], "timestamp": datetime.now(timezone.utc).isoformat(), "inference_config": inference_config or {}, "context": case.agent_context(), "ground_truth": case.ground_truth}
        for _ in range(self.max_tool_calls):
            response = self.backend.generate(messages, max_tokens=(inference_config or {}).get("max_tokens", 700), temperature=(inference_config or {}).get("temperature", 0.0))
            text = response.get("text", "")
            try:
                obj = _json(text)
                if all(k in obj for k in ("decision", "pattern", "risk_level")): break
            except (ValueError, json.JSONDecodeError): obj = {}
            tool_match = re.search(r"(?:TOOL|tool)\s*[:=]\s*([a-z_]+)(?:\s+|\n|$)(.*)", text)
            if not tool_match: break
            name = tool_match.group(1); args = {}
            try: args = json.loads(tool_match.group(2).strip())
            except json.JSONDecodeError: pass
            call = {"name": name, "arguments": args, "success": False}
            try: result = execute_tool(name, case.agent_context(), args); call.update({"success": True, "result": result})
            except Exception as exc: call["error"] = str(exc)
            trace["tool_calls"].append(call); messages.extend([{"role": "assistant", "content": text}, {"role": "tool", "content": json.dumps(call, default=str)}])
        else: obj = {}
        try: report = InvestigationReport(**{k: obj[k] for k in InvestigationReport.__dataclass_fields__ if k in obj})
        except Exception: report = InvestigationReport(summary="Invalid model output")
        trace.update({"report": asdict(report), "latency_seconds": time.perf_counter() - started, "tool_budget_hit": len(trace["tool_calls"]) >= self.max_tool_calls, "prompt_tokens": response.get("prompt_tokens", 0), "output_tokens": response.get("output_tokens", 0), "raw_response": response.get("text", "")})
        from pathlib import Path
        destination = Path(self.artifact_dir); destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{trace_id}.json").write_text(json.dumps(trace, default=str, indent=2))
        return trace
