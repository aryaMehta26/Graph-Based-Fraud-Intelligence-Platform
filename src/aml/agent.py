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

AGENTIC_SYSTEM_PROMPT = SYSTEM_PROMPT + """

The messages may also contain retrieved evidence from transaction_lookup, graph_lookup,
and community_lookup. Treat those tool results as the authoritative evidence source.
Use them when writing the evidence list, and do not add facts that are absent from the
case context or retrieved results."""

def _json(text):
    # Decode the first complete JSON object. Models occasionally append
    # repeated text after an otherwise valid report; a greedy regex makes the
    # whole response fail in that case.
    start = text.find("{")
    if start < 0:
        raise ValueError("model response did not contain JSON")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("model response did not contain valid JSON") from exc
    if not isinstance(obj, dict):
        raise ValueError("model response JSON was not an object")
    return obj

class AMLAgent:
    def __init__(self, backend, max_tool_calls=6, artifact_dir="artifacts/traces"):
        self.backend = backend; self.max_tool_calls = max_tool_calls; self.artifact_dir = artifact_dir

    def investigate(self, case: InvestigationCase, *, model_family: str, variant: str = "base", seed: int = 42, inference_config: Dict[str, Any] = None, prefetch_tools: bool = False) -> Dict[str, Any]:
        started = time.perf_counter(); trace_id = uuid.uuid4().hex
        system_prompt = AGENTIC_SYSTEM_PROMPT if prefetch_tools else SYSTEM_PROMPT
        context = case.agent_context()
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(context, default=str)}]
        trace = {"trace_id": trace_id, "case_id": case.case_id, "model_family": model_family, "variant": variant, "seed": seed, "tool_budget": self.max_tool_calls, "tool_calls": [], "timestamp": datetime.now(timezone.utc).isoformat(), "inference_config": inference_config or {}, "context": case.agent_context(), "ground_truth": case.ground_truth}

        # Controlled retrieval mode for the agentic benchmark. The model does not
        # choose the tools; the harness retrieves the same evidence package for every
        # model, records the calls, and lets the model synthesize the final report.
        if prefetch_tools:
            retrieval_plan = [("transaction_lookup", {}), ("graph_lookup", {}), ("community_lookup", {})]
            for name, args in retrieval_plan[:self.max_tool_calls]:
                call = {"name": name, "arguments": args, "success": False}
                try:
                    result = execute_tool(name, context, args)
                    call.update({"success": True, "result": result})
                except Exception as exc:
                    call["error"] = str(exc)
                trace["tool_calls"].append(call)
                messages.append({"role": "tool", "name": name, "content": json.dumps(call, default=str)})
            messages.append({"role": "user", "content": "Use the retrieved tool results above and return the required JSON report now."})

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
