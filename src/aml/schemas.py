"""Shared, dependency-light schemas for cases, reports, and traces."""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json

PATTERNS = {"FAN_IN", "FAN_OUT", "CYCLE", "STACK", "SCATTER_GATHER", "GATHER_SCATTER", "BIPARTITE", "RANDOM", "EMERGING_UNKNOWN", "NONE"}
RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

@dataclass
class InvestigationReport:
    decision: str = "SUSPICIOUS"
    pattern: str = "EMERGING_UNKNOWN"
    risk_level: str = "MEDIUM"
    confidence: float = 0.0
    evidence: List[Any] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    summary: str = ""

    def validate(self) -> List[str]:
        errors = []
        if self.decision not in {"SUSPICIOUS", "LEGITIMATE"}: errors.append("invalid decision")
        if self.pattern not in PATTERNS: errors.append("invalid pattern")
        if self.risk_level not in RISK_LEVELS: errors.append("invalid risk_level")
        if not isinstance(self.evidence, list): errors.append("evidence must be a list")
        if not isinstance(self.recommended_actions, list): errors.append("recommended_actions must be a list")
        return errors

@dataclass
class InvestigationCase:
    case_id: str
    context: Dict[str, Any]
    ground_truth: Dict[str, Any] = field(default_factory=dict)
    split: str = "benchmark"
    reference_report: Optional[Dict[str, Any]] = None

    def agent_context(self) -> Dict[str, Any]:
        """Return context with labels removed; labels stay evaluator-only."""
        blocked = {"is_laundering", "fraud_label", "pattern", "pattern_type", "ground_truth", "label"}
        def clean(value):
            if isinstance(value, dict): return {k: clean(v) for k, v in value.items() if str(k).strip().lower().replace(" ", "_") not in blocked}
            if isinstance(value, list): return [clean(v) for v in value]
            return value
        return clean(self.context)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), default=str)
