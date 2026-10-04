# DATA 298B final model selection

## Decision

Gemma is selected as the provisional final model for the backend demonstration.
This selection is based on the integrated four-model benchmark, where Gemma had
the strongest binary F1, specificity, and pattern Macro-F1 among the candidates.

## Integrated four-model benchmark

The benchmark used the layered case artifact generated from the test split:

```text
test transactions -> XGBoost score -> graph features -> Leiden/community
-> investigation case -> deterministic retrieval -> LLM report
```

Gemma led the aggregate comparison with binary F1 0.9639, specificity 0.70,
and pattern Macro-F1 0.2252. Phi-4 was fastest, while Qwen produced stronger
post-hoc evidence support but weaker binary decision performance.

## Final unseen validation

The 10-case Gemma validation completed with 5 suspicious and 5 legitimate cases.
Raw model output recalled all suspicious cases but flagged every legitimate case.
Deterministic report guardrails were then applied without rerunning inference:

```text
Recall:       1.00
Specificity:  0.60
False positives: 2 of 5 legitimate cases
```

The guardrails reject fan-in/fan-out labels that conflict with observed graph
topology, ignore a lone self-loop as sufficient evidence of a laundering cycle,
and downgrade suspicious decisions with neither structural nor XGBoost support.

## Limitations

- The final validation set contains only 10 cases and is preliminary.
- Pattern classification remains weak on minority patterns.
- Two borderline cases require analyst review because of high XGBoost scores or
  invalid model output.
- The legacy exact-string faithfulness metric is not used as a final grounding
  claim; the post-hoc claim audit is reported separately.
- This result should be described as a research prototype, not production-ready
  AML decisioning.

## Frozen local artifacts

- `model_comparison_integrated.csv`
- `gemma_validation_traces.jsonl`
- `gemma_validation_traces_guarded_v2.jsonl`
- `audit_summary.json`
- `case_audit.csv`

The dashboard is intentionally out of scope for this freeze.
