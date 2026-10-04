# DATA 298B backend

The 298B implementation extends the existing 298A files. The final selected LLM is the Gemma 4 31B AML fine-tuned model. Claude artifacts remain historical 298A baseline material and are excluded from the current dashboard report path.

## Final architecture

```text
Layer 1: Transaction Detection — XGBoost
Layer 2: Graph Intelligence — Neo4j
Layer 3: Community / Trend Intelligence — Leiden
Layer 4: AML Investigation — Gemma 4 31B AML fine-tuned model
```

The authoritative selection record is [`artifacts/final_benchmark/selected_model.json`](../artifacts/final_benchmark/selected_model.json). It preserves the exact integrated benchmark metrics and guarded unseen-validation metrics used for the final selection.

## Controlled experiment

The exact checkpoints are centralized in [`configs/models/registry.yaml`](../configs/models/registry.yaml). All model families use the same case format, system prompt, deterministic tools, maximum tool-call budget, output schema, and trace format. `base` and `finetuned` are represented by the backend/adapter selection; the evaluator consumes saved traces and never adds labels to agent context.

## Workflow

```text
python3 -m src.aml.dataset --input-dir artifacts --output artifacts/datasets/aml_cases.jsonl
python3 -m src.training.qlora_sft --model ministral --dataset artifacts/datasets/aml_cases.jsonl --output-dir artifacts/adapters/ministral --max-steps 100
python3 -m src.evaluation.benchmark_runner --traces artifacts/traces.jsonl --output-dir artifacts/evaluation
python3 -m src.evaluation.report --input-dir artifacts/evaluation --output-dir artifacts/evaluation
```

The training command requires the optional GPU stack in `requirements.txt`; dataset building and metric calculation do not require GPU libraries. The 31B/30B/27B targets should be run on a suitable high-memory GPU, while Ministral is the practical development smoke target.

## Trace contract

Each agent run writes a JSON trace under `artifacts/traces/` containing case ID, model family, variant, registry/inference metadata, tool calls and failures, raw response, structured report, latency, token counts, and ground truth for evaluator use. The ground truth is deliberately kept outside `context`.
