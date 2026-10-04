# DATA 298B — Two-Day Worklog and Final Handoff

**Branch:** `feature/298b-backend`  
**Selected model:** Gemma 4 31B AML fine-tuned model  
**Purpose:** Record the work completed while extending the original DATA 298A fraud platform into the DATA 298B layered AML investigation system.

## 1. Starting point

We started from the old `main` branch, which already contained the DATA 298A system:

- IBM HI-Medium AML transaction preprocessing.
- XGBoost fraud detection.
- Neo4j graph loading and graph analytics.
- Leiden/Louvain-style community analysis.
- A Streamlit dashboard.
- Historical Claude investigation outputs.

The old Claude reports remain historical 298A material. They are excluded from the primary DATA 298B investigator/report path.

## 2. Final DATA 298B architecture

```text
Layer 1: Transaction Detection          — XGBoost
Layer 2: Graph Intelligence              — Neo4j
Layer 3: Community / Trend Intelligence — Leiden
Layer 4: AML Investigation               — selected fine-tuned LLM
```

The end-to-end evidence path is:

```text
held-out transactions
→ XGBoost score
→ graph features
→ Leiden/community context
→ investigation-case construction
→ deterministic retrieval
→ LLM investigation report
```

The LLM report schema contains:

- suspicious/legitimate decision;
- fraud pattern;
- risk level;
- confidence;
- evidence;
- recommended analyst actions;
- summary.

## 3. Candidate models

The model registry is [`configs/models/registry.yaml`](configs/models/registry.yaml).

| Model | Checkpoint | Size | Used in final four-model comparison |
|---|---|---:|---|
| Ministral | `mistralai/Ministral-3-14B-Instruct-2512-BF16` | 14B | Yes |
| Phi-4 | `microsoft/phi-4` | 14B | Yes |
| Qwen | `Qwen/Qwen3.5-27B` | 27B | Yes |
| Gemma | `google/gemma-4-31B` | 31B | Yes |

Granite was kept in the registry and explored during development, but it was not one of the final four compared models.

## 4. Training and model preparation

The models were prepared in Google Colab using:

- [`notebooks/298B_LLM_Training_Colab.ipynb`](notebooks/298B_LLM_Training_Colab.ipynb)
- [`notebooks/298B_LLM_Training_Colab_Grounded.ipynb`](notebooks/298B_LLM_Training_Colab_Grounded.ipynb)

The training process used supervised fine-tuning/QLoRA-style adapters. The base models were not retrained from scratch. Model weights and adapters were too large for GitHub and were stored in Colab/Google Drive, in model-specific directories such as:

```text
DATA298B/adapters/{model}_full_grounded/
```

Training and inference fixes included:

- pre-rendering Ministral SFT conversations;
- accepting native Ministral tokenizer artifacts;
- supporting tokenizers without chat templates;
- enforcing concise JSON-only AML reports;
- adding Phi-4 training support;
- disabling unnecessary model thinking behavior where required.

The shared agent and schema code is in `src/aml/agent.py` and `src/aml/schemas.py`. The agent records model family, variant, tools, raw response, parsed report, latency, tokens, and evaluator-only ground truth.

## 5. Leakage-controlled case design

[`scripts/build_heldout_benchmark.py`](scripts/build_heldout_benchmark.py) and [`scripts/build_integrated_298b_benchmark.py`](scripts/build_integrated_298b_benchmark.py) build held-out cases.

The integrated builder reuses existing artifacts rather than retraining anything. It:

1. Reads the held-out test split.
2. Applies or joins the saved XGBoost score.
3. Joins graph features.
4. Joins Leiden/community features.
5. Builds case-level investigation records.
6. Preserves source transaction IDs and case IDs.
7. Stores labels and provenance for evaluation only.
8. Verifies that model-visible context excludes labels and provenance.

The final artifact was:

```text
/content/drive/MyDrive/DATA298B/heldout/integrated_cases.jsonl
```

It contained:

```text
50 total cases
40 suspicious
10 legitimate
5 cases for each of the 8 fraud patterns
10 NONE/legitimate cases
```

## 6. Upstream ML and graph layers

### XGBoost

The project preserves baseline and graph-enhanced XGBoost artifacts. PR-AUC is emphasized because the dataset is severely imbalanced; accuracy alone would be misleading.

### Neo4j

The graph ontology is:

```text
(Account)-[:SENT]->(Transaction)-[:RECEIVED_BY]->(Account)
```

Neo4j provides account degree and centrality features. The dashboard API now queries both outgoing and incoming relationships. This fixed the earlier issue where an incoming-only account appeared to have zero degree.

### Leiden/community analysis

Community context includes `community_id`, `community_size`, and `community_fraud_rate`. The final manifest reports 66,227 distinct Leiden communities.

## 7. Colab evaluation workflow

The main notebooks are:

- [`298B_LLM_Integrated_Four_Model_Final_Colab.ipynb`](notebooks/298B_LLM_Integrated_Four_Model_Final_Colab.ipynb) — final integrated four-model benchmark.
- [`298B_LLM_Evaluation_Colab.ipynb`](notebooks/298B_LLM_Evaluation_Colab.ipynb) — earlier evaluation workflow.
- [`298B_LLM_Agentic_Evaluation_Colab.ipynb`](notebooks/298B_LLM_Agentic_Evaluation_Colab.ipynb) — slower tool-use benchmark.
- [`298B_LLM_Metrics_Only_Colab.ipynb`](notebooks/298B_LLM_Metrics_Only_Colab.ipynb) — metrics from saved traces without loading models.
- [`298B_Final_Gemma_Validation.ipynb`](notebooks/298B_Final_Gemma_Validation.ipynb) — final unseen Gemma validation.

The normal workflow was:

1. Mount Drive.
2. Clone/upload the repository.
3. Install dependencies and authenticate with Hugging Face when required.
4. Verify the held-out artifact and adapter paths.
5. Load one model at a time.
6. Run the long model cell with progress output and flushed traces.
7. Resume completed cases after interruption.
8. Save traces and metrics to Drive.
9. Apply post-hoc audits/guardrails without rerunning inference.
10. Clear GPU memory and disconnect after outputs were safely saved.

Runtime problems handled during the work:

- Google Drive quota errors.
- Incorrect `/content` repository paths.
- Missing `bitsandbytes` for 4-bit loading.
- Gemma GPU out-of-memory pressure on a 40 GB runtime.
- Long-running cells that appeared silent because output was not flushed.
- Need to resume from existing JSONL traces instead of restarting completed cases.

Important Drive locations included:

```text
/content/drive/MyDrive/DATA298B/heldout/
/content/drive/MyDrive/DATA298B/final_benchmark/
/content/drive/MyDrive/DATA298B/final_validation_gemma/
```

Large adapters, model weights, full traces, and runtime caches were kept in Drive/Colab rather than GitHub.

## 8. Evaluation progression

### Four-model integrated benchmark

```text
50 shared held-out cases × 4 models
```

All models used the same cases, evidence format, system contract, deterministic retrieval, output schema, and evaluator.

### Agentic/tool-use benchmark

The slower benchmark evaluated the leading candidates with tool traces. It supported resuming completed cases and recorded transaction, graph, and community retrieval calls.

### Final unseen validation

After the four-model comparison, Gemma was selected and evaluated on 10 additional unseen cases:

```text
5 suspicious + 5 legitimate = 10 cases
```

## 9. Final dataset and detection results

The authoritative source is [`artifacts/final_benchmark/final_metrics_manifest.json`](artifacts/final_benchmark/final_metrics_manifest.json).

| Quantity | Final value |
|---|---:|
| Transactions | 31,898,238 |
| Accounts | 1,758,573 |
| Leiden communities | 66,227 |
| Fraud transactions | 35,230 |

Final detection metrics:

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline XGBoost | 0.304311 | 0.944734 | 0.017456 | 0.914213 | 0.034259 |
| Graph-enhanced XGBoost | 0.461665 | 0.984645 | 0.011098 | 0.991802 | 0.021950 |

Some older DATA 298A README/EDA tables contain earlier counts and metrics. Use the final manifest for the final presentation.

## 10. Four-model LLM results

These results are from the 50-case integrated benchmark.

| Model | Binary F1 | Specificity | Pattern Macro-F1 | Evidence supported | Unsupported claims | Avg latency (s) | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ministral | 0.898876 | 0.10 | 0.206310 | 0.504673 | 0.373832 | 83.27048 | 6.6338 |
| Phi-4 | 0.941176 | 0.50 | 0.127969 | 0.409314 | 0.492647 | 35.01395 | 9.55105 |
| Qwen | 0.898876 | 0.10 | 0.191046 | 0.637874 | 0.126246 | 172.59264 | 3.23189 |
| Gemma | **0.963855** | **0.70** | **0.225210** | 0.541833 | 0.354582 | 97.49147 | 3.47353 |

Gemma was selected because it led on binary F1, specificity, and pattern Macro-F1. Phi-4 was fastest. Qwen had stronger automated evidence support and fewer unsupported claims but weaker binary decision performance and higher latency.

Pattern classification was weak for every model and is disclosed as a limitation.

## 11. Final Gemma validation and guardrails

The selected model was evaluated on 10 unseen cases. Final guarded results:

| Metric | Result |
|---|---:|
| Cases | 10 |
| Suspicious | 5 |
| Legitimate | 5 |
| Recall | 1.00 |
| Specificity | 0.60 |
| False positives | 2 of 5 legitimate cases |

[`scripts/apply_report_guardrails.py`](scripts/apply_report_guardrails.py) applies deterministic post-processing to saved reports. The guardrails:

- reject fan-in/fan-out labels that conflict with observed topology;
- avoid treating one self-loop as sufficient proof of a laundering cycle;
- downgrade suspicious outputs with neither structural nor XGBoost support.

This did not rerun Gemma. It transformed saved traces and recomputed metrics.

## 12. Manual trace audit

[`artifacts/final_benchmark/manual_audit.md`](artifacts/final_benchmark/manual_audit.md) reviewed five cases per model:

- two legitimate cases;
- one FAN_IN case;
- one CYCLE case;
- one SCATTER_GATHER case.

The audit checked binary decisions, pattern correctness, evidence support, arithmetic consistency, and completeness. It found legitimate false positives, pattern confusion, arithmetic/derived-claim errors, and occasional incomplete/default reports.

The correct presentation is that Gemma is the best selected research-prototype model under the frozen benchmark—not a perfect or autonomous AML decision maker.

## 13. Metrics and audit code

- [`scripts/run_298b_fast_comparison.py`](scripts/run_298b_fast_comparison.py) — controlled no-tool comparison.
- [`scripts/recompute_298b_metrics.py`](scripts/recompute_298b_metrics.py) — recomputes metrics from saved traces without model weights.
- [`scripts/audit_saved_integrated_traces.py`](scripts/audit_saved_integrated_traces.py) — claim-level trace audit.
- [`scripts/apply_report_guardrails.py`](scripts/apply_report_guardrails.py) — deterministic report guardrails.
- [`scripts/run_298b_dry_run.py`](scripts/run_298b_dry_run.py) — no-model integration/tool smoke test.

## 14. Dashboard integration

The React/Vite dashboard is under `frontend/` and has six routes:

1. **Overview** — 31.9M transactions, 1.76M accounts, fraud statistics, architecture, status, selected Gemma model, and suspicious-case CTA.
2. **Transactions** — 50-row held-out preview, search/filtering, saved XGBoost score when available, network links, and investigator handoff.
3. **Network Analysis** — live read-only Neo4j neighborhoods, clickable nodes, incoming/outgoing edges, degree, centrality, and community fields.
4. **LLM Investigator** — selected Gemma report, decision, pattern, risk, confidence, evidence, and recommended actions.
5. **Model Comparison** — Detection Models tab and LLM Models tab.
6. **Reports** — current Gemma validation reports, with old Claude reports excluded from the primary path.

### Frozen and live behavior

Frozen dashboard artifacts:

- `frontend/src/data/data298b.json` — final DATA 298B metrics and selected model.
- `frontend/src/data/dashboard.json` — dashboard-safe transaction/account/report snapshot.
- `artifacts/final_benchmark/*` — selection, manifest, audit, and validation artifacts.

Live dashboard surface:

- Neo4j account-neighborhood queries through `scripts/dashboard_neo4j_api.py`.

The dashboard intentionally does not run new 31B Gemma inference when “Investigate” is clicked. New inference was performed in Colab, saved as traces, audited, and then exposed as deterministic reports. This makes the demo reproducible and avoids requiring every teammate to load a 31B model locally.

### Neo4j API

```text
GET /api/health
GET /api/neo4j/stats
GET /api/graph/{account_id}?limit=100
```

For account `810520F70`, the corrected live query returned:

```text
out_degree: 0
in_degree: 58
total_degree: 58
degree_centrality: 0.0000329813
```

Centrality values appear close to zero because they are normalized against approximately 1.76M accounts. The old `0.0000` display was rounding, and the old zero-degree result also ignored incoming relationships.

### Report routing fix

Reports now preserve the selected case ID when opening the investigator. For example:

```text
GEMMA-007 → validation_legitimate_002
GEMMA-008 → validation_legitimate_003
GEMMA-010 → validation_legitimate_005
```

If one of those legitimate cases is labeled suspicious, that is a model false positive, not a routing problem.

## 15. Important dashboard files

| File | Purpose |
|---|---|
| `frontend/src/App.tsx` | Six-route dashboard shell |
| `frontend/src/pages/OverviewPage.tsx` | Overview |
| `frontend/src/pages/TransactionsPage.tsx` | Transaction review |
| `frontend/src/pages/GraphPage.tsx` | Neo4j network analysis |
| `frontend/src/pages/LLMPage.tsx` | Gemma investigator |
| `frontend/src/pages/ModelCompPage.tsx` | Detection/LLM comparison |
| `frontend/src/pages/ReportsPage.tsx` | Validated reports |
| `frontend/src/state.ts` | Selected transaction/account/case handoff |
| `scripts/export_frontend_snapshot.py` | Regenerates dashboard snapshot |
| `scripts/dashboard_neo4j_api.py` | Read-only Neo4j API |
| `docs/298B_BACKEND.md` | Backend architecture |
| `docs/298B_DASHBOARD_HANDOFF.md` | Teammate setup |

## 16. Run instructions

Create a personal root `.env`; never commit it:

```text
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_local_password
NEO4J_DB=neo4j
```

Start Neo4j Desktop, then use two terminals:

```bash
# Terminal 1
python3 scripts/dashboard_neo4j_api.py

# Terminal 2
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 8443
```

Open:

```text
http://127.0.0.1:8443/#/overview
```

Checks:

```bash
curl http://127.0.0.1:8765/api/health
cd frontend && npm run build
cd .. && pytest -q
```

## 17. Recommended demo narrative

```text
Overview
→ Transactions
→ View details
→ View sender/receiver network
→ Network Analysis
→ Confirm live Neo4j context
→ LLM Investigator
→ Model Comparison
→ Reports
→ Select report
→ Open in investigator
```

Suggested explanation:

> XGBoost identifies suspicious transaction candidates. Neo4j supplies relational account and transaction structure. Leiden supplies community context. Deterministic retrieval assembles the evidence package. The fine-tuned Gemma investigator turns that package into a structured AML report. The dashboard replays validated traces while keeping the graph neighborhood live.

## 18. Git milestones

| Commit | Work |
|---|---|
| `e9e3f60` | Added Phi-4 training option |
| `01ec789` | Built leakage-checked held-out benchmark |
| `2c818b4` | Added four-model Colab comparison |
| `cb17afd` | Added trace-only metric recalculation |
| `2534367` | Added agentic top-two benchmark |
| `519c877` | Added resumable traces |
| `264d9ed` | Added integrated benchmark and guardrails |
| `f72b2a9` | Merged DATA 298B dashboard |
| `3e3b37f` | Connected DATA 298B metrics |
| `9501325` | Connected Neo4j API/Gemma status |
| `86d06c6` | Locked Gemma and reconciled artifacts |
| `eb6e865` | Fixed graph direction and report routing |
| `ac2b060` | Added teammate handoff documentation |
| `3a10ab5` | Expanded transaction preview to 50 rows |
| `b960383` | Kept investigator cards in one row |

The branch is pushed to:

```text
origin/feature/298b-backend
```

## 19. Final status and limitations

### Complete

- Four-model integrated benchmark.
- Gemma final selection.
- Ten-case unseen Gemma validation.
- Manual trace audit.
- Deterministic report guardrails.
- Frozen metrics and reports.
- React dashboard integration.
- Read-only Neo4j bridge.
- Report-to-investigator routing.
- Teammate documentation.

### Limitations to disclose

- Pattern classification remains weak.
- False positives remain: two of five legitimate final-validation cases were flagged after guardrails.
- Final unseen validation has only 10 cases.
- Browser clicks do not run new Gemma inference.
- Complete row-level SHAP explanations are not currently exposed in the React snapshot.
- Community information is shown in Network Analysis when available; there is no separate community tab.
- Live graph analysis requires a teammate’s local Neo4j database.
- The old Streamlit/Claude surfaces remain historical or parallel artifacts, not the primary DATA 298B UI.

This should be presented as a research prototype and analyst-assistance workflow, not autonomous production AML decisioning.

## 20. Source-of-truth rules

Use these files when numbers need to be reconciled:

1. `artifacts/final_benchmark/final_metrics_manifest.json` — final project-wide numbers.
2. `artifacts/final_benchmark/selected_model.json` — final model decision and rationale.
3. `frontend/src/data/data298b.json` — dashboard LLM comparison.
4. `frontend/src/data/dashboard.json` — dashboard snapshot and reports.
5. Saved traces and audit CSVs — case-level inspection.

Do not use old Claude reports, temporary Colab outputs, or earlier README tables as final DATA 298B results.

## 21. Teammate Codex prompt

```text
Pull feature/298b-backend from the Graph-Based-Fraud-Intelligence-Platform repository.

Set up the final DATA 298B dashboard. Do not retrain models, modify the Streamlit app, commit .env files, or mix old Claude/298A reports into the current UI. The selected model is Gemma 4 31B AML fine-tuned model. Use the frozen metrics and reports already in the repository.

Install Python requirements and frontend dependencies. Start Neo4j Desktop with the local database, create a personal .env with Neo4j credentials, start python3 scripts/dashboard_neo4j_api.py on port 8765, and start the frontend on port 8443.

Verify:
- /api/health returns online.
- /api/graph/810520F70 returns incoming degree 58 and total degree 58 when the same Neo4j dataset is loaded.
- the dashboard has Overview, Transactions, Network, LLM Investigator, Model Comparison, and Reports.
- Transactions shows the 50-row held-out preview.
- Reports → View → Open in investigator preserves the selected case ID.
- four-model comparison and final Gemma validation metrics come from frozen artifacts.
- npm run build passes.

Keep unrelated local files uncommitted and never expose secrets.
```
