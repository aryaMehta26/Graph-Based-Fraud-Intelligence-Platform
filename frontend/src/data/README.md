# Dashboard data boundary

`dashboard.json` is generated from the existing pipeline artifacts by:

```bash
.venv-fraud/bin/python scripts/export_frontend_snapshot.py
```

It reads the existing Parquet, graph-feature CSV, model metric JSON, and saved
LLM report artifacts. It does not run, retrain, or mutate the fraud pipeline.

When the additive read-only API is available, set `VITE_API_BASE_URL` and the
frontend will request `/api/dashboard/snapshot`, falling back to this snapshot
if the API is unavailable.

Planned API mappings:

| UI surface | Endpoint | Current source |
| --- | --- | --- |
| Overview | `GET /api/dashboard/summary` | processed Parquet/CSV and model JSON |
| Transactions | `GET /api/transactions` | scored test split |
| Graph | `GET /api/accounts/{id}` and `GET /api/graph/{id}` | graph features / Neo4j |
| Investigator | `GET /api/investigations` and `POST /api/investigations` | saved LLM artifacts / investigator script |
| Model comparison | `GET /api/models/comparison` | model metric JSON |
| Reports | `GET /api/reports` | `artifacts/llm_outputs/*.json` |

The current dashboard intentionally uses frozen artifacts for the transaction,
model-comparison, investigator, and report surfaces. The Network Analysis page
uses the read-only local Neo4j bridge (`scripts/dashboard_neo4j_api.py`) for
live account neighborhoods and falls back to the snapshot when Neo4j is not
available. The LLM Investigator displays saved Gemma validation reports; it
does not run new inference from the browser.
