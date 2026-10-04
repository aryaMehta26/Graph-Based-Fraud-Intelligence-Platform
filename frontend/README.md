# Fraud Intelligence Dashboard

This frontend is the Figma Make implementation integrated into the existing fraud-intelligence repository. It preserves the existing Python/Neo4j pipeline and adds six responsive dashboard routes:

- `#/overview`
- `#/transactions`
- `#/graph`
- `#/llm`
- `#/modelcomp`
- `#/reports`

## Run locally

From this directory:

```bash
npm install
npm run dev -- --host 127.0.0.1 --port 8443
```

The dashboard proxies `/api` requests to the read-only Neo4j bridge on
`http://127.0.0.1:8765`. Start Neo4j Desktop and, from the repository root, run:

```bash
python3 scripts/dashboard_neo4j_api.py
```

If port 8765 is already occupied, verify the existing bridge with
`curl http://127.0.0.1:8765/api/health` instead of starting a second copy.

For the normal pipeline workflow, run the repository launcher from the project root. It refreshes the snapshot, builds the dashboard, and serves it on localhost:

```bash
./scripts/run_local_dashboard.sh
```

Open [http://localhost:8443/#/overview](http://localhost:8443/#/overview).

The existing Streamlit application is unchanged and still starts with:

```bash
python3 -m streamlit run dashboard_app.py
```

## Pipeline data bridge

Refresh the frontend snapshot from the existing pipeline artifacts with:

```bash
cd ..
.venv-fraud/bin/python scripts/export_frontend_snapshot.py
```

The snapshot is written to `frontend/src/data/dashboard.json`. The adapter in `frontend/src/data/api.ts` will use `VITE_API_BASE_URL` when configured, and otherwise uses this generated snapshot.

The current demo uses the frozen DATA 298B artifacts for summary counts, model
metrics, selected Gemma reports, and transaction context. The Network Analysis
page uses live, read-only Neo4j neighborhood queries when the API bridge is
available and falls back to the checked-in snapshot when it is not. Reports
open the matching case in the LLM Investigator; they do not start new model
inference. This is intentional: the dashboard is a reproducible demo of the
frozen evaluation artifacts.
