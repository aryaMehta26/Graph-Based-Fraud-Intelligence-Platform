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
pnpm install
pnpm dev --host 127.0.0.1 --port 4173
```

For the normal pipeline workflow, run the repository launcher from the project root. It refreshes the snapshot, builds the dashboard, and serves it on localhost:

```bash
./scripts/run_local_dashboard.sh
```

Open [http://localhost:4173/#/overview](http://localhost:4173/#/overview).

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

Currently available from real pipeline outputs: summary counts, account graph features, community assignments, model metrics, and saved investigation reports. Live transaction filtering, live Neo4j graph queries, and live LLM investigation runs are clearly marked in the UI and remain mock/placeholder surfaces until corresponding backend endpoints exist. The planned endpoint contract is documented in `frontend/src/data/README.md`.
