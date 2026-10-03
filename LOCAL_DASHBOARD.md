# Local dashboard workflow

The Figma-based React dashboard runs as a local web application and is kept separate from the existing Python/Neo4j pipeline.

After running or refreshing the pipeline artifacts, start the dashboard from the repository root:

```bash
./scripts/run_local_dashboard.sh
```

Then open:

```text
http://localhost:4173/#/overview
```

The launcher performs three safe, read-only presentation steps:

1. Exports the current processed artifacts to `frontend/src/data/dashboard.json`.
2. Builds the React frontend.
3. Starts the local preview server.

The existing Streamlit application remains available separately:

```bash
python3 -m streamlit run dashboard_app.py
```

The dashboard uses real saved pipeline artifacts for summary metrics, transaction rows, account graph features, community assignments, model metrics, and investigation reports. Live Neo4j searches and new LLM runs remain unavailable until backend endpoints are added.
