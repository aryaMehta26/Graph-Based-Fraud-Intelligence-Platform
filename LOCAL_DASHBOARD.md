# Local dashboard workflow

The Figma-based React dashboard runs as a local web application and is kept separate from the existing Python/Neo4j pipeline.

After running or refreshing the pipeline artifacts, start the dashboard from the repository root:

```bash
python3 scripts/dashboard_neo4j_api.py

# In a second terminal
cd frontend
npm run dev
```

Then open:

```text
http://localhost:8443/#/overview
```

To refresh the browser snapshot before starting the frontend, run:

```bash
python3 scripts/export_frontend_snapshot.py
```

Then use `npm run build` for a production build or `npm run dev` for the demo.

The existing Streamlit application remains available separately:

```bash
python3 -m streamlit run dashboard_app.py
```

For the analyst demo, start the read-only Neo4j bridge separately:

```bash
python3 scripts/dashboard_neo4j_api.py
```

The dashboard uses real saved pipeline artifacts for summary metrics, transaction rows, account graph features, community assignments, frozen model metrics, and current Gemma validation reports. The Network page uses live read-only Neo4j retrieval when the bridge is running and falls back to the cached graph sample otherwise. The browser never starts new LLM inference.
