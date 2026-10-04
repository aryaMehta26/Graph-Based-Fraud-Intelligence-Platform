# DATA 298B dashboard handoff

The final dashboard is on `feature/298b-backend` and uses six routes:

- Overview
- Transactions
- Network Analysis
- LLM Investigator
- Model Comparison
- Reports

The selected model is Gemma 4 31B AML fine-tuned model. The dashboard reads
frozen DATA 298B artifacts for metrics and reports. The Network Analysis page
uses a read-only Neo4j API for live account neighborhoods.

## Teammate setup

```bash
git clone https://github.com/aryaMehta26/Graph-Based-Fraud-Intelligence-Platform.git
cd Graph-Based-Fraud-Intelligence-Platform
git checkout feature/298b-backend
npm --prefix frontend install
python3 -m pip install -r requirements.txt
```

Create `.env` in the repository root with local Neo4j credentials:

```text
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_local_password
NEO4J_DB=neo4j
```

Start Neo4j Desktop, then use two terminals:

```bash
# Terminal 1: read-only graph bridge
python3 scripts/dashboard_neo4j_api.py

# Terminal 2: dashboard
cd frontend
npm run dev -- --host 127.0.0.1 --port 8443
```

Open `http://127.0.0.1:8443/#/overview`.

Check the API before the demo:

```bash
curl http://127.0.0.1:8765/api/health
curl 'http://127.0.0.1:8765/api/graph/810520F70?limit=1'
```

The second request should include both incoming and outgoing graph metrics.
Reports should preserve the selected case when opening the investigator.

## Validation

```bash
cd frontend && npm run build
cd .. && pytest -q
```

The dashboard does not retrain models or run new Gemma inference. It is a
reproducible presentation layer over the frozen benchmark and validation
artifacts.
