# DATA 298B demo runbook

## Architecture shown in the demo

1. Layer 1: Transaction Detection — XGBoost
2. Layer 2: Graph Intelligence — Neo4j
3. Layer 3: Community / Trend Intelligence — Leiden
4. Layer 4: AML Investigation — Gemma 4 31B AML fine-tuned model

The dashboard reads frozen benchmark and validation artifacts. It does not load
model weights or start new Gemma inference from the browser.

## Start the stack

1. Open Neo4j Desktop and start the `neo4j` database. Confirm that Bolt is
   listening on `bolt://127.0.0.1:7687`.

2. From the repository root, start the read-only graph bridge:

   ```bash
   python3 scripts/dashboard_neo4j_api.py
   ```

3. In a second terminal, start the dashboard:

   ```bash
   cd frontend
   npm run dev
   ```

4. Open `http://127.0.0.1:8443/`.

## Verify the services

```bash
curl http://127.0.0.1:8765/api/health
curl http://127.0.0.1:8765/api/neo4j/stats
curl 'http://127.0.0.1:8765/api/graph/842B97DC0?limit=3'
```

The first response should show `"status": "online"`. The API only reads from
Neo4j and never writes to the database.

## Demo walkthrough

1. Open **Overview** and click **Explore suspicious cases**.
2. In **Suspicious Transactions**, click a row and choose **View details**.
3. Choose **View sender network** or **View receiver network**.
4. In **Account & Network Analysis**, confirm the `Neo4j connected` badge,
   account degree values, retrieved edges, and clickable neighbor nodes.
5. Choose **Investigate account** or return to the transaction drawer and
   choose **Investigate case**.
6. In **LLM Investigator**, show the same account, case, decision, pattern,
   risk level, confidence, retrieved evidence, and analyst recommendations.
7. Open **Model Comparison**, switch between **Detection models** and **LLM
   investigators**, and show the highlighted **Gemma 4 31B AML fine-tuned
   model**.
8. Open **Reports**, click **View** for a current `GEMMA-*` report, then use
   **Open in investigator**.

## What to say about the numbers

- The integrated comparison used 50 cases across four LLMs.
- The frozen dataset snapshot contains 31,898,238 transactions, 1,758,573
  accounts, and 66,227 Leiden communities.
- Gemma's integrated binary F1 was 0.963855, specificity was 0.70, and
  pattern Macro-F1 was 0.225210.
- Guarded unseen validation used 10 cases. Recall was 1.00 and specificity
  was 0.60 after deterministic report guardrails.
- These are research-prototype results. Borderline cases require analyst
  review.

## Troubleshooting

- If the network page shows `Cached artifact`, keep Neo4j Desktop running and
  restart the API bridge.
- If port 8765 is busy, stop the old API process before starting another one.
- If the frontend shows stale content, refresh the page after Vite rebuilds.
- Do not enable Evaluation Mode during the normal analyst demo. It is only for
  QA review of evaluator-only labels.
