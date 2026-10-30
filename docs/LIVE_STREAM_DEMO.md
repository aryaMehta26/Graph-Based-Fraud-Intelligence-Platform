# Live prediction demo

This optional local demo simulates incoming transactions. It does not change
the offline training pipeline, Parquet data, Neo4j, or saved model artifacts.

Start it from the repository root:

```bash
bash scripts/run_live_stream_demo.sh
```

Then run the React dashboard in another terminal:

```bash
./scripts/run_local_dashboard.sh
```

Open `http://localhost:4173/#/overview` or the Transactions page. The dashboard
will receive a new event every two seconds.

The transport is real HTTP Server-Sent Events. Predictions use the saved graph-
enhanced XGBoost model when it loads. Incoming transactions are synthetic demo
events. Replace `make_event()` with a Kafka, webhook, or database consumer for
a production stream.
