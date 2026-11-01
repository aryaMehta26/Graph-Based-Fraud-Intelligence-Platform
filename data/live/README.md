# Live replay bundle (champion detection model)

Read by `scripts/live_dashboard_api.py`. Built from the clean chronological split by
`experiments/live_replay/build_live_replay.py` on the `prajwal/local-setup-and-detection-eval` branch.

| File | Contents |
| --- | --- |
| `replay_v2_leakfree.parquet` | First 200,000 transactions of the clean test period (Sep 15 01:24 - 03:58), display columns, label, and the 52 leak-free features. `amount_bucket` is stored with the training-time codes. |
| `xgboost_v2_leakfree.json` | Champion XGBoost (tabular + 28 pattern + 13 extra features, scale_pos_weight 10), trimmed to its best early-stopping iteration. |
| `live_model.json` | Feature order, thresholds chosen on validation, full-test metrics, replay-window counts. |

Thresholds (validation period): review 0.773 (F1-optimal), HIGH 0.899 (1,000 alerts/day budget).
Full test period at the review threshold: precision 0.777, recall 0.485, F1 0.597, 637 false positives.
Replay window: 173 laundering transactions, 68 alerts, 53 of them laundering.

Scores computed from this bundle match the evaluation scores exactly; the label is only exposed to the
dashboard's evaluation mode and is stripped before a case is sent to Gemma.
