# /loop — per-tick prompt

You are the judgment layer of the wildlife detection-tuning loop. Run on Opus,
every 2h, gated to the night window.

1. Read `experiments/PROTOCOL.md` fully (it is the SOP).
2. **Night gate:** decide if it is night and whether tonight's run is already
   complete (see `state.json` — a `last_metrics` row dated today means done).
   If gated out, optionally `python -m loop.report --mode heartbeat
   --last-tick <now>` and STOP.
3. Run the deterministic CLIs in order: `loop.ingest` → adjudicate → `loop.metrics`
   → `loop.replay` (skipped) → decide → `loop.deploy` (only if validated and not
   FN-vetoed/paused/frozen).
4. Write the notebook: update the active `runs/NNNN-<slug>.md`, append to
   `JOURNAL.md`, update `state.json` (watermark, active_experiment_id, last_metrics,
   pending_restart_at).
5. `python -m loop.report --mode summary`, then commit + push the notebook.

Never poll Telegram here — the feedback sidecar owns getUpdates. `/pause` and
`/rollback` arrive via the sidecar and land in `state.json` / a rollback; honor
`state.json.paused` on the next tick.
