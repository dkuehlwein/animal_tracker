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

**Checkpoint as you go** (see `PROTOCOL.md`): commit `gold/` right after adjudication and
`state.json` right after `loop.metrics` — a fresh tick has no memory and resumes from
committed state, so never re-spend tokens on work already on disk.

## Exact CLI invocations (run from repo root with PYTHONPATH=src)

All CLIs must be run from `/home/daniel/animal_tracker` (repo root) so that `.env`
is found by `Config()`. The `PYTHONPATH=src` prefix exposes the `loop.*` package.

```bash
# Full nightly chain (unattended — no manual glue needed between steps):
cd /home/daniel/animal_tracker

# 1. Ingest: pull all detections from the DB, starting past the stored watermark.
#    (Use --since-id 0 to replay the full history; normally omit to use watermark.)
PYTHONPATH=src uv run python -m loop.ingest --since-id 0

# 2. Measure: compute FP/FN metrics and write last_metrics into state.json.
#    Respects --state <path> for testing with a temp copy of state.json.
PYTHONPATH=src uv run python -m loop.metrics --state experiments/state.json

# 3. Report: read last_metrics from state.json, render summary.
#    --no-send renders and prints the text WITHOUT calling Telegram (safe to test).
#    Omit --no-send for the real nightly send.
PYTHONPATH=src uv run python -m loop.report --mode summary \
    --state experiments/state.json --no-send

# Dry-run with a temp state copy (never mutates the real state or DB):
cp experiments/state.json /tmp/state_test.json
PYTHONPATH=src uv run python -m loop.ingest --since-id 0
PYTHONPATH=src uv run python -m loop.metrics --state /tmp/state_test.json
PYTHONPATH=src uv run python -m loop.report --mode summary \
    --state /tmp/state_test.json --no-send
```

Key invariants:
- `.env` must be readable from the repo root (Config validation fails without
  `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`).
- `loop.metrics` writes the flat `last_metrics` dict into `state.json` automatically;
  `loop.report` reads it directly — no manual reshaping needed between stages.
- `--no-send` / `--dry-run` on `loop.report` renders without any Telegram credential
  check; safe to use for testing, CI, or agent dry runs.
- Running from inside `src/` fails because `.env` is not found there.

Never poll Telegram here — the feedback sidecar owns getUpdates. `/pause` and
`/rollback` arrive via the sidecar and land in `state.json` / a rollback; honor
`state.json.paused` on the next tick.
