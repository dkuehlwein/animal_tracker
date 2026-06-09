# /loop — per-tick prompt

You are the judgment layer of the wildlife detection-tuning loop. Run on Opus,
every 2h, gated to the night window.

1. Read `experiments/PROTOCOL.md` fully (it is the SOP).
2. The night gate and heartbeat are handled **before** this session starts by
   `loop.nightgate` (a deterministic Python module). If you are reading this
   prompt, the gate has already passed — it is night and tonight's run is not
   yet done. Do NOT re-check the gate or send a heartbeat here.
3. Run the deterministic CLIs in order: `loop.ingest` → adjudicate → `loop.metrics`
   → `loop.replay` (skipped) → decide → `loop.deploy` (only if validated and not
   FN-vetoed/paused/frozen).
4. Write the notebook: update the active `runs/NNNN-<slug>.md`, append to
   `JOURNAL.md`, update `state.json` (active_experiment_id, last_metrics,
   pending_restart_at). Do NOT hand-write the watermark — `loop.metrics` persists
   it automatically.
5. `python -m loop.report --mode summary`. Then, as your FINAL actions, run
   `python -m loop.endtick` to mark tonight complete (stamps `last_tick_completed_day`
   so the remaining 2h ticks skip — one Opus session/night), then commit + push the
   notebook. Run `loop.endtick` ONLY after the whole workflow above succeeded — if you
   were interrupted or any step failed, do NOT run it; the next tick resumes from
   committed state.

**Checkpoint as you go** (see `PROTOCOL.md`): use
`python -m loop.checkpoint --message <msg>` after each expensive or irreversible
step. Specifically:
- After tier-2 adjudication (step 3): `python -m loop.checkpoint --message "tick: tier-2 adjudication done"`
- After `loop.metrics` (step 4): `python -m loop.checkpoint --message "tick: metrics written"`

A fresh tick has no memory and resumes from committed state, so never re-spend
tokens on work already on disk.

## Change levers — env first, code changes allowed with cause

- **Default to an env-var delta** via `loop.deploy` (bounded by `guardrails.BOUNDS`,
  rollback = restore `best_known_good`). Reach for it first whenever a tunable
  parameter can plausibly achieve the goal.
- **Code changes are allowed** when no env knob can address the root cause — e.g. a
  motion-detection algorithm fix like the MOG2 recurrent-frame issue (backlog #4).
  Only do so with a clear, recorded justification in the active `runs/NNNN-<slug>.md`.
  Keep the change minimal and reversible; we roll back via `git revert`.
- **Commit code changes separately**, with the experiment id in the message, e.g.
  `fix(motion): exp #4 (mog2-recurrent-frames) — reset MOG2 learning rate; <why>`.
  Record the resulting commit SHA in the run file so the change is auditable and
  revertible.
- A code change only goes live on a **camera restart**. Stamp `pending_restart_at`
  in `state.json` (same ~60-min-pre-sunrise window as an env deploy) so
  `apply_pending_deploy` reloads it; if you also ship an env delta, `loop.deploy`
  already stamps the restart for you.
- **Same gates still apply** to code changes: FN-veto, `paused`, feedback-starved
  freeze, one experiment at a time, and the volume collapse/explosion guardrail.

## Exact CLI invocations (run from repo root with PYTHONPATH=src)

All CLIs must be run from `/home/daniel/animal_tracker` (repo root) so that `.env`
is found by `Config()`. The `PYTHONPATH=src` prefix exposes the `loop.*` package.

```bash
# Full nightly chain (unattended — no manual glue needed between steps):
cd /home/daniel/animal_tracker

# 1. Ingest: pull all detections from the DB, starting past the stored watermark.
#    (Use --since-id 0 to replay the full history; normally omit to use watermark.)
PYTHONPATH=src uv run python -m loop.ingest --since-id 0

# 2. Measure: compute FP/FN metrics, write last_metrics into state.json, AND
#    advance state["watermark"] to the new watermark automatically.
#    Respects --state <path> for testing with a temp copy of state.json.
PYTHONPATH=src uv run python -m loop.metrics --state experiments/state.json

# 3. Report: read last_metrics from state.json, render summary.
#    --no-send renders and prints the text WITHOUT calling Telegram (safe to test).
#    Omit --no-send for the real nightly send.
PYTHONPATH=src uv run python -m loop.report --mode summary \
    --state experiments/state.json --no-send

# 4. Checkpoint (after any expensive/irreversible step — stages experiments/ only):
PYTHONPATH=src uv run python -m loop.checkpoint --message "tick: metrics written"

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
- `loop.metrics` writes the flat `last_metrics` dict into `state.json` automatically
  AND advances `state["watermark"]` — no manual reshaping or watermark writes needed.
- `--no-send` / `--dry-run` on `loop.report` renders without any Telegram credential
  check; safe to use for testing, CI, or agent dry runs.
- Running from inside `src/` fails because `.env` is not found there.
- Do NOT hand-write the watermark in `state.json` — `loop.metrics` owns it.
- Do NOT send a heartbeat — `loop.nightgate` owns heartbeats on skipped ticks.
  `loop.endtick` is the completion stamp: YOU run it as your final step (step 5),
  only after a fully successful tick. It is deliberately NOT run by systemd, so an
  interrupted or incomplete tick never false-marks the night done — the next tick
  just resumes from committed state.

Never poll Telegram here — the feedback sidecar owns getUpdates. `/pause` and
`/rollback` arrive via the sidecar and land in `state.json` / a rollback; honor
`state.json.paused` on the next tick.
