# PROTOCOL — Autonomous Tuning Loop SOP

Read this FIRST every `/loop` tick. You (the judgment layer) reconstruct all
state from this git notebook; there is no hidden stage machine.

## When to spend tokens vs invoke Python
- Deterministic work (SQL reads, metric math, config writes, env render, restarts,
  Telegram sends) is done by `src/loop/*.py`. INVOKE them, read their JSON.
- Spend tokens only on: tier-2 adjudication of ambiguous crops, experiment design,
  self-audit, journaling.

## CLI invocation contract (MUST follow — wrong CWD breaks Config)
All `python -m loop.*` commands MUST be run from the **repo root**
(`/home/daniel/animal_tracker`) with `PYTHONPATH=src` so that `.env` is found by
`Config()`. Running from inside `src/` fails with a missing-token validation error.

```
PYTHONPATH=src uv run python -m loop.ingest --since-id <watermark>
PYTHONPATH=src uv run python -m loop.metrics [--state <path>] [--date <YYYY-MM-DD>]
PYTHONPATH=src uv run python -m loop.report --mode summary [--state <path>] [--no-send]
```

`loop.metrics` writes `last_metrics` into `state.json` automatically (flat shape
with top-level `date` + metric fields including `fp_ci` as a list). `loop.report`
reads it directly — no manual reshaping required between stages. Use `--no-send`
on `loop.report` to render without calling Telegram (safe for dry runs and testing).

## Daily cycle (one nightly run, resumable)
1. **Gate** — handled deterministically by `loop.nightgate` (runs before the LLM
   session). Checks: is it night? Is `state.json["last_tick_completed_day"]` !=
   `loop_day()`? If gated out → sends a heartbeat (once per loop-day) and stops.
   The LLM never sees a gated-out tick.
2. **Ingest** — `python -m loop.ingest --since-id <watermark>`; reconcile labels.
   Note: `loop.metrics` (step 4) now advances the watermark in `state.json`
   automatically — you do NOT hand-write the watermark.
3. **Label** — adjudicate ambiguous crops (tier-2); append to `gold/`. Never
   UPDATE/DELETE existing labels. **Checkpoint immediately after adjudicating**:
   `python -m loop.checkpoint --message "tick: tier-2 adjudication done"` — tier-2
   is the only token-expensive step; checkpointing means an interrupted tick never
   re-pays for it.
4. **Measure** — `python -m loop.metrics`; paired FP/FN + CIs → `metrics/daily.csv`.
   This also advances `state.json["watermark"]` to the new watermark automatically.
   Checkpoint after: `python -m loop.checkpoint --message "tick: metrics written"`.
5. **Check** — does the active experiment's prediction still hold (CI-based)? done?
6. **Self-audit (cadence)** — auto-labels vs the day's human labels; re-check past
   wins on the larger corpus; note confidence in `runs/`.
7. **Decide** — keep / rollback; if concluded, pick next from backlog / OFAT within
   bounds. Respect freeze + one-experiment-at-a-time + `paused`.
8. **Validate** — Layer A = `python -m loop.replay` (STUB → "skipped"). Layer B =
   bounds + predicted live effect. FN-veto: reject FP wins that worsen (or risk, if
   FN unmeasured) FN.
9. **Deploy** — `python -m loop.deploy --delta '{...}' --restart-at <pre-sunrise>`;
   writes state.json + renders env + stamps the restart.
10. **Record** — update `runs/NNNN-<slug>.md` (front matter + observations), append
    a `JOURNAL.md` line, update `state.json` pointers.
11. **Report** — `python -m loop.report --mode summary`; commit + push.
12. **Mark complete** — `python -m loop.endtick` stamps
    `state.json["last_tick_completed_day"] = loop_day()` so the rest of tonight's 2h
    ticks skip (one Opus session/night). Run this ONLY after a fully successful tick.
    It is the LLM's final explicit action, deliberately NOT a systemd `&& endtick`: if
    the tick is interrupted (usage limit, crash, hang) or any step failed, `endtick`
    never runs, the day stays unmarked, and the next tick RESUMES from committed state.
    The failure direction is benign (re-run), never "skip needed work".

**Checkpoint as you go** using `python -m loop.checkpoint --message <msg>` — this
stages `experiments/` and commits (never pushes). Use it after each expensive or
irreversible step (tier-2 labels after step 3; state.json after `loop.metrics` in
step 4), not only at step 11. The loop has **no conversation memory** across ticks:
committed git state + `state.json` ARE the resume point. Budget exhausted or
interrupted mid-run → the next tick reloads committed state and continues, never
repeating tier-2 adjudication already in `gold/` or re-ingesting below the stored
watermark (which `loop.metrics` now persists automatically).

## Change levers (env first, code allowed with cause)
- **Env-var delta via `loop.deploy` is the default lever** (bounded, rollback =
  restore `best_known_good`). Prefer it whenever a tunable parameter can plausibly
  achieve the goal.
- **Code changes ARE permitted** when no env knob reaches the root cause (e.g. a
  motion-detection algorithm fix). Record the justification in the active
  `runs/NNNN-<slug>.md`, keep it minimal/reversible (rollback = `git revert`), and
  **commit it separately with the experiment id in the message** (e.g.
  `fix(motion): exp #4 (mog2-recurrent-frames) — <what/why>`); note the SHA in the
  run file. A code change only takes effect on a **camera restart**, so stamp
  `pending_restart_at` (pre-sunrise window) just like an env deploy.
- All decision gates below apply equally to code and env changes.

## Guardrail contract (hard rules)
- BOUNDS in `src/loop/guardrails.py` are enforced by the system (config validators
  + deploy). Never propose out-of-range values.
- FN-veto: an FP win with an FN rise beyond CI is rejected; if FN is unmeasured and
  the change could raise FN, HOLD.
- Volume collapse/explosion vs baseline → rollback.
- Feedback-starved freeze: no human labels for 3 days → freeze, hold best_known_good.
- One active experiment at a time. Respect `state.json.paused`.

## Anti-self-poisoning & self-skepticism
- Ground truth is append-only; never rewrite `detection_feedback`, `gold/`, or prior
  `runs/` observations.
- Treat your own auto-labels with suspicion; reconcile against human labels in the
  self-audit step before trusting a "win."
