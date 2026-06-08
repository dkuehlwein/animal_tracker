# Design: Autonomous Detection-Tuning Loop (ADR-004 Phase 4)

**Date:** 2026-06-08
**Status:** Approved for planning
**Parent design:** `docs/ADR-004-autonomous-tuning-loop.md` (converged v0.7)
**Implements:** ADR-004 Phase 4 (autonomous loop) + a *stubbed* Phase 3 seam (Layer-A replay).

This spec turns ADR-004's converged design into an implementable scope. It does
not re-debate the ADR; it resolves the ADR's remaining open questions for Phase 4
and defines concrete components, interfaces, data flow, and build order. Read the
ADR first for the *why*; this doc is the *what to build*.

---

## Decisions locked for this build

These were the ADR's open questions; resolved here with the user:

1. **Scope:** Build the full autonomous loop now (not a report-only first slice).
   The known Phase-2 improvements are *seeded as the loop's initial experiment
   backlog* so it has real work from night one rather than waiting for data.
2. **Deploy mechanism (ADR OQ#1):** *Restart before sunrise.* The loop writes the
   chosen config; a deploy step `systemctl restart`s the camera service ~60 min
   before sunrise so MOG2 warmup (~5 min) finishes while it is still dark. No
   hot-reload machinery (matches today's "Config built once at startup" reality).
3. **Cadence (ADR OQ#3):** `/loop` fires **every 2h**, but each tick is *gated* to
   the sunset→sunrise window and to "tonight's run not yet complete." First tick of
   the night does the work; later ticks are cheap no-ops unless the first got stuck
   / usage-limited, in which case they resume from the committed notebook.
4. **Layer-A replay harness (Phase 3):** **STUBBED** for this build. `replay.py`
   exposes the real interface but returns a "skipped (not implemented)" result.
   Replay-gated experiments (e.g. `unknown_species_threshold`) are *parked* in the
   backlog until the stub is filled in. The loop runs motion/gate experiments
   live-only, validated by bounds + predicted-effect + FN-veto.
5. **Defaults:** loop runs on **Opus**; **daily** heartbeat (with the report);
   feedback-starved freeze at **N = 3 days**; deploy restart **~60 min before
   sunrise**.

Inherited unchanged from ADR-004 (do not re-open): Pro/Max subscription auth
(training opted out); execution via on-Pi Claude Code `/loop`; notebook committed
to `main`; event key = `detections.id`; OEC = paired FP+FN with CIs; FN-veto;
one active experiment at a time; garden images never enter the repo.

---

## Architecture overview

Two layers, deliberately separated so tokens are spent only on judgment:

- **Deterministic local Python** (`src/loop/`): unit-tested, token-free. Does all
  the heavy/cheap work — SQL reads, metric math, config writes, restarts, Telegram
  I/O. The agent *invokes* these as CLI entrypoints and reads their JSON output.
- **Judgment layer** (`experiments/loop.md` + `PROTOCOL.md`): the prompt `/loop`
  executes. Spends tokens only on tier-2 adjudication, experiment design,
  self-audit, and journaling. Reconstructs all state from the git notebook each
  tick (implicit resumption — no bespoke stage machine).

```
on-Pi `claude` session ──/loop (every 2h, night-gated)──┐
                                                         │ invokes
                                                         ▼
              src/loop/*.py  (deterministic CLIs, JSON in/out)
                 │ reads/writes              │ reads/writes
                 ▼                           ▼
        data/detections.db (WAL)      experiments/ (git notebook on main)
                                             │ renders
                                             ▼
                              experiments/deployed_config.env
                                             │ read at startup (overlay)
                                             ▼
                                wildlife_system.py (restarted pre-sunrise)
```

---

## Components

All new runtime code lives under `src/loop/`. Each module is single-purpose,
independently testable, and exposes a `python -m loop.<name>` CLI that prints a
JSON result to stdout (so the agent can consume it without parsing logs).

### `src/loop/ingest.py`
- Pull `detections` rows and `detection_feedback` rows newer than the last
  ingested `detections.id` (watermark stored in `state.json`).
- Reconcile labels per detection: **human > tier-2 > tier-1**. Tier-1 =
  `animals_detected` (MegaDetector). Tier-2 = cloud-vision adjudication (a field
  written by the judgment layer, not by this module). Latest human row wins.
- Output: list of `{detection_id, reconciled_label, tier1, tier2, human, motion
  features, hour_of_day, gate_would_suppress}`.
- Pure read; never writes labels (anti-self-poisoning: ground truth is append-only
  and owned by the feedback sidecar + gold/).

### `src/loop/metrics.py`
- Compute, over a chosen window, **paired FP and FN with Wilson 95% CIs**, total
  volume, breakdown by hour and by machine-labeled FP cause.
  - **FP** (from the captured/labeled set): triggers a human/tier-2 marked
    `false_positive` ÷ labeled triggers.
  - **FN** (from the timelapse audit channel): animals found in timelapse frames
    that never produced a detection ÷ animal-present timelapse frames. *Until the
    timelapse audit is wired with a detector pass, FN is reported as "unmeasured"
    rather than 0* — a zero would falsely clear the FN-veto.
- Append one row per day to `experiments/metrics/daily.csv` (idempotent: a re-run
  for the same date overwrites that date's row, never duplicates).

### `src/loop/replay.py`  *(STUB this build)*
- Interface: `replay(candidate_config, labeled_set) -> ReplayResult`.
- This build returns `ReplayResult(status="skipped", reason="not implemented")`.
- Documented as the Layer-A seam (re-run MegaDetector+classifier over saved
  high-res images with candidate thresholds, score vs labels). Filling it in is a
  later task; the rest of the system treats `status="skipped"` as "no offline
  evidence available → this experiment cannot be validated offline yet."

### `src/loop/deploy.py`
- Input: a candidate config delta (only keys the loop is allowed to tune).
- **Validate against bounds in code** (see `guardrails`/`config`): reject
  out-of-range → no deploy, log the rejection.
- Update `state.json`: set `deployed`, push previous onto history, keep
  `best_known_good`.
- **Render `experiments/deployed_config.env`** from `state.json.deployed`
  (e.g. `MOTION_THRESHOLD=2500`).
- **Schedule the restart** for ~60 min before next sunrise (compute via
  `SunChecker`): write a one-shot `systemd-run --on-calendar` / timestamp the
  desired restart in `state.json` and let the deploy step (run by a later tick, or
  a tiny pre-sunrise systemd timer) execute `systemctl restart
  wildlife-camera.service`. **Chosen mechanism:** a dedicated
  `wildlife-deploy.timer` + `wildlife-deploy.service` (oneshot) that, pre-sunrise,
  checks `state.json` for a pending deploy and restarts the camera if so. Keeps the
  restart off the `/loop` session's critical path.
- `rollback()`: restore `best_known_good` → re-render env → flag pending restart.

### `src/loop/report.py`
- Build the Telegram daily summary (FP **and** FN, active experiment status,
  decision-pending, top FP cause, asks-to-label) from `metrics/daily.csv` +
  `state.json`. Format mirrors ADR-004 §"Daily summary over Telegram."
- Send via the existing `NotificationService` (send-only path; no getUpdates
  conflict with the feedback sidecar).
- **Heartbeat:** the daily summary *is* the deadman ping; additionally emit a
  terse "loop alive, last tick OK @ <ts>" if a run completes without a full
  summary (e.g. a no-op resume tick). Silence ⇒ the human notices no daily report.
- **Veto commands:** `/pause` and `/rollback` are handled by the **feedback
  sidecar** (it already runs `telegram.ext.Application`); they write a flag to
  `state.json` (`paused: true`) / call `deploy.rollback()`. `report.py` only
  *sends*; the sidecar *receives* (one polling process, as in Phase 1).

### `src/loop/guardrails.py`
- **Bounds:** a single source-of-truth dict of allowed ranges for tunable params,
  consumed both by `config.py` field validators (load-time enforcement) and by
  `deploy.py` (pre-write rejection). Out-of-range is rejected by the *system*, not
  merely discouraged.
- **Fast guards:** detect capture-volume collapse (~0) or explosion vs the trailing
  baseline, or camera-loop crash signals → mark for immediate `rollback()`.
- **FN-veto:** given before/after metrics, reject any change whose FP improvement
  comes with an FN increase beyond CI noise. If FN is "unmeasured," a motion/gate
  change that could plausibly raise FN is **held** (not deployed) rather than
  guessed.
- **Feedback-starved freeze:** if no fresh human labels for **N=3 days**, freeze
  tuning and hold `best_known_good`.

### `config.py` changes
- Add an overlay env file: each tunable sub-config (`MotionConfig`, `SpeciesConfig`,
  `PerformanceConfig`) gets `env_file=('.env', 'experiments/deployed_config.env')`.
  pydantic-settings precedence: real OS env > last env_file (`deployed_config.env`)
  > first (`.env`) > defaults. So the loop's deployed params override `.env`, while
  a human OS env var still overrides everything (manual override preserved). A
  missing overlay file is ignored (fresh checkout / pre-first-deploy is safe).
- Add **bounds field validators** to `MotionConfig` (and any other tunable class),
  importing the ranges from `guardrails`. An out-of-range deployed value raises at
  `Config()` construction → the service refuses to start on a bad config, which is
  the desired hard guardrail.

### `experiments/` git notebook (committed to `main`)
```
experiments/
  PROTOCOL.md            # SOP a fresh /loop session reads FIRST
  loop.md                # the operational prompt /loop executes
  JOURNAL.md             # thin append-only chronological index linking run files
  LEARNINGS.md           # distilled firm conclusions (semantic memory)
  state.json             # deployed config, active experiment id, SEEDED BACKLOG,
                         #   baselines, bounds, best_known_good + history,
                         #   ingest watermark, paused flag, pending-deploy stamp
  runs/NNNN-<slug>.md    # per-experiment record: YAML front matter (structured,
                         #   greppable) + narrative body. See "Experiment records".
  metrics/daily.csv      # FP/FN/volume time series
  gold/                  # immutable human-verified labels + frozen eval set
  deployed_config.env    # rendered overlay (gitignored; regenerated from state.json)
```
- `deployed_config.env` is **gitignored** (single source of truth = `state.json`,
  which is committed). Everything else under `experiments/` is committed to `main`.
- **Seeded backlog** in `state.json` (the user's Phase-2 hypotheses), each tagged
  with how it can be validated this build:
  - `notification_gate_live` — flip the shadow gate to route no-animal triggers to
    a review channel. *Live-validated*, FN-veto-gated. (Highest value.)
  - `unknown_species_threshold 0.5→0.75` — *replay-gated → parked* until `replay.py`
    is real (label-only change, no offline scorer yet).
  - `roi_masking` — restrict motion ROI. *Live-validated*, incremental.

### Experiment records (`runs/NNNN-<slug>.md`)

One markdown file per experiment, **authored by the judgment layer** (not by the
deterministic Python — keeps markdown-parsing out of the tools). The YAML front
matter is the structured, greppable record ("did we try X?" = grep front matter);
the body is the narrative. This unifies the old `runs/NNNN.json` + the
per-experiment portion of `JOURNAL.md`.

Front matter fields:
```yaml
id: 7
slug: notification-gate-live
status: running          # proposed | running | concluded | rolled_back | parked
validation: live         # live | replay | parked
hypothesis: "Route no-animal triggers to a review channel; cuts FP w/o raising FN"
param_delta: { notification_gate: "shadow -> live" }
predicted_effect: { fp_rate: "-15pp", fn_risk: "low" }
created: 2026-06-10
started: 2026-06-11
concluded: null
decision: null           # keep | rollback | inconclusive
baseline: { fp_rate: 0.38, fp_ci: [0.30, 0.46], fn: unmeasured }
result:   { fp_rate: null, fp_ci: null, fn: null }
confidence: null
```
Body sections: `## Hypothesis & method`, `## Daily observations`, `## Decision &
rationale`. Append-only in spirit (results/decision filled in over the
experiment's life; prior observations never rewritten — anti-self-poisoning).

**Ownership boundary:**
- `runs/NNNN-<slug>.md` — agent writes (front matter + body).
- `state.json` — deterministic tools only; holds live pointers (`active_experiment_id`,
  deployed config, backlog, watermark, `best_known_good` + history). References
  experiments by `id`; never duplicates their record.
- `JOURNAL.md` — thin chronological index linking run files; cross-experiment notes.
- `metrics/daily.csv` — deterministic tools only.

The seeded backlog entries in `state.json` are *proposed* experiments (id +
slug + validation tag); when the loop starts one it creates the matching run file.

### `loop.md` + `PROTOCOL.md`
- `PROTOCOL.md`: the SOP. How to resume from the notebook, the daily cycle steps,
  the anti-self-poisoning + self-skepticism rules, the guardrail contract, when to
  spend tokens vs invoke Python.
- `loop.md`: the concrete per-tick prompt: check night-gate + "tonight done?"; if
  work needed, run the deterministic CLIs in order, do the judgment steps, write
  the notebook, commit + push, send the report.

### systemd units (committed, same pattern as existing services)
- `wildlife-loop.service` — keeps the on-Pi `claude --continue`/`/loop` session
  alive; restarts it across reboots and re-arms the 7-day `/loop` expiry.
- `wildlife-deploy.timer` + `wildlife-deploy.service` — pre-sunrise oneshot that
  applies a pending deploy (restart the camera) off the loop's critical path.
- README documents installing both (as with `wildlife-camera`/`wildlife-feedback`).

---

## Daily cycle (one nightly run, resumable)

```
1. Gate    — SunChecker says night? tonight's run already in state.json? else stop.
2. Ingest  — loop.ingest: new detections + feedback past the watermark; reconcile.
3. Label   — judgment: adjudicate ambiguous crops (tier-2), update gold/ appends.
4. Measure — loop.metrics: paired FP/FN + CIs → daily.csv.
5. Check   — does the active experiment's prediction still hold (CI-based)? done?
6. Self-audit (cadence) — compare auto-labels vs the day's human labels; re-check
             past "wins" on the larger corpus; note confidence in runs/.
7. Decide  — keep / rollback; if concluded, pick next from seeded backlog / OFAT
             within bounds. Respect freeze + one-experiment-at-a-time + paused.
8. Validate— Layer A = loop.replay (STUB → "skipped"). Layer B = bounds + predicted
             live effect. FN-veto: reject FP wins that worsen (or risk, if FN
             unmeasured) FN.
9. Deploy  — loop.deploy: write state.json + render env + stamp pending restart for
             pre-sunrise (wildlife-deploy.timer executes it).
10. Record — update runs/NNNN-<slug>.md (front matter + observations), append the
             JOURNAL.md index line, update state.json pointers.
11. Report — loop.report: Telegram daily summary (FP+FN) + heartbeat; commit + push.
```
Budget exhausted mid-run → next 2h tick resumes from committed state.

---

## Error handling

- Every `src/loop` CLI exits non-zero with a JSON `{error: ...}` on failure; the
  agent treats a failed step as "do not proceed to deploy this tick."
- `deploy.py` is the only writer of live config; it is atomic (write temp →
  rename) and always leaves `best_known_good` intact for rollback.
- Bad deployed config ⇒ `Config()` raises ⇒ service won't start ⇒
  `wildlife-deploy.service` detects the failed restart and rolls back to
  `best_known_good`, then alerts via Telegram.
- SQLite access is WAL read-only from the loop except the feedback sidecar's
  appends; no schema writes from the loop.
- Anti-self-poisoning: the loop never UPDATEs/DELETEs `detection_feedback` or
  `gold/`; only appends/derives.

---

## Testing strategy

TDD per module, all against seeded SQLite fixtures + temp dirs (reuse
`tests/conftest.py` isolation):
- `test_loop_ingest.py`: watermark advances; reconciliation precedence
  (human>tier2>tier1); latest human row wins; no writes to feedback.
- `test_loop_metrics.py`: Wilson CI math on known counts; FN reported "unmeasured"
  when no audit data; daily.csv idempotent per date.
- `test_loop_replay.py`: stub returns `status="skipped"`; interface shape stable.
- `test_loop_deploy.py`: out-of-bounds rejected (no write); state.json history +
  best_known_good maintained; env render matches deployed; rollback restores.
- `test_loop_report.py`: summary renders FP+FN + asks from fixtures; heartbeat
  text; `paused` suppresses tuning lines.
- `test_loop_guardrails.py`: bounds dict; volume collapse/explode detection;
  FN-veto rejects FP-win/FN-rise; freeze after N=3 days no labels.
- `test_config.py` additions: overlay precedence (env > deployed_config.env >
  .env > default); out-of-range deployed value raises at construction; missing
  overlay file is safe.
- Existing suite stays green.

---

## Out of scope (later)

- Real Layer-A replay harness + RDE (fills the `replay.py` stub) — and with it the
  `unknown_species_threshold` experiment.
- Bayesian optimization (coarse-grid/OFAT only at this data volume).
- Motion-layer *offline* eval (ADR Phase 5).
- Tier-2 cloud-vision adjudication automation beyond the judgment-layer prompt
  (the plumbing field exists in `ingest`; the agent fills it).
- USB SSD migration, MegaDetector V6 upgrade, seasonal re-baselining.

---

## Build order (for the implementation plan)

1. `guardrails` bounds dict + `config.py` overlay env + bounds validators (+ tests).
2. `ingest` + `metrics` (seeded DB fixtures) (+ tests).
3. `replay` stub (+ test).
4. `report` + heartbeat (+ tests).
5. `deploy` + rollback + `state.json` schema (+ tests).
6. `experiments/` scaffold: `state.json` (with seeded backlog), `PROTOCOL.md`,
   `loop.md`, `JOURNAL.md`, `LEARNINGS.md`, `gold/`, `metrics/`, `runs/` (with one
   example `NNNN-<slug>.md` documenting the front-matter schema), gitignore for
   `deployed_config.env`.
7. systemd: `wildlife-loop.service`, `wildlife-deploy.timer`/`.service`; README +
   feedback-sidecar `/pause` `/rollback` handling.
8. End-to-end dry run on the Pi (loop runs, reports, deploys to a no-op/bounded
   change, rolls back) before arming nightly.
