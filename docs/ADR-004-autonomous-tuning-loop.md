# ADR-004: Autonomous Detection-Tuning Loop

**Status**: Proposed (design doc / v0.1)
**Date**: 2026-06-07
**Decision Makers**: Daniel Kuehlwein + Claude
**Technical Story**: Stand up a self-improving daily loop that reviews each day's
detections, runs experiments, and tunes the motion-detection configuration to
reduce false positives (first) and false negatives (later) — with a human
feedback channel and a daily summary over Telegram.
**Related**: ADR-002 (Two-Stage Detection Pipeline), ADR-003 (Multi-Frame Burst)

---

## Context and Problem Statement

After months of operation the system still produces too many **false positives**
(captures with no animal — wind-blown vegetation, shadows, lighting changes) and
**false negatives** (animals that never trigger a capture). Tuning is currently
manual guesswork because:

1. **There is no ground truth.** The `detections` table stores `species_name`,
   `confidence_score`, and `motion_area`, but nothing that records whether a
   capture was *actually* an animal. Without labels, no objective metric exists.
2. **MegaDetector is not trustworthy as the arbiter.** Months of observation show
   it makes meaningful mistakes (misses small/distant animals, occasional false
   detections). Optimizing toward MegaDetector would teach the loop to reproduce
   MegaDetector's errors.
3. **False positives and false negatives need different data.** FPs are reviewable
   from images we already have. FNs are invisible — the animal never triggered a
   capture, so no image exists to review.
4. **Experiments aren't evaluable live.** You can't A/B test parameters on a live
   garden; the wildlife won't repeat on command, and the environment drifts daily
   (weather, season, light), confounding any naive before/after comparison.
5. **The agent has no memory.** The intended operator of this loop is an LLM agent
   that runs as ephemeral sessions with no persistent state of its own.

### The meta-goal

This is explicitly an experiment in **autonomous problem-solving**: can an LLM
agent run a coherent, improving, multi-day research program against a real
physical system? The hard part is therefore not the tuning math — it is
**durable memory, honest evaluation, and safe autonomy**.

---

## Decision Drivers

1. **Ground truth must come from the human**, captured with near-zero effort.
2. **Privacy**: garden images must not leave the local network by default and
   must never be committed to git (even a private repo) — thousands of images
   accumulate and the repo is the wrong store.
3. **Self-contained data flow**: the cloud cannot reach the Pi on the home LAN;
   the design must not depend on inbound access to the home network.
4. **Honest evaluation**: changes must be validated against a fixed labeled set
   before deployment, not judged by confounded day-to-day live comparisons.
5. **Safe autonomy**: bounded parameters, auto-rollback, one-change-at-a-time, and
   a human veto.
6. **Durable, self-documenting memory**: a fresh agent session must be able to
   resume the research program from the repo alone.
7. **Human-facing reporting**: a daily experiment summary delivered over Telegram.

---

## Decision Outcome

Build an **on-device autonomous tuning loop** that runs nightly on the Raspberry
Pi (host-agnostic; can also run on the desktop with SSH access to the Pi). It
reviews the day's detections, reconciles human + machine labels, designs and
evaluates a parameter experiment offline against a labeled replay corpus, deploys
the winning change within guardrails, records everything to a git-backed lab
notebook, and sends a daily summary to Telegram.

### Where it runs — host-agnostic, default on the Pi

The loop is a lightweight local process plus the Claude Code CLI. The heavy
reasoning is a cloud API call; **the model does not run on the Pi.** Local work is
limited to SQLite queries, pandas-style metrics, and replaying the OpenCV motion
detector over saved frames.

- **Pi 5 (8GB) feasibility**: confirmed adequate. The only CPU-bound part is
  replay sweeps over saved frames, which run overnight while the camera is idle
  (the Pi already runs SpeciesNet at ~17s/image). RAM is not a constraint.
- **Desktop alternative**: faster replay sweeps and keeps the API key off an
  always-on box; requires the desktop to be on and SSH to the Pi.
- The implementation reads a local data directory + SQLite DB and needs only
  **outbound** internet (Anthropic API + Telegram). No inbound access required.

### Privacy boundary (hard requirement)

| Data | Leaves the Pi? | Where it lives |
|------|----------------|----------------|
| Garden images / burst frames | **No** (except existing Telegram notifications) | Local disk only |
| Replay corpus (raw motion frames) | **No** | Local disk only |
| Detection metadata, labels, metrics | Yes (non-image) | git repo (lab notebook) |
| Experiment journal / config / state | Yes (non-image) | git repo |
| Daily summary | Yes (text, maybe 1-2 example thumbnails *if* explicitly enabled) | Telegram |

**Cloud-vision relabeling (sending images to the Anthropic API) is OFF by
default.** The autonomous decisions operate over **metadata + human labels**.
Heavy vision stays local (MegaDetector) plus the human's own eyes. Cloud vision
is an opt-in tier for occasional ambiguous-case adjudication only.

### Three-tier labeling (reconciled, human wins)

| Tier | Labeler | Cadence | Trust | Role |
|------|---------|---------|-------|------|
| 1 | MegaDetector (on Pi, live) | every capture | weak/noisy | a *feature*, cheap triage — never truth |
| 2 | Stronger local model (optional) / cloud vision (opt-in) | daily batch | good | pre-fill + adjudicate ambiguous cases |
| 3 | **Human via Telegram buttons** | whenever you glance | **gold** | overrides everything |

Reconciliation rule: human label > tier-2 > tier-1. Disagreements between tiers
are themselves logged — they reveal where MegaDetector fails (dawn light, rain,
distance) and feed active-learning sample selection.

### Human feedback over Telegram

Each notification gains an inline-keyboard row:

```
[ ✅ Animal ]  [ ❌ False positive ]  [ 🐦 Wrong species ]
```

A tap fires a callback handled by the Pi's bot (which already holds the bot
token) and writes a row to a new `detection_feedback` table. Zero extra effort —
you already review these notifications. This is the keystone: nothing downstream
is trustworthy without real labels.

**Active learning**: the loop prioritizes *which* unlabeled detections to surface
for your review — those nearest the current decision boundary (motion_area close
to threshold, low-confidence MegaDetector) yield the most information per tap.

False-negative feedback (later phase): a daily low-FPS timelapse digest where you
can tap any animal the detector missed.

### Daily summary over Telegram

Once per day the loop sends a concise report:

```
🦊 Daily Tuning Report — 2026-06-07
Detections: 47  |  Labeled by you: 12  |  FP rate (labeled): 38% → target <15%
Active experiment #007: threshold 2000→2500 (day 2/4)
  Offline replay: would keep 9/10 animal events, drop 18/26 FPs ✅
  Live so far: FP rate 38%→24%, 0 confirmed missed animals
Decision pending: 2 more days of data before keep/rollback.
Top FP cause (machine-labeled): wind-blown vegetation (61%)
Asks: please label the 5 pinned borderline captures 🙏
```

The summary is generated from the lab notebook + metrics, so it doubles as a
human-readable view of the agent's memory.

### Evaluation — offline replay corpus is the core

To make experiments *evaluable* despite a drifting environment:

1. **Record raw motion frames locally** — not just triggered captures. Continuous
   full-rate storage is too large (~15-25 GB/day at 640×480 5fps), so capture:
   - **buffered clips around every trigger** (already have these frames), and
   - **near-miss clips** when `motion_area` lands in a shoulder band
     (e.g. 50-100% of threshold) — the events that *almost* fired, and
   - a **low-FPS continuous timelapse** for false-negative audits.
2. **Label that corpus** (human + reconciled machine labels) into animal-events
   vs false-positive-events vs near-misses.
3. **Replay deterministically**: re-run `MotionDetector` over the corpus with any
   candidate config and measure exactly how many real events it would keep and
   how many FPs it would drop. This removes environmental confounding — the corpus
   is fixed.
4. **Live data is confirmation, not proof.** Deploy only changes that win offline;
   then watch live metrics as a sanity check.

> ⚠️ **Selection-bias caveat**: the corpus only contains events the *current*
> config triggered (plus near-misses). It under-represents true negatives and
> hard false-negatives. Near-miss + timelapse capture partially corrects this;
> the loop must treat offline FN estimates as lower bounds.

### Autonomous loop — daily cycle

```
NIGHTLY (camera idle):
  1. Ingest   — pull new detections + human feedback from SQLite.
  2. Label    — reconcile tier1/2/3; update corpus labels.
  3. Measure  — compute FP rate, FN proxy, volume, by-hour, by-cause; append
                to metrics/daily.csv.
  4. Check    — did the active experiment's prediction hold? Enough samples yet?
  5. Decide   — keep / rollback the active experiment; if concluded, design the
                next one (OFAT: one parameter at a time, within bounds).
  6. Validate — replay the candidate config over the labeled corpus. Abort if it
                doesn't beat the current config offline.
  7. Deploy   — write new config; signal the Pi service to reload (canary).
  8. Record   — append to JOURNAL.md, write runs/NNNN.json, update state.json.
  9. Report   — send the Telegram daily summary. Commit + push the notebook.
```

### Durable memory — git-backed lab notebook (non-image only)

```
experiments/
  PROTOCOL.md       # the SOP. A fresh agent session reads this FIRST.
  JOURNAL.md        # append-only narrative ("Day 12: FP 31%, tried X because…")
  state.json        # deployed config, active experiment, baselines, guard limits
  runs/NNNN.json    # per-experiment: hypothesis, config delta, predicted effect,
                    #   replay result, live result, decision, confidence
  metrics/daily.csv # FP rate, FN proxy, volume, by-hour time series
```

`PROTOCOL.md` is what makes stateless autonomy coherent: it encodes the rules
below so any fresh session resumes the program correctly. Garden images are
**never** part of this tree.

### Safety / guardrails for full autonomy

- **Bounded parameters** — each tunable has a hard min/max in `state.json`.
- **One change at a time (OFAT)** — attributable effects; commit for N days or
  until a minimum labeled-sample count before judging.
- **Offline gate** — never deploy a change that didn't beat current config on the
  replay corpus.
- **Guard metrics + auto-rollback** — revert immediately if capture volume
  collapses to ~0 or explodes, or the FN proxy spikes.
- **Canary + confirmation window** before locking a change in.
- **Human veto** — a Telegram command (e.g. `/pause`, `/rollback`) halts or
  reverts the loop at any time, even in autonomous mode.
- **No thrashing** — at most one active experiment; no daily flip-flopping.

---

## Required Changes (high level)

1. **DB**: new `detection_feedback` table; persist `animals_detected`,
   MegaDetector box count/confidence, and richer motion metadata (contour count,
   largest contour, foreground pixels) that `motion_detector.py` already computes
   but currently discards.
2. **Telegram**: inline feedback buttons + callback handler (Pi-side); daily
   summary sender; veto commands.
3. **Capture**: near-miss frame logging + low-FPS timelapse writer (local disk,
   size-bounded with rotation).
4. **Replay harness**: offline runner that executes `MotionDetector` over a
   labeled corpus and reports keep/drop metrics for a given config.
5. **Loop runner**: the nightly orchestrator + git-backed notebook + PROTOCOL.md.
6. **Config**: surface new tunables + bounds; the loop edits config via the
   existing env-var / config-file mechanism and triggers a service reload.

---

## Phased Roadmap

- **Phase 0 — Design (this doc + PROTOCOL.md).**  ← current
- **Phase 1 — Ground truth**: Telegram feedback buttons, `detection_feedback`
  table, richer logging. *Nothing downstream is trustworthy without this.*
- **Phase 2 — Corpus capture**: near-miss + timelapse local recording (bounded).
- **Phase 3 — Replay harness**: offline parameter evaluation.
- **Phase 4 — Autonomous loop**: nightly runner, git notebook, guardrails,
  daily Telegram summary. FP-focused first.
- **Phase 5 — False negatives**: timelapse review feedback, FN metrics, extend
  tuning objective.

---

## Consequences

### Positive
- Objective, human-grounded metrics replace guesswork.
- Experiments become cheap and confound-resistant via offline replay.
- Images never leave the Pi; memory (non-image) is durable and self-documenting.
- A real test of autonomous, multi-day agentic problem-solving.

### Negative / Risks
- Human labels are sparse → slow convergence; mitigated by active-learning sample
  selection and the daily "please label these" ask.
- Corpus selection bias under-represents false negatives until Phase 5.
- Autonomy risk (a bad config degrading capture) mitigated by offline gate,
  guard metrics, auto-rollback, and human veto.
- On-Pi replay is CPU-bound; mitigated by nightly batch + optional desktop host.

### Neutral
- The loop edits config through the existing override mechanism; no new config
  engine needed.

---

## Open Questions

1. **Loop host**: Pi nightly vs desktop-with-SSH — start on Pi, revisit if replay
   sweeps are too slow.
2. **Tier-2 labeler**: which (if any) local model beyond MegaDetector, and whether
   to ever enable opt-in cloud vision for adjudication.
3. **Corpus retention**: how many days of near-miss/timelapse frames to keep, and
   rotation policy, given SD-card wear and capacity.
4. **Git data branch**: keep the notebook in the main repo on a docs/experiments
   path, or a separate `experiments` branch/repo.
5. **Scheduling/auth**: how the Claude Code CLI is invoked nightly on the Pi
   (cron/systemd timer) and how its API key is stored securely.

---

## References

- ADR-002 (Two-Stage Detection Pipeline), ADR-003 (Multi-Frame Burst Capture)
- [MegaDetector](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md)
- [SpeciesNet](https://github.com/google/cameratrapai)
- [Camera Trap ML Survey — Dan Morris](https://agentmorris.github.io/camera-trap-ml-survey/)
- *Best-practices research synthesis to be folded in (in progress).*

---

**Document Version**: 0.1
**Last Updated**: 2026-06-07
**Next Review**: After best-practices research lands and Phase 1 scoping.
