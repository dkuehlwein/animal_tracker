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

**Auth = Pro/Max subscription (decided).** This has two consequences baked into
the design:
- *Session/token limits*: the subscription enforces ~5-6h rolling reset windows,
  so the loop must be **checkpointed and resumable** (see "Resumable execution"),
  not a single long-running session. A systemd timer fires a fresh CLI session at
  sunset and again at reset boundaries (~2-3× per night); each session drains the
  next pending stages until its budget runs out, commits, and exits.
- *Data terms*: under consumer subscription terms, conversations **may be used for
  training unless you opt out**. Since image crops are sent for adjudication, the
  account must have **"help improve Claude / model training" turned OFF**
  (Settings → Privacy) as a prerequisite. (An API key under commercial terms +
  ZDR would avoid both the reset windows and the training-data exposure, but was
  not chosen.) Policies change — verify against Anthropic's current data-usage
  docs before relying on this.

### Resumable execution (subscription-aware)

Most of the loop spends **zero Claude tokens** — ingest, metric computation,
MegaDetector relabel, replay sweeps, deploy, and rollback checks are deterministic
local Python that run unconditionally on the systemd timer. Claude tokens are only
spent on the *judgment* layer: image adjudication, designing the next experiment,
and writing the journal narrative.

Tonight's run is modeled as an **idempotent stage machine** in `state.json`
(`ingested → labeled → measured → adjudicated → decided → validated → deployed →
reported`). Each stage marks itself done as it completes; a restarted session
reads `PROTOCOL.md` + `state.json` and resumes at the first incomplete stage, so a
session dying mid-run (token exhaustion) loses no work.

### Privacy boundary (hard requirement)

| Data | Leaves the Pi? | Where it lives |
|------|----------------|----------------|
| Garden images / burst frames | **No** (except existing Telegram notifications) | Local disk only |
| Replay corpus (raw motion frames) | **No** | Local disk only |
| Detection metadata, labels, metrics | Yes (non-image) | git repo (lab notebook) |
| Experiment journal / config / state | Yes (non-image) | git repo |
| **Image crops for adjudication** | Yes (only ambiguous cases, MegaDetector crops not full frames) | Anthropic API |
| Daily summary | Yes (text, maybe 1-2 example thumbnails *if* explicitly enabled) | Telegram |

**Cloud-vision adjudication is enabled but used surgically.** It was initially
scoped off-by-default, but human taps are themselves noisy and the loop needs a
competent set of eyes to resolve tier disagreements — so it is a real Tier-2 step
(see below). Exposure is bounded by design: send **MegaDetector crops, not full
frames**, and **only on ambiguous/low-agreement cases** the active-learning step
surfaces — not every capture. The bulk of autonomous decisions still run over
**metadata + labels**; vision is the tie-breaker, not the workhorse.

### Three-tier labeling (reconciled, human wins)

| Tier | Labeler | Cadence | Trust | Role |
|------|---------|---------|-------|------|
| 1 | MegaDetector (on Pi, live) | every capture | weak/noisy | a *feature*, cheap triage — never truth |
| 2 | Cloud vision (crops, ambiguous cases) | daily batch | good | adjudicate tier disagreements + pre-fill |
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

### Evaluation — three layers that defeat the two big traps

Research flagged two failure modes that naive replay/eval would walk straight
into: **selection bias** (you only have frames the current config triggered on, so
optimizing against them just reproduces the current config and grows blind spots)
and **environmental confounding** (one camera, so before/after comparisons credit
weather/season changes to your config change). The design counters both:

**(a) An independent random-capture corpus — the FN ground truth.** Capture frames
on a **random schedule decoupled from motion triggering** (plus near-miss frames
in the 50-100% shoulder band, plus a low-FPS timelapse). This is the *only* way to
estimate false negatives and to recover the true scene distribution. It is held
**strictly separate** from the active-learning queue and is used **only for
evaluation** — actively-sampled (boundary-hugging) frames are unrepresentative and
must never be the eval set.

**(b) Offline replay before any deploy.** Re-run `MotionDetector` over the labeled
corpus with the candidate config and estimate keep/drop on real events vs FPs —
*before* it touches the live camera. Because the current trigger is a hard,
deterministic threshold, valid counterfactual replay requires injecting a little
**stochastic triggering near the boundary** (so logged decisions have non-degenerate
propensities); use self-normalized IPS / doubly-robust estimators to reweight
old-config logs for the new config.

**(c) Interleaving instead of before/after — the confound killer.** Rather than
"deploy config B for N days then compare to A," **randomly assign each motion event
(or short time-block) to config A or B** so both experience identical weather/light.
This converts a confounded comparison into a clean randomized one on a single
device, and is the primary live-attribution method. Difference-in-differences
against an environment control (e.g. SpeciesNet-confirmed event counts) is the
fallback when interleaving isn't possible.

**Optimize an OEC, not one metric.** The objective is an **F-beta over
human-verified events** (precision *and* recall), guarded so the agent can't "win"
by triggering never or always. Single noisy daily deltas are weak evidence —
require replication, and watch for novelty/transient effects that decay.

> ⚠️ Offline FN estimates from the triggered corpus alone are **lower bounds**; the
> random-capture corpus (a) is what makes them trustworthy.

### Autonomous loop — daily cycle

```
NIGHTLY (camera idle):
  1. Ingest   — pull new detections + human feedback from SQLite.
  2. Label    — reconcile tier1/2/3; update corpus labels.
  3. Measure  — compute FP rate, FN proxy, volume, by-hour, by-cause; append
                to metrics/daily.csv.
  4. Check    — did the active experiment's prediction hold (interleaved A/B
                result)? Enough verified-event replication yet?
  5. Decide   — keep / rollback the active experiment; if concluded, propose the
                next config via Bayesian optimization (noise-aware surrogate over
                past results, within bounds). OFAT only for a quick single-knob
                sanity check.
  6. Validate — replay the candidate config over the random-capture corpus (OEC =
                F-beta). Abort if it doesn't beat the current config offline.
  7. Deploy   — push config as an interleaved A/B (per-event A-vs-B assignment),
                not a wholesale switch; service reloads.
  8. Record   — append to JOURNAL.md, write runs/NNNN.json, update state.json.
  9. Report   — send the Telegram daily summary. Commit + push the notebook.
```

### Durable memory — git-backed lab notebook (non-image only)

```
experiments/
  PROTOCOL.md       # the SOP. A fresh agent session reads this FIRST.
  JOURNAL.md        # append-only narrative ("Day 12: FP 31%, tried X because…")
  LEARNINGS.md      # distilled semantic memory: durable, firm conclusions only
  state.json        # deployed config, active experiment, baselines, guard limits,
                    #   best-known-good config + full history (O(1) "tried this?")
  runs/NNNN.json    # per-experiment: hypothesis, config delta, predicted effect,
                    #   replay result, live result, decision, confidence
  metrics/daily.csv # FP rate, FN proxy, volume, by-hour time series
  gold/             # human-verified labels + frozen eval set — IMMUTABLE
```

Two memory tiers (per agent-memory research): **episodic** (per-day raw
observations in `JOURNAL.md`/`runs/`) and **semantic** (firm, distilled
conclusions consolidated into `LEARNINGS.md` once an experiment concludes — keeps
session context small). `PROTOCOL.md` encodes the SOP so any fresh session resumes
the program correctly.

**Anti-self-poisoning rule**: human labels and the gold eval set are
**append-only and immutable** — the agent may never overwrite ground truth or its
own past results, only append. Only *interpretations* evolve. Garden images are
**never** in this tree.

### Safety / guardrails for full autonomy

- **Bounded parameters** — each tunable has a hard min/max in `state.json`.
- **OEC + guardrails** — optimize F-beta over verified events; guardrail metrics
  that *can't go backward*: false-negative rate must not rise, capture volume must
  not collapse-to-~0 or explode, the camera loop must not crash.
- **Offline gate** — never deploy a change that didn't beat current config on the
  random-capture corpus.
- **Auto-rollback as a paved path** — keep the previous known-good config and
  revert atomically on a *sustained* guardrail breach (not a single bad hour).
- **Interleaved canary** — new config runs as a per-event A/B against the
  incumbent, so a bad config only affects its share of events while it's measured.
- **Replication before commit** — single noisy daily deltas aren't enough; require
  repeated agreement (and constrain the BO surrogate's hyperparameters so it
  doesn't overfit noise at low sample counts).
- **Human veto** — a Telegram command (e.g. `/pause`, `/rollback`) halts or
  reverts the loop at any time, even in autonomous mode.
- **No thrashing** — at most one active experiment at a time.

---

## Research Findings (2025-2026) folded into this design

Two best-practices reviews (camera-trap FP/FN reduction; autonomous experimentation
& agent memory) informed the design. Highlights and how they land here:

### Camera-trap / motion detection
- **Motion detection cannot classify cause** — switching frame-diff↔MOG2 does *not*
  fix vegetation/wind/shadow FPs (one deployment confirmed the algorithm choice was
  irrelevant). The fix is **ROI masking + a downstream AI filter**, which this
  system already does. *Lean into MegaDetector-as-filter; don't chase a magic
  motion algorithm.*
- **Timelapse plays double duty**: (1) FN safety net for slow/cold/edge animals
  that never trip motion; (2) FP filter — compare a trigger against a recent
  timelapse reference frame and discard if not meaningfully different. Both fold
  into the random-capture/timelapse channel.
- **Repeat Detection Elimination (RDE)**: an offline batch pass that flags
  detections recurring in nearly the same bounding box across many images (a branch,
  a sunlit rock) and removes them — very effective for static-camera stationary FPs.
  Added as a nightly offline step.
- **MOG2 tuning levers** (validate on-site): keep `history`≥500 (✓ current);
  consider raising `varThreshold` (current `background_threshold`=40) only if
  vegetation FPs persist; **beware high `learningRate`** — it absorbs slow animals
  into the background, a direct FN source; add a **directional-consistency / minimum
  displacement** check to reject back-and-forth wind jitter.
- **Thresholds (flagged against current config)**: keep MegaDetector recall-high
  at ~0.2 as a blank filter (✓ current `min_detection_confidence`=0.2); but
  `unknown_species_threshold`=0.5 is **likely too low** — field guidance for
  species labels is ~0.75-0.80 to suppress mis-IDs. *Candidate early experiment.*
  Never reuse thresholds across detector/classifier versions.
- **MegaDetector V6-compact (~2.3M params) runs on a Pi 4B/5** — a future option to
  run a *better* on-device FP filter than the v5-era bundled detector. Noted; the
  user already distrusts MD as truth, so it stays a feature regardless.
- **Reference projects**: AddaxAI/EcoAssist (filter blanks → verify ~10%),
  PyTorch-Wildlife (MDv6+SpeciesNet), `ratsakatika/camera-traps` (close analog: 4G
  + Telegram alerts), Timelapse (reference-frame technique).

### Autonomy / methodology (already reflected above)
- Selection-bias death spiral → independent **random-capture eval corpus**.
- Daily before/after is confounded → **interleave configs per event**.
- Deterministic threshold breaks counterfactual replay → **stochastic boundary
  triggering** + self-normalized IPS / doubly-robust.
- **Bayesian optimization** (noise-aware, low-sample-constrained) as the search
  engine; OFAT only for quick sanity checks.
- **OEC (F-beta)** + immutable guardrails so the agent can't game a single metric.
- **Episodic→semantic memory** with append-only, immutable gold labels (avoid
  memory self-poisoning).

---

## Required Changes (high level)

1. **DB**: new `detection_feedback` table; persist `animals_detected`,
   MegaDetector box count/confidence, and richer motion metadata (contour count,
   largest contour, foreground pixels) that `motion_detector.py` already computes
   but currently discards.
2. **Telegram**: inline feedback buttons + callback handler (Pi-side); daily
   summary sender; veto commands.
3. **Capture**: near-miss frame logging + **random-schedule capture** + low-FPS
   timelapse writer (local disk, size-bounded with rotation); optional **stochastic
   boundary triggering** for valid counterfactual replay.
4. **Replay harness**: offline runner that executes `MotionDetector` over the
   random-capture corpus and reports OEC (F-beta) keep/drop metrics for a config;
   plus a nightly **RDE** pass to strip recurring stationary FPs.
5. **Interleaving**: per-event A/B config assignment in the live loop.
6. **Loop runner**: the nightly orchestrator + git-backed notebook + PROTOCOL.md +
   resumable stage machine.
7. **Config**: surface new tunables + bounds; the loop edits config via the
   existing env-var / config-file mechanism and triggers a service reload.

---

## Phased Roadmap

- **Phase 0 — Design (this doc + PROTOCOL.md).**  ← current
- **Phase 1 — Ground truth**: Telegram feedback buttons, `detection_feedback`
  table, richer logging. *Nothing downstream is trustworthy without this.*
- **Phase 2 — Corpus capture**: random-schedule + near-miss + timelapse local
  recording (bounded), the independent eval corpus.
- **Phase 3 — Replay harness**: offline parameter evaluation (OEC) + RDE pass.
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
   sweeps are too slow. *(Auth resolved: Pro/Max subscription, training opt-out.)*
2. **Corpus retention**: how many days of random/near-miss/timelapse frames to
   keep, and rotation policy, given SD-card wear and capacity.
3. **Git data branch**: keep the notebook in the main repo on a docs/experiments
   path, or a separate `experiments` branch/repo.
4. **Scheduling cadence**: exact systemd-timer firing times after sunset and how
   many restarts per night map to the subscription reset windows.
5. **MegaDetector upgrade**: adopt V6-compact on-device as a better FP-filter
   feature, or keep the bundled v5-era detector?
6. **Stochastic-trigger budget**: how much boundary randomization is acceptable
   (it briefly trades a few live FPs/FNs for valid offline evaluation).

---

## References

**Project**
- ADR-002 (Two-Stage Detection Pipeline), ADR-003 (Multi-Frame Burst Capture)

**Camera-trap / detection**
- [MegaDetector](https://github.com/microsoft/MegaDetector) ·
  [RDE postprocessing](https://github.com/agentmorris/MegaDetector/tree/main/megadetector/postprocessing/repeat_detection_elimination)
- [SpeciesNet / cameratrapai](https://github.com/google/cameratrapai)
- [Camera Trap ML Survey — Dan Morris](https://agentmorris.github.io/camera-trap-ml-survey/)
- [AddaxAI / EcoAssist](https://github.com/PetervanLunteren/AddaxAI) ·
  [PyTorch-Wildlife](https://github.com/microsoft/CameraTraps) ·
  [ratsakatika/camera-traps](https://github.com/ratsakatika/camera-traps)
- [OpenCV MOG2](https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html) ·
  smart camera-trap / PIR FN study [PMC9185543](https://pmc.ncbi.nlm.nih.gov/articles/PMC9185543/),
  [PMC4623860](https://pmc.ncbi.nlm.nih.gov/articles/PMC4623860/) ·
  WiseEye confirmatory sensing [PMC5226779](https://pmc.ncbi.nlm.nih.gov/articles/PMC5226779/)

**Autonomy / methodology**
- Active learning & uncertainty sampling [Encord](https://encord.com/blog/active-learning-machine-learning-guide/),
  [PLOS One 2025](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327694)
- Selection-bias / feedback loops [CEUR](https://ceur-ws.org/Vol-3442/paper-18.pdf),
  [sampling bias in AL, arXiv:2109.06321](https://arxiv.org/pdf/2109.06321)
- Offline policy eval / counterfactual [eugeneyan](https://eugeneyan.com/writing/counterfactual-evaluation/),
  [IPS self-normalized, arXiv:2509.00333](https://arxiv.org/abs/2509.00333)
- Confounding / DiD [everydaycausal](https://www.everydaycausal.com/twfe-did.html);
  guardrails & OEC [Statsig](https://statsig.com/blog/what-are-guardrail-metrics-in-ab-tests)
- Few-Shot HITL Refinement [Nature Sci Reports 2025](https://www.nature.com/articles/s41598-025-87046-z)
- Agent memory / lab-notebook [Memory for Autonomous LLM Agents, arXiv:2603.07670](https://arxiv.org/html/2603.07670v1),
  governed memory / self-poisoning [arXiv:2603.11768](https://arxiv.org/html/2603.11768v1)

---

**Document Version**: 0.2
**Last Updated**: 2026-06-07
**Next Review**: After best-practices research lands and Phase 1 scoping.
