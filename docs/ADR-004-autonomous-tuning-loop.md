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

**Auth = Pro/Max subscription (decided).** Two consequences:
- *Session/token limits*: ~5-6h rolling reset windows. Handled by the `/loop`
  execution model below — **not** a custom checkpoint/stage machine.
- *Data terms*: under consumer subscription terms conversations may be used for
  training unless opted out. The account owner has **opted out of model training**,
  satisfying the prerequisite for sending image crops for adjudication. Policies
  change — re-verify against Anthropic's current data-usage docs periodically.

### Execution model — driven by Claude Code `/loop` on the Pi

The loop is driven by Claude Code's native **`/loop`** feature running in a
**persistent session on the Pi itself** — *not* a custom systemd timer + resume
state machine (an earlier draft over-engineered this), and *not* a cloud Routine.

Why on-Pi `/loop` and not a cloud Routine: Claude Code **Routines run in
Anthropic's cloud with a fresh repo clone**, so they cannot see the Pi's local
images — which breaks local-image privacy and on-device crop adjudication. A
session running *on the Pi* keeps images local and can crop+adjudicate directly.

How it works:
- A long-lived `claude` session on the Pi runs `/loop` (hourly, or dynamic
  interval). Each tick the agent reads the git-tracked notebook, checks *"is it
  after sunset and has tonight's run completed?"* (via `utils.SunChecker` + the
  notebook), and if not, performs the next chunk of work.
- **Resumption is implicit.** Every tick reconstructs context from the git
  notebook, so a session killed by a usage-limit reset simply continues on the next
  tick (or via `claude --continue`, which restores loops within 7 days). No
  bespoke stage tracking — the git notebook *is* the durable state.
- **Token economy still matters**: the heavy deterministic work (ingest, metrics,
  MegaDetector relabel, replay sweeps, RDE) is plain local Python the agent
  *invokes*; Claude tokens are spent only on the judgment layer (crop adjudication,
  designing the next experiment, writing the journal). So a tick does cheap Python
  first, then spends tokens on decisions — and if the budget runs out mid-judgment,
  the next tick resumes from committed state.
- **OS glue (minimal)**: a small systemd unit keeps/restarts the looping session
  across reboots and re-arms the loop (recurring loops auto-expire after 7 days).
  This is the only non-native piece, and it is far smaller than the prior design.

The operational prompt the loop executes lives in `experiments/loop.md` (a runtime
artifact created during implementation — *not* a second design doc; see "one ADR").

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

### Evaluation — the accumulated random-capture corpus is the evaluator

Two failure modes would sink a naive eval: **selection bias** (you only have frames
the current config triggered on, so optimizing against them just reproduces the
current config and grows blind spots) and **environmental confounding** (one
camera, so live before/after credits weather/season changes to your config change).

An earlier draft tried to fix confounding with live **interleaving** (per-event
A/B) plus **stochastic boundary triggering** + IPS reweighting. **Both are dropped**
— interleaving needs more daily volume than this garden produces (some days yield
only a handful of captures, giving per-arm samples with no power), it only applies
cleanly to decision-stage params (switching MOG2 history/learning-rate per event
corrupts the background model), and it adds real complexity. The cheaper, stronger
answer makes them unnecessary:

**The independent random-capture corpus does the job.** Capture frames on a
**random schedule decoupled from motion triggering** (plus near-miss frames in the
50-100% shoulder band, plus a low-FPS timelapse), label them (human + reconciled
machine), and keep this set **strictly separate** from the active-learning queue —
used **only for evaluation**. Then:

- **Replay every candidate config over this fixed labeled set** and score an
  **OEC = F-beta over verified events** (precision *and* recall, so the agent can't
  "win" by triggering never/always). Because it's the *same frames* for every
  config, there is **zero weather confound** — no interleaving needed. Because the
  corpus is unbiased by construction (random, not trigger-gated), there is **no
  propensity problem** — no stochastic triggering or IPS needed.
- **Low daily volume is handled by accumulation**: the corpus grows across days, so
  evaluation draws on the whole accumulated set, not one thin day. A quiet day adds
  little but breaks nothing.
- It is also the **only** honest estimator of false negatives (animals the live
  config missed but that appear in random captures).

**Live data = loose multi-day sanity check, not proof.** After deploying a
replay-validated config, watch live metrics *accumulated over many days* (never a
single-day delta) purely to confirm the offline prediction roughly holds; treat
divergence as a prompt to investigate, not an automatic verdict.

> ⚠️ FN estimates from the triggered corpus alone are **lower bounds**; the random
> corpus is what makes them trustworthy.

### Autonomous loop — daily cycle

Each `/loop` tick after sunset, if tonight's run hasn't completed:
```
  1. Ingest    — pull new detections + human feedback from SQLite.
  2. Label     — reconcile tier1/2/3; adjudicate ambiguous crops; update corpus.
  3. Measure   — FP rate, FN estimate (random corpus), volume, by-hour/by-cause;
                 append to metrics/daily.csv.
  4. Check     — does the active experiment's offline prediction still hold against
                 accumulated live data? Enough accumulated evidence to conclude?
  5. Self-doubt— on cadence (see memory): re-verify own past labels/decisions
                 against the immutable gold set; assume own auto-labels may err.
  6. Decide    — keep / rollback the active experiment; if concluded, propose the
                 next config via Bayesian optimization (noise-aware surrogate over
                 past results, within bounds). OFAT only for a quick single-knob
                 sanity check.
  7. Validate  — replay candidate over the random-capture corpus (OEC = F-beta).
                 Abort if it doesn't beat the current config offline.
  8. Deploy    — write config within bounds; service reloads (then loose multi-day
                 live sanity check, not a per-day verdict).
  9. Record    — append JOURNAL.md, write runs/NNNN.json, update state.json.
 10. Report    — send Telegram daily summary; commit + push the notebook to main.
```
Token-cheap steps (1,3,4,7) are plain Python the agent invokes; token-spend
concentrates in 2,5,6,9. If the budget runs out mid-run, the next tick resumes
from the committed notebook.

### Durable memory — git-backed lab notebook, committed to `main`

The notebook lives **in the main repo, committed directly to `main`** — we are
running live experiments and want the record in the canonical history, not on a
side branch. (The `/loop` session reads and pushes it on `main`.)

```
experiments/
  PROTOCOL.md       # the SOP. A fresh agent session reads this FIRST.
  loop.md           # the operational prompt /loop executes (runtime artifact)
  JOURNAL.md        # append-only narrative ("Day 12: FP 31%, tried X because…")
  LEARNINGS.md      # distilled semantic memory: durable, firm conclusions only
  state.json        # deployed config, active experiment, baselines, guard limits,
                    #   best-known-good config + full history (O(1) "tried this?")
  runs/NNNN.json    # per-experiment: hypothesis, config delta, predicted effect,
                    #   replay result, live result, decision, confidence
  metrics/daily.csv # FP rate, FN estimate, volume, by-hour time series
  gold/             # human-verified labels + frozen eval set — IMMUTABLE
```

Two memory tiers (per agent-memory research): **episodic** (per-day raw
observations in `JOURNAL.md`/`runs/`) and **semantic** (firm, distilled conclusions
consolidated into `LEARNINGS.md` once an experiment concludes — keeps session
context small). `PROTOCOL.md` encodes the SOP so any fresh session resumes correctly.

**Anti-self-poisoning rule**: human labels and the gold eval set are
**append-only and immutable** — the agent may never overwrite ground truth or its
own past results, only append. Only *interpretations* evolve.

**Self-skepticism is mandatory (bake the chance of AI error into the loop).** The
agent runs the labeling and the tuning, so its own errors can silently compound.
PROTOCOL.md therefore mandates a periodic **self-audit** (e.g. weekly, or every N
experiments):
- Re-label a fresh random sample and compare against its *own* prior auto-labels to
  estimate its current error rate; surface disagreements to the human.
- Re-check that past "wins" actually held up on the now-larger corpus (catch
  regressions to the mean / transient effects mistaken for improvements).
- Explicitly state confidence and assumptions in `runs/`, and prefer "I might be
  wrong because…" over false certainty. A claimed improvement inside the noise band
  is treated as *not proven*, not as success.

Garden images are **never** in this tree.

### Safety / guardrails for full autonomy

- **Bounded parameters** — each tunable has a hard min/max in `state.json`.
- **OEC + guardrails** — optimize F-beta over verified events; guardrail metrics
  that *can't go backward*: false-negative rate must not rise, capture volume must
  not collapse-to-~0 or explode, the camera loop must not crash.
- **Offline gate** — never deploy a change that didn't beat current config on the
  random-capture corpus.
- **Auto-rollback as a paved path** — keep the previous known-good config and
  revert atomically on a *sustained* guardrail breach (not a single bad hour).
- **Replication before commit** — single noisy daily deltas aren't enough; require
  accumulated agreement (and constrain the BO surrogate's hyperparameters so it
  doesn't overfit noise at low sample counts).
- **Self-audit cadence** — the periodic self-skepticism review above is a guardrail:
  the agent must actively look for its own errors, not assume it's right.
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
- Selection-bias death spiral → independent **random-capture eval corpus** (the
  primary evaluator).
- Daily before/after is confounded → solved by replaying every config over the
  *same fixed corpus*. **Interleaving + stochastic-triggering + IPS were considered
  and dropped** as unjustified complexity for this low-volume single camera (the
  random corpus already removes the confound and the propensity problem).
- **Bayesian optimization** (noise-aware, low-sample-constrained) as the search
  engine; OFAT only for quick sanity checks.
- **OEC (F-beta)** + immutable guardrails so the agent can't game a single metric.
- **Episodic→semantic memory** with append-only, immutable gold labels, plus a
  mandated **self-audit** cadence (avoid memory self-poisoning *and* AI-error drift).

---

## Required Changes (high level)

1. **DB**: new `detection_feedback` table; persist `animals_detected`,
   MegaDetector box count/confidence, and richer motion metadata (contour count,
   largest contour, foreground pixels) that `motion_detector.py` already computes
   but currently discards.
2. **Telegram**: inline feedback buttons + callback handler (Pi-side); daily
   summary sender; veto commands.
3. **Capture**: near-miss frame logging + **random-schedule capture** + low-FPS
   timelapse writer (local disk, size-bounded with rotation).
4. **Replay harness**: offline runner that executes `MotionDetector` over the
   random-capture corpus and reports OEC (F-beta) keep/drop metrics for a config;
   plus a nightly **RDE** pass to strip recurring stationary FPs.
5. **Loop runner**: `experiments/loop.md` + `PROTOCOL.md`; a persistent on-Pi
   `claude` session driven by `/loop`; a small systemd unit to keep it alive and
   re-arm the loop across reboots / 7-day expiry. State lives in the git notebook
   on `main` (no custom resume machine).
6. **Config**: surface new tunables + bounds; the loop edits config via the
   existing env-var / config-file mechanism and triggers a service reload.

---

## Phased Roadmap

- **Phase 0 — Design (this ADR; `PROTOCOL.md`/`loop.md` written at Phase 4).** ← current
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
- The random-capture corpus needs time to accumulate before FN estimates are solid.
- Autonomy risk (a bad config degrading capture) mitigated by offline gate,
  guard metrics, auto-rollback, self-audit, and human veto.
- On-Pi replay is CPU-bound; mitigated by nightly batch + optional desktop host.
- `/loop` requires a live on-Pi session and recurring loops expire after 7 days →
  mitigated by a small systemd keep-alive/re-arm unit.

### Neutral
- The loop edits config through the existing override mechanism; no new config
  engine needed.

---

## Open Questions

1. **Corpus retention**: how many days of random/near-miss/timelapse frames to
   keep, and rotation policy, given SD-card wear and capacity.
2. **`/loop` cadence**: hourly fixed vs dynamic interval; and the keep-alive/re-arm
   approach for the 7-day loop expiry and reboots (systemd unit specifics).
3. **MegaDetector upgrade**: adopt V6-compact on-device as a better FP-filter
   feature, or keep the bundled v5-era detector?
4. **Host fallback**: if on-Pi replay sweeps prove too slow, move the loop to the
   desktop-with-SSH (the design is host-agnostic).

*Resolved:* auth = Pro/Max subscription (training opt-out done); execution = on-Pi
`/loop`; notebook in git on `main`; single ADR (this doc).

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

**Claude Code execution**
- [Run prompts on a schedule (`/loop`)](https://code.claude.com/docs/en/scheduled-tasks.md) ·
  [Routines](https://code.claude.com/docs/en/routines.md) ·
  [Headless mode](https://code.claude.com/docs/en/headless.md) ·
  [Sessions / resume](https://code.claude.com/docs/en/sessions.md)

---

**Document Version**: 0.3
**Last Updated**: 2026-06-07
**Next Review**: Phase 1 scoping.
