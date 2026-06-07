# ADR-004: Autonomous Detection-Tuning Loop

**Status**: Design converged (v0.6) — Phase 0 complete, Phase 1 ready to build.
**Date**: 2026-06-07
**Decision Makers**: Daniel Kuehlwein + Claude
**Technical Story**: Stand up a self-improving daily loop that reviews each day's
detections, runs experiments, and tunes the motion-detection configuration to
reduce false positives (first) and false negatives (later) — with a human
feedback channel and a daily summary over Telegram.
**Related**: ADR-002 (Two-Stage Detection Pipeline), ADR-003 (Multi-Frame Burst)

---

## ⏯️ Status & how to resume (read this first)

This design was developed over several sessions with two best-practices research
passes and two critical code-grounded reviews. It has **converged**; the next
session should start **building Phase 1**, not re-debating the design.

- **Where we are**: Phase 0 (design) done. Design verdict from review: converged,
  implementable, all code citations verified accurate.
- **What to build next**: Phase 1 — see **Required Changes** + **Phased Roadmap**
  (the "smallest correct slice" is spelled out in the Phase 1 bullet). In short:
  `detection_feedback` table + richer logging → **Telegram receive path** (the real
  work: `telegram.ext.Application` + `CallbackQuery` handler + inline buttons) →
  shadow-mode gate logging → lightweight timelapse FN-audit writer. **Keep sending
  all detections** (gate stays shadow-only in Phase 1).
- **Settled decisions** (do not re-open): Pro/Max subscription auth (training
  opted out); execution via on-Pi Claude Code `/loop`; lab notebook in git on
  `main`; single ADR; evaluation split into Layer A (offline on captured images) vs
  Layer B (motion/MOG2, tuned live); FP *and* FN tracked from Phase 1; event key =
  `detections.id`; embed `detection_id` in Telegram `callback_data`.
- **⚠️ One decision still OPEN (needed at the start of Phase 1)**: should the
  Telegram receive path run as a **separate sidecar process** (recommended — keeps
  the async camera loop untouched, SQLite WAL handles two writers) or as a
  **concurrent asyncio task** inside `wildlife_system.run()`? The user deferred this;
  confirm before coding the receive path.
- **Decisions deferred (not needed until later phases)**: hot config-reload vs
  "restart at next sunrise" (Phase 4); USB SSD vs SD (monitor wear); MegaDetector
  V6 upgrade; seasonal corpus re-baselining.

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

### Evaluation — two layers, evaluated differently (not one magic corpus)

The detector is two stages with very different evaluability. Conflating them (an
earlier draft did) led to the false claim that one replay corpus cleanly evaluates
everything. It doesn't. Split them:

**Layer A — Notification / species layer (MegaDetector + classifier on the
captured high-res images).** This **is** faithfully offline-evaluable: every motion
trigger saves high-res JPEGs, so we can re-run MD + the classifier with any
thresholds and re-decide "notify / route-to-review / drop", then score against
human labels. No MOG2 state involved — it's a pure function of the saved image.
This is where the cheapest, highest-leverage FP win lives:

> **Notification gate (current gap, but ship it in *shadow mode* first).** Today the
> system notifies on *every* motion trigger, including when MegaDetector finds no
> animal (`wildlife_system.py:343-347`, unconditional `send_notification` at `:603`).
> The eventual fix routes no-animal triggers to a separate "review" channel — *not* a
> hard delete (MD misses real animals → FN risk).
> **Initially, keep sending *all* detections to the main channel** and run the gate
> in **shadow mode**: it records what it *would* have suppressed but suppresses
> nothing. You label everything, including no-animal triggers, which yields both FP
> examples *and* the exact cases the gate would wrongly hide (MD's own misses = its
> FN cost). Flip the gate from shadow → live only once labels show its FN cost is
> acceptable. This is the "measure before you deploy" rule applied to the biggest
> single change.

**Layer B — Motion-capture layer (MOG2 deciding whether to grab a frame).** This is
**not** faithfully replayable: MOG2 is stateful/path-dependent
(`motion_detector.py:19-23,143`), triggering is multi-frame
(`consecutive_required`), and the 640×480 motion frames aren't even saved (only
high-res bursts are, `wildlife_system.py:219-223`). So motion-layer tuning is
**investigated live, day by day**, not via clean offline A/B. Two distinct error
types, measured from different sources:
- **Motion FP (junk triggers)** — visible in the *captured* set: a trigger that MD
  /human says has no animal. Correlate with `motion_area`, contour, time-of-day to
  guide `motion_threshold` etc. — but adjust incrementally and watch live, since we
  can't replay MOG2 exactly.
- **Motion FN (missed animals)** — *by definition not in the captured set*. The
  **only** way to see these is an **independent capture channel**: a low-FPS
  timelapse / random-schedule grab that an audit pass (MD + human) scans for animals
  the live config never triggered on. This is why FN tracking cannot wait.

**OEC tracks both sides, from Phase 1.** The objective is reported as **paired
FP and FN** (an F-beta with β chosen per phase, *plus the raw counts*), and **any
change that cuts FP but raises FN is vetoed**. FP reduction is half the coin; the
review must always show the other half.

**Time-of-day:** daytime-only operation is correct *scope*, not a gap. The one
residual is intra-day light (dawn/dusk long shadows vs flat noon) — so we
**timestamp every detection/sample** to *enable* time-of-day stratification later,
without acting on it pre-emptively.

**Statistics:** with small daily counts, decisions use **confidence intervals**
(Wilson/bootstrap on keep/drop counts), and the gate requires the CI to clear, not
a point estimate. A delta inside the noise band is "not proven", never a win.

### Autonomous loop — daily cycle

Each `/loop` tick after sunset, if tonight's run hasn't completed:
```
  1. Ingest    — pull new detections + human feedback from SQLite; scan the
                 independent timelapse/random channel for missed animals (FN audit).
  2. Label     — reconcile tier1/2/3; adjudicate ambiguous crops; update labels.
  3. Measure   — paired FP and FN (with CIs), volume, by-hour/by-cause; append to
                 metrics/daily.csv.
  4. Check     — does the active experiment's prediction still hold against
                 accumulated evidence (CI-based)? Enough evidence to conclude?
  5. Self-doubt— on cadence: re-verify own labels/decisions against the immutable
                 gold + the day's fresh human labels; assume own labels may err.
  6. Decide    — keep / rollback; if concluded, propose the next config by
                 coarse-grid / OFAT within bounds (BO only later, if enough data).
  7. Validate  — Layer A (notify/species): replay candidate over captured images
                 vs labels. Layer B (motion): cannot replay MOG2 — gate on bounds +
                 predicted live effect. VETO any FP win that worsens FN.
  8. Deploy    — write config within enforced bounds; **restart the service** (no
                 hot-reload exists — see below), incurring a ~5-min warmup; then a
                 loose multi-day live check, not a per-day verdict.
  9. Record    — append JOURNAL.md, write runs/NNNN.json, update state.json.
 10. Report    — Telegram daily summary (FP *and* FN); commit + push to main.
```
Token-cheap steps (1,3,4) are plain Python the agent invokes; token-spend
concentrates in 2,5,6,7,9. If the budget runs out mid-run, the next tick resumes
from the committed notebook.

> **Deploy = restart, not reload (code reality).** `Config()` is built once at
> startup (`wildlife_system.py:36`) and never re-read; there is no hot-reload path.
> So a config change requires a **process restart**, which resets MOG2 and triggers
> `warmup_seconds=300`. Either implement a real reload (signal → rebuild
> `MotionDetector`) or treat "deploy = restart at next sunrise" as an explicit,
> budgeted step. The earlier "service reloads / no new config engine needed" claim
> was wrong.

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
The strongest check is the **fresh daily human labels via Telegram** — because new
gold arrives every day, *systematic* agent bias (consistently wrong the same way)
gets caught by comparison against the human, which self-comparison alone cannot do.
On top of that, PROTOCOL.md mandates a periodic **self-audit**:
- Compare the agent's auto-labels against the day's human labels to estimate its
  current error rate and detect bias, not just drift.
- Re-check that past "wins" still hold on the larger corpus (catch regression to
  the mean / transient effects mistaken for improvements).
- State confidence + assumptions in `runs/`; a delta inside the CI noise band is
  *not proven*, never a success.

Garden images are **never** in this tree.

### Safety / guardrails for full autonomy

- **Bounded parameters enforced in code** — bounds live in `state.json` *and* are
  enforced at config load via `MotionConfig` field validators (today there are
  none, `config.py:50-66`). An out-of-range value must be rejected by the system,
  not merely discouraged — the agent voluntarily respecting a JSON file is not a
  guardrail for a fully autonomous loop.
- **FN veto** — any change that lowers FP but raises FN (beyond CI noise) is
  rejected. FN must not go backward; this is the dominant guardrail given the
  project goal.
- **Crude fast guards** — capture volume collapsing to ~0 or exploding, or the
  camera loop crashing, trigger immediate rollback (these catch gross failures fast;
  FN rise is slower to detect, see risk below).
- **Auto-rollback as a paved path** — keep the previous known-good config; revert
  atomically on a *sustained* breach.
- **Human-feedback-starved safe mode** — if no fresh human labels arrive for N days
  (vacation), **freeze tuning** and hold the known-good config; do not keep editing
  live config blind.
- **Telegram heartbeat / deadman** — a periodic "loop alive, last tick OK" ping;
  silence is itself an alert (covers `/loop` 7-day expiry, stuck ticks, dead session).
- **Human veto** — `/pause`, `/rollback` halt or revert at any time.
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
- Selection-bias → notify/species layer is replayed on captured images; motion-FN
  needs an **independent timelapse/random channel** (captured set can't show misses).
- **Interleaving + stochastic-triggering + IPS were considered and dropped** as
  unjustified complexity for this low-volume single camera.
- **Coarse-grid / OFAT is the primary search method** at this data volume; Bayesian
  optimization deferred until there are enough evaluated configs for a real surrogate
  (single-digit/low-tens of noisy observations would mostly fit noise).
- **OEC = paired FP+FN with CIs** so the agent can't game one metric; FN-veto.
- **Episodic→semantic memory** with append-only immutable gold; daily human labels
  catch *systematic* bias that self-comparison can't.

---

## Required Changes (high level)

1. **DB**: new `detection_feedback` table. **Event key = `detections.id`** — no
   extra dedup work: a burst already collapses to one `process_detection` → one row
   (`wildlife_system.py:554-604`), and cooldown-suppressed motion (`:543-552`) never
   reaches the DB (it's a Layer-B FN-audit concern, not a dedup concern). Persist
   `animals_detected` (already available, `species_identifier.py:173`), MD box
   count/confidence, and the richer motion metadata **already computed in
   `MotionResult`** (`contour_count`, `largest_contour_area`, `foreground_pixel_count`,
   `motion_detector.py:279-290`) but currently dropped in `log_detection` — so this
   is pure schema+plumbing, not new computation. Derive `time_of_day` from the
   existing `timestamp`. (`detections.db` is *already* gitignored via `data/`; the
   real new work is exporting CSV/JSON snapshots to `experiments/`. DB small writes
   are the main SD-wear driver → batch commits, WAL.)
2. **Notification gate (shadow first)**: keep sending *all* detections to the main
   channel initially; the `animals_detected==False` gate only *records* what it would
   suppress. Flip to live (→ review channel, not hard drop) once labels show the FN
   cost is acceptable.
3. **Telegram — a new *receive* path (the largest Phase-1 component).**
   `NotificationService` is **send-only** today (wraps `telegram.Bot`, no
   `Application`/polling/callback handler). The inline FP/wrong-species buttons need a
   long-running `telegram.ext.Application` polling for `callback_query` updates →
   `UPDATE detection_feedback` by id. **Recommended: a separate sidecar process**
   sharing the SQLite DB (WAL handles two writers), keeping the async detection loop
   untouched. **Embed `detection_id` in `callback_data`** at send time — the row is
   already written before `send_notification` (`:162` then `:603`), so just thread the
   returned id through. Also: a "we missed this" FN-report path; daily summary (FP
   *and* FN); veto + heartbeat.
4. **Capture (FN audit)**: **lightweight** low-FPS (~1 frame / 10–30s) **640×480
   grayscale** timelapse + near-miss logging (~100–250 MB/day), size-bounded with
   rotation. **SD card is fine to start** (images already live there); monitor card
   wear and move the DB/corpus to USB SSD only if wear shows up.
5. **Config safety**: add `MotionConfig` bounds validators (load-time enforcement);
   decide hot-reload vs "restart at next sunrise" for deploys.
6. **Replay harness (Layer A)**: offline runner over captured images for the
   notify/species decision; nightly **RDE** pass (with a guard so habitual-path
   animals aren't suppressed as static FPs).
7. **Loop runner**: `experiments/loop.md` + `PROTOCOL.md`; persistent on-Pi `claude`
   session driven by `/loop`; small systemd unit for keep-alive / 7-day re-arm.
   State lives in the git notebook on `main` (no custom resume machine).

---

## Phased Roadmap

- **Phase 0 — Design (this ADR; `PROTOCOL.md`/`loop.md` written at Phase 4).** ← current
- **Phase 1 — Ground truth (FP *and* FN from day one)**. Smallest correct slice, in
  order: (1) `detection_feedback` table (key = `detections.id`, derive time-of-day,
  reuse the existing 10-min `reference_frame` correlation for the camera-stability
  stamp, `wildlife_system.py:62-65,520-530`, persist the discarded `MotionResult`
  fields); (2) **the receive path** — `telegram.ext.Application` + `CallbackQuery`
  handler + inline keyboard (the real work); (3) shadow-mode gate logs the
  would-suppress decision, send behaviour unchanged; (4) lightweight 640×480
  timelapse FN-audit writer (independent, lands last). **Keep sending all
  detections.** *Nothing downstream is trustworthy without this.* (Retrofitting the
  event key / time-of-day / FN channel later would waste the early corpus.)
- **Phase 2 — Static 80/20 wins, measured**: flip the notification gate live once its
  FN cost is acceptable (→ review channel), ROI masking, reference-frame
  differencing, `unknown_species_threshold` 0.5→~0.75. Run each as a *measured
  experiment* (FP win vs FN cost) using Phase-1 labels.
- **Phase 3 — Layer-A replay harness** + RDE: offline tuning of the notify/species
  decision on captured images.
- **Phase 4 — Autonomous loop**: on-Pi `/loop` runner, git notebook, guardrails,
  daily summary; tunes Layer A offline + motion-layer live, both FP and FN.
- **Phase 5 — Motion-layer offline eval (optional)**: only if the residual warrants
  tuning MOG2 itself — capture ordered motion-res clips + per-clip warmup.

---

## Consequences

### Positive
- Objective, human-grounded metrics (FP *and* FN) replace guesswork.
- The notify/species layer is faithfully offline-evaluable on captured images.
- The cheap static wins (notification gate, ROI, threshold) likely deliver most of
  the FP reduction early, before the autonomous machinery exists.
- Images never leave the Pi; memory (non-image) is durable and self-documenting.
- A real test of autonomous, multi-day agentic problem-solving.

### Negative / Risks
- **Motion-layer (MOG2) can't be faithfully replayed** → those params are tuned
  live/incrementally, not by clean offline A/B (Phase 5 if needed).
- **FN-detection latency vs rollback**: FN rises are slow to observe (need the audit
  channel to accumulate), so a bad config could under-trigger for days before
  rollback. The FN-veto + freeze-on-starvation + fast volume guards mitigate but
  don't eliminate this; it's the dominant residual risk.
- **Non-stationarity**: accumulating the corpus across seasons blends distributions;
  needs recency-weighting / seasonal re-baselining (and the gold set can age out).
- **Camera movement** (bump/refocus/sag) invalidates ROI/RDE/region labels → needs a
  framing-stability check (reference-frame correlation) shipped in Phase 1.
- Human labels are sparse → slow convergence; daily Telegram labelling is the main
  mitigation, plus a feedback-starved safe mode when you're away.
- `/loop` needs a live on-Pi session, expires after 7 days, and the keep-alive unit
  becomes a single point of failure → Telegram heartbeat/deadman makes silence loud.
- Running an authenticated Claude session on an always-on Pi is a credential surface
  (separate from image privacy) — protect the session token at rest.

---

## Open Questions

1. **Deploy mechanism**: implement hot config-reload (signal → rebuild
   `MotionDetector`) vs accept "restart at next sunrise" with the warmup cost?
2. **SD wear (monitored, not blocking)**: start on the SD card (images already there);
   log card lifetime/health and alert on Telegram; move DB/corpus to USB SSD only if
   wear climbs. Keep the audit channel low-res/low-rate and DB commits batched.
3. **`/loop` cadence**: hourly fixed vs dynamic interval; systemd keep-alive / 7-day
   re-arm specifics.
4. **Seasonal re-baselining**: corpus recency-weighting policy and when the gold set
   "ages out".
5. **MegaDetector upgrade**: adopt V6-compact on-device, or keep the bundled detector?

*Resolved:* auth = Pro/Max subscription (training opt-out done); execution = on-Pi
`/loop`; notebook in git on `main`; single ADR; eval split into Layer A (offline on
captured images) vs Layer B (motion, live); FP and FN tracked from Phase 1; static
80/20 wins run as measured experiments before the autonomous loop.

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

**Document Version**: 0.7
**Last Updated**: 2026-06-07
**Next Review**: After Phase 1 implementation.

### Changelog
- **0.7** — Added "Status & how to resume" section for clean session handoff;
  recorded the still-open receive-architecture decision; added a pointer from
  CLAUDE.md. No design changes.
- **0.6** — Second review folded in (verdict: converged, ready to build). Elevated
  the Telegram **receive path** to a first-class Phase-1 component (today's service
  is send-only); event key simplified to `detections.id`; noted motion metadata is
  already computed (plumbing only), DB already gitignored (real work = CSV export),
  and the camera-stability stamp can reuse the existing reference frame. Sharpened
  the smallest-correct Phase-1 slice.
- **0.5** — Notification gate ships in *shadow mode* first: keep sending all
  detections so we collect FP and gate-FN-cost labels before going live. SD-card
  concern downgraded from blocker to monitored-risk (audit channel kept
  lightweight; DB writes are the real wear driver; SSD only if wear shows).
- **0.4** — Opus review folded in. Eval split into Layer A (notify/species, offline
  on captured images) vs Layer B (motion/MOG2, not faithfully replayable → live).
  Corrected: system currently notifies on *every* trigger → add review-channel gate.
  FP *and* FN tracked from Phase 1; FN-veto; static 80/20 wins promoted to Phase 2.
  Code-grounded fixes: "deploy = restart not reload", bounds enforced in code,
  gitignore the DB, corpus on external storage, coarse-grid/OFAT over BO, CIs on the
  gate, heartbeat + feedback-starved safe mode, camera-stability check, seasonal
  non-stationarity. Daytime-only confirmed as correct scope (+ time-of-day stamp).
- **0.3** — `/loop` execution model; dropped interleaving/IPS; notebook on `main`;
  self-skepticism. **0.2** — best-practices research. **0.1** — initial design.
