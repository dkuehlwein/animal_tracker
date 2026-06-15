---
id: 1
slug: notification-gate-live
status: concluded          # proposed | running | concluded | rolled_back | parked
validation: live          # live | replay | parked
hypothesis: "Flag no_animal+unclassifiable triggers as likely-FP; cuts FP-review burden w/o raising FN"
param_delta: { notification_gate: "shadow -> live", realized_as: "same-channel REVIEW labeling (NO_ANIMAL+UNCLASSIFIABLE)" }
predicted_effect: { fp_rate: "-15pp", fn_risk: "low" }
created: 2026-06-08
started: 2026-06-11
concluded: 2026-06-15
decision: keep            # keep | rollback | inconclusive
baseline: { fp_rate: 0.798, fp_ci: [0.700, 0.870], fn: unmeasured }
result:   { fp_recall_of_prefix: 0.99, clean_stream_animal_purity: 0.94, fn: 0 }
confidence: high
---

## Hypothesis & method

The shadow notification gate already records, per detection, whether it *would*
suppress (no animal found by MegaDetector) via `detections.gate_would_suppress`.
This experiment flips the gate from shadow to live: no-animal triggers route to a
review channel instead of the main channel. We expect a large FP reduction in the
main channel with low FN risk (MegaDetector misses are rare and caught by the
review channel). Validation is live (no offline replay this build); FN-veto
applies — because FN is currently *unmeasured*, a change that could raise FN is
HELD rather than auto-deployed.

## Daily observations

(Append-only. The agent records each night's FP/FN deltas and CI overlap here.
Never rewrite prior observations — anti-self-poisoning.)

### 2026-06-08 — baseline established; deploy HELD on infra gap

First real loop tick. Ingested 84 detections (watermark 0 → 84). Baseline:

- **FP rate 0.798** (67/84), 95% CI [0.700, 0.870].
- **FN unmeasured** (no FN-eliciting ground truth yet).
- **Volume 84 triggers/night** → set `baselines.volume_per_night = 84`.

Shadow-gate signal (cross-tab of `reconciled_label` × `gate_would_suppress`),
strongly in favour of this experiment's hypothesis:

| reconciled_label | gate suppresses | gate keeps | gate=null |
|---|---|---|---|
| animal (12) | 0 | 11 | 1 |
| false_positive (66) | 65 | 1 | 1 (id 2, human-labelled) |
| wrong_species (5) | 5 | 0 | 0 |

→ Gate would suppress **70/72 FP+wrong_species (97%)** and **0/12 true animals**
(zero FN on the labelled corpus). This is the clearest available FP lever.

**Decision: HOLD (cannot deploy).** The live notification gate does not exist
yet — `gate_would_suppress` is computed only inside `loop.ingest` as a shadow
metric. There is no gate in `notification_service.py`, no config field/env var,
and no `guardrails.BOUNDS` entry, so `loop.deploy --delta` would reject any gate
key (`validate_param`: "not a tunable parameter"). This is a **mechanical /
infra blocker, not an FN-veto** — the shadow data shows zero FN risk. Wiring the
gate live (notification_service gate + review-channel route + config/env +
BOUNDS + deploy path) is a code change for a normal dev session, not an
autonomous 2 h tick. Flagged to Daniel in tonight's summary. Experiment stays
`proposed`; `active_experiment_id` remains null. No numeric OFAT was launched —
`motion_area` does not separate animals (800–2185) from FPs (800–2000+), so a
`MOTION_THRESHOLD` bump would risk FN with no support.

### 2026-06-11 — shipped as SAME-CHANNEL LABELING (not routing)

Deployed the gate as **labeling, not routing**: review-class detections
(`NO_ANIMAL`, `UNCLASSIFIABLE`) now get a `🔍 REVIEW — likely false positive`
header prepended to their Telegram caption in the existing channel. Nothing is
dropped or moved → **FN-safe** (a misclassified real animal is still fully
visible with its feedback buttons, merely labeled "review"). This sidesteps the
two blockers that held the routing variant: no second Telegram channel needed,
and no FN-veto (suppression was the vetoed part; labeling is not suppression).

- Predicate: `is_review_detection(status)` in `data_models.py`
  (`{NO_ANIMAL, UNCLASSIFIABLE}`). The DB `gate_would_suppress` shadow column is
  unchanged (still `not animals_detected`) — deliberately independent.
- Toggle: `PERFORMANCE_REVIEW_PREFIX_ENABLED` (default true); instant off-switch.
- Commit: `31d3bc6`. Reversal: `git revert` + env flag.
- Spec: `docs/superpowers/specs/2026-06-11-review-prefix-notification-design.md`.

Goes live on the next camera restart (code change). Experiment #1 is realized as
this labeling variant; a future real channel split remains optional follow-up.

### 2026-06-11 (night tick) — VALIDATED on new data; promoted to running + brought LIVE

Promoted #1 from `proposed` to `running` and stamped the restart that actually
activates it. **The committed labeling code (commit 31d3bc6) was dormant** — the
camera only reloads code when `wildlife-deploy.timer` (daily 03:30 CEST) finds a
due `pending_restart_at`, and that stamp was `null`. So the shipped feature had
never run. Stamped `pending_restart_at = 2026-06-12T03:00:00+02:00` → the 03:30
deploy timer restarts `wildlife-camera.service`, REVIEW labeling live for 06-12
daytime captures. `active_experiment_id` → 1.

**Validation on tonight's 109 triggers (06-11, watermark 356→465; 40 human
labels — not feedback-starved).** Applied the live predicate
`is_review_detection(status)` = status ∈ {NO_ANIMAL, UNCLASSIFIABLE} to the
reconciled corpus:

| detection_status | FP | animal | wrong_species |
|---|---|---|---|
| no_animal       | 57 | 3 | 1 |
| unclassifiable  | 32 | 0 | 0 |
| identified      | 1  | 15 | 0 |

- **FP recall of the REVIEW prefix: 89/90 = 99%.** Only ONE FP slips through
  unlabeled — an `identified` misclassification (SpeciesNet named a species in
  vegetation). The prior no-animal-gate-only design (06-10) had ~33% FP recall;
  adding the `unclassifiable` class is what closes the gap.
- **Unprefixed ("clean") stream is 15/16 = 94% true animals** — the residual FP is
  the single `identified` one above.
- **Zero FN.** 3/18 true animals (the `no_animal` animals — MegaDetector misses)
  receive a REVIEW prefix but are STILL fully shown with feedback buttons. Cosmetic
  cost, not suppression. FN-veto satisfied by construction (labeling ≠ dropping).

Note: this labeling does NOT change `detection_status` or the FP-rate metric — it
is a notification-UX change. Its "result" is the recall/precision of the predicate
(above), measured offline; going live adds the operator benefit (Daniel sees which
notifications to scrutinise) without any metrics movement to wait on. Off-switch:
`PERFORMANCE_REVIEW_PREFIX_ENABLED=false`. Reversal: `git revert 31d3bc6`.

Gates: not paused, not frozen (40 human labels). No env delta, no volume change
(labeling doesn't suppress). One-experiment-at-a-time preserved — #4 concluded
this tick (see runs/0002).

### 2026-06-15 — CONCLUDED (keep). Daniel: same channel is the final design.

Daniel (via /remote-control): **"consider the second fp channel as solved.
routing it to the same channel with the pr fix is good enough. an actual second
channel just adds overhead. I am not clicking on two channels."**

This resolves the only remaining open thread on #1. The same-channel REVIEW-prefix
variant (commit 31d3bc6, live since 06-12) is the **accepted final design** — the
"future real channel split" noted on 06-11 is **dropped, not deferred**. Re-confirmed
location-agnostic on the 06-13/06-14 new-scene corpus (status mix no_animal 62,
unclassifiable 1, identified 5 → prefix still isolates the FP mass; see JOURNAL
2026-06-15). No code change, no restart — the feature is already live.

## Decision & rationale

**KEEP — concluded as a success (high confidence).** The notification-gate goal is
met in its accepted form: a same-channel 🔍 REVIEW prefix on `status ∈ {NO_ANIMAL,
UNCLASSIFIABLE}` flags **99% of FP (89/90)** while the unprefixed stream stays **~94%
true animals**, at **0 FN** (the 3 animals MegaDetector misses still arrive fully with
feedback buttons — labeling ≠ suppression, so the FN-veto is satisfied by construction).
Validated live across two camera locations. The second-channel routing variant is
explicitly out of scope per Daniel — a second channel adds operator overhead with no
benefit he wants. Off-switch remains `PERFORMANCE_REVIEW_PREFIX_ENABLED=false`;
reversal `git revert 31d3bc6`. Distilled into LEARNINGS.md.
