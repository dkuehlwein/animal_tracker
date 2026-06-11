---
id: 1
slug: notification-gate-live
status: proposed          # proposed | running | concluded | rolled_back | parked
validation: live          # live | replay | parked
hypothesis: "Route no-animal triggers to a review channel; cuts FP w/o raising FN"
param_delta: { notification_gate: "shadow -> live" }
predicted_effect: { fp_rate: "-15pp", fn_risk: "low" }
created: 2026-06-08
started: null
concluded: null
decision: null            # keep | rollback | inconclusive
baseline: { fp_rate: 0.798, fp_ci: [0.700, 0.870], fn: unmeasured }
result:   { fp_rate: null, fp_ci: null, fn: null }
confidence: null
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

## Decision & rationale

(Filled in when the experiment concludes: keep / rollback / inconclusive, with
the CI-based reasoning and the FN-veto outcome.)
