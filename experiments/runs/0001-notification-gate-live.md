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
baseline: { fp_rate: null, fp_ci: null, fn: unmeasured }
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

## Decision & rationale

(Filled in when the experiment concludes: keep / rollback / inconclusive, with
the CI-based reasoning and the FN-veto outcome.)
