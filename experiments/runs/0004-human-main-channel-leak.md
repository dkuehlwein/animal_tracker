---
id: 5
slug: human-main-channel-leak
status: proposed          # proposed | running | concluded | rolled_back | parked
validation: parked        # live | replay | parked — pending Daniel's product/privacy call
hypothesis: "Humans the classifier IDs as 'homo species' with detection_status=identified bypass the REVIEW prefix (which only fires on NO_ANIMAL/UNCLASSIFIABLE) and leak to the MAIN channel as if a real wildlife identification. Route human/homo identifications to REVIEW (or suppress) to stop main-channel human alerts. Zero FN risk to wildlife (humans are not target animals)."
param_delta: null         # no env lever — REVIEW routing is taxon-blind in code (is_review_detection / _REVIEW_STATUSES); a fix is a CODE change
predicted_effect: { fp_rate: "main-channel human leaks -> 0", fn_risk: "none for wildlife (suppressing human alerts cannot hide an animal)" }
created: 2026-06-30
decision: pending         # product/privacy dimension is Daniel's call (cf. no-second-channel precedent)
confidence: high          # leak mechanism confirmed in code; two visually-confirmed human leaks on disk
---

## Finding (2026-06-30 tick, batch 1143-1167)

Three animal-tier rows tonight:
- **1147** (10:20, `aves;...;bird`, conf 0.81) — genuine. Frame shows a dark
  blackbird (Amsel) on the ground bottom-left. Human tapped ✅ animal. Correctly
  surfaced to main channel. No correction.
- **1163** (19:28, `mammalia;primates;hominidae;homo;;homo species`, conf 0.82) —
  **real human**. Frame is an unmistakable close-up of a person (brown hair, blue
  shirt) bending right in front of the camera, motion-blurred.
- **1167** (19:48, same `homo species` rollup, conf 0.92) — **real human** at the
  right frame edge (dusk, motion blur).

## Leak mechanism (confirmed in code, not assumed)

`src/data_models.py`: `_REVIEW_STATUSES = {NO_ANIMAL, UNCLASSIFIABLE}` and
`is_review_detection(status)` is taxon-blind. `src/wildlife_system.py:446` only
prepends the `🔍 REVIEW` header when `is_review_detection(status)` is true.
A `homo species` classification produces `detection_status = identified`, which is
**not** a review status → no prefix → the photo lands in the MAIN channel exactly
like a real bird ID. So 1163/1167 alerted Daniel's main channel as wildlife.

This is the same class as the 06-27 leak audit (classifier-FP "animal" rollups
bypassing the REVIEW prefix). 06-28's humans did NOT leak because their status was
not `identified`; tonight's two were `identified` at high conf → they did.

## Metrics caveat (do not silently fold in)

tier-1 maps `identified` → `animal`, so 1163/1167 reconcile as **animal**, not FP.
With no human tap on them they are NOT in tonight's fp_count. Reconciled fp = 22/25;
true operational FP (humans are non-wildlife) = 24/25. Per the standing rule
(auto-labels are not truth, headline = human-only), the reported headline is
**fp_human 5/6 = 0.83**; the 2-human-leak undercount is footnoted, not relabelled
via tier-2 (avoids self-poisoning the reconciled series). See JOURNAL 2026-06-30.

## Proposed fix (code change, minimal/reversible) — PENDING DANIEL

Add the human/homo taxon to REVIEW routing (or suppress human alerts entirely).
Smallest change: in `is_review_detection` / the notification path, treat a `homo`/
`homo species` classification as review-eligible regardless of `detection_status`,
so human detections get the `🔍 REVIEW` prefix instead of a clean main-channel ID.

**Why this is parked, not deployed tonight:** whether a wildlife camera should
alert on humans at all is a product/privacy decision (REVIEW-tag vs suppress vs
leave) that is Daniel's to make — same posture as the no-second-channel call. The
mechanism and impact are settled; the policy choice is not. Flagged in tonight's
verdict. If Daniel says go, this becomes a running code experiment with the fix
committed separately (`fix(notify): exp #5 ...`) + a pre-sunrise restart stamp.
