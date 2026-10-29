---
id: 43
slug: review-sample-rate-075
status: active
validation: live   # env delta; replay stub. Evidence is offline: every sampled-out burst since 10-09 is labelled
hypothesis: "Review sampling at 0.5 now loses whole animal visits at roughly the ~1/month re-open bar exp #41 set. Every lost visit so far (5468, 5477, 5844/5845) is a dusk or dawn blackbird that SpeciesNet gave no box (#31/#36), and in late autumn dusk falls inside capture hours. Raising the rate to 0.75 should roughly halve visit-level losses for about +2 REVIEW pings/day."
created: 2026-10-29
promoted_from: "tick of 2026-10-28: 5844/5845, a dusk blackbird visit, were both sampled out with no sent sibling, so the visit was lost (frames re-inspected 10-29 with a brightened contact sheet: blackbird clearly visible in every frame). Held one night by the feedback-starved freeze, which lifted 10-29 when 44 human labels arrived."
confidence: low-medium   # 3 lost visits in ~89 days; n is small, the seasonal argument carries part of the weight
delta: {"PERFORMANCE_REVIEW_SAMPLE_RATE": 0.75}   # was 0.5
commit: none   # env-only
restart_at: 2026-10-30T03:25:00+01:00
---

## Evidence (DB + labels, rows 5590-5850, 10-09..10-29, 19 days)

- 154 review-class bursts, 80 sampled out, and all 80 are labelled (tier-2 or human).
- 5 sampled-out animals: 5717/5720 (10-21) had sent siblings within 120s, so the visit was delivered.
  5841 (10-27) had no sibling within 120s, but 5836-5840 of the same visit went to MAIN
  minutes earlier, so it was delivered. **5844/5845 (10-28) is the only visit lost.**
- Exp #41 (07-27..10-10): 2 lost visits in 70 days (5468, 5477, both dusk/dawn blackbirds).
- Combined: 3 lost visits in ~89 days ≈ 1/month, exactly #41's re-open bar ("exceed ~1/month").
- Cost: 80 sampled-out in 19 days ≈ 4.2/day at 0.5. Going to 0.75 sends half of those, about **+2 REVIEW/day**.
  Going to 1.0 costs +4.2/day. The long-run ratio (3 lost visits per ~564 sampled-out pings ≈ 1/188)
  is still worse than the 1/155 Daniel judged not worth it. That argues against jumping straight to 1.0.

## Decision

Take a middle OFAT step, 0.5 → 0.75. It stays inside BOUNDS (0.0-1.0). It is a volume lever, so the FN-veto
does not apply: it only adds sends and cannot raise FN. It changes no captures, so the volume guardrail
does not apply. Not paused, slot empty, freeze lifted (44 human labels 10-28/10-29, all agreeing with tier-2).

Per PROTOCOL "Review sampling", fp_rate is not affected, and this experiment must never be credited with an fp_rate change.

## Prediction / exit criteria (observe ≥21 days, to ~2026-11-20)

- REVIEW pings per active day rise ~1.5x vs the 10-09..10-29 baseline (sent review-class ≈ 74/19 ≈ 3.9/day → ~5.8/day).
- Keep if: no visit is lost to sampling in the window, and Daniel keeps labelling REVIEW pings (no sign of fatigue).
- Revert to 0.5 if: Daniel sends `/rollback`, or human labelling of REVIEW pings stalls (freeze trips) while the rate is raised.
- Escalate to 1.0 only if a visit is still lost to sampling at 0.75.

## Observations
