---
id: 44
slug: review-sample-rate-1
status: active
validation: live   # env delta; replay stub
hypothesis: "Review sampling at 0.75 still loses whole animal visits (5909, 11-14), and in the 0.75 window every sampled-out burst (4/4) was a blackbird. In late autumn, review-class volume is low and its sampled-out tail is mostly low-light birds SpeciesNet does not box (#31/#36). Setting the rate to 1.0 removes visit loss to sampling for ~+0.25 REVIEW pings/day."
created: 2026-11-14
promoted_from: "exp #43 pre-registered exit: 'escalate to 1.0 only if a visit is still lost to sampling at 0.75' (runs/0030 night 16)"
confidence: medium   # cost side is measured (4 extra pings in 16 nights); benefit n is small but 4/4
delta: {"PERFORMANCE_REVIEW_SAMPLE_RATE": 1.0}   # was 0.75
commit: none   # env-only
restart_at: 2026-11-15T03:25+01:00
---

## Evidence (rows 5851-5909, 10-30..11-14, rate 0.75)

- 21 review-class bursts, 4 sampled out: 5852, 5877, 5884, 5909. All 4 are animals (tier-2, frames checked).
- 3 were delivered via a sibling burst; 5909 (male blackbird, 14:57) had no sibling and the visit was lost.
- Lost visits to sampling since 07-27: 5468, 5477, 5844/5845, 5909 = 4 in ~110 days, and 2 of them in the last 17 days.

## Decision

Escalate per exp #43's pre-registered criterion. In BOUNDS (0.0-1.0). Volume lever: only adds sends, cannot raise FN
(FN-veto N/A) and changes no captures (volume guardrail N/A). Not paused, no freeze, slot freed by concluding #43.
Per PROTOCOL "Review sampling", never credit this with an fp_rate change. At 1.0 the sampling gate is effectively off;
it stays in code (0 sampled out is the intended outcome, not a zero-firings retirement case while the knob is in use).

## Prediction / exit criteria (observe ≥14 days, to ~2026-11-28)

- `review_sampled_out` = 0 on every row after the 11-15 03:30 restart (verifies the deploy).
- REVIEW pings rise by only ~0.25/active day vs the 0.75 window (15 pings / 16 nights ≈ 0.9/night).
- Keep if Daniel keeps labelling REVIEW pings (no freeze tied to a real ping stall) and no `/rollback`.
- Revert to 0.75 if a feedback-starved freeze trips while pings sat unlabelled (fatigue), or on `/rollback`.
- Revisit downward only when spring volume returns (sampling was a summer-volume fix: ~44 REVIEW/night in July).

## Observations

### Night 1 (2026-11-15, rows 5910-5928)

- Deploy verified: camera restarted 11-15 03:30:11 (systemctl); `review_sampled_out`=0 on all 11 review-class rows, no `[REVIEW-SAMPLE]` log lines.
- 19 triggers: 7 HUMAN (5913-5916, 5918, 5919, 5926; two via `[HUMAN-SWEEP]`), 1 `[HUMAN-PROXIMITY]` mute (5917, a hand at the top-left edge, no animal), 10 REVIEW sent (after deferral).
- Tier-2 (frames viewed): 5927 + 5928 = blackbird left of the ornament at dusk (15:39/15:40, sharpness 4.6/4.8, MD no box), the #31/#36 class again. The other 8 REVIEW = bamboo/grass wind FPs.
- Counterfactual at 0.75 (`is_review_sampled_out(id, 0.75)`): 5920, 5922, 5923, 5924 would have been held back, all FPs. Both blackbird bursts would have been sent anyway. **Cost today: +4 pings, 0 visits saved.** One windy day ≫ predicted +0.25/day; one day is no trend. Keep observing to ~11-28.

### Night 2 (2026-11-16, rows 5929-5934)

- 6 triggers: 5929 HUMAN (HUMAN-SWEEP), 5931-5933 blackbird to MAIN, 2 REVIEW sent (5930, 5934: bamboo wind FPs, tier-2 from frames).
- Counterfactual at 0.75: only 5930 (fraction 0.980, FP) held back; 5934 (0.303) sent anyway. **Cost today +1 ping, 0 visits saved. Running total: +5 pings, 0 visits saved.**
- Feedback-starved freeze tripped (last human label 11-13 17:25 UTC). The literal revert criterion is met (12 unlabelled pings since then), but held by judgement: only 5 of the 12 are due to 1.0, and Daniel's 3-4-day batch labelling produced freezes at the same rate under 0.75. **Pre-registered tightening:** no human label by the 11-18 tick → revert to 0.75.

### Night 3 (2026-11-17, rows 5935-5949)

- 15 triggers, 0 ERROR. 7 HUMAN muted (5941, 5943-5948, 14:28-14:34: a person gardening near the lens, cutting back plants; 5948 frames checked, person only). 5942 `[HUMAN-PROXIMITY]` mute (frames checked: empty garden, no animal). 7 REVIEW sent (5935-5940, 5949): all human `false_positive` (bamboo wind; MD boxes the ornament at 0.22-0.40 on raw `blank`, or no box).
- **Freeze lifted:** Daniel labelled 26 rows on 11-17 ~17:16-17:20 UTC (5911-5940, 5949), clearing the whole backlog including all 12 pings that were unlabelled at the 11-16 freeze. The pre-registered 11-18 revert criterion passes: the 11-16 freeze was his normal batch rhythm, not fatigue.
- Self-audit: tier-2 matches the human label on all 13 overlapping rows (5911/5912/5920-5925/5930/5934 FP; 5927/5931 animal; 5928 tier-2 `animal` vs human `animal_wrong_id`, still an animal).
- Counterfactual at 0.75: only 5939 (fraction 0.986, FP) would have been held back. **Cost today: +1 ping, 0 visits saved. Running total over 3 nights: +6 pings, 0 visits saved.** Keep observing to ~11-28.
