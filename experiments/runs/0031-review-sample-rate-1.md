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

### Night 4 (2026-11-18, rows 5950-5969)

- 20 triggers, 0 ERROR, 0 HUMAN, 0 `[HUMAN-PROXIMITY]`. 18 MAIN: 5950-5963 (10:28-10:45, blackbird visit, 14 bursts; 5954 frames checked: two blackbirds by the ornament) and 5966-5969 (15:30-15:32, dusk blackbird, 5968 checked). 2 REVIEW sent after the 240s deferral: 5964 (13:35) and 5965 (14:41), `no cv result`, MD boxes 0.23-0.24, frames checked: empty garden, bamboo wind FPs (tier-2 `false_positive`).
- `review_sampled_out`=0 on both review-class rows. Counterfactual at 0.75: fractions 0.306 / 0.258, so both would have been sent anyway. **Cost today: +0 pings, 0 visits saved. Running total over 4 nights: +6 pings, 0 visits saved.**
- No human labels yet for 5950-5969 (last human label 11-17 17:19 UTC; no freeze). Keep observing to ~11-28. No deploy.

### Night 5 (2026-11-19, rows 5970-5972)

- 3 triggers, 0 ERROR, 0 HUMAN, 0 `[HUMAN-PROXIMITY]`, 0 MAIN. All 3 REVIEW sent after the 240s deferral: 5970 (11:09), 5971 (13:11), 5972 (15:05), all `no cv result` with raw top-1 `blank`, below the sharpness floor (dark, luma < 70 so no BLUR mute). Frames checked: empty garden, MD boxes 0.23-0.40 on the static stones, the blue pipe and the left trellis edge; motion point in the bamboo. Tier-2 `false_positive` x3.
- 5970 `person_confidence` 0.31 (< 0.5 gate) is the left-edge trellis strip; no person in any frame; the burst human sweep ran and found nothing.
- `review_sampled_out`=0 on all 3. Counterfactual at 0.75: 5970 (0.868) and 5972 (0.858) would have been held back, both FPs; 5971 (0.282) sent anyway. **Cost today: +2 pings, 0 visits saved. Running total over 5 nights: +8 pings, 0 visits saved.**
- Self-audit: Daniel labelled all of 5950-5969 on 11-18 19:29-19:32; tier-2 agrees on all 4 overlaps (5964/5965 FP, 5954 animal, 5968 tier-2 `animal` vs human `animal_wrong_id`, still an animal). No freeze.
- Low volume (3 vs the stale 27/night baseline) is a quiet, dark late-autumn day, not caused by this change: sampling acts after capture and logging, so it cannot reduce triggers. Camera service up since 11-15 03:30, monitoring normally. Keep observing to ~11-28. No deploy.

### Night 6 (2026-11-20, row 5973)

- 1 trigger, 0 ERROR, 0 HUMAN, 0 `[HUMAN-PROXIMITY]`, 0 MAIN. 5973 (13:13) NO_ANIMAL, MD 0 boxes (11 sub-floor detections), pc 0.06, sharpness 15.6 (above floor), REVIEW sent after the 240s deferral (sendPhoto 13:17:54). Frames checked: empty garden, bamboo/grass wind; intra-burst diff only 424 px, centred. Tier-2 `false_positive`.
- `review_sampled_out`=0. Counterfactual at 0.75: fraction 0.562, so it would have been sent anyway. **Cost today: +0 pings, 0 visits saved. Running total over 6 nights: +8 pings, 0 visits saved.**
- Self-audit: Daniel labelled 5970-5972 on 11-20 08:45; tier-2 agrees 3/3 (all FP). No freeze.
- Tick note (infra, no behaviour change): first tier-2 insert failed (`DatabaseManager` needs a `Config`), so `loop.metrics` ran without it; restored state.json + daily.csv from the pre-metrics commit and re-ran `loop.metrics` (watermark rewritten by the tool, not by hand). Keep observing to ~11-28. No deploy.

### Night 7 (2026-11-21, rows 5974-5975)

- 2 triggers, 0 ERROR, 0 MAIN, 0 `[HUMAN-PROXIMITY]`. 5974 (11:17) UNCLASSIFIABLE (`no cv result`, raw top-1 `blank` 0.82), MD box 0.24 on the left trellis strip, pc 0.05, sharpness 9.8 (below floor, dark so no BLUR mute); REVIEW sent after the 240s deferral (sendMediaGroup 11:22:08). Frames checked: empty garden, motion point in the bamboo. Tier-2 `false_positive`. 5975 (11:40) HUMAN (classifier `human` 0.91), `[HUMAN-GATE]` muted; frame checked: a person's legs walking past the lens, no animal. 22 min after 5974, so outside the deferral window.
- `review_sampled_out`=0. Counterfactual at 0.75: fraction 0.216, so it would have been sent anyway. **Cost today: +0 pings, 0 visits saved. Running total over 7 nights: +8 pings, 0 visits saved.**
- Self-audit: Daniel labelled 5973 `false_positive` on 11-21 06:24; tier-2 agrees. No freeze. Keep observing to ~11-28. No deploy.

### Night 8 (2026-11-22, rows 5976-5984)

- 9 triggers, 0 ERROR, 0 MAIN, 0 `[HUMAN-PROXIMITY]`. 5976 (09:51) UNCLASSIFIABLE (`no cv result`, raw top-1 `blank` 0.88), MD boxes 0.34 (left trellis strip) and 0.27 (pond edge), pc 0.04, sharpness 7.0 (below floor, dark so no BLUR mute); REVIEW sent after the 240s deferral. Frames checked: **a male blackbird at the pond edge** inside the 0.27 box. Daniel labelled it `animal` (11-22 11:51 UTC). 5977-5984 (10:12-10:17) are 8 HUMAN bursts (2 escalated by `[HUMAN-SWEEP]`), all `[HUMAN-GATE]` muted; the first is 21 min after 5976, so outside the deferral window.
- **First visit saved by this experiment.** `review_sampled_out`=0. Counterfactual at 0.75: fraction 0.864, so 5976 would have been held back. No other burst between 09:00 and 10:12, so no sibling would have delivered it and the visit would have been lost. **Cost today: +1 ping, 1 visit saved. Running total over 8 nights: +9 pings, 1 visit saved.** This matches the pre-registered mechanism: a dark-frame blackbird that SpeciesNet boxes weakly and does not classify.
- Delivery note (infra, not behaviour): `sendMediaGroup` returned 400 `image_process_failed` at 09:55:53. The existing fallback (`sendMessage` + `sendDocument`, both 200) still delivered it, and Daniel labelled it. This is the only such error in all rotated logs; logged as a watch item, no fix.
- Self-audit: Daniel labelled 5974 `false_positive` (11-22 11:51); tier-2 agrees. No freeze. Keep observing to ~11-28. No deploy.

### Night 9 (2026-11-23, rows 5985-5986)

- 2 triggers, 0 ERROR, 0 HUMAN, 0 `[HUMAN-PROXIMITY]`, 0 review-class. Both MAIN, sent immediately: 5985 (10:10:02) IDENTIFIED `;aves;;;;;bird` 0.83, MD 0.82 box; 5986 (10:10:36) IDENTIFIED `;aves;;;;;bird` 0.72, MD boxes 0.57 + 0.26 (the 0.26 is the yellow/red ornament). Frames checked: a blackbird at the pond's left edge (5985) and then by the right-hand rocks (5986), so one visit. Both below the sharpness floor (6.5/6.4, dark morning). This does not matter for MAIN rows. Tier-2 `animal` x2.
- No review-class row, so the sample rate decided nothing tonight. **Cost today: +0 pings, 0 visits saved. Running total over 9 nights: +9 pings, 1 visit saved.**
- Self-audit: no new human labels since last tick (5974/5976 were audited on night 8). The last human label was 11-22 11:51, so no freeze. Keep observing to ~11-28. No deploy.

### Night 10 (2026-11-24, rows 5987-5988)

- 2 triggers, 0 ERROR, 0 HUMAN, 0 `[HUMAN-PROXIMITY]`, 0 review-class. Both MAIN, sent immediately (sendMediaGroup 200 at 09:51:57): 5987 (09:51:44) IDENTIFIED `;aves;;;;;bird` 0.91, MD 0.87 box at the bottom edge; 5988 (09:52:15) IDENTIFIED `;aves;;;;;bird` 0.68, MD 0.64 box by the left trellis. Frames checked: a **magpie** (black/white, blue wing sheen), first walking past close to the lens and then on the grass at left. One visit. Both below the sharpness floor (5.1/5.1, dark morning), which does not matter for MAIN rows. Tier-2 `animal` x2.
- No review-class row, so the sample rate decided nothing tonight. **Cost today: +0 pings, 0 visits saved. Running total over 10 nights: +9 pings, 1 visit saved.**
- Self-audit: no new human labels since night 8 (last 11-22 11:51 UTC, ~2 days, so under the 3-day freeze bar). Keep observing to ~11-28. No deploy.

### Night 11 (2026-11-25, row 5989)

- 1 trigger, 0 ERROR, 0 HUMAN, 0 `[HUMAN-PROXIMITY]`, 0 review-class. 5989 (10:28:29) IDENTIFIED `;;;;;;animal` rollup 0.60 (raw top `blank` 0.49), MD boxes 0.60 / 0.24 / 0.23, sent to MAIN. Frames checked (brightened crops, frames 1/3/5): the 0.60 box is a **static grey rock next to the blue pipe**, the 0.24 box is the yellow/red ornament, and the 0.23 box is the left trellis. Nothing moves between frames, and frost is on the grass. Motion point is in the bamboo, so this is wind. Tier-2 `false_positive`. It is the known `;;;;;;animal`-rollup MAIN FP class (rock/ornament boxes). Sharpness 6.2, below the floor, but `[BLUR]` covers review-class only.
- No review-class row, so the sample rate decided nothing tonight. **Cost today: +0 pings, 0 visits saved. Running total over 11 nights: +9 pings, 1 visit saved.**
- Self-audit: no new human labels (last 11-22 11:51 UTC, 3 days ago). **FEEDBACK-STARVED FREEZE tripped** → `feedback_starved_since` = 2026-11-25. Hold config. Exp #44 stays live (no change to ship), observing to ~11-28. No deploy.
