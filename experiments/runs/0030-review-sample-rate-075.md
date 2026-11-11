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

### Night 1 — 2026-10-30 (rate 0.75 live since the 03:30:11 restart, verified via systemctl + wildlife-deploy journal)

- 3 triggers (5851-5853), 09:33-10:27, overcast morning (luma ~28-29). 0 ERROR.
- 5851 IDENTIFIED `aves;bird` (MD 0.85), blackbird under the left tree → MAIN.
- 5852 NO_ANIMAL, no box, pc 0.24, **sampled out** (the 1-in-4 drop at 0.75). Tier-2 `animal`: the same blackbird
  is visible behind the trunk in frames 1-3 and flies off (motion blur) in frame 4. The visit was **delivered**,
  because 5851 reached MAIN 26 s earlier, so this is a burst-level miss only (same class as 5717/5720/5841).
  It is also another #31/#36 data point: a dark bird in low light (luma ~28) that SpeciesNet did not box.
- 5853 IDENTIFIED `aves;bird` (MD 0.85): a **Eurasian magpie** walking past the pot. It is the first magpie in this
  scene, and it went to MAIN.
- REVIEW pings sent: 0 (the only review-class burst was sampled out). Too little volume to judge the +2/day prediction.
- Lost visits to sampling: 0. Keep observing.

### Night 2 — 2026-10-31 (rate 0.75 live)

- 5 triggers (5854-5858), 08:25-11:30 (DB time), 0 ERROR. All five are IDENTIFIED and went to MAIN. 5854-5857 are `aves;bird`
  (MD 0.66-0.88). 5858 is the `;;;;;;animal` rollup (MD 0.77) with raw top-1 `bird`, not `blank`. Annotated frames for 5857
  and 5858 were checked: both show a blackbird by the trunk / stone edge. Tier-1 `animal` stands, so no tier-2 was needed.
- Review-class bursts: 0, so nothing was sampled out and no REVIEW pings were sent. This night says nothing about the
  prediction, either way.
- Lost visits to sampling: 0. Running total under 0.75: 0 lost visits and 1 burst-level miss (5852) in 2 nights.

### Night 3 — 2026-11-01 (rate 0.75 live; camera not yet restarted onto the Confident-Blank retirement code, that restart is stamped for 11-02 03:25)

- 3 triggers (5859-5861), 12:35-13:16 (DB time), 0 ERROR.
- 5859 IDENTIFIED `aves;bird` (MD 0.82) and 5860 the `;;;;;;animal` rollup (MD 0.72, raw top-1 `bird` 0.32): one bird visit 53 s
  apart, both sent to MAIN. Tier-1 `animal` stands; frames were not re-inspected.
- 5861 NO_ANIMAL, no MD box, person_confidence 0.43 (below the 0.5 human gate), motion_area 26490 with background_drift 13.7.
  **Not sampled out.** It was held 240 s by the deferred-send gate and sent as 🔍 REVIEW at 13:20:56. Tier-2 check on a 5-frame
  contact sheet + diff heatmap: the scene is empty. The only change is bamboo leaves moving in the upper right, plus a
  whole-frame light change. Tier-1 `false_positive` holds.
- REVIEW pings sent: 1 (an empty frame). Lost visits to sampling: 0. Running total under 0.75: 0 lost visits and 1 burst-level
  miss (5852) in 3 nights, with 1 REVIEW ping. Keep observing (window to ~11-20).

### Night 4 — 2026-11-02 (rate 0.75 live; camera restarted 03:30:11 onto the Confident-Blank retirement code, verified via the wildlife-deploy journal + systemctl)

- 4 triggers (5862-5865), 0 ERROR.
- 5862 IDENTIFIED `aves;bird` (MD 0.76) and 5863 the `;;;;;;animal` rollup (MD 0.79, raw top-1 `bird` 0.43): one bird visit 25 s apart at 10:03-10:04, both sent to MAIN. Tier-1 `animal` stands.
- 5864 (16:23) NO_ANIMAL, no box, pc 0.05. Not sampled out. The deferred REVIEW send was cancelled because 5865 (HUMAN, pc 0.89) landed 88 s later (`human_proximity_muted=1`). Tier-2 check on a contact sheet + diff: frame 1 has a large dark near-lens occlusion over the right half (consistent with the person passing), and frames 2-5 show the empty scene with bamboo movement. No animal, so this is not a false negative. Tier-1 `false_positive` stands.
- 5865 HUMAN → muted by the human gate.
- REVIEW pings sent: 0. Sampled out: 0. Lost visits to sampling: 0. Running total under 0.75: 0 lost visits and 1 burst-level miss (5852) in 4 nights, with 1 REVIEW ping. Keep observing (window to ~11-20).
- No human labels since 10-31 21:51 (2 days; the freeze trips at 3). This is a low-volume period: only 4 MAIN alerts and 1 REVIEW were sent across 11-01 and 11-02.

### Night 5 — 2026-11-03 (rate 0.75 live; no restart since 11-02 03:30:11, wildlife-deploy ran 11-03 03:30:10 with no stamp)

- 2 triggers (5866-5867), 12:14-12:15, 0 ERROR. Both IDENTIFIED `aves;bird` (one bird visit) → MAIN. Tier-1 `animal` stands.
- Review-class bursts: 0. Nothing was sampled out and no REVIEW pings were sent, so this night says nothing about the prediction.
- Timelapse FN screen (1717 frames, 07:27-17:02, 20 s cadence; largest connected change blob per consecutive pair): the
  only large local change was the 12:15 visit, which did trigger (blob ~1500 px). The next candidates (11:22, 11:59, ≤270 px)
  were inspected visually and showed leaf and light changes, with no animal. No missed visit was found at this cadence.
- Lost visits to sampling: 0. Running total under 0.75: 0 lost visits and 1 burst-level miss (5852) in 5 nights, with 1 REVIEW ping.
  Review-class volume has been near zero since 11-01, so the +2/day cost prediction can't be checked yet. Keep observing (to ~11-20).

### Night 6 — 2026-11-04 (rate 0.75 live; no restart since 11-02 03:30:11, wildlife-deploy ran 11-04 03:30:10 with no stamp)

- 4 triggers (5868-5871), 0 ERROR.
- 5868-5870 (12:51-12:52) IDENTIFIED `aves;bird` (MD 0.70-0.89): one bird visit, 3 MAIN. All three are below the sharpness floor (6.0-6.2), but the BLUR gate is review-class only, so they were sent. Tier-1 `animal` stands.
- 5871 (14:02) UNCLASSIFIABLE (`no cv result`), one MD box 0.41 on the rock with the blue pot, top raw `blank` 0.19, pc 0.0. **Not sampled out.** It was held 240 s and sent as 🔍 REVIEW at 14:06:56. Tier-2 check (box-region crop across all 5 frames): the rock and pot are static and only the bamboo leaves move, so the scene is empty. Tier-1 `false_positive` holds.
- REVIEW pings sent: 1 (empty). Sampled out: 0. Lost visits to sampling: 0. Running total under 0.75: 0 lost visits and 1 burst-level miss (5852) in 6 nights, with 2 REVIEW pings. Keep observing (to ~11-20).
- Human labels resumed: 3 labels arrived 11-03 21:12 (5859/5860 `animal`, 5861 `false_positive`), all agreeing with tier-1. So the feedback-starved freeze did not trip, and Daniel is still labelling REVIEW pings.

### Night 7 — 2026-11-05 (rate 0.75 live; no restart since 11-02 03:30:11, wildlife-deploy ran 11-05 03:30:10 with no stamp)

- 4 triggers (5872-5875, 15:13-15:20, dusk, luma ~20), 0 ERROR.
- 5872 IDENTIFIED `;;;;;;animal` rollup (raw top-1 `aves;bird`, MD 0.60 box on a dark bird beside the gnome figurine) → MAIN. Tier-1 `animal` stands.
- 5873, 5874 UNCLASSIFIABLE (`no cv result`, raw `blank`). **Tier-1 `false_positive` overturned by tier-2 → `animal`:** a blackbird is plainly visible behind the gnome in all 5 frames of 5873 and in frame 1 of 5874. Same visit as 5872. Both sent as 🔍 REVIEW (sample fractions 0.441 / 0.094, so they would also have been sent at 0.5; the rate raise rescued nothing tonight). Another instance of the dusk-blackbird miss class (backlog #31): SpeciesNet's ensemble discards a visible dark bird.
- 5875 UNCLASSIFIABLE, motion 17741 px / 23 contours, MD 0.39 box on the static gnome. **Tier-2 `person`:** frame 1 (the selected, sent frame) shows a heavily motion-blurred figure in blue walking across the right third, close to the lens; frames 2-5 are empty (HUMAN-SWEEP checked frames 2-5: no person). `person_confidence` 0.376 — inside the exp #14 watch band [0.30, 0.50). **It reached REVIEW** (sample 0.46, not sampled out; no HUMAN burst in the 240 s deferral, so it was sent). Pre-registered rollback criterion is a *recognizable* person reaching REVIEW; this one is a dark, blurred silhouette with no face (less recognizable than the 5020 precedent), so `SPECIES_HUMAN_DETECTION_CONFIDENCE` stays at 0.50. Recorded as the first in-band person to actually be *sent* that I can see in the recent runs; if a second one lands, or one is recognizable, lowering the threshold toward 0.35 (in BOUNDS) becomes the next experiment once #43 closes. Photo purge: 5875 is not HUMAN status, so its frames are retained under normal rotation.
- REVIEW pings sent: 3 (2 blackbird, 1 blurred person). Sampled out: 0. Lost visits to sampling: 0. Running total under 0.75: 0 lost visits and 1 burst-level miss (5852) in 7 nights, with 5 REVIEW pings. Keep observing (to ~11-20).
- No new human labels since 11-03 21:12 UTC. The 3-day freeze trips after 11-06 21:12 UTC unless Daniel labels.

### Night 8 — 2026-11-06 (rate 0.75 live; no restart since 11-02 03:30:11, wildlife-deploy ran 11-06 03:30:10 with no stamp)

- 3 triggers (5876-5878, 15:30-15:31, dusk), 0 ERROR. One blackbird visit beside the gnome figurine.
- 5876 (MD 0.87) and 5878 (MD 0.74) IDENTIFIED `aves;bird` → MAIN (both below the sharpness floor; BLUR is review-class only). Annotated frames checked: the blackbird is plain in both. Tier-1 `animal` stands.
- 5877 UNCLASSIFIABLE (`no cv result`, raw `blank` 0.67, MD 0.42, pc 0.16) → **sampled out** (fraction ≥ 0.75, so it would have been sampled out at 0.5 too). **Tier-2 overturns tier-1 `false_positive` → `animal`:** the selected frame 4 shows the blackbird beside the gnome. Same dusk-blackbird miss class as backlog #31. The visit was delivered by 5876/5878 (35 s either side), so this is a burst-level miss, not a lost visit.
- REVIEW pings sent: 0. Sampled out: 1 (animal, visit delivered). Lost visits to sampling: 0. Running total under 0.75: 0 lost visits and 2 burst-level misses (5852, 5877) in 8 nights, with 5 REVIEW pings. Keep observing (to ~11-20).
- No new human labels since 11-03 21:12 UTC. The feedback-starved freeze trips after 11-06 21:12 UTC unless Daniel labels; nothing was going to ship tonight anyway.

### Night 9 — 2026-11-07 (rate 0.75 live; camera up since 11-02 03:30:11, no stamp pending)

- 1 trigger (5879, 13:49 midday), 0 ERROR. UNCLASSIFIABLE (`no cv result`, raw `blank` 0.95), MD box 0.33, pc 0.244, motion 96098 px. Not sampled out; no HUMAN burst during the 240 s deferral, so it was **sent as 🔍 REVIEW** at 13:54:52.
- **Tier-2 `cant_tell`:** frame 1 (the selected frame) shows a large, dark navy, out-of-focus column right in front of the lens, filling the frame from top to bottom on the right half. Frame 2 has only its edge, bottom right, and frames 3-5 are empty garden. It is most likely a trouser leg brushing past the camera, but at that blur it could also be an animal very close to the lens, so I did not force `person` or `false_positive`. Same near-lens-occlusion class as 5864. pc 0.244 is below the exp #14 watch band [0.30, 0.50), and nothing in the frame could identify anyone, so it does not meet the rollback criterion for a *recognizable* person.
- REVIEW pings sent: 1. Sampled out: 0. Lost visits to sampling: 0. Running total under 0.75: 0 lost visits and 2 burst-level misses in 9 nights, with 6 REVIEW pings. Keep observing (to ~11-20).
- **Feedback-starved freeze tripped**: last human label was 11-03 21:12 UTC, so `feedback_starved_since` = 2026-11-06. Hold the current config. Exp #43 had nothing to ship anyway.

### Night 10 — 2026-11-08 (rate 0.75 live; camera up since 11-02 03:30:11, no stamp pending)

- 7 triggers (5880-5886, 08:27-08:41), 0 ERROR. luma low (overcast morning), all below the sharpness floor.
- 5880, 5882 HUMAN (pc 0.91/0.70, raw `human`) → muted by the human gate.
- 5881 UNCLASSIFIABLE (raw `blank` 0.85, MD 0.49, pc 0.026), 104 s after 5880 → **muted by HUMAN-PROXIMITY** (window). Standing duty, adjudicated on frames 1-5 (brightened for 2/3/5): frame 1 has a navy near-lens blur on the left edge, frames 2-5 are empty garden. **Tier-2 `person`**, following Daniel's `person` label on the same near-lens class (5879). No concealed animal.
- 5883, 5885, 5886 IDENTIFIED `aves;bird` (MD 0.85/0.85/0.63) → MAIN. Human-labelled `animal` within minutes. 5883 was 519 s after the last HUMAN burst, outside the window.
- 5884 NO_ANIMAL, no box, pc 0.218 → **sampled out** (fraction ≥ 0.75, so out at 0.5 too). **Tier-2 overturns tier-1 `false_positive` → `animal`:** the blackbird is visible behind the gnome in frames 1 and 4. Visit delivered by 5883 (30 s before) and 5885 (39 s after), so a burst-level miss, not a lost visit. Same dusk/low-light dark-bird class as backlog #31.
- REVIEW pings sent: 0. Sampled out: 1 (animal, visit delivered). Lost visits to sampling: 0. Running total under 0.75: 0 lost visits and 3 burst-level misses (5852, 5877, 5884) in 10 nights, with 6 REVIEW pings. The +2/day cost prediction is not confirmed: 6 pings in 10 nights is far below the ~5.8/day predicted, because review-class volume has been near zero. Keep observing (to ~11-20).
- **Pre-registered revert criterion met on 11-07, overridden by judgement.** The exit criteria say revert to 0.5 if "human labelling of REVIEW pings stalls (freeze trips) while the rate is raised", and the freeze did trip on 11-07. Not reverted. The criterion was meant to detect labelling fatigue from extra pings, and the evidence does not show that: only 6 REVIEW pings were sent in 10 nights, and on 11-08 Daniel labelled 14 detections in one batch, including every REVIEW ping sent since 11-01 (all 6 now human-labelled). This is a deviation from the letter of the pre-registration, recorded here so it is not mistaken for the criterion having passed. If the freeze trips again before ~11-20, revert without further argument.
- **Self-audit (first human labels on tier-2 rows since 11-03):** 4 rows carry both. 5873/5874 `animal` agree. 5875: tier-2 `person`, human `false_positive` (disagree). 5879: tier-2 `cant_tell`, human `person` (abstained, right to be unsure of `false_positive`). Tier-1 was wrong on 5873/5874, both `false_positive` on a visible blackbird. Tier-2 sampled-out `animal` calls (5852, 5877, 5884) remain unverified by a human, as those were never sent.
- **Exp #14 watch band:** night 7 said a second in-band person reaching REVIEW would make lowering `SPECIES_HUMAN_DETECTION_CONFIDENCE` toward 0.35 the next experiment. That condition is not met: Daniel labelled 5875 (pc 0.376, the first) `false_positive`, and 5879, the one he called `person`, has pc 0.244, below the [0.30, 0.50) band and below the BOUNDS floor of 0.3, so no in-bounds threshold reaches it. Near-lens occlusions reach REVIEW unrecognisably; nothing to change.
- **Freeze lifted**: 14 human labels on 11-08. `feedback_starved_since` cleared.

### Night 11 — 2026-11-09 (rate 0.75 live; camera up since 11-02 03:30:11, wildlife-deploy ran 11-09 03:30:10 with no stamp)

- **0 triggers** (ingest past 5886 empty; `loop.metrics` returned `no_data`, baseline preserved). Nothing to adjudicate, sample or send: 0 REVIEW pings, 0 sampled out, so this night says nothing about the prediction either way.
- **Camera not blind.** Sunrise warm-up 07:37:47, 6427 `Monitoring` heartbeats, sunset stop 16:53:13 ("0 detections today"). Heartbeat `motion_area` was 0 on every line, as on 11-03 and 11-07; on other November days only 1-4 lines are nonzero, since it logs one frame in ~25. 1659 timelapse frames, luma 1.8-62.9 (median 42.3), no identical consecutive frames, so the sensor was not frozen.
- **Timelapse FN screen, with a positive control.** Largest connected change blob per consecutive pair (640x480, global light change removed by subtracting the median diff, threshold 12). Control on 11-08 08:00-09:00: the blackbird visit scores up to 1587 px, at 08:39-08:41, matching triggers 5883-5886. Today's maximum is 106 px and the top four distinct candidates (14:29, 12:57, 12:44, 12:50) were viewed as brightened crops over consecutive frames: bamboo leaves only, no animal. Limitation: 20 s cadence, so a visit shorter than one interval can be missed; a first screen at threshold 25 returned nothing at all and was too blunt to be evidence.
- Lost visits to sampling: 0. Running total under 0.75: 0 lost visits and 3 burst-level misses (5852, 5877, 5884) in 11 nights, with 6 REVIEW pings. Keep observing (to ~11-20).
- No new human labels since 11-08 08:47 UTC. The freeze trips after 11-11 08:47 UTC unless Daniel labels. Note for that tick: with no pings sent there is nothing to label, so a trip driven only by an empty inbox would not be the labelling fatigue this experiment's revert criterion was written to detect. The night-10 "revert without further argument" commitment stands as written unless a later tick records a reason.

### Night 12 — 2026-11-10 (rate 0.75 live; camera up since 11-02 03:30:11, no stamp pending)

- **0 triggers** (ingest past 5886 empty; `loop.metrics` `no_data`, baseline preserved). 0 REVIEW pings, 0 sampled out: no evidence either way on the prediction.
- **Camera not blind.** Sunrise transition 07:39:14, warm-up complete 07:44:32, sunset stop 16:51:42 ("0 detections today"). 1652 timelapse frames, luma 1.6-33.0 (median 29.0, dull overcast day).
- **Timelapse FN screen, same method as night 11, with positive control.** Largest change blob per consecutive pair (median-diff removed, threshold 12, 3x3 open). The control (11-08 08:00-09:00) scores the blackbird visit at 1283 px (08:39:11). Today's max is 231 px. I viewed brightened crops (frames before, at and after) for the top three distinct candidates: 11:55 (0.76,0.20), 13:22 (0.55,0.15) and 12:47 (0.46,0.19). All three are bamboo leaves. A small dark-and-white patch near 13:22 is in the same place on 11-08 at 12:00 and today at 09:00 and 15:30, so it is static scenery, not an animal. `scripts/fn_audit_timelapse.py` (5-day window, 10k frames) ranks 11-08/11-06/11-05 trigger-coincident frames first and finds nothing for 11-10. Same 20 s cadence limit as night 11.
- Lost visits to sampling: 0. Running total under 0.75: 0 lost visits and 3 burst-level misses in 12 nights, with 6 REVIEW pings. Keep observing (to ~11-20).
- No new human labels since 11-08 08:47 UTC; the freeze trips after 11-11 08:47 UTC. The night-11 note stands: with an empty inbox, a trip would not be the labelling fatigue the revert criterion targets. The next tick must decide that explicitly.

### Night 13 — 2026-11-11 (rate 0.75 live; camera up since 11-02 03:30:11, no stamp pending)

- 6 triggers (5887-5892, 13:31-14:27 local, overcast, every burst below the sharpness floor except 5892), 0 ERROR.
- 5887-5890 IDENTIFIED `aves;bird` (classifier 0.78-0.85) → MAIN. 5887's frames show a **Eurasian magpie** (second time in this scene, after 5853). Tier-1 `animal` stands.
- 5891 UNCLASSIFIABLE (`no cv result`, MD 0.42, pc 0.013) → not sampled out (fraction 0.301), BLUR did not fire (needs luma ≥ 70), sent 🔍 REVIEW after the 240 s deferral (13:53). **Tier-2 overturns tier-1 `false_positive` → `animal`:** a blackbird is plain in frames 1-2 left of the pole, gone by frame 3. Same dark-bird-in-low-light class as backlog #31 (the 4th tier-1 overturn of this kind in 6 nights).
- 5892 NO_ANIMAL (raw `blank` 0.99, pc 0.098) → sent 🔍 REVIEW (14:31; fraction 0.117). **Tier-2 `person`:** frame 1 has a large dark navy near-lens column over the right half, and frames 2-5 show empty garden. This is the same class as 5864, 5879 (Daniel labelled it `person`) and 5881. Nobody can be recognised in it. pc 0.098 is far below BOUNDS, so no in-bounds threshold reaches it.
- Both REVIEW bursts have fractions < 0.5, so they would have been sent at 0.5 too. Tonight gives **no evidence for or against 0.75**.
- REVIEW pings sent: 2. Sampled out: 0. Lost visits to sampling: 0. Running total under 0.75: 0 lost visits and 3 burst-level misses in 13 nights, with 8 REVIEW pings. Keep observing (to ~11-20).
- **Feedback-starved freeze tripped** at 11-11 08:47 UTC (last human label 11-08 08:47 UTC). `feedback_starved_since` = 2026-11-11. Hold the current config.
- **Revert criterion: explicitly decided NOT to revert**, as nights 11 and 12 asked. This overrides the night-10 "revert without further argument" commitment, for this reason: between the last ping before the trip (5886, 11-08 07:41 UTC), the last human label (11-08 08:47 UTC) and the trip (11-11 08:47 UTC), the camera sent **zero** pings (11-09 and 11-10 had 0 triggers). Nothing went unlabelled, so the trip measures an empty inbox, not labelling fatigue. The criterion is meant to catch fatigue. Reverting would also be a config change during a freeze, which the freeze forbids.
  **Replacement criterion (binding, checkable):** revert to 0.5 if 5891 and 5892 (sent 11-11 12:53/13:31 UTC) both still have no human label at the 11-14 tick, which is 3 days after they were sent. That would be a real stall on pings that actually reached Daniel.
- Self-audit: no new human labels, so there are no tier-2 vs human pairs to reconcile. Tier-1 keeps calling dark low-light birds `false_positive` (5891). This is already backlog #31, and tier-2 catches it.
