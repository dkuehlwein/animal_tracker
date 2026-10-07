---
id: 40
slug: confident-blank-autumn-fn
status: running
validation: live   # env delta; replay stub, so validated by re-scoring every blank_confidence_muted row
hypothesis: "The Confident-Blank gate (exp #29) no longer separates empty gardens from animals. In autumn light SpeciesNet scores dim frames holding a visible blackbird as `blank` at 0.93-0.99, inside the range of truly empty frames (0.92-0.99). No in-bounds threshold keeps the gate's pre-registered max(animal)+0.02 margin, so the threshold goes to the BOUNDS ceiling 1.0, which turns the mute off in practice."
created: 2026-10-03
promoted_from: "tick of 2026-10-03 (first after the 09-26..10-02 outage): 3 of the 5 Confident-Blank mutes in the window hold a clearly visible blackbird (5468, 5477, 5495)."
confidence: medium   # FN evidence is direct (frames inspected); n=3 animals out of 19 lifetime mutes
delta: {"PERFORMANCE_BLANK_CONFIDENCE_MUTE_THRESHOLD": 1.0}   # was code default 0.92
commit: none   # env-only
restart_at: 2026-10-04T03:25:00+02:00
---

## Window

First tick since 2026-09-25 (OAuth outage, see JOURNAL 2026-10-03). Ingest
5456-5543: 88 triggers across 09-26..10-03. Of those, 18 HUMAN (suppressed), 57 IDENTIFIED,
12 UNCLASSIFIABLE, 1 NO_ANIMAL. Human labels: 5 (all 10-02/10-03; 4 animal, 1 cant_tell).
The camera restarted 10-03 08:53 onto the decide() refactor, so rows 5520+ ran the new gate chain.

Tier-2 (22 rows, CLAHE-enhanced crops of the sharpest frame): every review-class
and every `;;;;;;animal` row. Animals: 5459 5462 5463 5470 5471 5483 5485 5489
5531 5533 (`;;;;;;animal`, blackbirds), and among review-class bursts 5468 5472 5477
5492 5495. False positive: 5488 5515 5516 5529. cant_tell: 5480 (dark shape behind
the tree trunk), 5501 (faint ghost, top right). Person: 5504 (NO_ANIMAL, legs at the
left edge, already human-proximity muted, which is correct).

## Finding: Confident-Blank is muting blackbirds

| id | time | blank score | sampled out | content |
|---|---|---|---|---|
| 5468 | 09-27 17:19 | 0.9366 | yes | blackbird, sharp, left of pond |
| 5477 | 09-29 08:43 | 0.9317 | yes | blackbird, sharp, by the gnome |
| 5495 | 09-30 17:43 | 0.9885 | no | blackbird in flight (blurred) and a second bird by the pond |
| 5516 | 10-02 14:52 | 0.9798 | no | empty |
| 5529 | 10-03 14:00 | 0.9709 | no | empty |

Lifetime: 19 mutes since 09-18. I re-inspected the 12 earlier ones whose frames
are still on disk (5425, 5432 have rotated out) with the same CLAHE crops. All of
them are genuinely empty, so the prior tier-2 `false_positive` labels stand.
Lifetime record: 16 empty, 3 animal (16%). All 3 animals arrived in the last week, in dim
light (`below_sharpness_floor=1`, dawn/dusk), which fits the season changing. At launch the
animal ceiling was 0.8475. It is now 0.9885, above both truly-empty rows in this window.

Run 0021's pre-registered response to a gate FN is: "raise the threshold strictly above
that row's score (in bounds), or disable if no in-bounds value would have prevented it." The gate's own margin rule
(max+0.02 = 1.0085) is out of bounds. The best in-bounds value is the ceiling, 1.0.
Disabling via `0` is human-only (CLAUDE.md), and at 1.0 the mute can only fire on a score that
is exactly 1.0.

## Gates

- FN-veto: passes by construction. Raising the threshold can only un-mute bursts.
- Volume: of the 16 empty lifetime mutes, 12 were not sampled out. Those would have
  been REVIEW sends, so the extra REVIEW volume is about 12 over 16 days (~0.75/day)
  for 1 recovered animal (5495). 5468 and 5477 would still have been dropped by the 0.5
  review sampling. No volume explosion: baseline is 27 triggers/night and this
  adds less than one REVIEW message a day.
- Freeze: not starved (human labels 10-03). Not paused. Slot was free (null).
- Review sampling is untouched (volume lever, not an FP lever).

## Rollback / next

Rollback: `loop.deploy --rollback` or redeploy `0.92`. Observation: 7 nights. The
gate is now effectively unexercised. If it records zero firings at
window close, PROTOCOL says retire (remove the code), not keep. That is the expected
conclusion. Watch REVIEW volume in the meantime, plus whether Daniel labels the
newly-unmuted bursts.

Separate observation, no action: 2 of the 3 animals would still have been lost to
review sampling (0.5). The other sent review-class animals tonight were 5472 and 5492, so REVIEW is
currently catching real blackbirds that MegaDetector misses in dim light.
Raising the sample rate on this FN evidence is in-bounds, but it is a separate experiment and
must wait for this slot to close.

## 2026-10-04 — night 1 (live since the 03:30 restart, verified via systemctl)

Ingest 5544-5563: 20 triggers, all daylight 09:46-14:47. 7 HUMAN (suppressed),
12 IDENTIFIED (blackbirds; Daniel labelled 6 of them `animal`), 2 UNCLASSIFIABLE
(both sampled out). fp_rate 0.077 [0.014-0.333] over 13 labelled; fp_human 0/6.

**Exp #40 had no firing opportunity.** Neither review-class row scored near the
old 0.92 line: 5554 raw `blank` 0.445, and 5552's raw top was `aves` 0.178. So
the change neither helped nor hurt tonight. Window continues (night 1 of 7).

Tier-2 (CLAHE crops): 5546 `person` (`;;;;;;animal` IDENTIFIED, raw top blank 0.72;
a knitted sleeve fills the right edge, already human-proximity muted, which is correct). 5552 `animal`:
a clear blackbird by the gnome, **lost to review sampling**. 5554 `false_positive`
(empty, pond spray running).

**Review-sampling FN evidence grows.** 5552 is the third sampled-out real
animal in 8 days (5468, 5477, 5552). Mitigation: 5552 sat between 5551 and 5553
(both IDENTIFIED blackbird, sent, human-labelled `animal`, 30s either side), so
Daniel still saw the visit. The sampling lost a photo, not the event. Queued as
backlog #41 for when this slot frees. PROTOCOL allows raising the rate on
genuine FN evidence, but one experiment at a time.

## 2026-10-05 — night 2

Ingest 5564-5568: 5 triggers, all in one blackbird visit 17:33-17:37 (dim, luma ~25,
all below the sharpness floor). Camera healthy (up since the 10-04 03:30 restart, warm-up
07:44, sunset stop 19:00, no errors). 0 HUMAN. 3 IDENTIFIED (`aves` bird) and 2 UNCLASSIFIABLE.
Daniel labelled 4 `animal` (5564 5565 5567 5568). fp_rate 0/5, fp_human 0/4.

**First counterfactual firing for exp #40, and it supports the change.** 5566 had raw top-1
`blank` @ **0.9875**, which is above the old 0.92 threshold. At the old setting the Confident-Blank gate would have
muted it. Tier-2 (CLAHE crops of frames 1/3/5) shows a **blackbird clearly visible**:
motion-blurred in frame 1 and sharp, perched left of the tree trunk, in frames 3 and 5. That makes
**4 animals at blank ≥0.93** (5468 0.937, 5477 0.932, 5495 0.989, 5566 0.988) against the
2 recent empties at 0.971/0.980, so the classes still interleave. With the threshold at 1.0,
`blank_confidence_muted=0` was correct.

But 5566 was still lost: **review sampling dropped it** (`[REVIEW-SAMPLE]`, rate 0.5). This is the 4th
sampled-out real animal in 9 days (5468, 5477, 5552, 5566). As with 5552, the visit itself
reached Daniel: 5565 (−115 s) and 5567 (+55 s) were sent and human-labelled `animal`. So
sampling lost a photo, not the event. 5564 shows the opposite case: an UNCLASSIFIABLE REVIEW send
that Daniel labelled `animal`. REVIEW keeps catching blackbirds that MegaDetector misses in dim light.

Decision: HOLD, night 2 of 7. The gate is now inert by design (no score can exceed 1.0). Retiring the code is
still the expected conclusion at window close, and tonight's row adds evidence for it. Backlog #41 is updated
with 5566. It still waits for this slot.

## 2026-10-06 — night 3

Ingest 5569-5572: 4 triggers. Camera healthy: up since the 10-04 03:30 restart, deploy unit not failed, no errors.
- 5569 (14:23) HUMAN, pc 0.84, suppressed `[HUMAN-GATE]`.
- 5570 (14:25) NO_ANIMAL, `[HUMAN-PROXIMITY]`-muted (window). Standing duty: tier-2 shows it is empty, so the mute was correct.
- 5571 (14:43) UNCLASSIFIABLE, sent to REVIEW.
- 5572 (17:36) IDENTIFIED `aves` blackbird, a clear bird left of the trunk, sent to MAIN.

No human labels today (the last ones were 10-05), so the loop is not starved. fp_rate 2/3, all on auto/tier-2 labels; fp_human n=0.

**Second counterfactual firing for exp #40, and this one went the other way.** 5571 had raw top-1 `blank` @ **0.9686**,
above the old 0.92 threshold. Tier-2 (CLAHE crops of frames 1/3/5) shows an empty garden with the fountain running.
The old gate would have muted it *correctly*. At 1.0 it was sent as a REVIEW message, so the change cost one extra
REVIEW ping. That matches the ~0.75/day volume prediction.

Window ledger after 3 nights: 1 animal recovered (5566 @0.9875, though it was then sampled out), 1 empty sent
(5571 @0.9686). Lifetime ≥0.92 scores: 4 animals (0.932-0.989) and 3 empties (0.9686-0.980). The classes still interleave,
so there is no case for a lower threshold. HOLD, night 3 of 7. The expected conclusion is unchanged: retire the gate code at window close.
5570 tier-2 `false_positive`, 5571 tier-2 `false_positive` (appended to detection_feedback).

## 2026-10-07 — night 4

**Zero triggers today.** Ingest past 5572 returned nothing. `loop.metrics` returned `no_data` and kept the 10-06 baseline.
I checked that this was a quiet day, not a blind camera:
- The camera has been up since the 10-04 03:30 restart, the deploy unit is not failed, and the logs show no errors. Sunrise warm-up
  ran 07:42-07:47 and sunset stop came at 18:56 ("0 detections today"). `motion_area` was 0 on all ~7.8k monitoring lines.
  The only `[DIAG-MOTION]` entries were sub-100 px contours.
- Timelapse (2013 frames today): the framing matches 10-06 (14:50 frames compared by eye), nothing blocks the lens, and luma moved
  normally through the day (24→70→1). So the camera was seeing a live, changing scene.
- `scripts/fn_audit_timelapse.py` over 10-03..10-07 (10k frames): none of the top-25 transient-object candidates is from 10-07.
  Every listed candidate sits next to a real trigger or is under 50 px. No animal that failed to trigger shows up.

**Exp #40: no firing opportunity** (no review-class rows). Window ledger unchanged: 1 animal recovered (5566) vs 1 empty
sent (5571). HOLD, night 4 of 7.

Starvation watch: the last human labels were on 10-05. If tomorrow also brings none, the 3-day feedback-starved freeze applies.
On a zero-trigger day there was nothing to label, so this is not a sign of Daniel disengaging.

Infra (no slot): `loop.report` was replaying the previous day's `last_metrics` as "Last night: N images" on `no_data` nights.
Fixed in 7727350: it now renders a no-new-images line instead.
