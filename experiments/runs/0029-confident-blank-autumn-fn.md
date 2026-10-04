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
