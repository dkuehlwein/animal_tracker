---
id: 21
slug: burst-human-sweep
status: running
validation: live   # shipped as a code change; measured against both real leak bursts before deploy
occupies_active_slot: true  # exp #18 (runs/0014) concluded this tick; this takes the slot
hypothesis: "Every human/privacy gate in this system judges exactly ONE frame per burst — the sharpest — but sharpness is uncorrelated with whether a person is visible. A burst whose sibling frames diverge from the selected one is not represented by that frame's verdict, so a person can be present, saved to disk, and sent to REVIEW without any gate ever scoring a person box. Re-identifying the most divergent siblings of a review-class burst closes the hole."
created: 2026-09-09
promoted_from: "found during night-5 tier-2 adjudication of exp #18: burst 5119 reached REVIEW with a recognisable child's face in its saved frames."
confidence: high   # root cause reproduced directly on the real frames, and the fix verified end to end on both leaks
commit: f73a8ae
restart_at: 2026-09-10T03:25:00+02:00
---

## The leak

**Burst 5119, 2026-09-09 14:20:50, `unclassifiable`, `person_confidence=0.0`,
sent to REVIEW.** Its saved frames contain a child at close range, face clearly
recognisable in frames 1 and 2.

Re-running SpeciesNet on each frame of that burst:

| frame | status | person_conf | Laplacian variance |
|---|---|---|---|
| frame1 | **human** | 0.894 | 13.41 |
| frame2 | **human** | 0.881 | 12.99 |
| frame3 | **human** | 0.032 | 12.80 |
| frame4 | **human** | 0.000 | 10.21 |
| frame5 | unclassifiable | 0.000 | **13.57 ← selected** |

Four of five frames classify as human. The one that does not won best-frame
selection by **13.57 vs 13.41** — a 1.2% sharpness margin — and that single
frame is the only thing the privacy gate ever saw.

Nothing downstream could recover. The person boxes were never scored, so the
burst never became HUMAN-status; and because **the entire day had zero
HUMAN-status rows**, the Human-Proximity gate (window and density), the
Deferred REVIEW Send gate and the human-adjacent retention purge were all
inert for want of an anchor. Blur gate: above floor. Scene gate: 0.747, far
under threshold. Sampling gate: not sampled out. Every layer of the stack was
functioning exactly as designed and the burst still went out.

**Burst 5096 the same morning (09:22:11, `no_animal`, pc 0.178) is a second
instance**: frame1 holds a leg in dark trousers at arm's length, the selected
frame is empty pond. Re-identified, frame1 returns `human` (via the
raw-classifier homo arm, pc 0.0). Daniel labelled 5096 `false_positive` from
the notification — correctly, for the frame he was shown. The human labeller
sees the same single frame the gate does, so this leak class is invisible to
human feedback too, which is why it survived two months.

Root cause, stated plainly: **the privacy gate's unit of analysis (one frame)
is smaller than the privacy risk's unit of analysis (one burst, five frames,
all retained on disk).** Nine gates deep, and the leak was in the sampling.

## The fix — burst human sweep

On a review-class result (`NO_ANIMAL`/`UNCLASSIFIABLE` — no animal found),
measure how far each sibling frame diverges from the selected one, then
re-identify the most divergent siblings and adopt the first result that fires
the human gate. Divergence = fraction of pixels differing by >40 levels on a
240x135 grayscale downsample (~1 ms/frame).

Adopting the swept result rather than bolting on a new flag is deliberate: all
downstream human handling is status-driven, so suppression, the `human` DB
status and the 48h photo-retention purge extend to the whole burst for free,
with no new column and no new precedence rule.

- `human_sweep_divergence_threshold` (default **0.03**, env
  `PERFORMANCE_HUMAN_SWEEP_DIVERGENCE_THRESHOLD`, bounds [0.0, 1.0])
- `human_sweep_max_frames` (default **2**, env
  `PERFORMANCE_HUMAN_SWEEP_MAX_FRAMES`, bounds [0, 4])

Either at 0 disables the sweep. Fails open (keeps the single-frame result) on
any error; the only possible effect is routing MORE bursts to HUMAN.

## Threshold: measured, not guessed

Divergence over **142 review-class bursts with frames still on disk**
(2026-09-04..09):

| rank | burst | divergence | contents |
|---|---|---|---|
| 1 | 5119 | 0.2146 | **person (face)** |
| 2 | 5096 | 0.1694 | **person (leg)** |
| 3 | 4987 | 0.0819 | empty, sun/shade |
| 4 | 4939 | 0.0673 | empty |
| 5 | 5070 | 0.0551 | empty |

Both leaks rank 1 and 2, above every empty burst in the corpus. `T=0.03`
keeps a >5x margin under both and sweeps 14/142 = **~10%** of review-class
bursts (~3–5 per night at ~10 s each). Rounding *down* is the safe direction
here — a lower threshold costs latency, not privacy — which is the opposite of
the scene gate's rule and worth not confusing.

Why divergence is the right trigger, structurally: it is not a person
detector. It measures whether the selected frame *represents* the burst. A
near-identical burst is one where the single-frame verdict does generalise; a
divergent burst is one where it demonstrably does not. That is exactly the
precondition for this leak class, and it is why the 29 correctly-caught
HUMAN-status bursts of 09-07/08 span the full divergence range (0.0001–0.35)
without that being a counter-example: in every one of them the person was on
the selected frame and was caught there.

## Cost and the FN gates

- **FN (animals):** the sweep only converts review-class → HUMAN. Review-class
  already means no animal was found on the selected frame, and the existing
  gate already gives HUMAN precedence over a confident animal. No animal alert
  that fires today stops firing.
- **FN (blind time):** ~10 s on ~10% of review-class bursts, ≈30–50 s per
  night against a ~10 h active window. Negligible, and bounded by
  `human_sweep_max_frames`.
- **Volume:** removes ~2 review-class sends/night out of 6–14. Not a collapse.
- Feedback-starved: no (18 human labels today). Paused: no.

## Verification

- 592/592 deterministic tests pass; 9 new ones cover divergence, escalation,
  the near-identical no-op, the max-frames cap, both rollback levers, missing
  `sharpness_info`, and end-to-end suppression + `human` DB status.
- The unreadable-frame test caught a real bug: importing SpeciesNet pulls in
  yolov5, which **replaces `cv2.imread` with a variant that raises on a
  missing path** instead of returning None. `_frame_divergence` now catches
  it, so one unreadable sibling drops itself from the sweep instead of
  aborting it.
- End-to-end against the real frames, real config, real model: burst 5119 →
  `human` (pc 0.894) and burst 5096 → `human`, each on its **first** swept
  frame. Both leaks would have been suppressed and their photos purged at 48 h.

## Scope deliberately not taken

The sweep runs on review-class bursts only, not on IDENTIFIED ones. An
IDENTIFIED burst with a person in a sibling frame still sends an animal alert
whose photo is the animal, so nothing recognisable is transmitted; only the
on-disk retention window is affected, which is second-order. Revisit if such a
burst is ever observed — tonight's four bird bursts had divergence
0.0010–0.0034 and would not have swept anyway.

## Prediction for night 6

≤1 person-in-non-selected-frame burst reaches REVIEW; sweep fires on ~10% of
review-class bursts; no change to animal alerts. A `[HUMAN-SWEEP]` log line
plus a `human`-status row with a sub-threshold `person_confidence` is the
signature to look for.

---

## Night 1 live (2026-09-10) — clean, but no positive test available

Deploy confirmed applied: `wildlife-deploy.service` restarted the camera at
**03:30:07** (`{"restarted": true, "reason": "applied deploy stamped
2026-09-10T03:25:00+02:00"}`), so f73a8ae has been live for the whole active
window.

**Night totals:** 8 triggers, all review-class (`no_animal`), 0 identified,
0 HUMAN-status, 0 below-floor, 0 scene-gate mutes, 5 sampled out, 3 sent.
Tier-2: 8/8 empty pond → `false_positive`. Zero human labels arrived today
(last human labels 2026-09-09, n=18) — not feedback-starved (needs 3 days).

**The sweep fired zero times, and that is the correct behaviour.** Recomputing
`_frame_divergence` offline over all 4 siblings of each of the 8 bursts gives a
per-burst maximum of **0.0000–0.0005** — 60x below `T=0.03`. No burst had a
candidate, so `_burst_human_sweep` returned before its first log statement.
Cost tonight: 0 extra identifications, 0 added blind time, 0 escalations, 0
false escalations.

**Liveness was verified without adding instrumentation.** A silent sweep and an
*inert* sweep look identical in the log, which matters for a privacy fix on its
first night. Resolved from artifacts already on disk rather than by adding a
counter: `sharpness_score` is non-NULL on all 8 rows, so `sharpness_info` was
non-None inside `process_detection`; `all_frame_paths` is set unconditionally in
the same dict literal that carries it (`wildlife_system.py:712-720`), and that
same object is threaded 1444 → 1458 → 985 → 375; and the offline recomputation
reproduces the silence exactly. The sweep ran on all 8 bursts and correctly
found nothing.

**No positive test was possible.** Nobody entered the garden today — zero
HUMAN-status rows — so the leak class the sweep exists for did not occur. The
experiment stays `running`; it needs a day with people in the garden before the
prediction above can be scored. Absence of a `[HUMAN-SWEEP]` line tonight is
absence of the *hazard*, not absence of the *fix*.

**FN audit (backlog #20 standing duty), 10k timelapse frames 09-06..09-10:**
exactly one 2026-09-10 candidate in the top 25 — 12:48:58, blob 2002 px at
(0.58, 0.13), coincident with trigger 5130 (+4 s). Inspected: the upper half of
the scene brightens between the 12:48:22 and 12:49:18 neighbours — a sun-out
illumination transition, the dominant false alarm this detector's own
calibration note warns about. **No missed animal today.** The zero-animal
reading for 2026-09-10 rests on a tested instrument, not on absence of evidence.

### Measured negative: intra-burst stillness is NOT an FP lever

Tonight's bursts were triggered by real motion (areas 885–2344, `diff_from_bg`
~30) yet were *static across the burst itself*, and capture latency rules out
"the subject already left" — frame 1 lands **205 ms** after the confirmed
detection. The tempting inference is that a burst whose frames don't move
contains no subject, and that intra-burst divergence — already computed for
free by exp #21 — is a cheap FP discriminator.

Measured before proposing, over all 279 on-disk bursts (max divergence of the
selected frame against its siblings, the same statistic the sweep thresholds):

| bucket | n | min | med | p75 | p90 | max |
|---|---|---|---|---|---|---|
| IDENTIFIED (animal) | 4 | 0.0118 | 0.0119 | 0.0131 | 0.0131 | 0.0131 |
| HUMAN status | 10 | 0.0001 | 0.0071 | 0.0341 | 0.0744 | 0.0744 |
| review-class, labelled `false_positive` | 180 | 0.0000 | 0.0014 | 0.0107 | 0.0233 | 0.2146 |
| review-class, unlabelled | 82 | 0.0000 | 0.0000 | 0.0001 | 0.0028 | 0.0139 |

The four confirmed blackbird bursts sit at **0.0118–0.0131**, i.e. between the
FP distribution's p75 (0.0107) and p90 (0.0233). There is no cut that mutes a
useful share of FPs without muting the only animal bursts this scene has ever
produced — the lever is FN-vetoed on measured data, not on assumption. Recorded
so a later tick does not re-derive it. (Note this table uses max-over-siblings;
the night-5 entry above quoted 0.0010–0.0034 for the same four bursts under a
different reduction. Both are far under `T=0.03`, so the sweep's threshold
margin is unaffected either way.)

Two second-order confirmations from the same table: the sweep's operating point
is ~10/180 ≈ **5.6%** of review-class bursts, matching the pre-deploy estimate;
and the two known person leaks (0.2146, 0.1694) still rank 1 and 2 corpus-wide,
above every empty burst — while both are *labelled* `false_positive` by the
human, because he judged the one frame he was shown. That is the leak class
being invisible to human feedback, exactly as documented above, and it is why
tier-2 and not the label column is the instrument for scoring this experiment.

**Decision: keep, unchanged.** No env delta, no code change, no restart stamped.

---

## Night 2 live (2026-09-11) — still inert, still no positive test; the tick's find was elsewhere

**Night totals:** 6 triggers, all review-class (4 `no_animal`, 2
`unclassifiable`), 0 identified, 0 HUMAN-status, 0 below-floor, 0 scene-gate
mutes, 2 sampled out, 4 sent. Tier-2: 6/6 empty pond → `false_positive`. Three
human labels arrived at 07:11 on 09-10 rows (5126, 5127, 5133 — all
`false_positive`), so the feedback channel is alive and the 3-day starvation
clock is reset.

**Sweep fired zero times, correctly.** Offline recompute of
`_frame_divergence` over all 4 siblings of each burst: per-burst maxima
0.0002–0.0128, every one under `T=0.03`. The closest (5134 at 0.0128) is a
sun/shade shift across the burst. Second night with no candidate; second night
with nobody in the garden. The experiment still has no positive test and stays
`running`.

**But one of tonight's bursts was a person-shaped burst after all — just not
this experiment's leak class.** Burst 5137 carried a homo-sapiens *raw
classifier* top-1 at 0.512 with a 0.291 person box and still routed to
`unclassifiable`. Its frames contain no person (adjudicated), so it is a
phantom, not a leak — but chasing why no human gate fired turned up that exp
#9's raw-homo trigger has been disabled since the day it shipped by a sentinel
misparse in its own guard. Opened and shipped as exp #23 (`runs/0016`, commit
479e0ac). Note what the sweep would have done had 5137 held a real person in a
sibling frame: divergence 0.0058, no candidate, no sweep. The two mechanisms
cover different failure modes and neither subsumes the other.

**FN audit (backlog #20 standing duty), 10k timelapse frames 09-07..09-11:**
exactly one 2026-09-11 candidate in the top 25 — 12:23:26, blob 897 px at
(0.37, 0.16), coincident with trigger 5135 (+4 s), already captured. No missed
animal today.

**Decision: keep, unchanged.** No env delta, no code change against #21.
