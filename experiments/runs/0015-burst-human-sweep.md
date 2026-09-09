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
