---
id: 10
slug: review-volume-reduction
status: running           # deployed 2026-07-26 by human directive, outside a nightly tick
validation: live          # scene gate ENABLED T=0.97 + review sampling 0.25
occupies_active_slot: false  # NOT a loop experiment — a human-directed config change.
                             # exp #9 (runs/0008) keeps active_experiment_id; the
                             # "one active experiment at a time" rule is not in play here.
                             # The loop's duty for THIS doc is monitoring, not arbitration.
hypothesis: "REVIEW-channel notification volume (~44/night) is not worth its false-negative yield: over 2026-07-12..26 it cost ~618 REVIEW messages to surface 4 missed animals (~1 catch per 155 pings). Cutting volume ~6x via two independent, already-built mute paths (scene-unchanged gate at T=0.97, plus a deterministic 1/4 sampling gate) restores a manageable channel. Accepted risk: FN detection power falls with label supply; the post-enable adjudication duty is the compensating control."
created: 2026-07-26
promoted_from: "Daniel, direct instruction 2026-07-26 — not loop-originated. Supersedes the scene-gate FN-veto HOLD recorded in PROTOCOL.md 'Scene-gate ownership'."
confidence: medium        # volume effect is well-estimated; FN cost is explicitly unmeasured by construction
---

## Origin — human directive, not a loop decision

This change did **not** come from a nightly tick and did not pass the FN-veto
gate. Daniel asked for it directly on 2026-07-26 after reviewing the two-week
REVIEW-channel numbers below, and explicitly accepted the FN risk. Recorded
here so the loop treats it as settled prior art rather than re-litigating it.

See `PROTOCOL.md` → "Scene-gate ownership" (SUPERSEDED 2026-07-26 block) and
"Review sampling".

## The measurement that motivated it (2026-07-12 .. 2026-07-26, 14 days)

| quantity | value |
|---|---|
| review-class bursts (`no_animal` + `unclassifiable`) | 679 (613 + 66) |
| actually sent to Telegram | 618–679 (~44/night; peak 112 on 07-23) |
| human-labelled | 128 (20.7% of sent) |
| → `false_positive` | 107 |
| → `person` (human-gate misses) | 18 |
| → **`animal_wrong_id` = false negatives** | **4** |
| → `animal` | 0 |

The 4 FNs: ids 2011 / 2025 / 2028 (all 2026-07-14) and 3116 (2026-07-24). All
`animal_wrong_id` on `no_animal` rows; 3 of 4 are a single-day cluster. Frames
are all past retention, so none could be re-inspected. For contrast the MAIN
channel logged 75 `identified` bursts with 33 human `animal` confirmations over
the same window — the review channel is a marginal FN-catcher, not the primary
one.

Note the "sent" figure is a range, not a point: `below_sharpness_floor=1` does
not imply muted (exp #8's `blur_mute_min_luma=70` un-mutes dark below-floor
bursts), and luma is not persisted, so the exact send count is unrecoverable
from the DB. 618 is the lower bound.

## Change 1 — scene gate ENABLED at T=0.97

Validator re-run 2026-07-26 (`scripts/validate_scene_gate.py`): 35 on-disk
review-class frames scored, **all still `unlabeled`** (`human_animal` n=0,
`human_fp` n=0). Distribution: min 0.7376 / median 0.9534 / max 0.9793.

Mute yield by threshold on that corpus:

| T | muted |
|---|---|
| 0.950 | 20/35 (57%) |
| 0.960 | 16/35 (46%) |
| 0.965 | 10/35 (29%) |
| **0.970** | **6/35 (17%)** |
| 0.975 | 2/35 (6%) |
| 0.980 | 0/35 (0%) |

`T=0.97` is the pre-registered placeholder — not a number invented to fit. It
sits just under the observed ceiling (0.9793), so only near-identical frames
mute, and it is the conservative end of the range that still does real work.
The locked `max(animal similarity) + 0.02` rule remains uncomputable (empty
animal bucket) and is formally superseded, not satisfied.

Low-texture diagnostic re-checked: 0 flagged pairs (both frames std < 5.0);
frame std min 34.6 / median 55.9. The daytime-only coverage gap from Task 5
persists — the scored corpus is 2026-07-25..26 daytime frames, so dusk/dark
behaviour is still unexercised. Weighted conservative accordingly.

## Change 2 — review sampling at 0.25

New `PERFORMANCE_REVIEW_SAMPLE_RATE` (default 0.25, in `BOUNDS` [0.0, 1.0]).
Deterministic selection: SHA-256 of the detection id → first 8 bytes → uint64 →
/2**64, send iff below the rate. Deterministic rather than `random` so the
decision is reproducible and unit-testable, and so re-running analysis over the
corpus reproduces exactly which bursts were sent.

Precedence is now **Human > Blur > Scene > Sampling**; sampling only ever
applies to review-class statuses and fails open (sends) on a missing detection
id. Suppressed bursts are still species-ID'd and DB-logged with the new
`review_sampled_out` column — the corpus is unaffected, only the inbox.

## Predicted effect

679 review-class/14d → scene gate −17% → ~563 → sampling ×0.25 → **~141 over
14 days ≈ 10/night**, down from ~44/night. Roughly a 4.4x reduction.

Both gates are notification-layer only. `fp_rate` is label-conditioned and
should be **unchanged by construction** — if it moves materially, that is a
bug, not a win, and should be investigated as such.

## Accepted risk + compensating control

FN detection power drops with label supply. At full volume the channel yielded
4 FNs/14d; at ~1/4 sampling the expectation is ~1/14d, and the scene gate
removes a further slice. **FN measurement on the review channel is now weak by
design.** This was Daniel's explicit call.

Compensating control — the `scene_gate_muted=1` adjudication duty in
PROTOCOL.md is now the primary safety net rather than a secondary check, since
no pre-validation exists. Every muted burst gets adjudicated each tick for a
concealed animal; a hit is an FN-veto event handled the same tick (raise `T`
above that frame's recorded `scene_similarity`, or disable). Sampled-out bursts
retain their frames on disk for the normal retention window and are equally
adjudicable via `review_sampled_out=1`.

## Exit criteria

- **Keep** if REVIEW volume lands near the ~10/night prediction and adjudication
  of muted/sampled-out bursts turns up no concealed animals over ~1 week.
- **Raise the rate / raise T** on any positive FN evidence (a real animal found
  in a `scene_gate_muted=1` or `review_sampled_out=1` burst). This is the
  correct response to evidence — it is *not* the correct response to merely
  having fewer labels, which is the intended effect.
- **Rollback levers**: `PERFORMANCE_REVIEW_SAMPLE_RATE=1.0` and/or
  `PERFORMANCE_SCENE_GATE_ENABLED=0`, + service restart. Both independent.

## Observations

_(pending first night post-deploy)_
