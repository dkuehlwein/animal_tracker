---
id: 29
slug: confident-blank-mute
status: running
validation: live   # code change; replayed against the shipped predicate over the full corpus before deploy
occupies_active_slot: false  # mechanism repair in the scene-gate family (backlog #17); exp #21 keeps the slot
hypothesis: "The Scene-Unchanged Gate asks 'do these pixels look like a recently-confirmed empty scene?' and that question does not separate animals from empty gardens. The classifier already answers a better one — 'is this frame blank?' — and its confidence in that answer does separate. Mute review-class bursts whose raw top-1 is SpeciesNet's generic blank verdict at high confidence."
created: 2026-09-17
promoted_from: "night of 2026-09-17: 21 of 23 triggers were wind in the bamboo, the scene gate muted zero of them, and measuring why produced both halves of this finding."
confidence: high   # FN side is measured against every animal-labelled row in the corpus, not inferred
commit: 3c2856c
restart_at: 2026-09-18T03:25:00+02:00
---

## Tonight

23 triggers. One real person (5322, `person_confidence` 0.663, close-up of a leg
and dark trousers) — caught by the primary human gate and suppressed, exactly as
designed. One phantom (5329, HUMAN status from the ensemble's `homo` taxon arm at
0.534 with `person_confidence` 0.0; all five frames are an empty garden). That is
backlog #16's known, ruled-on failure mode — **do not re-derive a score floor for
it**; the measurement that refuted the floor still stands, and tonight's phantom
cost nothing (the next review-class burst was 611 s later, outside the 240 s
proximity window, so it armed no mute).

The remaining 21 were all review-class and all the same thing: wind in the bamboo
canopy. Frame-differencing every frame of every burst against a median background
put the largest non-annotation blob at 507 px (5327 frame5) and 421 px (5318
frame4); both crops inspected at full resolution and both are leaves. Zero
animals tonight.

Standing duties: blur gate muted 0 (nothing below floor). Human-proximity muted 0.
Scene gate muted 0 — similarities 0.8299–0.9111 against T=0.97.

## The scene gate's discriminator, finally measured — and refuted

Backlog #17 has carried "the threshold is still unvalidatable" since 2026-09-04,
when `scene_similarity` began being recorded for **every** status precisely so the
animal side of the distribution could accumulate. It has now accumulated enough to
answer the question, and the answer is not the one the gate needs:

| bucket | n | min | median | max |
|---|---|---|---|---|
| human-labelled `animal`/`animal_wrong_id` | 10 | 0.7886 | 0.8759 | 0.9202 |
| human-labelled `false_positive` | 139 | 0.5335 | 0.8952 | 0.9675 |
| all review-class | 884 | 0.4817 | 0.8972 | 0.9812 |

The animal bucket sits **entirely inside** the FP bucket, and its median is
*below* the FP median. The pre-registered rule `max(animal)+0.02` would give
T=0.9402 — 0.02 above a measured real blackbird at 0.9202 and above four more at
0.9104–0.9170. Any threshold low enough to mute tonight's bamboo storm (≤0.911)
would sit inside the animal distribution.

This is the same shape as backlog #16: the signal is real, the classes are
interleaved, and no threshold fixes that. So T=0.97 stays exactly where the
2026-07-26 human override put it — safe and inert — and the question changes
from "which threshold?" to "which discriminator?". **Backlog #17's open question
is now answered by measurement, not left unvalidatable.**

## The discriminator that does separate

The classifier's raw top-1 is recorded independently of the rolled-up ensemble
label. On a burst MegaDetector finds no box in, it scores the whole frame, and on
an empty garden it returns SpeciesNet's generic `<uuid>;;;;;;blank` with a
confidence in that emptiness. Over all 182 review-class rows corpus-wide carrying
that label with a recorded score:

| bucket | n | scores |
|---|---|---|
| ever labelled `animal`/`animal_wrong_id` | 5 | 0.6431, 0.7454, 0.8023, 0.8297, **0.8475** |
| human-labelled `false_positive` | 42 | median 0.9219, max 0.9825 |
| ever labelled `person` | 7 | max **0.9116** (4915) |

At **T=0.92**: 52/182 muted (29%), and of the 29 muted rows carrying any label,
all 29 are `false_positive`. Zero animals. Zero people — the gate sits above the
highest person-labelled row too, so it can only help the privacy side. Margin over
the animal ceiling is 0.0725, 3.6× the protocol's pre-registered `max+0.02`.
Replayed once more against the *shipped* `utils.is_blank_label` predicate after
implementation, not just the scratch query: same 52, same zero.

Honest limits, stated rather than buried:
- n=5 on the FN side, and three of the five are July `animal_wrong_id` rows. The
  two recent ones (5176, 5214 — the blackbird by the gnome) score 0.6431/0.7454,
  well clear.
- It mutes **0 of tonight's 19** eligible bursts (tonight's blank scores top out
  at 0.917). This is not a fix for tonight; it is a corpus-wide 29% cut that
  tonight's data happened to fall just below. Claiming otherwise would be
  crediting it with a win it did not deliver.
- It shrinks human label supply further, on top of the 0.5 sampling rate. Per
  PROTOCOL that shrinkage is the intended effect of a volume lever, not a
  feedback-starved freeze.

## Why this is not a new experiment slot

Exp #21 keeps the active slot. This is a repair of the scene-gate mechanism's root
cause — the gate exists to mute "nothing is there" review-class bursts and tonight
it was measured incapable of doing so at any in-bounds threshold — in the same
family as backlog #17, and follows the precedent of #23/#26/#27 shipping as
mechanism repairs. The framing is recorded here so it is auditable rather than
assumed.

## Change

`3c2856c`, restart-gated 2026-09-18T03:25. `utils.is_blank_label` (sentinel
segments count as empty, per exp #23); `PerformanceConfig.blank_confidence_mute_threshold`
= 0.92, env `PERFORMANCE_BLANK_CONFIDENCE_MUTE_THRESHOLD`, `0.0` = disabled;
`blank_confidence_muted` BOOLEAN column on the initial INSERT; the gate inserted
between Blur and Scene in the notification chain. `loop.guardrails.BOUNDS` floors
the loop's own future deploys at **0.87** = `0.8475 + 0.02`, so this loop can
never tune itself down into a known false negative. 34 tests added, 664 pass.

Rollback: `PERFORMANCE_BLANK_CONFIDENCE_MUTE_THRESHOLD=0` + restart, or
`git revert 3c2856c`.

## What to watch

The gate's own mutes are now a nightly adjudication duty, same shape as the blur
and scene paths: every `blank_confidence_muted=1` burst gets inspected for a
concealed animal. A real animal in one is an FN-veto event — respond by raising
the threshold strictly above that row's `top_species_score` (in bounds), or by
disabling if no in-bounds value would have prevented it. On current data that duty
should fire roughly 1 burst per 3–4 nights.
