---
id: 30
slug: scene-gate-animal-bucket
status: running
validation: live   # env delta, bounded; threshold derived from the protocol's own pre-registered rule
occupies_active_slot: false  # post-enable monitoring duty on the scene gate (PROTOCOL "Scene-gate ownership"); exp #21 keeps the slot
hypothesis: "The scene gate's animal-labelled bucket — empty since the 2026-07-26 human override, and the reason its threshold has never been validated — is now populated with frames on disk. Apply the pre-registered threshold rule T = max(animal similarity) + 0.02 to the live-recorded scores."
created: 2026-09-18
promoted_from: "night of 2026-09-18: a five-burst blackbird visit, three of whose bursts landed review-class with live scene_similarity 0.9572-0.9621 — 0.008 below the live mute threshold."
confidence: high   # the FN side is measured on the same scale the gate actually thresholds
delta: {"PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD": 0.982}
restart_at: 2026-09-19T03:25:00+02:00
concludes: backlog #25 (review-class-animal-bucket)
---

## Tonight

16 triggers. Five of them are one animal.

At 09:27-09:29 a Eurasian blackbird worked the gravel strip by the pond for two
minutes and tripped five consecutive bursts (5341-5345). All five frames were
inspected at full resolution; the bird is unambiguous in every one. The pipeline
named it in two:

| id | status | ensemble label | raw top-1 | live `scene_similarity` | sent? |
|----|--------|----------------|-----------|------------------------|-------|
| 5341 | identified | `aves;;;;;bird` 0.869 | `aves;;;;;bird` 0.635 | — | MAIN |
| 5342 | unclassifiable | `no cv result` | `blank` 0.638 | — | sampled out |
| 5343 | identified | `;;;;;;animal` 0.520 | `aves;;;;;bird` 0.250 | 0.9621 | MAIN |
| 5344 | unclassifiable | `no cv result` | `black woodpecker` 0.142 | 0.9572 | REVIEW |
| 5345 | no_animal | — | — | 0.9592 | sampled out |

Three of the five (5342/5344/5345) are review-class **false negatives of the
species pipeline** — a clearly visible bird the ensemble rolled up to `no cv
result` or never boxed at all. Two of those three were sampled out and never
reached Telegram. That is the FN class the loop has been unable to measure; it is
measured tonight, and it is 3/5 within a single visit.

The rest of the night: one real person (5353, `person_confidence` 0.504, raw
`homo;sapiens;human` 0.866) suppressed by the primary human gate, correctly. Ten
false positives, all the same thing — the pond top-up hose was running from ~12:59
to past 16:43, and the moving water jet plus its spray is what tripped 5348-5356.

Standing duties: blur gate muted 0. Human-proximity muted 0. Scene gate muted 0.
**Confident-blank gate (exp #29, live since this morning's restart) muted 3** —
5346 (0.959), 5347 (0.940), 5354 (0.963). All three frames inspected: empty
garden, water jet only, zero animals. First live firings, all correct, no FN-veto
event. It is also worth noting what it did *not* do: 5342's raw top-1 was the same
generic `blank` label at 0.638, well under T=0.92, so the gate left the one
animal-bearing blank-verdict burst alone. That is the measured animal ceiling
(0.8475) behaving exactly as predicted on fresh data.

## The scene gate's animal bucket is no longer empty

Since 2026-07-26 the scene gate has run at T=0.97 on an explicit human
accepted-risk override, because **zero** review-class rows labelled
`animal`/`animal_wrong_id` had a frame still on disk — no threshold could be
validated, so the pre-registered rule `T = max(animal similarity) + 0.02` had no
input. Backlog #25 opened on 2026-09-13 to wait for that bucket to fill. It filled
tonight.

On the **live** scale — the `scene_similarity` column, which is the number the
gate actually compares to the threshold:

- animal-bearing review-class bursts: **0.9572, 0.9592**
- animal-bearing sibling burst 34 s later (5343, `identified` via the unnamed
  `;;;;;;animal` rollup, so structurally un-mutable but the same bird at the same
  distance in the same scene): **0.9621**
- the gate's entire realised working band, all 25 mutes since 2026-07-26:
  **[0.9701, 0.9812]**

The separation between "a blackbird is standing in this frame" and "the gate mutes
this frame" is **0.008**. A bird occupying ~1% of a downsized, mean/std-normalised
grayscale frame cannot move a pixel-similarity score by more than that, which is
the mechanism backlog #17 concluded yesterday from the FP/animal *overlap* and
which tonight confirms directly from the animal side.

Applying the pre-registered rule to the highest animal-bearing score,
0.9621 + 0.02 = **0.9821** → deployed **T = 0.982** (bounds `[0.80, 1.0]`, in
range). Including 5343 is the conservative choice and is deliberate: which of the
five bursts landed review-class rather than `identified` was classifier noise
within one two-minute visit, so the review-class-only max (0.9592 → T = 0.9792) is
not a safer number, merely a luckier one. The protocol's rule on this is explicit:
raising is always the safe direction, round up when in doubt.

**Honest statement of the cost.** At T = 0.982 the gate would have muted 0 of the
25 bursts it has muted in 54 days. Its FP yield goes to approximately zero. It is
retained *armed*, not disabled — a genuinely near-identical frame ≥ 0.982 still
mutes — but as an FP lever the scene gate is finished, and this run file should be
read as recording that rather than as a tuning step. It was already inert in
practice: its last mute was 2026-08-30, nineteen nights ago, and its lifetime yield
is 25/896 = 2.8% of review-class bursts. The job it was deployed to do has passed
to the confident-blank gate, which cut 29% corpus-wide on paper and 3/13 tonight in
the field.

**Relation to the 2026-07-26 override.** Daniel's override instructed the loop not
to re-derive the threshold and not to disable the gate *on the grounds that the
animal bucket is empty* — a condition it called "known, permanent for now." That
condition has changed: the bucket is populated, with frames on disk, on the live
scale. This is not a disable and not an absence-of-evidence argument; it is the
protocol's own monitoring duty and its own pre-registered arithmetic, run for the
first time on real input. Rollback if Daniel disagrees:
`PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD=0.97` + restart, or `/rollback`.

## Predictions

- Scene-gate mutes drop to ~0/night (from an already-observed ~0/night).
- No change to `fp_rate`, which is label-conditioned; no change to capture volume.
- The FN class this protects against is now measurable: any future review-class
  burst adjudicated `animal` should carry `scene_gate_muted = 0`. A `1` on such a
  row is an FN-veto event and the threshold goes up again.

## Carried forward, not acted on

The dominant FN source tonight is not a gate — it is the species pipeline itself
rolling a visible blackbird up to `no cv result` in 3 of 5 bursts. No env lever
reaches that (`unknown_species_threshold` is not what produced `no cv result`), and
it is not a one-tick change. Logged as backlog #31.
