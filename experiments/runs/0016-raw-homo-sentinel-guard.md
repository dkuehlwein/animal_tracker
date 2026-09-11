---
id: 23
slug: raw-homo-sentinel-guard
status: running
validation: live   # code change, restart-gated; verified end-to-end on the real leak frames before deploy
occupies_active_slot: false  # defect repair of exp #9's shipped mechanism, not a new tuning lever — exp #21 keeps the slot
hypothesis: "Exp #9's raw-classifier homo-leak trigger has been inert since it shipped. Its guard `_is_specific_animal_taxon` tests genus/species segments for literal non-emptiness, but SpeciesNet writes the string 'no cv result' into EVERY taxonomy segment when the crop is unreadable — so the unclassifiable sentinel reads as a genus+species, the guard reports 'confident specific animal', and the trigger is disabled on precisely the label shape it was built for. Treating sentinel segments as empty restores the measured behaviour exp #9 shipped."
created: 2026-09-11
promoted_from: "found during night-2 tier-2 adjudication of exp #21: burst 5137 carried a homo-sapiens raw classifier top-1 at 0.512 and still routed to `unclassifiable`."
confidence: high   # root cause is a one-line predicate misread, reproduced on the real frames with the real model
commit: 479e0ac
restart_at: 2026-09-12T03:25:00+02:00
---

## What tonight's burst showed

**Burst 5137, 2026-09-11 15:15:21, `unclassifiable`, `person_confidence=0.291`,
`top_species_raw = ...;hominidae;homo;sapiens;human` @ 0.512.**

Three human-gate triggers exist and none fired:

| trigger | value | fires? |
|---|---|---|
| MegaDetector person box ≥ 0.50 | 0.291 | no |
| `homo` segment in the **ensemble** label | `...;no cv result;...` | no |
| exp #9 raw-classifier homo top-1 (`raw_homo_leak`) | raw top-1 **is** homo | **should have — did not** |

The third is guarded by `not _is_specific_animal_taxon(ensemble_prediction)`,
so a confident, specific animal ID is never overridden. The guard reads
`parts[-3]` (genus) and `parts[-2]` (species) and asks whether both are
non-empty. For the production unclassifiable label

```
f2efdae9;no cv result;no cv result;no cv result;no cv result;no cv result;no cv result
```

genus = `"no cv result"` and species = `"no cv result"` — both non-empty
strings. The guard therefore reports **a confident specific animal**, and the
raw-homo trigger is suppressed. The docstring already says the function must
return False for "an unclassifiable sentinel"; the code never did.

## This is not a new leak class — it is exp #9 never having worked

`SELECT ... WHERE top_species_raw LIKE '%homo%'` returns 5 rows corpus-wide:

| id | date | status | person_conf | raw score | ensemble shape | label |
|---|---|---|---|---|---|---|
| 1852 | 07-11 | unclassifiable | 0.000 | 0.474 | `no cv result` sentinel | — |
| 1988 | 07-13 | identified | 0.103 | 0.591 | `;;;;;;animal` | — |
| 2548 | 07-21 | unclassifiable | 0.059 | 0.573 | `no cv result` sentinel | **human `person`** |
| 4613 | 08-23 | unclassifiable | 0.268 | 0.499 | `no cv result` sentinel | — |
| 5137 | 09-11 | unclassifiable | 0.291 | 0.512 | `no cv result` sentinel | tier-2 `false_positive` |

Four of five carry the sentinel shape, including **2548 — one of the three
bursts named in exp #9's own hypothesis, and the only one Daniel labelled
`person`**. The fix exp #9 shipped on 2026-07-21 (c366087) would not have
caught its own motivating example. Exp #9 concluded on 07-27 with "the new
raw-homo top-1 trigger never fired (0 rows carry it)" and read that silence as
specificity. It was the guard misfiring, and two more sentinel-shaped bursts
(4613, 5137) have passed through since.

## The fix

`_is_specific_animal_taxon` now normalises sentinel segments to empty
(`no cv result`, `blank`) before the non-emptiness test. One predicate, no new
config knob, no new column, no precedence change. The never-override guard for
real animal IDs is untouched and still tested.

Second, observability only: a HUMAN result's metadata now carries the raw top-1
(`top_species_raw`/`top_species_score` are therefore written on HUMAN rows).
Without it a raw-homo suppression is indistinguishable in the DB from a
person-box one, and that is exactly the row a later tick must audit.

**Commit `479e0ac`**, restart-gated `2026-09-12T03:25:00+02:00`.

## Verification

- 595/595 deterministic tests pass; 3 new. The e2e regression uses the real
  7-segment sentinel label — the pre-existing exp #9 test used `"unclassifiable"`
  as a *single-segment* string, which passes the guard for the wrong reason
  (`len(parts) < 3`), which is why the bug survived a TDD ship.
- **End to end on the real frames, real config, real model:**
  `capture_20260911_151508_frame1.jpg` → `status=human`, conf 0.512,
  `person_confidence=0.291`. Pre-fix that same frame returned `unclassifiable`.

## Cost, measured

Precision is not perfect and the run file should say so. Of the four sentinel
rows this now converts to HUMAN: 2548 is a confirmed person; 5137 is **empty
pond** — adjudicated tonight, frames 1 and 3 show no one — i.e. a phantom homo
classification at 0.512. 1852 and 4613 have no frames left on disk.

So the trigger will occasionally suppress an empty burst. That is the cost exp
#14 named (phantom HUMAN rows arm the proximity window, the 1800 s density
counter and the deferral cancel), and it is worth paying here at a completely
different order of magnitude: exp #14 demoted **182** phantom HUMAN rows; this
fires on ~1 burst every 3–4 weeks. One phantom anchor per month can mute a
handful of review-class empty-pond bursts, which are false positives anyway.

Against that: a missed person is a recognisable face retained on disk through
the full `max_images` rotation and eligible for a REVIEW send. Asymmetric, and
the asymmetry runs the same direction as every other privacy call this loop has
made.

## Gates

- **FN-veto (animals):** raw-homo top-1 = 5 rows DB-wide, **zero real animals**
  (exp #9 measured this corpus-wide; tonight's addition is an empty frame, not
  an animal). The trigger cannot fire on a confident specific animal ID — guard
  intact, tested. An UNCLASSIFIABLE burst never fires a MAIN animal alert, so no
  animal notification that fires today stops firing. The residual cost is that
  such a burst no longer reaches REVIEW to be human-labelled `animal` — ~1 burst
  per 3–4 weeks of FN *detection* power, not of FN performance.
- **Volume:** ≤1 review-class send removed per month. Nowhere near the collapse
  floor (2.7 vs baseline 27).
- **Feedback-starved:** no — 3 human labels arrived today (07:11, on 09-10 rows).
- **Paused:** no. **Slot:** exp #21 keeps it; see front matter.
- **Confound with exp #21:** both route review-class bursts to HUMAN. They are
  separable in the record: the sweep logs `[HUMAN-SWEEP]` and escalates via a
  *sibling* frame, this one logs "Human detected via raw-classifier homo leak"
  and now writes a homo `top_species_raw` on the HUMAN row.

## Prediction for night 1 (2026-09-12)

Zero or one firing — this is a ~1-per-month path. The signature is a
`human`-status row with `person_confidence < 0.5` and a homo `top_species_raw`,
plus the raw-classifier-homo-leak log line. No change to animal alerts, no
change to REVIEW volume beyond at most one burst. If it fires, adjudicate the
frames: a person confirms the repair, an empty pond is a phantom to be counted
against the exp #14 ledger, not silently accepted.
