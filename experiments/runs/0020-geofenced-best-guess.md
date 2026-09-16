---
id: 28
slug: geofenced-best-guess
status: running
validation: live   # code change, restart-gated; caption text only — no routing, no FP/FN surface
occupies_active_slot: false  # notification-caption quality fix; the animal-bucket slot (exp #21) is unchanged
hypothesis: "The 'Best guess' caption line reads the classifier's RAW top-1 (metadata['top_classifier_prediction']) — the label before geofencing and before rollup. That is the one prediction we know is unfiltered for region, so on the rare bursts that actually contain an animal it renders either a species that cannot occur in Germany or a tautology of the verdict itself. metadata['best_geofenced_species'] — the first species-level candidate in the classifier top-k that SpeciesNet's own geofence allows in DEU/NW — is already computed on every identification and simply never shown. Preferring it turns the line from misinformation into the correct species at zero routing risk."
created: 2026-09-16
promoted_from: "tier-2 adjudication of 2026-09-16: bursts 5307/5308/5309, three Eurasian blackbird visits to the pond — the first real animals in weeks — each captioned with a wrong or empty best guess."
confidence: high   # measured directly against the three bursts' own saved frames; the fix is caption-only and fails open
commit: 838e5f6
env_delta: {}
restart_at: 2026-09-17T03:25:00+02:00
---

## Tonight — 11 triggers, and the animal bucket is no longer empty

| status | n | outcome |
|---|---|---|
| `identified` | **3** | MAIN alert, all three a real blackbird ✅ |
| `human` (suppressed) | 4 | all genuine people, no Telegram |
| `unclassifiable` | 3 | empty scene / bamboo |
| `no_animal` | 1 | empty scene / bamboo |

Eleven triggers is a quiet day (baseline 27, but 09-10 and 09-11 ran 8 and 6),
and every burst still has all frames on disk. All 11 were adjudicated.

**Three real animals, correctly alerted.** At 11:04, 11:04 and 11:06 a male
Eurasian blackbird (*Turdus merula* — yellow bill visible in 5309) worked the
gravel at the pond edge. All three routed to `identified` and reached MAIN.
Daniel had already human-labelled all three `animal` before this tick ran; my
independent tier-2 read agrees, which is the first direct human/tier-2 agreement
check on animal-class rows in weeks. **This is the first non-empty animal
bucket since the camera was re-aimed (exp #18), and it says the pipeline's
positive path works end to end.**

**Zero false negatives.** The four review-class bursts (5312, 5313, 5314, 5315)
are all wind in the bamboo and the pond surface — frame-differenced, the
largest moving blob in any of them is 936 px of leaf. 5313 was human-labelled
`false_positive` by Daniel; my tier-2 read of the other three agrees in kind.

**Zero privacy leaks.** All four `human` rows are genuine people at close range
(legs, trousers, a shirt) during two short garden visits. Two of them —
5310 at `person_confidence` 0.436 and 5317 at 0.199 — sit *below* the 0.5
MegaDetector threshold and were caught only by the raw-homo top-1 path
(exp #9/#23), which is exactly the demoted band exp #27 was built for. No
review-class burst tonight carried `person_confidence >= 0.3`, so **exp #27
shipped at 03:30 today and remains unobserved** — as does exp #26, which saw no
unnamed-animal row. Both carry forward.

## The defect: the right answer was computed and thrown away

The three blackbird alerts carried these captions:

| burst | verdict line | best-guess line as sent |
|---|---|---|
| 5307 | 🐦 bird (87%) | `Best guess: Blue whistling-thrush (42%)` |
| 5308 | 🐦 bird (79%) | `Best guess: Bird (55%)` |
| 5309 | 🐦 bird (83%) | `Best guess: Bird (40%)` |

*Myophonus caeruleus* is a Himalayan and Southeast Asian species. It cannot
occur in Bonn. The other two lines restate the verdict.

Re-running SpeciesNet in-tick against those three saved frames shows why, and
shows the fix was already sitting in the metadata dict:

```
5307  top-1              blue whistling-thrush      0.422
      best_geofenced     common blackbird           0.062   ← correct
5308  top-1              bird (generic rollup)      0.549
      best_geofenced     common blackbird           0.029   ← correct
5309  top-1              bird (generic rollup)      0.396
      best_geofenced     common blackbird           0.032   ← correct
```

`SpeciesIdentifier._find_best_geofenced_species` walks the classifier's top-k
and returns the first species-level candidate SpeciesNet's own geofence map
allows in DEU/NW. It runs on **every** identification and its result is stored
in `metadata['best_geofenced_species']`. `_best_guess_line` never read it.

Note the score inversion this exposes: the geographically correct answer sits
at 3–6% while an impossible congener sits at 42%. The classifier is not
region-aware; the geofence is the only thing that is. Ranking by raw score is
therefore the wrong selection rule for this line specifically, which is why the
ensemble rolls the verdict up to `bird` in the first place.

## Change

`_best_guess_line` (src/wildlife_system.py), commit `838e5f6`:

1. Candidate order is now `best_geofenced_species`, then the raw top-1 as a
   fallback when no in-region candidate exists (unchanged behaviour there).
2. A guess that merely repeats the ensemble's own rollup common name is
   suppressed ("Best guess: Bird" under a 🐦 bird verdict).
3. Everything else is untouched: same generic-sentinel suppression, same
   low-score-is-the-point display, same try/except so a caption error can never
   block a notification.

Why a code change and not an env delta: there is no knob here. The selection
source of a caption line is not a tunable parameter.

Why this passes the gates: it is **caption text only**. It changes no routing,
no mute, no threshold, no DB column, and no notification's existence — only the
words inside one already-sent message. FN-veto is trivially satisfied (the FN
surface is untouched); the volume guardrail is untouched; bounds do not apply.

Four tests added (geofenced-preferred, raw fallback when no in-region
candidate, tautology suppression, never-crash on a malformed
`best_geofenced_species`). Full suite: **630 passed**.

## Corpus context

Over the last 127 `identified` rows carrying a recorded `top_species_raw`:

- **83** would render the tautological `Best guess: Bird` under a `bird` verdict;
- **~5** name a species that cannot occur in Germany — blue whistling-thrush,
  chinese monal, black agouti (×2), american robin;
- the remainder either refine the verdict correctly (corvus species, domestic
  cat, common blackbird ×3) or carry no line.

So roughly 70% of the line's firings today are noise or wrong. I cannot
retro-verify what `best_geofenced_species` would have shown on those 83 rows —
their frames have rolled off disk and the column was never persisted — so the
evidence for the fix's *correctness* is tonight's 3/3, not the 83. That is a
small n, honestly stated. The direction, however, is not in doubt: a candidate
filtered by the region's own geofence cannot be worse than one that ignores it,
and when the top-1 is itself in-region and species-level the two sources return
the same prediction by construction.

## Prediction

The next burst whose ensemble verdict is a generic rollup shows a
regionally-possible species name, or no best-guess line at all — never a
Himalayan thrush and never "Best guess: Bird". Watch for it on the next
`identified` row; with animals this scarce that may take several nights.

Follow-up worth considering (NOT shipped): persist `best_geofenced_species`
alongside `top_species_raw` so this line's accuracy becomes measurable over
time instead of re-derivable only while frames survive on disk. Deliberately
not done tonight — one change at a time, and a new column is exactly the
instrumentation the protocol says to justify before adding.

## Other standing duties

- **Scene gate**: 0 mutes tonight, and 0 in 17 days. The three review-class
  bursts that got a similarity score ran 0.618–0.854 against T=0.97. Still
  inert per backlog #17; threshold not re-derived, per the 2026-07-26 override.
- **Blur gate**: 5314 and 5315 were below the sharpness floor and muted; both
  adjudicated, both empty. No animal concealed by the blur mute.
- **Sampling gate**: held at 0.5. Both sampled-out bursts (5312, 5315)
  adjudicated, both empty. No FN evidence, so no reason to raise the rate.
- **Human-proximity / deferral**: nothing muted; no review-class burst landed
  near a human burst tonight.
