# Telegram Feedback Label Redesign — Design

**Date:** 2026-07-09
**Status:** Approved (Approach A)

## Problem

The feedback keyboard has three buttons (✅ Animal / ❌ False positive / 🐦 Wrong species).
Labellers hit cases the vocabulary doesn't cover — right box but wrong animal, wrong box
but an animal is present, an animal with no box at all, a person in frame, a frame too
blurry to judge — and improvise. The known result (see memory note "wrong_species label
is heterogeneous"): `wrong_species` is used for both "human in frame" and "animal but
wrong ID", which muddies analysis. Metrically, `wrong_species` buys nothing today: the
tuning loop's headline fp_rate only distinguishes `false_positive` from everything else.

## Purpose of labels (decided)

Labels exist to (a) cleanly drive fp_rate (animal-or-not on the trigger axis) and
(b) surface false negatives (animal present but pipeline said NO_ANIMAL/UNCLASSIFIABLE).
Species-ID quality is kept as a single coarse bit (partially actionable via
`unknown_species_threshold` / geofencing). Box quality (localization) is deliberately
**not** tracked — not actionable, pure labeller burden.

Interaction budget: exactly one tap per notification, up to 5 buttons in 2 rows.

## New keyboard

```
Row 1:  ✅ Animal      🐦 Animal, wrong ID      👤 Human
Row 2:  ❌ Nothing there        🤷 Can't tell
```

Row 1 = "something real was there"; row 2 = "nothing / unusable". Spatial grouping does
the explaining.

### Stored labels and wire codes

| Button | Wire code | Stored label | Meaning |
|---|---|---|---|
| ✅ Animal | `a` | `animal` | Real animal, ID acceptable (or no ID offered). Covers wrong-box-but-animal and no-box-but-animal cases. |
| 🐦 Animal, wrong ID | `wid` | `animal_wrong_id` | Real animal, but the named species is wrong. |
| 👤 Human | `p` | `person` | A person is (part of) what triggered/appears. Signal for `human_detection_confidence` tuning (these are sub-threshold persons that slipped the gate). |
| ❌ Nothing there | `fp` | `false_positive` | No animal, no person. |
| 🤷 Can't tell | `ct` | `cant_tell` | Frame unusable (blur/dark); labeller explicitly declines to judge. |

Naming: the stored label is `person`, **not** `human` — `human` already means the
labeller tier (`source='human'`, per-tier `human` bucket) throughout the loop code.

Legacy: `ws` → `wrong_species` stays in `CODE_TO_LABEL` **parse-only** so buttons on
old messages still in the channel keep working; it is removed from the displayed
keyboard. Existing `wrong_species` rows are untouched (known-heterogeneous legacy).
No backfill/migration.

## Semantics in the tuning loop

### Reconciliation (`loop/ingest.py`)

Unchanged: `reconciled = human or tier2 or tier1`. All new labels are non-empty strings,
so a human `cant_tell` **wins** reconciliation — this is intentional: a human looked and
declared the frame unusable, so tier-2 (Claude) and tier-1 (MegaDetector) auto-labels
must not backfill it (memory note: auto-labels are not truth).

### Metrics (`loop/metrics.py`)

- **fp_rate denominator**: rows with `reconciled_label` not None **and not
  `cant_tell`**. `cant_tell` rows are excluded entirely — they are neither FP nor
  real-trigger evidence.
- **FP numerator**: unchanged — `reconciled_label == "false_positive"` only.
- `animal`, `animal_wrong_id`, `person`, and legacy `wrong_species` all count as real
  (non-FP) triggers. No other metrics change.
- **Per-tier partition**: a row whose winning label is `cant_tell` is skipped in
  `_per_tier_partition` (not counted in any bucket), preserving the documented
  invariant `n_human + n_claude + n_md == labeled_triggers` against the new
  denominator definition. (Only human-sourced rows can be `cant_tell`; the check may
  be applied uniformly anyway.)

### FN visibility

No new code needed: an `animal`/`animal_wrong_id` human label on a row whose
`detection_status` is `no_animal`/`unclassifiable` **is** the false-negative signal,
derivable by joining `detection_feedback` with `detections`. Surfacing it in the daily
report is out of scope for this change.

### Person signal

`person`-labelled rows joined with the `person_confidence` observability column give
the distribution needed to evaluate/tune `human_detection_confidence`. Query-time
analysis only; no code in this change.

## Changes by file

1. **`src/feedback_protocol.py`** — add `wid`/`p`/`ct` to `CODE_TO_LABEL`; keep `ws`
   parse-only; `_BUTTONS` becomes two rows; `build_feedback_keyboard` returns a 2-row
   `InlineKeyboardMarkup`.
2. **`src/telegram_feedback.py`** — `_LABEL_CONFIRMATION` entries for the new labels
   (keep `wrong_species` entry for legacy taps).
3. **`src/loop/metrics.py`** — exclude `cant_tell` from the labeled denominator and
   from per-tier buckets; update the invariant comment.
4. **`src/loop/ingest.py`** — no behavior change; extend the vocabulary comment.
5. **Tests** — `parse_callback_data` accepts new codes and legacy `ws`, rejects
   unknowns; keyboard is 2 rows / 5 buttons with correct callback_data; confirmation
   mapping covers all 5 + legacy; metrics: `cant_tell` excluded from denominator and
   buckets, invariant holds, `person`/`animal_wrong_id` count as non-FP.
6. **Docs** — CLAUDE.md: describe the feedback keyboard + label vocabulary (currently
   undocumented there); note the `cant_tell` metrics exclusion.

## Out of scope

- Box-quality / localization labels.
- Two-tap flows or free-text species correction.
- Backfill of legacy `wrong_species` rows.
- Daily-report FN line from status×label joins (future).
- Tier-2 (Claude) auto-label vocabulary changes.
