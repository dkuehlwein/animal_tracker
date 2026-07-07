---
id: 6
slug: blur-gate-false-negative
status: concluded         # proposed | running | concluded | rolled_back | parked
validation: live          # live | replay | parked
hypothesis: "The sharpness floor (min_sharpness_threshold=11.0) in _capture_and_select_best_frame silently discarded low-light/dusk captures — returning (None, None) before DB logging or notification — causing unrecorded false negatives. E.g. the 07-07 ~19:10 pond bird: motion detected 17x, all 4 captured bursts discarded (best-frame sharpness 8.6-9.4 < 11.0), zero DB rows, zero notification. Fix: below-floor bursts now flow to species ID + DB logging; alert if an animal is found, mute if not (so REVIEW volume doesn't rise)."
param_delta: null         # no env lever reaches this — the silent-drop was a control-flow bug (early return before DB write), not a threshold value; the floor (11.0) is unchanged, only what happens below it
predicted_effect: { fn_risk: "down — dusk/low-light animal captures that were previously silently dropped now reach species ID, DB logging, and (if an animal is found) the owner via Telegram", fp_rate: "unchanged — blurry bursts with no animal found are still not sent to Telegram, only DB-logged, so REVIEW-channel volume does not rise" }
created: 2026-07-07
decision: "Ship: below-floor bursts always get species ID + a DB row (never silently dropped). Notification layer alerts if an animal is found (even blurry), mutes if not (review-class status + below floor). Human-privacy gate (exp #5 / runs/0004) takes precedence — a blurry human burst is suppressed as HUMAN, not re-alerted as a blur-mute exception."
confidence: high          # mechanism confirmed in code (early return in _capture_and_select_best_frame preceded process_detection); triggering incident visually observed by Daniel same night
---

## Triggering incident (2026-07-07, ~19:10)

Daniel watched a bird bathing at the pond in person. The system's motion detector
fired **17 times** on the activity and captured **4 full 5-frame bursts** to disk —
but every burst's best-frame sharpness score (8.6-9.4) fell below
`config.performance.min_sharpness_threshold` (11.0, dusk lighting). Under the
pre-fix code path, `_capture_and_select_best_frame` returned `(None, None)` the
moment the sharpness check failed, **before** `process_detection` (species ID +
DB write) or any notification ever ran. Result: **zero DB rows, zero Telegram
notification** for an event the owner watched happen — a false negative invisible
to every downstream signal (metrics, loop tuning, human feedback), because there
was no row to label and no alert to miss. The burst frames themselves were the
only surviving evidence, and only because they hadn't yet aged out of the
`max_images` retention window.

This is the same class of failure the loop's own `PROTOCOL.md` warns about
generally ("FN structurally unmeasured" has held multiple experiments, e.g.
`runs/0003-roi-masking.md`) — except here the unmeasurability wasn't inherent to
the domain, it was a **specific control-flow bug**: a below-floor sharpness score
short-circuited the entire pipeline instead of just skipping the notification.

## Mechanism (confirmed in code, not assumed)

`src/wildlife_system.py::_capture_and_select_best_frame` (around line 253) captures
a burst, saves all frames, and picks the sharpest via
`SharpnessAnalyzer.select_sharpest_frame`. Previously, if the selected frame's score
was below `min_sharpness_threshold`, the method returned `(None, None)` and the
caller never invoked `process_detection` — no DB row, no notification, no trace.

The fix keeps the below-floor burst in the pipeline: it still returns a real
`best_frame_path` and a `sharpness_info` dict with a new `below_sharpness_floor`
flag, and logs (rather than aborts):

> "Best frame sharpness (8.6) below threshold (11.0) - processing anyway (blur gate
> no longer discards bursts)"

`_process_and_notify_detection` (around line 523) now always calls
`process_detection` — species ID + DB logging happen unconditionally for every
captured burst, sharp or not. Only the **notification** decision reads
`below_sharpness_floor`:

- `is_human` (human-privacy gate, `runs/0004`) is checked first and always wins —
  a blurry human burst is suppressed as HUMAN, not re-alerted as a blur exception.
- `is_blurry_review` = below the sharpness floor **and** the detection status is
  review-class (`NO_ANIMAL` / `UNCLASSIFIABLE`, i.e. no animal was found in the
  blurry frame). These are DB-logged but **not** sent to Telegram — this is what
  keeps REVIEW-channel volume flat despite every below-floor burst now reaching
  species ID.
- Otherwise (an animal *was* found, even in a blurry frame) — full notification
  path, exactly as any sharp-frame animal detection.

## Predicted effect on the loop's numbers

- **DB rows/day will rise, roughly ~2x.** Bursts that used to vanish silently
  (no row at all) are now logged every time, regardless of sharpness. This is a
  **measurement change**, not a behavior regression — the loop must not read a
  jump in `total_triggers` / DB row count as trigger-volume growth or a config
  anomaly.
- **Telegram notification volume will drop further**, compounding with the human
  suppression from `runs/0004`: blurry-no-animal bursts that previously would have
  become a `NO_ANIMAL`/`UNCLASSIFIABLE` REVIEW-tagged Telegram message (on the
  rare occasion they cleared the old sharpness floor) are now muted at the
  notification layer instead. Net effect on `fp_rate` (measured over sent/labeled
  notifications) should be flat to slightly improved — the denominator shrinks
  along with the numerator.
- **fn_risk goes down**, not up: this closes an unmeasured false-negative channel
  (dusk/low-light animals that used to be discarded with zero trace) without
  loosening any threshold that gates real animals away from the owner.

See `experiments/JOURNAL.md` (2026-07-08 entry) for the combined loop-facing note
covering both this fix and `runs/0004`'s human-suppression fix — the loop's
"stock config" volume/rollback baselines assume neither shift and must not
mistake either for an anomaly.
