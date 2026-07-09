---
id: 7
slug: dusk-short-exposure
status: concluded         # proposed | running | concluded | rolled_back | parked
validation: live          # live | replay | parked
hypothesis: "Auto-exposure picks long exposures at dusk/low light to compensate for reduced illumination, and long exposure blurs any motion in frame — this is the root cause behind the sub-11.0 sharpness scores that runs/0005 found (07-07 8.6-9.4, silently dropped; 07-08 19:33-19:35 10.0-10.4, now alerting but still marginal). Biasing auto-exposure toward shorter exposures at dusk (accepting more sensor noise/gain in exchange) should raise dusk-hour sharpness scores above the 11.0 floor without materially changing midday exposure behavior (midday light is already bright enough that AE isn't choosing long exposures)."
param_delta: { "CAMERA_AE_EXPOSURE_MODE": "unset (libcamera default, effectively \"normal\") -> \"short\" (shipped as the new CameraConfig default)" }
predicted_effect: { fn_risk: "down — dusk/low-light animal captures should score further above the sharpness floor, so fewer end up in the below_sharpness_floor + no-animal-found mute path, and the ones that do alert do so with a sharper (more useful) photo", fp_rate: "no direct effect — this changes auto-exposure behavior, not motion detection or species classification; expected flat" }
created: 2026-07-09
decision: "Ship CameraConfig.ae_exposure_mode default \"short\" (env CAMERA_AE_EXPOSURE_MODE, validated normal|short|long), applied only in auto-exposure mode via libcamera's AeExposureModeEnum, as a code default rather than a require-Daniel-to-opt-in env var — the mechanism (long AE exposure -> blur -> sub-floor sharpness) is a direct, confirmed-in-code cause of two already-observed false-negative-adjacent incidents (07-07 silent drop, 07-08 marginal below-floor alerts), and the fix is reversible with a single env var + restart if it misfires."
confidence: medium        # mechanism is a well-known camera/AE tradeoff and matches both dusk incidents' timing, but the actual sharpness-score shift has not yet been measured on live dusk data — the Task 1 sharpness_score/below_sharpness_floor DB columns needed to measure it only start recording from 2026-07-09 onward
---

## Motivating evidence (from `runs/0005-blur-gate-false-negative.md`)

Two dusk/low-light incidents, both traced to below-floor best-frame
sharpness, both pond-bird activity around the same hour:

- **2026-07-07 ~19:10**: 4 captured bursts, best-frame sharpness **8.6-9.4** vs
  the 11.0 floor. Under the pre-`runs/0005` code these were silently dropped —
  zero DB rows, zero notification (the false negative that forced the blur-gate
  fix in the first place).
- **2026-07-08 19:33-19:35**: 4 more pond-bird bursts, best-frame sharpness
  **10.0-10.4** — closer to the floor than 07-07 but still below it. With the
  `runs/0005` fix live, all four now alert (detections 1738-1741) instead of
  vanishing. But the underlying scores are still marginal: a slightly darker
  evening, or a slightly faster-moving animal, would put the next batch of
  dusk captures back below the floor.

Both incidents share a mechanism, not just a symptom: dusk light is too dim
for the camera's auto-exposure to hold a short exposure at a usable
brightness, so it lengthens exposure time instead — and a longer exposure
integrates more motion blur from any moving subject (a bird bathing, in
both cases). The blur-gate fix (`runs/0005`) makes the *consequence* (a
below-floor burst) survive to the DB and, if an animal is found, to
Telegram. It does not touch the *cause*. This run addresses the cause.

## Mechanism (confirmed in code)

`src/camera_manager.py::PiCameraManager._initialize_camera` only sets
`ExposureTime`/`AnalogueGain` controls when both are configured (manual
exposure mode); when either is `None` (the current production config — see
CLAUDE.md "auto-exposure enabled"), the auto-exposure algorithm is free to
choose its own exposure time based on scene brightness, with no bias toward
short vs. long exposures. libcamera exposes `AeExposureMode` (`Normal`,
`Short`, `Long`) precisely to bias that choice, trading exposure time against
sensor gain/noise.

The fix (`_apply_ae_exposure_mode`, only reached in the auto-exposure branch)
sets `controls["AeExposureMode"]` to the enum mapped from
`config.camera.ae_exposure_mode` (new field, default `"short"`). Wrapped so a
missing `libcamera` module or an unsupported/renamed enum member logs a
warning and leaves `controls` unmodified rather than raising — camera startup
must never crash on this (`Global Constraint`).

## Predicted effect on the loop's numbers

- **Dusk-hour `sharpness_score` values should rise.** This is now directly
  measurable, for the first time, via the Task 1 DB columns
  (`sharpness_score`, `below_sharpness_floor`) which start recording from
  2026-07-09 onward (NULL before). Compare the dusk-hour (roughly 19:00-20:30
  local, matching both incidents above) distribution of `sharpness_score`
  before vs. after this deploy once enough post-2026-07-09 dusk data
  accumulates — a rise, and fewer/no `below_sharpness_floor=1` rows at dusk,
  is the win condition. **A rise in dusk sharpness is the expected, intended
  effect of this fix — the loop must not read it as an anomaly.**
- **Midday sharpness should be flat.** Auto-exposure at midday brightness is
  not exposure-time-constrained the way dusk is, so `AeExposureMode=Short`
  should have little to no effect on already-short midday exposures. A
  midday `sharpness_score` regression (scores dropping) would suggest the
  short-exposure bias is trading away too much brightness/gain even in good
  light, and would be forcing evidence to reconsider or narrow the fix (e.g.
  time-of-day-gated rather than always-on).
- **No expected effect on `fp_rate`** — this changes camera exposure
  behavior, not motion detection thresholds or species classification, so
  the FP/animal entanglement conclusions from exp #3/#4 (ROI, motion
  magnitude) are unaffected.
- **Photo usability should improve** even for below-floor alerts that still
  occur: a sharper (even if still-below-floor) photo is more useful to
  Daniel and to future tier-2 visual adjudication than the 07-07/07-08
  frames were.

## Success metric

Dusk-hour (approx. 19:00-20:30) `sharpness_score` distribution shifts above
`min_sharpness_threshold` (11.0) post-deploy, with **no midday regression**
in the same column. This is a data check for a future loop tick once enough
post-2026-07-09 dusk-hour rows exist — not something this run doc can confirm
at write time, since the observability columns needed to measure it are only
now coming online (see `experiments/JOURNAL.md`, 2026-07-09 entry).

## Rollback lever

`CAMERA_AE_EXPOSURE_MODE=normal` + `sudo systemctl restart
wildlife-camera.service`. Single env var, no schema/code rollback needed —
`ae_exposure_mode` is validated to `normal|short|long` and defaults to
`"short"` only in code; setting it back to `"normal"` restores libcamera's
un-biased auto-exposure behavior on the next restart.
