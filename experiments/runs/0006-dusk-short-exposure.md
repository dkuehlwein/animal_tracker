---
id: 7
slug: dusk-short-exposure
status: rolled_back       # proposed | running | concluded | rolled_back | parked
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

---

## Resolution — 2026-07-09 loop tick: ROLLED BACK (same day as ship)

**Verdict: the fix cannot achieve its own success metric, and it moves the
gating statistic the wrong way. Reverted to `CAMERA_AE_EXPOSURE_MODE=normal`.**

This run's success metric was "dusk-hour `sharpness_score` shifts above the
11.0 floor, with no midday regression." Both halves failed on first
measurement, and the first half fails *structurally*, not just empirically.

### What was measured

The observability columns only start at 2026-07-09, so the pre/post comparison
was done offline against **saved frames** (`data/images/`, 537 burst frames
covering 07-07, 07-08 and 07-09 — AE flipped to `short` at the 15:49 restart).
Method validated first: recomputing Laplacian variance on the 12 rows that DO
have `sharpness_score` reproduced the stored value within ±0.3 (id 1786 ±2.2).

Per-hour medians (`lap` = Laplacian variance = the production `sharpness_score`;
`luma` = mean grayscale level):

| day | hour | AE | n | lap | luma |
|---|---|---|---|---|---|
| 07-08 | 15 | normal | 10 | 19.62 | 113.1 |
| 07-08 | 16 | normal | 52 | 19.18 | 112.9 |
| 07-08 | 19 | normal | 24 | 10.19 | 68.9 |
| 07-09 | 15 | short | 10 | 18.14 | 100.3 |
| 07-09 | 16 | short | 17 | 15.07 | 100.7 |
| 07-09 | 17 | short | 23 | 8.64 | 77.3 |
| 07-09 | 18 | short | 12 | 7.83 | 65.7 |
| 07-09 | 19 | short | 5 | 9.76 | 52.0 |

1. **AE=short is genuinely applying.** `data/logs/wildlife.log` shows
   `Auto-exposure mode: short` at both restarts (15:49:40, 16:56:46), no
   warning. Pixels agree: at the matched 19h hour, luma fell **68.9 → 52.0**.
   Shorter exposure ⇒ darker frame. The mechanism works.
2. **`sharpness_score` did not rise at dusk — it fell** (19h: 10.19 → 9.76),
   and post-deploy 17:30–18:49 sat at **7.1–9.1**, i.e. *below* the 11.0 floor
   for 6 of the 12 post-deploy rows.
3. **The pre-registered midday-regression trigger fired.** 16h: 19.18 → 15.07.
   This run doc committed in advance to treating that as "forcing evidence to
   reconsider or narrow the fix."

### Why this was doomed by construction (the important part)

`sharpness_score` is **Laplacian variance**, a whole-frame *contrast* statistic.
Scale an image's luminance by k and its Laplacian variance scales by ~k². So
`sharpness_score` is a function of scene brightness, not only of blur.

`AeExposureMode=Short` **lowers scene brightness by design**. Therefore it
*necessarily* lowers `sharpness_score` at any given ambient light level. The
fix and its success metric point in opposite directions: the run proposed to
raise a number by applying a change that mathematically lowers it.

Supporting (weaker) evidence: binning all frames by luma and comparing at
matched brightness, AE=short frames had *lower* lap in every bin
(13.4→9.8, 10.2→8.6, 20.2→12.7, 20.2→15.3) — the opposite of what reduced
motion blur would predict, and consistent with the ISP applying stronger
noise reduction at the higher analogue gain that AE=short forces. **Caveat,
stated plainly: every matched-luma bin is confounded by time-of-day and
subject, `n_short` is 67 frames from a single afternoon, and conditioning on
luma conditions on a mediator of the treatment. This bin comparison is
suggestive, not decisive.** The rollback does not rest on it — it rests on
(2), (3) and the structural argument above.

Also observed while adjudicating: the confirmed-FN frames (id 1718) are
*uniformly* soft across the entire scene, foreground and background alike.
That is focus/contrast, not motion blur — further evidence that a global
Laplacian floor is not measuring the thing this run assumed it measured.

### Why roll back rather than leave it and keep watching

The blur gate (`runs/0005`) mutes a burst that is below the sharpness floor
**and** has no animal found — DB-logged, no Telegram. By depressing
`sharpness_score` across the board, AE=short makes that mute path strictly
more reachable. A real animal the classifier misses in a darker frame is then
silently dropped: exactly the false-negative class `runs/0005` existed to
close. `fn_rate` is still `"unmeasured"` (`loop.metrics` only computes it from
an `fn_audit` timelapse pass that is not implemented).

Guardrail contract: *"if FN is unmeasured and the change could plausibly raise
FN, HOLD."* AE=short is such a change. FN-veto applies → rollback.

Secondary benefit: we have **zero** AE=normal frames at hours 17–18 (retention
keeps only ~100 bursts and is already at cap), so the matched-hour baseline
this run needs to be evaluated properly does not exist and cannot be recovered
retroactively. Running AE=normal collects it.

### What was NOT disproven

Whether AE=short actually reduces *motion blur* is still unknown. Laplacian
variance cannot separate blur from contrast, so this run's instrument was
never able to answer its own question. If AE=short is revisited, it needs a
blur-specific, brightness-invariant measure, and it should not be evaluated
against a raw-Laplacian floor.

### Action taken

- `CAMERA_AE_EXPOSURE_MODE=normal` appended to `.env` (the rollback lever this
  run doc specifies). **`.env` is gitignored**, so that edit is not visible in
  git — it is recorded here and in `JOURNAL.md` deliberately. Backup at
  `.env.bak.20260709`. `Config().camera.ae_exposure_mode` verified == `normal`.
  This was NOT routed through `loop.deploy`: `guardrails.BOUNDS` holds only
  numeric `(low, high)` ranges and rejects `CAMERA_AE_EXPOSURE_MODE` as "not a
  tunable parameter", so `state.deployed` stays `{}` and no
  `deployed_config.env` is rendered. (Teaching a string/enum param to
  `BOUNDS` is a guardrail change and was out of scope mid-tick.)
- `pending_restart_at = 2026-07-10T03:00:00+02:00`; `wildlife-deploy.timer`
  fires 03:30 CEST and `apply_pending_deploy` restarts on any due stamp,
  independent of whether a delta was rendered.
- Status → `rolled_back`; reopened as backlog id 7 `running` to gather the
  17–19h AE=normal baseline.
- New backlog id 8 `sharpness-floor-is-a-brightness-gate` filed as the actual
  root cause (see below).

### The finding worth keeping (unconfounded)

Over 470 pre-deploy frames (AE=normal, multiple days), the probability a frame
falls below the 11.0 floor is a step function of brightness alone:

| luma bin | P(lap < 11.0) | n |
|---|---|---|
| 0–40 | 100.0% | 100 |
| 40–60 | 0.0% | 2 |
| 60–80 | 71.4% | 28 |
| 80–100 | 0.0% | 60 |
| 100–130 | 0.0% | 278 |

`min_sharpness_threshold` is, operationally, a **light-level gate**. At dusk
almost everything is "below floor" no matter how sharp it is. That — not the
auto-exposure mode — is what governs whether a dusk burst can be silently
muted, and it is what backlog id 8 should fix.
