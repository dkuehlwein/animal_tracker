---
id: 8
slug: sharpness-floor-is-a-brightness-gate
status: running          # proposed | running | concluded | rolled_back | parked
validation: parked        # live | replay | parked  (design/data phase — not yet deployed)
hypothesis: "min_sharpness_threshold=11.0 is applied to raw Laplacian variance, a whole-frame contrast statistic that scales with scene brightness. So the 'blur gate' is operationally a LIGHT-LEVEL gate: at dusk nearly every burst is below-floor regardless of actual motion blur. Because the blur-gate MUTE path (below_sharpness_floor AND no-animal-found -> no Telegram) is therefore reachable mainly as a function of darkness, real dusk animals the classifier misses are silently dropped — the exact unobservable-FN class runs/0005 existed to close, reopened by the dusk data. Fix: stop letting darkness alone trigger the mute."
created: 2026-07-10
promoted_from: "backlog id 8 (filed 2026-07-09 at runs/0006 rollback); now the active experiment after exp #7 (AE mode) concluded — AE bias is not the lever, the floor statistic is"
confidence: high          # the brightness->floor relationship is a measured step function over 470+ frames across multiple days and both AE modes; the mechanism (Laplacian variance ~ luminance^2) is analytic
---

## Why this is now the active experiment

Exp #7 (`runs/0006`) tested whether biasing auto-exposure shorter would lift
dusk sharpness above the 11.0 floor. It cannot, and was rolled back; the
reopened AE=normal baseline (07-10, 17–18h sharpness 6.9–9.4) confirmed AE mode
is indistinguishable from AE=short at dusk. **The floor statistic, not the
exposure mode, decides whether a dusk burst is muted.** That is this experiment.

## The measured problem (unconfounded, from `runs/0006` + 07-10 data)

P(Laplacian variance < 11.0) as a function of frame luma, over 470 pre-deploy
AE=normal frames spanning multiple days:

| luma bin | P(lap < 11.0) | n |
|---|---|---|
| 0–40   | 100.0% | 100 |
| 40–60  | 0.0%   | 2 |
| 60–80  | 71.4%  | 28 |
| 80–100 | 0.0%   | 60 |
| 100–130| 0.0%   | 278 |

Above luma ~80 essentially nothing is below floor; below luma ~60 essentially
everything is. `min_sharpness_threshold` is a **light-level gate**. The floor
was intended to catch motion blur; it actually catches darkness.

## Why it matters (the FN this reopens)

The blur-gate mute path is `src/wildlife_system.py::_should_send_notification`
→ `is_blurry_review`: suppress Telegram when `below_sharpness_floor` AND the
status is review-class (no animal found). At dusk `below_sharpness_floor` is
true almost unconditionally, so any dusk burst the classifier fails to find an
animal in is silently dropped. If the classifier missed a real animal in a dark
frame, that is an **unobservable false negative** — muted means never labelled.

07-10 window: the mute path fired 4× (ids 1787 morning, 1845/1846/1848 dusk),
all below-floor no-animal/unclassifiable. One below-floor *animal* (1847,
18:23) alerted correctly — the gate does the right thing when an animal is
found; the risk is only in the no-animal branch, where "no animal" may be a
classifier miss on a dark frame rather than a true empty scene.

## Recommended fix (minimal, reversible, FN-reducing) — pending Daniel's OK on volume

**Brightness-gate the MUTE, not the notification.** Only let the mute path fire
when low sharpness plausibly means blur — i.e. in adequate light. When the
scene is dark (low luma), a below-floor score is explained by darkness, so the
burst should flow through as a normal 🔍 REVIEW notification instead of being
silently muted.

Concretely:
- Compute mean-gray **luma** for the selected best frame (cheap; already have
  the frame in `_capture_and_select_best_frame`) and add it to `sharpness_info`.
- Add `is_blurry_review`'s condition a `luma >= brightness_floor` term (a new
  config knob, e.g. `blur_mute_min_luma`, default ~70 per the step function
  above). Below that luma, do not mute.
- Keep the raw Laplacian floor as-is for the animal-alert path (unchanged) and
  for the DB `below_sharpness_floor` column (still recorded, still honest).

This is a **code change** (no env knob reaches it), so: commit separately with
the experiment id, note the SHA here, and stamp a pre-sunrise `pending_restart_at`
since it only takes effect on camera restart. Rollback = `git revert` + restart.

**FN-veto stance:** this change *lowers* FN risk (fewer dark bursts silently
muted), so the FN-veto does not block it. The only cost is a small REVIEW-volume
increase — bounded to the below-floor no-animal *dark* bursts (07-10: the 3 dusk
ones, 1845/1846/1848). All of it is 🔍 REVIEW-tagged, in-channel.

**Why HELD tonight, not shipped:** (1) REVIEW-channel volume is Daniel's
standing product lever (he rejected a 2nd channel to keep volume manageable), so
a change that raises it — even modestly — should get his explicit nod first;
(2) it needs luma plumbed into `sharpness_info` + a new validated config knob +
TDD, which per CLAUDE.md is subagent-driven work, not a rushed end-of-tick edit.
There is no fire forcing it tonight: FP rate is stable and 100% REVIEW-tagged
with zero main-channel leak. Flagged in tonight's verdict for Daniel's greenlight;
ready to implement next tick.

## Alternatives considered (recorded, not chosen)

- **(a) brightness-normalized sharpness** (`lap / gray_var`) as the floor
  statistic. Cleaner in theory but replaces a calibrated threshold with an
  uncalibrated one — needs a fresh distribution study before it can ship
  without risking either FN (too strict) or REVIEW spam (too loose). Higher
  risk than the luma-gate for the same FN benefit.
- **(c) drop the mute path entirely**, rely only on the REVIEW prefix. Simplest
  and maximally FN-safe, but raises REVIEW volume at *all* light levels, not
  just dark ones — every genuinely-blurry daylight no-animal burst would then
  notify too. The luma-gate is (c) restricted to the cases where the floor is
  actually lying, so it buys the same FN safety at a fraction of the volume cost.
