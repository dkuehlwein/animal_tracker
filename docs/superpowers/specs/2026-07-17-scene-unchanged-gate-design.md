# Scene-Unchanged Gate — Design Spec

**Date:** 2026-07-17
**Status:** Approved by Daniel (interactive session)
**Goal:** Cut Telegram REVIEW noise by muting review-class detections whose frame
is near-identical to recent known-empty views of the scene — automating the
comparison Daniel does by eye ("compare to earlier images; even at low Telegram
resolution that's usually good enough").

## Problem

The pipeline judges each burst in isolation. MOG2 background subtraction runs
only on the low-res motion stream at trigger time; SpeciesNet sees a single
burst. Nothing asks "does this frame differ from the last known-empty view?"
Result: recurring empty-pond REVIEW notifications (the dominant FP class,
fp_rate ~0.8 MD-auto on recent nights).

## Decisions (locked with Daniel)

1. **Live gate, built now** — not deferred to the nightly loop's backlog queue.
2. **Mute from day one** — matched detections are DB-logged, not sent to
   Telegram. No shadow mode. Kill switch via env + restart.

## Behavior

**Hook point:** notification layer of `WildlifeSystem.process_detection`.
Applies ONLY to review-class statuses (NO_ANIMAL / UNCLASSIFIABLE) that would
otherwise send a 🔍 REVIEW message.

**Gate precedence (unchanged order):** human/privacy gate → blur mute → scene
gate. The scene gate only sees above-sharpness-floor review-class bursts.
IDENTIFIED animals are never touched. HUMAN handling unchanged. The
blank-identified→main FP class stays with the nightly loop's queued routing
fix; once blank routes to review-class, this gate covers it automatically.

**Decision rule:** compare the burst's best frame against the best frames of
recent empty-verdict bursts (the reference set). Near-identical to ANY
reference → status-quo scene → mute (no Telegram; DB row records the score and
the mute). Otherwise notify as today.

**Fail-open invariant:** every failure direction must be "noise stays", never
"animal hidden". No qualifying reference (quiet period, restart with stale DB
rows, camera nudged, lighting regime change), unreadable reference file,
comparison error → gate passes through and the REVIEW message sends.

## Reference set

- No new image storage. References are `image_path` best frames already on
  disk from recent bursts whose outcome was empty (review-class result).
- Rolling window: last K=3 empty-verdict bursts within H=6 hours (both
  configurable). Seeded from the DB on startup; maintained in-process
  afterward.
- A burst muted by the scene gate itself is still an empty verdict and may
  join the reference set (it was near-identical to an existing reference, so
  drift is bounded by the recency window).
- HUMAN-status bursts are never references (frames get purged; privacy).

## Comparison metric & offline validation

- Comparison on downscaled grayscale, ~64×64, luma-normalized (matches the
  "Telegram resolution is enough" observation and is robust to global
  exposure/AE shifts at dusk).
- Exact metric (mean-abs-diff vs aHash Hamming vs SSIM-lite) and threshold are
  chosen empirically by an offline validation script
  (`scripts/validate_scene_gate.py`) run against the ~100 retained bursts
  joined with `detection_feedback` labels **before the gate goes live**.
- **FN-veto acceptance rule:** zero labeled animals (`animal`,
  `animal_wrong_id`) may score as "unchanged" at the chosen threshold;
  maximize FP-mute yield subject to that. If no threshold satisfies it, the
  gate ships disabled pending more data.
- Script is committed so the nightly loop can re-run it as labels accumulate
  and tune the threshold via its normal env-delta lever.

## Configuration

Follows existing `PerformanceConfig`/env patterns:

- `PERFORMANCE_SCENE_GATE_ENABLED` (bool, default true) — kill switch.
- `PERFORMANCE_SCENE_GATE_THRESHOLD` (float) — similarity cutoff, default from
  offline validation.
- `PERFORMANCE_SCENE_GATE_REF_COUNT` (int, default 3) and
  `PERFORMANCE_SCENE_GATE_REF_MAX_AGE_HOURS` (float, default 6).
- Threshold (and enabled flag) added to `loop/guardrails.py` BOUNDS so the
  nightly loop may tune within limits.
- Rollback lever: `PERFORMANCE_SCENE_GATE_ENABLED=false` + service restart.

## Observability

Mirrors the blur-gate/ADR-004 pattern — two new nullable columns on
`detections`, populated at decision time (frames age out, scores must not):

- `scene_similarity` (REAL) — best similarity vs the reference set (NULL when
  gate didn't evaluate: non-review-class, disabled, no references).
- `scene_gate_muted` (BOOLEAN) — true when the gate suppressed the
  notification.

Round-trip through `DatabaseManager.log_detection` and
`WildlifeSystem.process_detection` like the existing observability columns.
Muted bursts keep their frames under normal retention so the nightly loop can
adjudicate them.

## Nightly loop integration

- JOURNAL.md entry + PROTOCOL.md note: the scene-gate mute path exists and the
  loop OWNS its monitoring — adjudicate every `scene_gate_muted` burst for
  concealed animals nightly, exactly like the blur-mute path. A concealed
  animal in this path is an FN-veto event: loop rolls the threshold down or
  disables via env.
- Loop may re-run `scripts/validate_scene_gate.py` and tune the threshold
  within BOUNDS. No greenlight required (per 2026-07-17 autonomy protocol).

## Risks

- **Ripples/rain/wind:** frames differ more → fewer mutes. Safe direction
  (noise stays).
- **Dusk lighting drift:** luma normalization + 6h recency window bound it;
  residual drift → fewer mutes (safe).
- **Camera nudge/refocus:** all references differ → fail-open (safe); window
  ages out old references within 6h.
- **Reference poisoning:** an animal wrongly in the reference set could match
  a later frame of the same animal. Bounded: references come only from
  empty-verdict bursts, and the loop's nightly adjudication of the mute path
  catches it (FN-veto → threshold down / disable).

## Testing

- Unit: comparison function (identical frames, global luma shift, small
  animal-blob synthetic diff, shifted/nudged frame), reference-set maintenance
  (window eviction, DB seeding, HUMAN exclusion), config validation, and
  notification-precedence tests (human > blur > scene; identified never
  muted; fail-open paths).
- Deterministic suite: `uv run pytest tests/ -v` (subagent-run).
- Offline validation against labeled corpus is the live-enable gate.
