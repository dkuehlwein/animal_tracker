# Loop Journal

Thin, append-only chronological index. One line per event, linking run files.
Cross-experiment notes live here; per-experiment detail lives in `runs/NNNN-<slug>.md`.

- 2026-06-08 — Notebook scaffolded. Seeded backlog: #1 notification-gate-live (live),
  #2 unknown-species-threshold (parked/replay), #3 roi-masking (live).
- 2026-06-08 — First loop tick. Baseline: FP 0.798 (67/84, CI [0.700,0.870]),
  FN unmeasured, volume 84/night. Deploy HELD: #1 gate not wired into live
  notification path / BOUNDS — infra blocker, not FN-veto. Shadow gate would cut
  70/72 FP (97%) at 0/12 animal loss. See runs/0001-notification-gate-live.md.
- 2026-06-09 00:04 — No-op tick (night). No new detections since watermark 84
  (DB still 84; latest data 06-08). metrics re-ingests from the watermark, so it
  measured 0 new triggers and transiently wrote a degenerate 0-trigger 06-09 row;
  reverted — restored 06-08 baseline in state.last_metrics and dropped the 0/0
  06-09 row from daily.csv (a 0-vs-84 row would falsely read as volume-collapse /
  FP→0 to future ticks). No decision, no deploy (#1 still infra-blocked; FN
  unmeasured). Sent heartbeat. Baseline unchanged.
- 2026-06-09 02:00 — No-op tick (night). Still no new detections since watermark 84
  (ingest --since-id 84 → 0 rows; DB max id == 84). Skipped metrics per the
  watermark-noop rule (it re-ingests from the watermark and would clobber the 06-08
  baseline). state.last_metrics holds 06-08 FP 0.798 (67/84); daily.csv unchanged
  (single 06-08 row). No decision, no deploy (#1 still infra-blocked; FN unmeasured;
  not paused). Sent heartbeat.
- 2026-06-09 04:00 — No-op tick (night, ~04:00 CEST). Still no new detections since
  watermark 84 (DB max id == 84; detection_feedback == 17). Skipped metrics per the
  watermark-noop rule (re-ingests from watermark; would clobber the 06-08 baseline
  with a 0-trigger row). state.last_metrics holds 06-08 FP 0.798 (67/84); daily.csv
  unchanged (single 06-08 row). No decision, no deploy (#1 still infra-blocked; FN
  unmeasured; not paused). Sent heartbeat.
- 2026-06-09 — USER NOTE (Daniel, reviewing the 06-08 baseline FP triggers): "I am
  very surprised by the alarms — these are all pretty much the same images. Why
  didn't they get filtered out by the background comparison?" Observation: a large
  share of the 67 FP appear to be recurrent, near-identical static scenes that MOG2
  background subtraction should have absorbed into the background model. Open
  question for a future tick: why aren't repeated/static frames suppressed? Candidate
  causes to investigate — central-region weighting re-amplifying the same edge motion,
  motion_threshold (500px) low enough that residual MOG2 noise clears it, MOG2 learning
  rate / history=500 vs trigger cadence, or shadow/lighting drift. Filed as backlog #4
  (mog2-recurrent-frames). Potentially high-impact FP reduction if confirmed.
- 2026-06-09 (night tick) — First new-data day. Ingested 185 detections (watermark
  84→269, all 06-09 daytime h7–18; 47 human-labeled). Measured FP 0.616 (114/185,
  CI [0.544,0.683]) vs 06-08 0.798 — but NOT a validated win (see self-audit).
  SELF-AUDIT (critical): tier-1 auto-labels agree with humans only 17/47 (36%),
  biased toward calling FP "animal" (24/30 disagreements); human dist 43 FP /
  4 wrong_species / 0 confirmed animals. → reconciled FP rate is an UNDERESTIMATE;
  no auto-label-based FP "win" is trustworthy. This label-trust gap gates every FP
  experiment. Promoted #4 (mog2-recurrent-frames) to running/diagnosis. #4 finding:
  motion features do NOT separate FP from animal (motion_area med 1143 vs 1142; 0/114
  FP near the 500px threshold) → threshold tuning is FN-vetoed & futile. Recurrence
  hypothesis needs scene-recurrence instrumentation (ROI centroid or ROI aHash) not
  currently logged — observability-only, zero FN risk, but multi-file + schema
  migration → flagged to Daniel for design review rather than shipped blind tonight.
  Gate #1 re-confirmed (92 suppress: 88 FP + 4 wrong_species, 0 animal = 100% prec)
  but still infra-blocked. Decision: HOLD, no deploy. Not paused, not frozen
  (47 human labels today). See runs/0002-mog2-recurrent-frames.md.
- 2026-06-09 (late, human-directed) — CORRECTION to the tick above. Scene recurrence
  WAS testable on existing data (no instrumentation needed): aHash over the saved
  frames showed 80% of adjacent triggers near-identical and 100 triggers collapsing
  into ~15 visual scenes (one 62-trigger scene over ~6h). Viewing the frames: a fixed
  sunlit garden with a swinging bird-feeder + wind-blown vegetation + moving sun-dapple
  — recurring REAL motion, NOT static scenes MOG2 "failed to absorb." MOG2 detects
  inter-frame change, not scene novelty; wind motion is non-periodic and the ~45s
  no-sampling gap after each trigger (cooldown + species-ID) keeps it from being
  learned. Levers: scene-recurrence dedup gate (aHash, live, no schema), SpeciesNet
  no-animal gate (#1), or vegetation-motion suppression — not "fix MOG2." Loop prompt
  updated (79ccd37) to check existing data before proposing instrumentation. See
  runs/0002 correction section.
- 2026-06-11 (human-directed) — Shipped exp #1 (notification-gate-live) as
  SAME-CHANNEL LABELING: 🔍 REVIEW header on NO_ANIMAL/UNCLASSIFIABLE captions
  (`is_review_detection` in data_models; `PERFORMANCE_REVIEW_PREFIX_ENABLED`
  default on). FN-safe (labels, doesn't drop/route) → no 2nd channel, no FN-veto.
  gate_would_suppress column untouched. Live on next camera restart. See
  runs/0001-notification-gate-live.md.
- 2026-06-10 (night tick) — Second new-data day; 87 triggers (watermark 269→356,
  06-10 h6–20; 31 human-labeled). FIXED the label-trust meta-blocker: cross-tabbing
  detection_status vs human labels isolated a single unidirectional error —
  `unclassifiable` (MegaDetector boxes a region, classifier can't ID) was mapped to
  tier-1 "animal" but is 27/27 false_positive across all history (the camera boxing
  wind-blown vegetation / the swinging feeder). Changed _STATUS_TO_TIER1
  ["unclassifiable"]="false_positive" in src/loop/ingest.py (commit 8f3ff01) — a
  metrics-reconciliation change only, zero FN risk, no camera restart. Effect:
  tier-1↔human concordance 29%→74% (06-10), 36%→64% (06-09); de-biased FP 0.724
  (06-09) / 0.874 (06-10) vs the masked 0.616/0.678 — the earlier "improvement" was a
  labelling artifact, true FP is HIGH and trending UP. last_metrics recomputed: FP
  0.874 (76/87, CI [0.788,0.928]), trustworthy. Recurrence re-confirmed on today's
  frames (87→~32 scenes, 49% adjacent near-identical, top scenes 0 'animal'); largest
  12-trigger scene mostly missed by the no-animal gate (2/12) → scene-dedup is
  complementary. Gate #1 today: 51 suppress, 0 animal (precision holds) but only ~33%
  FP recall (misses the unclassifiable FP class). Strongest lever = route
  detection_status∈{no_animal,unclassifiable} to a REVIEW channel (routing not
  suppression: the 6 no_animal wrong_species are real animals); still infra-blocked on
  a 2nd Telegram channel Daniel must provision. Decision: HOLD on camera deploy, no
  pending_restart. Not paused, not frozen. See runs/0002-mog2-recurrent-frames.md.
- 2026-06-11 (night tick) — Third new-data day; 109 triggers (watermark 356→465,
  06-11 h6–18; 40 human labels — not feedback-starved). **FP 90/109 = 0.826, CI
  [0.744, 0.885], trustworthy** — indistinguishable from 06-10's 0.874 (overlapping
  CIs); FP stably HIGH, FN structurally unmeasured. Two decisions this tick:
  (1) **CONCLUDED exp #4 (mog2-recurrent-frames)** — a diagnosis-only experiment
  (param_delta always null). Its three findings are stable across 3 nights:
  recurrence = REAL motion (swinging feeder/wind/sun-dapple), NOT static scenes MOG2
  failed to absorb; motion features don't separate FP from animal → all MOTION/ROI
  sensitivity tuning FN-vetoed & futile (no env lever in BOUNDS reaches the root
  cause); the actionable lever it surfaced was shipped as exp #1's labeling. Decision:
  inconclusive-as-deploy, diagnosis-successful, closed. (2) **PROMOTED exp #1
  (notification-gate-live) to running and brought it LIVE.** The committed REVIEW-
  labeling code (31d3bc6) was DORMANT — the camera only reloads code when
  wildlife-deploy.timer (03:30 CEST) finds a due pending_restart_at, which was null,
  so the shipped feature had never run. Stamped pending_restart_at=2026-06-12T03:00
  → camera restarts at the 03:30 timer, REVIEW labeling live for 06-12. Validated on
  tonight's 109 triggers: is_review_detection (status∈{NO_ANIMAL,UNCLASSIFIABLE})
  flags **89/90 FP = 99% recall** (1 FP slips through, an identified-misclassification);
  unprefixed stream is **15/16 = 94% true animals**; **0 FN** (3/18 animals get a
  cosmetic REVIEW prefix but are still fully shown). active_experiment_id 4→1. Not
  paused, not frozen, no env delta, no volume change. See runs/0001 & runs/0002.
- 2026-06-13 16:25 CEST — **LOCATION CHANGE / RE-BASELINE (human-driven, /remote-control).**
  Daniel physically moved the camera to a NEW location. wildlife-camera was stopped
  09:21 for the move (status=143 = SIGTERM, not a crash) and is now restarted clean
  (camera init OK, warmup armed). All old empirical state is OLD-SCENE and invalid for
  the new field of view, so the loop was paused and re-baselined rather than allowed to
  diff new data against stale baselines:
  - `paused: true` AND `wildlife-loop.timer` disabled+stopped (hard pause — nightgate
    does not honor `paused`, only the report banner does, so the timer is the real gate).
  - `baselines.volume_per_night: 84 → 0` (= "no baseline yet" per guardrails.check_volume;
    avoids false volume-collapse/explosion vs the old 84/night).
  - `last_metrics: {06-11 FP 0.826, 90/109} → null` (old-scene FP rate retired; first
    new-scene data tick will repopulate it).
  - `watermark: 465 → 470` (= current max detections.id) so the last old-location triggers
    are NOT ingested into the new baseline. New-scene triggers (id > 470) start fresh.
  - exp #3 roi-masking hypothesis annotated: old ROI geometry no longer applies; re-derive
    from new-scene FP patterns before proposing live. #1 REVIEW-labeling stays live
    (location-agnostic). Old DB (469 dets, 177 labels) + 1.4G images KEPT as archive.
  TO RESUME: let the new scene accumulate ~2-3 nights of triggers + Telegram labels, set
  a fresh volume_per_night baseline, then `paused: false` + `sudo systemctl enable --now
  wildlife-loop.timer`.
- 2026-06-15 (manual tick, /remote-control — RESUME after location change). Daniel
  confirmed enough new-scene data has accumulated; ran the loop by hand and re-armed the
  timer. **First new-scene metrics**: 68 triggers (id 471–538, 06-13 h16+ partial + 06-14
  full; watermark 470→538), 39 fresh human labels (not feedback-starved). **FP 60/68 =
  0.882, CI [0.785, 0.939], trustworthy** — new location's FP is just as HIGH as the old
  scene (06-11 was 0.826, overlapping CIs). FN still structurally unmeasured. New-scene
  status mix: no_animal 62, unclassifiable 1, identified 5 → REVIEW-labeling (#1, live,
  location-agnostic) still cleanly flags the FP mass (63/68 = status∈{no_animal,
  unclassifiable}). **Decisions**: (a) set fresh `baselines.volume_per_night = 42` (06-14
  full-day count; was 0 = no-baseline after the move) so guardrails.check_volume is armed;
  (b) `paused: false`, re-enabled wildlife-loop.timer — autonomous nightly cadence resumes;
  (c) NO deploy / no env delta / no pending_restart this tick — exp #4's conclusion still
  holds (motion features don't separate FP from animal; no env lever in BOUNDS reaches the
  root cause), and the high-leverage lever (route REVIEW→2nd Telegram channel) stays
  infra-blocked on Daniel provisioning a channel. **Next candidate**: exp #3 (roi-masking)
  is now unblocked — 2 nights of new-scene FP frames exist on disk to re-derive the ROI
  geometry from; still `proposed`, to be designed on a future tick (or on request).
  Note: metrics dates this backfill row 2026-06-15 (run-day) though the data is 06-13/06-14.
  Manual tick stamped via loop.endtick as loop-day **2026-06-14** (loop_day = (now−12h).date,
  run at 06:50 UTC) — so this catches up the never-completed 06-12/13/14 paused window.
  **Tonight's timer fire is loop-day 06-15 ≠ 06-14 → it RUNS** the first post-resume
  autonomous tick (ingests 06-15 daytime triggers, id>538; will overwrite the cosmetic
  06-15 CSV row — the durable new-scene resume number 60/68 lives here in JOURNAL).
  See runs/0001-notification-gate-live.md.
- 2026-06-15 (manual, /remote-control) — **CONCLUDED exp #1 (notification-gate-live),
  decision=keep.** Daniel's call: "consider the second fp channel as solved. routing it
  to the same channel with the pr fix is good enough... I am not clicking on two channels."
  The same-channel 🔍 REVIEW-prefix variant (31d3bc6, live since 06-12) is the ACCEPTED
  FINAL design; the future-channel-split follow-up is DROPPED, not deferred. Re-confirmed
  on new-scene data: prefix flags 99% FP, ~94% clean-stream animal purity, 0 FN, location-
  agnostic. No code/restart (already live). active_experiment_id 1→null (slot free). First
  LEARNINGS.md entries written (gate + exp #4 motion-feature findings). Next candidate
  remains exp #3 (roi-masking), proposed/unblocked. See runs/0001 + LEARNINGS.md.
- 2026-06-15 (autonomous tick, loop-day 06-15 — FIRST post-resume timer fire). Healthy,
  no-action tick. Ingested id 539–557 (watermark 538→557): **19 daytime triggers (hours
  10–17), all 19 human-labeled (NOT feedback-starved), FP 17/19 = 0.895, CI [0.686,0.971],
  trustworthy; FN unmeasured.** On-baseline (new-scene resume was 0.882; old scene 0.826 —
  all CIs overlap). No volume anomaly (partial-day daytime window; baseline 42 is full-night).
  Status mix: no_animal 16, unclassifiable 2, identified 1 → live REVIEW prefix (#1) flags
  18/19, clean stream = the 1 identified. No tier-2 needed (all crops human-labeled).
  **Decision: KEEP — no deploy, no env delta, no restart, active_experiment_id stays null.**
  Rationale: (a) no active experiment; (b) metrics on-baseline, no anomaly; (c) candidate
  exp #3 (roi-masking) is double-gated — ROI is NOT an env lever (guardrails.BOUNDS has only
  MOTION_{THRESHOLD,MIN_CONTOUR_AREA,CONSECUTIVE_REQUIRED,MIN_COLOR_VARIANCE} + SPECIES_
  UNKNOWN_THRESHOLD), so it's a code change, AND it inherently risks raising FN (edge animals
  masked) while FN is structurally unmeasured → FN-veto = HOLD on deploy; (d) exp #4 already
  concluded motion-threshold-family env knobs don't separate FP from animal, so no in-BOUNDS
  env delta has an expected FP win without FN risk. Next: exp #3 needs an FN-safety story
  (measure/bound FN, or a conservative center-preserving ROI) before it can deploy; new-scene
  FP frames keep accumulating nightly so the spatial ROI diagnostic isn't time-boxed away.
  See runs/0001 + LEARNINGS.md.
- 2026-06-16 (autonomous tick, loop-day 06-16). Healthy, no-action KEEP tick.
  Ingested id 558–599 (watermark 557→599): **42 daytime triggers (hours 12–19), 42/42
  labeled (40 human + 2 confident auto, NOT feedback-starved). FP 32/42 = 0.762,
  CI [0.615,0.865], trustworthy; FN unmeasured.** On-baseline (06-15 was 17/19=0.895
  small-sample; CIs overlap), **volume 42 == baseline 42** → no collapse/explosion.
  Label split: 32 false_positive / 6 wrong_species / 4 animal → **10/42 triggers had a
  real animal present** (~24% true-motion rate). No tier-2 needed (2 auto-labeled rows
  had decisive tier1 status; nothing genuinely ambiguous).
  **New observation (reaffirms exp #1 same-channel):** all 6 `wrong_species` rows have
  detection_status=`no_animal` but human=`wrong_species` → these are REAL ANIMALS that
  SpeciesNet returned no_animal on. Under the live REVIEW prefix, status=no_animal gets
  the 🔍 REVIEW prefix, so 6 real animals landed in the prefixed (likely-FP) stream — but
  because the prefix is SAME-CHANNEL, Daniel still saw + labeled them (zero info loss). A
  2nd-channel split would have HIDDEN these 6 in an FP channel; same-channel + prefix kept
  them visible. Concrete vindication of Daniel's 06-15 same-channel decision (exp #1, keep).
  These 6 are a sliver of *classification*-FN signal (triggered animal → no_animal), a
  different axis from motion-FN (animals that never triggered, still structurally unmeasured).
  **Decision: KEEP — no deploy, no env delta, no restart, active_experiment_id stays null.**
  Rationale: (a) no active experiment; (b) metrics on-baseline, no anomaly; (c) BOUNDS env
  levers are MOTION_{THRESHOLD,MIN_CONTOUR_AREA,CONSECUTIVE_REQUIRED,MIN_COLOR_VARIANCE} +
  SPECIES_UNKNOWN_THRESHOLD — none has an expected FP win without FN risk (exp #4 settled the
  motion knobs; the no_animal-on-real-animal miss is MegaDetector's detection threshold, which
  is NOT in BOUNDS, and SPECIES_UNKNOWN_THRESHOLD governs unknown-vs-named, not animal-vs-none);
  (d) exp #3 (roi-masking) still double-gated — code change + raises FN while FN unmeasured →
  FN-veto = HOLD. Next candidate unchanged: exp #3 needs an FN-safety story (bounded/center-
  preserving ROI) before deploy; new-scene FP frames keep accumulating so the spatial ROI
  diagnostic isn't time-boxed. Plateau is genuine: REVIEW prefix handles FP UX, no clean env
  lever, motion-FN unmeasurable from trigger data. See runs/0001-notification-gate-live.md.
- 2026-06-17 (autonomous tick, loop-day 06-17). **Productive tick — concluded exp #3
  (roi-masking) with a data-backed FN-safety diagnostic, ending a 3-tick deferral.**
  Ingested id 600–608 (watermark 599→608): 9 daytime triggers (hours 8–16), 9/9 labeled
  (8 human + 1 confident auto), NOT feedback-starved. FP 5/9 = 0.556, CI [0.27,0.81]
  (wide, small-sample), trustworthy; FN unmeasured. Volume 9 < baseline 42 but partial
  daytime window + nothing deployed → no collapse guardrail (no deploy to roll back).
  **Spatial ROI diagnostic (in-tick throwaway script over saved burst frames):** diffed
  consecutive frames → motion centroid for 70 labeled detections with frames on disk
  (18 real-animal incl wrong_species, 52 FP; 204 rows aged out by retention or no blob).
  FP and animal centroids are spatially ENTANGLED, both center-weighted (median ~0.5/0.5).
  **No edge band removes FP without removing comparable-or-more animals:** left15%
  8%FP/17%animals (hurts animals more), right15% 12%/11% (wash), top15% 10%/11% (wash),
  bottom15% 0 animals but only 1/52 FP (negligible). → No zero-observed-FN ROI exists;
  the FN-veto that held exp #3 for 3 ticks is now backed by measurement, not assumption.
  **Decision: KEEP (no deploy/delta/restart, active_experiment_id stays null) AND
  conclude exp #3 → not viable in current scene.** Significance: third axis on which
  FP and animals refuse to separate at the trigger (exp #4 = motion magnitude, exp #3 =
  motion location) → trigger-side FP suppression is a genuine plateau; the live
  notification-layer REVIEW prefix (exp #1) is vindicated as the right mitigation (sorts
  FP post-trigger via SpeciesNet at zero motion-FN cost). Backlog now: #1 concluded(live),
  #2 parked(replay), #3 concluded(not-viable), #4 concluded. Remaining lever is
  post-trigger (exp #2, parked on real replay.py). See runs/0003-roi-masking.md.
- 2026-06-18 (autonomous tick, loop-day 06-18). **No-action KEEP — genuine plateau, no
  deployable lever.** Ingested through id 645 (watermark 608→645): 37 triggers, 37/37
  labeled → feedback-rich, NOT starved (no freeze). FP 30/37 = 0.811, CI [0.66,0.91],
  trustworthy; FN unmeasured. Volume 37 ≈ baseline 42 (within normal range; no
  collapse/explosion guardrail). active_experiment_id stays null; nothing deployed →
  nothing to roll back. **Decision rationale:** backlog is fully settled on trigger-side
  levers — #1 concluded/live (REVIEW prefix), #3 concluded/not-viable (ROI entangled,
  06-17), #4 concluded (motion knobs don't separate FP from animal); the only open item
  is #2 (raise SPECIES_UNKNOWN_THRESHOLD 0.5→0.75), which is post-trigger and parked on a
  real `replay.py` (Layer-A validation is still a STUB→"skipped"). With no env knob whose
  expected FP win lacks FN risk, and the live REVIEW prefix already sorting the 0.81
  trigger-FP post-hoc at zero motion-FN cost, the disciplined output is KEEP. **Next
  substantive step is engineering, not a per-tick delta:** build `replay.py` so exp #2 can
  be replay-gated and the loop regains a validation lever — flagged for a dedicated build,
  not half-done in a 2h tick. See runs/0001-notification-gate-live.md.
