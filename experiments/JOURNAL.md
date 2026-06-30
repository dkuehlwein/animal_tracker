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
- 2026-06-19 (autonomous tick, loop-day 06-19). **No-action KEEP — a human-dominated
  day; the headline "FP dropped" is a metric artifact, not a real improvement.** Ingested
  id 646–719 (watermark 645→719): 74 daytime triggers (hrs 7–18), 74/74 labeled →
  feedback-rich, NOT starved (no freeze). `loop.metrics` reports FP **24/74 = 0.324**,
  CI [0.23,0.44], trustworthy; FN unmeasured. Taken at face value that's a big drop from
  06-18's 0.81 — but it is **not** a genuine FP reduction. Reconciled labels:
  24 false_positive, **47 wrong_species** (44 of them detection_status=no_animal,
  gate_would_suppress=true), 3 animal. `wrong_species` is excluded from `fp_count`, so a
  large cohort of unwanted triggers is hidden from the headline metric.
  **In-tick frame check (6 saved frames spanning 08:33 / 12:16 / 13:06 / 13:23 / 17:28 /
  18:34, all within retention):** every `wrong_species` frame shows a **person** working
  at the garden pond (net over the pond, blue pump/tool, bare legs/shorts). 06-19 was an
  **all-day human pond-maintenance/gardening session** (dense burst hrs 12–13, ~38
  triggers), not wildlife. So the day's true unwanted-trigger rate is ≈ **96% (71/74:
  24 FP + 47 human)**, with only **3 genuine wildlife IDs** all day.
  **Decision: KEEP (no deploy/delta/restart; active_experiment_id stays null; nothing
  deployed → nothing to roll back).** Rationale: (a) no lever — a human and an animal are
  indistinguishable at the motion trigger (the exact FP/animal entanglement concluded in
  exp #3 ROI 06-17 and exp #4), and a one-off gardening session is transient and
  non-recurring, so no env knob or code change is warranted; (b) volume 74 > baseline 42
  is fully explained by the human session (extra triggers), not a deploy/regression — no
  collapse/explosion guardrail applies; (c) the live REVIEW prefix (exp #1) already routed
  the 44 no_animal human triggers to the 🔍 REVIEW lane, behaving as designed.
  **Two honesty/measurement flags for Daniel (NOT acted on unilaterally):** (1) the
  `wrong_species` label is **heterogeneous** — the 06-17 diagnostic treated it as "real
  animal," but today's 47 are unambiguously **human**. Because `wrong_species` is dropped
  from both `fp_count` and the animal bucket, the headline FP rate can swing widely on how
  this cohort is bucketed; a metric-policy decision (separate "human/non-target" bucket?)
  would make the rate trustworthy on mixed days. (2) Today carries **no tuning signal** —
  a human-dominated day tells us nothing new about FP/animal separation, which remains the
  established plateau. Backlog unchanged: #1 concluded/live, #2 parked (replay.py), #3
  concluded/not-viable, #4 concluded. Next substantive step is still engineering
  (build replay.py to unpark exp #2), not a per-tick delta. See runs/0001-notification-gate-live.md.
- 2026-06-20 (autonomous tick, loop-day 06-20). **No-action KEEP — second consecutive
  human-dominated garden day; FP 0.78 is on the established plateau, no new signal.**
  Ingested id 720–770 (watermark 719→770): 51 daytime triggers (hrs 8–20), 51/51 labeled.
  `loop.metrics` reports FP **40/51 = 0.784**, CI [0.65, 0.88], trustworthy; FN unmeasured.
  That is squarely on-baseline (06-18 was 0.81), NOT a regression — volume 51 vs baseline
  42 is mildly elevated and fully explained by human activity, no collapse/explosion
  guardrail applies. Status mix: no_animal 36, unclassifiable 5, identified 10. Only 1
  human label today (id 741 = wrong_species), so the FP count is driven by reliable tier-1
  auto-labels (no_animal/unclassifiable → false_positive).
  **In-tick frame check (all 51 frames on disk, within retention):** sampled no_animal FP
  frames show **people working in the garden** — id 728 (10:00) a person's body/leg at
  frame-right, id 738 (10:31) a person carrying a coil/basket across frame, id 741 (10:35,
  the lone human label) the same. The **net-over-pond setup from 06-19 is still present**,
  so this reads as a continuation of the same pond/garden-maintenance activity, not
  wildlife. A few real birds were captured (the 10 `identified`, e.g. id 720/722/723/735/736).
  **aHash recurrence test (exp #4 re-check): 26 visual clusters from 51 frames — no single
  dominant recurrent static scene.** Crucially the largest clusters (0,1,2) each MIX
  `identified` (animal) with `no_animal`/`unclassifiable` frames — animals and FP share the
  same garden background, the exact spatial/visual entanglement concluded in exp #3 (ROI,
  06-17) and exp #4. So MOG2 recurrent-frame suppression still offers no clean separation.
  **Decision: KEEP (no deploy/delta/restart; active_experiment_id stays null; nothing
  deployed → nothing to roll back).** Rationale: (a) no lever — a human and an animal are
  indistinguishable at the motion trigger; transient garden activity is non-recurring, so
  no env knob or code change is warranted; (b) volume within normal range, no guardrail
  breach; (c) the live REVIEW prefix (exp #1) already routes the 41 no_animal/unclassifiable
  triggers to the 🔍 REVIEW lane, behaving as designed. Not feedback-starved (heavy human
  labeling 06-19 + 1 today; <3 days, no freeze). Backlog unchanged: #1 concluded/live,
  #2 parked (replay.py), #3 concluded/not-viable, #4 concluded. Next substantive step
  remains engineering — build replay.py to unpark exp #2 — not a per-tick delta.
  See runs/0001-notification-gate-live.md.
- 2026-06-21 (autonomous tick, loop-day 06-21). **No-action KEEP — third consecutive
  human/garden day; FP 0.96 is elevated but explained by sustained pond-maintenance
  activity + wind-blown foreground grass, no new tuning signal, no safe lever.**
  Ingested id 771–847 (watermark 770→847): 77 daytime triggers (hrs 7–19), 77/77
  labeled, **zero human labels today**, no tier-2 crops to adjudicate (all tier-1
  auto-labels). `loop.metrics`: FP **74/77 = 0.961**, CI [0.89, 0.99], trustworthy;
  FN unmeasured. Higher than the 06-18/06-20 ~0.78–0.81 plateau, driven by only 3
  `identified` animals (id 776, 805, 845) against 74 no_animal/unclassifiable. Volume
  77 vs baseline 42 is ~1.8× — elevated but well under the 5× explosion guardrail
  (210), no collapse/explosion breach.
  **In-tick aHash recurrence test (exp #4 re-check, all 77 frames on disk):** 35
  clusters from 77 frames. Unlike 06-19/06-20, the three largest clusters are
  large, time-localized and **pure-FP** (cluster 0 n=10 all no_animal hrs 14–15;
  cluster 1 n=10 9×no_animal+1×unclassifiable hr 15; cluster 2 n=8 all no_animal
  hrs 16–17). Visual inspection of representatives: **cluster 1 shows a person's
  arm/body at frame-left** (human garden activity); clusters 0 and 2 show the same
  garden scene with the net-over-pond grid (present since 06-19) and tall foreground
  grass. The aHash "recurrence" is just the **shared static background** — the
  triggering motion inside each frame differs (wind-blown grass, people passing),
  i.e. genuine pixel change MOG2 correctly fires on, NOT an identical recurrent
  frame it failed to absorb. So exp #4's recurrent-frame suppression still has no
  purchase here.
  **Decision: KEEP (no deploy/delta/restart; active_experiment_id stays null;
  nothing deployed → nothing to roll back).** Rationale: (a) no safe lever — the
  FP mass is human + wind-grass motion, both entangled with the 3 real animals;
  raising motion_threshold would risk the small birds and FN is unmeasured, so the
  FN-veto/HOLD applies on data; transient garden activity is non-recurring so no env
  knob or code change is warranted; (b) volume within range, no guardrail breach;
  (c) the live REVIEW prefix (exp #1) routes the 74 no_animal/unclassifiable triggers
  to the 🔍 REVIEW lane as designed. Not feedback-starved (last human labels 06-20;
  1 day, <3, no freeze). Backlog unchanged: #1 concluded/live, #2 parked (replay.py),
  #3 concluded/not-viable, #4 concluded. Next substantive step remains engineering —
  build replay.py to unpark exp #2 — not a per-tick delta. See
  runs/0001-notification-gate-live.md.
- 2026-06-22 (autonomous tick, loop-day 06-22). **No-action KEEP — 4th consecutive
  human/garden day; FP 0.61 (well below yesterday's 0.96, on the ~0.6–0.8 plateau),
  strong human feedback, no new tuning signal, no safe lever.**
  Ingested id 848–909 (watermark 847→909): 62 daytime triggers (hrs 9–19), 62/62
  labeled, **44 human labels today** (22 wrong_species, 20 false_positive, 2 animal) —
  NOT feedback-starved. No tier-2 crops to adjudicate (tier2 empty; all tier-1 auto or
  human-labeled). `loop.metrics`: FP **38/62 = 0.613**, CI [0.49, 0.72], trustworthy;
  FN unmeasured. Status mix: 47 no_animal, 10 unclassifiable, 5 identified. The 22
  human wrong_species (heterogeneous, excluded from fp_count) absorbed much of what
  yesterday's pure auto-labels counted as FP — explaining the drop from 0.96 to 0.61.
  Volume 62 vs baseline 42 ≈ 1.5× — elevated but far under the 5× explosion guardrail
  (210); no collapse/explosion breach.
  **In-tick aHash recurrence test (exp #4 re-check, all 62 frames on disk):** 35
  clusters from 62 frames — fragmented, no dominant recurrent scene. Largest cluster
  (n=7, all no_animal, hrs 15–16) is the shared static garden background, not identical
  recurrent frames MOG2 failed to absorb. Cluster 3 (hr 11) mixes an `identified`
  animal with no_animal+unclassifiable on the same scene, re-confirming the FP/animal
  spatial entanglement (exp #3) and that exp #4 recurrent-frame suppression has no
  purchase here.
  **Decision: KEEP (no deploy/delta/restart; active_experiment_id stays null;
  nothing deployed → nothing to roll back).** Rationale: (a) no safe lever — FP mass
  is human garden activity + wind-grass motion, entangled with the few real animals;
  raising motion_threshold risks the small birds and FN is unmeasured, so FN-veto/HOLD
  applies on data, not assumption; transient garden activity is non-recurring so no env
  knob or code change is warranted; (b) volume within range, no guardrail breach;
  (c) the live REVIEW prefix (exp #1) routes the 57 no_animal/unclassifiable triggers
  to the 🔍 REVIEW lane as designed. Not feedback-starved (44 human labels today).
  Backlog unchanged: #1 concluded/live, #2 parked (replay.py), #3 concluded/not-viable,
  #4 concluded. Next substantive step remains engineering — build replay.py to unpark
  exp #2 — not a per-tick delta. See runs/0001-notification-gate-live.md.
- 2026-06-23 (autonomous tick, loop-day 06-23). **No-action KEEP — 5th consecutive
  human/garden day; FP 0.45 (below yesterday's 0.61, now beneath the ~0.6–0.8
  plateau), strong human feedback, no new tuning signal, no safe lever.**
  Ingested id 910–951 (watermark 909→951): 42 daytime triggers (hrs 9–19), 42/42
  labeled, **41 human labels today** (23 wrong_species, 18 false_positive) — NOT
  feedback-starved. No tier-2 crops to adjudicate (all tier-1 auto or human-labeled).
  `loop.metrics`: FP **19/42 = 0.452**, CI [0.31, 0.60], trustworthy; FN unmeasured.
  Status mix: 36 no_animal, 5 unclassifiable, 1 identified. The 23 human wrong_species
  (heterogeneous, excluded from fp_count) again absorbed much of what pure auto-labels
  would have counted as FP — the FP decline 0.96→0.61→0.45 over the last three days is
  a labeling artifact (human reclassification), not a detector improvement; the scene
  is the same garden/human activity. Volume **42 = exactly baseline (42)** — no
  collapse/explosion breach.
  **In-tick aHash recurrence test (exp #4 re-check, all 42 frames on disk):** 35
  clusters from 42 frames — fragmented, largest cluster only n=4 (hr 12, mixed
  no_animal+unclassifiable, shared static garden background, not identical recurrent
  frames MOG2 failed to absorb). No dominant recurrent scene → exp #4 recurrent-frame
  suppression still has no purchase here.
  **Decision: KEEP (no deploy/delta/restart; active_experiment_id stays null;
  nothing deployed → nothing to roll back).** Rationale: (a) no safe lever — FP mass
  is human garden activity + wind-grass motion, entangled with the rare real animals
  (1 identified today); raising motion_threshold risks the small birds and FN is
  unmeasured, so FN-veto/HOLD applies on data, not assumption; transient garden
  activity is non-recurring so no env knob or code change is warranted; (b) volume at
  baseline, no guardrail breach; (c) the live REVIEW prefix (exp #1) routes the 41
  no_animal/unclassifiable triggers to the 🔍 REVIEW lane as designed. Not
  feedback-starved (41 human labels today). Backlog unchanged: #1 concluded/live, #2
  parked (replay.py), #3 concluded/not-viable, #4 concluded. Next substantive step
  remains engineering — build replay.py to unpark exp #2 — not a per-tick delta. See
  runs/0001-notification-gate-live.md.

## 2026-06-24 (loop-day 06-24) — no-action KEEP
- `loop.ingest`/`loop.metrics`: 14 new triggers since watermark 951 (ids 952–965).
  FP **12/14 = 0.857**, CI [0.60, 0.96], trustworthy; FN unmeasured. Status mix: 11
  no_animal, 1 unclassifiable, 2 identified. Volume **14** — below baseline 42 but
  within historical range (9–109; cf. 06-15=19, 06-17=9); nothing deployed so no
  collapse-rollback applies.
- Only **2 human labels** today (1 animal id 952, 1 wrong_species id 953) vs 41–44 the
  prior days, so today's auto-labels dominate. The high FP rate is the *same* garden
  scene seen *without* human wrong_species reclassification — the mirror-image of the
  06-21→06-23 FP decline (0.96→0.61→0.45), confirming that swing was a labeling
  artifact, not a detector change. **Not feedback-starved** (2 labels today; the
  3-consecutive-zero-days rule does not trigger). No tier-2 crops to adjudicate (12 FP
  are tier-1 auto, 2 are human ground truth).
- **In-tick aHash recurrence test (exp #4 re-check, all 14/14 frames on disk):** 11
  fragmented clusters, largest only n=2. Notable hr-20 burst of 8 FP (ids 958–965) does
  NOT form one recurrent static scene — it splits into n=2 pairs + singletons. No
  dominant recurrent frame MOG2 should have absorbed → exp #4 recurrent-frame
  suppression still has no purchase here.
- **Decision: KEEP** (no deploy/delta/restart; active_experiment_id stays null;
  nothing deployed → nothing to roll back). No safe trigger lever — FP mass is garden
  activity entangled with the rare real animals (1 identified, 1 wrong_species today);
  raising motion_threshold risks small birds and FN is unmeasured, so FN-veto/HOLD
  stands on data. The live REVIEW prefix (exp #1) routes the 12 no_animal/unclassifiable
  triggers to the 🔍 REVIEW lane as designed. Backlog unchanged: #1 concluded/live, #2
  parked (replay.py), #3 concluded/not-viable, #4 concluded. Next substantive step
  remains engineering (build replay.py to unpark exp #2), not a per-tick delta. See
  runs/0001-notification-gate-live.md.

## 2026-06-25 (loop-day 06-25) — no-action KEEP
- `loop.ingest`/`loop.metrics`: 8 new triggers since watermark 965 (ids 966–973).
  FP **5/8 = 0.625**, CI [0.31, 0.86], trustworthy; FN unmeasured. Status mix: 3
  no_animal at hr14/16/16 + 1 no_animal hr19 + 1 unclassifiable hr13 (the 5 FP),
  3 identified. Volume **8** — below baseline 42 but within historical range (9–109;
  cf. 06-17=9, 06-24=14); nothing deployed so no collapse-rollback applies.
- **3 human labels** today (id 966 animal/TP, ids 967+968 wrong_species) → **not
  feedback-starved** (3-consecutive-zero-days rule does not trigger). No tier-2 crops
  to adjudicate: the 5 FP are tier-1 auto (no_animal/unclassifiable), the 3 identified
  are human ground truth.
- **In-tick aHash recurrence test (exp #4 re-check, all 8/8 frames on disk):** 5
  clusters (Hamming ≤10), largest n=3 = `[967, 968, 969]` — the two human-labeled
  wrong_species crops + the hr-13 unclassifiable FP. That co-clustering suggests 969 is
  the *same animal* the classifier couldn't pin down (an animal present, not a recurrent
  static scene MOG2 should have absorbed). `[971, 972]` pair at hr16; 970 and 973
  singletons. The hr-19 outlier det 973 has `motion_area=16307` (vs ~800–1100 for the
  rest) but is a lone event, not recurrent. No dominant recurrent frame → exp #4
  recurrent-frame suppression still has no purchase here.
- **Decision: KEEP** (no deploy/delta/restart; active_experiment_id stays null; nothing
  deployed → nothing to roll back). No safe trigger lever — FP mass is garden activity
  entangled with the rare real animals (1 identified id 966, 2 wrong_species today);
  raising motion_threshold risks small birds and FN is unmeasured, so FN-veto/HOLD
  stands on data. The live REVIEW prefix (exp #1) routes the 5 no_animal/unclassifiable
  triggers to the 🔍 REVIEW lane as designed. Backlog unchanged: #1 concluded/live, #2
  parked (replay.py), #3 concluded/not-viable, #4 concluded. Next substantive step
  remains engineering (build replay.py to unpark exp #2), not a per-tick delta. See
  runs/0001-notification-gate-live.md.

## 2026-06-26 (loop-day 06-26) — no-action KEEP
- `loop.ingest`/`loop.metrics`: 42 new triggers since watermark 973 (ids 974–1015).
  FP **40/42 = 0.952**, CI [0.84, 0.99], trustworthy; FN unmeasured. Volume **42 =
  baseline** exactly (no collapse/explosion; nothing deployed regardless). Status mix:
  35 no_animal + 4 unclassifiable + 3 identified. Hours concentrated 15–17 (28 triggers)
  and 8–10 (12) — daytime garden activity.
- **5 human labels** today (979 FP, 980 animal, 981 FP, 982 animal, 986 FP) → **not
  feedback-starved**. The 40 FP = 39 tier-1 auto (no_animal/unclassifiable) + det 981
  (classifier-identified but human-labeled FP); 2 TP (980, 982 identified+human-animal).
  As on 06-21→25, the high FP rate is auto-label-dominated, not a detector regression.
  No tier-2 crops to adjudicate (5 are human ground truth; rest are tier-1 auto).
- **In-tick aHash recurrence test (exp #4 re-check, 41/42 frames on disk):** 18
  fragmented clusters, largest **n=6 = [997,998,1003,1004,1005,1006]** at hr16–17 (the
  closest thing to a recurrent scene, but still a minority of 42). Crucially the
  human-labeled animals co-cluster with FPs: `[975,977,981,982]` mixes FP 981 + animal
  982, and `[976,979,980]` mixes FP 979 + animal 980. An aHash-keyed recurrent-scene
  suppressor would therefore drop real animals too — same entanglement exp #3 found
  spatially. Exp #4 recurrent-frame suppression still has no clean purchase here.
- **Decision: KEEP** (no deploy/delta/restart; active_experiment_id stays null; nothing
  deployed → nothing to roll back). No safe trigger lever: FP mass is daytime garden
  activity visually entangled with the rare real animals; raising motion_threshold risks
  small birds and FN is unmeasured, so FN-veto/HOLD stands on data. The live REVIEW
  prefix (exp #1) routes the 39 no_animal/unclassifiable triggers to the 🔍 REVIEW lane
  as designed. Backlog unchanged: #1 concluded/live, #2 parked (replay.py), #3
  concluded/not-viable, #4 concluded. Next substantive step remains engineering (build
  replay.py to unpark exp #2), not a per-tick delta. See runs/0001-notification-gate-live.md.
- 2026-06-28 — Tick over batch ids 1016–1040 (25 triggers, 06-27 13:00–15:46;
  watermark 1015→1040). Status mix: 11 no_animal + 9 unclassifiable + 5 identified.
  **Tier-2 adjudication of the 5 "identified" (ids 1033–1037, 15:16–15:22, one
  SpeciesNet rollup UUID):** frames on disk show the SAME static garden scene
  (wild grass + bamboo bush + ground mesh); triggering motion is wind-bent
  bamboo/grass; the only salient object is a small fixed bright-blue blob (static
  man-made, not a bird — unmoved across all 5 bursts/6 min). No animal present →
  all 5 = classifier-FP. Wrote 5 append-only `source='tier2'` FP labels
  (feedback rows 532–536). Re-ran metrics (reset watermark 1040→1015 to reprocess
  the batch with the new tier-2 labels; metrics re-advanced it to 1040).
- 2026-06-28 — Metrics: **FP 25/25 = 1.00**, CI [0.87, 1.0], trustworthy; FN
  unmeasured. Partition: n_human=0, n_claude=5 (5 FP, tier-2 mine), n_md=20 (20 FP,
  MegaDetector tier-1). Volume 25 < baseline 42 (lower, but nothing deployed →
  natural daytime variation, no volume-guardrail action). **0 human labels** this
  loop-day; 06-27 also 0 → 2 consecutive label-free days. Feedback-starved freeze
  triggers at 3 → one day from freeze; flagged to Daniel in verdict.
- 2026-06-28 — **Finding (feeds B1 + exp #2): SpeciesNet's generic "animal" rollup
  (`<uuid>;;;;;;animal`, blank genus/species, top-level common name) yields
  status=IDENTIFIED, which is NOT in `_REVIEW_STATUSES`={no_animal,unclassifiable}
  (data_models.is_review_detection). So these classifier-FP BYPASS the exp #1 🔍
  REVIEW prefix and reach the MAIN channel as if real sightings — tonight 5 of them.
  The rollup is a stable, parseable signal (recent identified rows: 11×`;;;;;;animal`,
  2×`aves;;;;;bird` class-level rollups vs real `…homo;sapiens;human`). Candidate
  lever: extend the REVIEW set to flag blank/class-level rollups (notification-layer
  only, ZERO FN risk — notification still sends, just with REVIEW header; mirrors
  exp #1 architecture). Lower-risk than exp #2 (raise UNKNOWN_THRESHOLD 0.5→0.75,
  still parked pending replay.py). Recorded as evidence, NOT deployed tonight.
- 2026-06-28 — **Decision: KEEP / HOLD** (no deploy/delta/restart;
  active_experiment_id stays null; nothing deployed → nothing to roll back). FP mass
  is daytime garden vegetation movement, visually+spatially entangled with the rare
  real animals (exp #3/#4 concluded); no safe trigger lever and FN unmeasured →
  FN-veto/HOLD stands on data. The blank-rollup→main-channel leak is a real
  notification-quality gap but deserves a designed run-file (B1 owns it in worktree
  loop-fn-audit), not an end-of-tick reflex. Backlog unchanged: #1 concluded/live,
  #2 parked, #3 concluded, #4 concluded.
- 2026-06-28 (2nd batch, same loop-day — prior tick committed+pushed e902e33 but
  was interrupted before `loop.endtick`, so the night stayed unmarked and this tick
  resumed; 51 NEW daytime detections 1041–1091 had accrued past watermark 1040).
  **Metrics: FP 45/51 = 0.882**, CI [0.77, 0.94], trustworthy; FN unmeasured.
  Partition all tier-1 MegaDetector (n_md=51, n_human=0, n_claude=0). The 6 non-FP
  are tier1="animal" rows (1042,1043,1045,1046,1047,1048, hours 10–11) — tier-2
  adjudicated: all 6 frames clearly show a PERSON (legs/dark trousers close to lens,
  Daniel in garden). SpeciesNet classified them `homo;sapiens;human` /
  `homo;;homo species` — CORRECT, so reconciled "animal"/non-FP stands; no tier-2 FP
  override written (these are genuine human triggers, not vegetation FP). **Contrast
  with the 1st-batch 5 blank `;;;;;;animal` rollups that leaked to MAIN channel:
  THIS batch's 6 animal-tier rows are all confidently+correctly human → no
  notification-quality leak this batch.** That reinforces the B1/exp-#2 finding is
  specifically about *blank/class-level* rollups, not human rows.
- 2026-06-28 — **Decision: KEEP / HOLD** (no deploy/delta/restart; active_experiment_id
  stays null; nothing deployed → nothing to roll back). FP mass is daytime garden
  vegetation + people, no safe trigger lever (exp #3/#4 concluded), FN unmeasured →
  FN-veto/HOLD stands on data. **0 human feedback labels again → 2 consecutive
  label-free days (06-27, 06-28); feedback-starved freeze trips at 3 → one more
  label-free day freezes the loop.** Flagged to Daniel in verdict. Backlog unchanged.
- 2026-06-29 — Batch 1092-1142 (51 daytime triggers, hrs 8-19). FP 47/51 = 0.922
  (CI [0.815,0.969]); 45 no_animal + 2 unclassifiable = FP-tier, 4 identified/animal.
  **Tier-2 (frames on disk): the 4 main-channel 'identified;aves;bird' alerts
  (1096-1099, 13:11-13:16, conf 0.75-0.85) are ALL genuine — a real blackbird (Amsel)
  visiting the garden bird bath, clearly perched/bathing in 1098/1099. No
  classifier-FP main-channel leak this batch (contrast 06-27 leak audit).** The 2
  unclassifiable (1134 18:20 = person in garden at frame edge; 1142 19:12 = dusk
  vegetation, no animal) are true FP and correctly REVIEW-prefixed. All 6
  adjudications confirm tier1 → no tier-2 corrections / no reconciled-label change.
  **Decision: KEEP / HOLD** — no active experiment, nothing deployed, FN unmeasured,
  no safe trigger lever (exp #3/#4 concluded); FP mass is the known daytime
  garden-movement pattern handled by the REVIEW prefix (exp #1), not trigger-side.
  **Feedback: tonight's batch again n_human=0; human taps on 06-28 (05:39, labeling
  the prior batch) were the last calendar feedback — experimentation stays effectively
  frozen (best_known_good={}, already stock config, so freeze is operationally moot).**
  Backlog unchanged. Positive signal: classifier correctly surfaced real birds to main
  channel while REVIEW-gating the human/dusk FP.
- 2026-06-30 — Batch 1143-1167 (25 daytime triggers, hrs 9-19). Reconciled FP 22/25
  = 0.88 (CI [0.70,0.96]). **6 HUMAN labels this batch (1143-1148: 5 false_positive +
  1 animal) → feedback drought BROKEN; the 3-label-free-day freeze (06-27/06-28 were
  2 of the 3) is averted.** Headline (human-only): fp_human 5/6 = 0.83 (CI
  [0.44,0.97]). FN unmeasured; error_count 0. Tier-2 (frames on disk) on the 3
  animal-tier rows: 1147 (10:20, aves;bird, conf 0.81) = genuine blackbird on the
  ground, human-confirmed ✅, no correction. **1163 (19:28) & 1167 (19:48), both
  classifier rollup `mammalia;primates;hominidae;homo;;homo species` (conf 0.82/0.92)
  = REAL HUMANS** — 1163 a close-up of a person bending in front of the camera, 1167 a
  person at the dusk frame edge. **MAIN-CHANNEL LEAK: detection_status=identified is
  NOT in _REVIEW_STATUSES={NO_ANIMAL,UNCLASSIFIABLE} (data_models.is_review_detection
  is taxon-blind; wildlife_system.py:446), so both alerted Daniel's MAIN channel as if
  a real wildlife ID — no 🔍 REVIEW prefix.** Same class as the 06-27 leak audit;
  06-28's humans did NOT leak because their status wasn't `identified`, tonight's two
  were. **Metrics caveat:** tier-1 maps identified→animal, so 1163/1167 reconcile as
  animal (not FP) and, lacking a human tap, are excluded from fp_count — reconciled
  22/25 UNDER-counts; true operational FP = 24/25. Per standing rule (auto-labels not
  truth; headline=human-only) I did NOT tier-2-relabel them to FP (avoids poisoning the
  reconciled series); footnoted only. **Decision: KEEP / HOLD** — no active experiment,
  nothing deployed (best_known_good={}, stock), FN unmeasured, no safe trigger lever
  (exp #3/#4 concluded: motion knobs can't separate FP from animal). **New backlog
  exp #5 (human-main-channel-leak, runs/0004): route homo/human IDs → REVIEW (or
  suppress) regardless of status; code change, minimal/reversible, ZERO FN risk to
  wildlife. PARKED pending Daniel's product/privacy call (alert on humans at all? cf.
  no-second-channel) — flagged in tonight's verdict.** Positive: feedback returned and
  the classifier correctly ID'd both the real bird and the humans (the gap is routing,
  not classification).
