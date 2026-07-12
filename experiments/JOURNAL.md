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

## 2026-07-01 — KEEP/HOLD (no change): 68 triggers, 65 FP (95.6%); exp #5 leak-watch CLEAN

Batch 1168–1235 (68 triggers, watermark 1167→1235). Headline **fp_human 8/10 =
0.80** (2 human-confirmed real blackbirds 1179/1180 = animal; 8 human FP). Auto:
fp_md 57/57=1.0, tier-2 0/1 (1196). Reconciled fp 65/68=0.956. FN unmeasured. **10
human labels — feedback healthy, drought stays broken (2 nights running).**

**Exp #5 (human-main-channel-leak) leak-watch: CLEAN tonight.** All three
`status=identified` main-channel alerts (1179 10:21, 1180 10:25, 1196 12:43) were
`aves;;;;;bird` — real blackbirds, NOT homo/human. 1179/1180 human-confirmed animal;
1196 tier-2 visually confirmed (dark blackbird at pond, lower-left frame). No
homo-taxon leak occurred, so no new evidence forcing the exp #5 policy call — it
stays PARKED pending Daniel's product/privacy decision (runs/0004 unchanged in
substance; leak-watch noted). REVIEW gate held: all 65 FP were no_animal/
unclassifiable → correctly 🔍 REVIEW-prefixed, none leaked to main.

**Decision: HOLD, no deploy.** active_experiment=null, deployed={} (stock). Volume
68 vs baseline 42 is elevated (bright-summer-daytime garden movement, hours 7–20)
but not an explosion, and with nothing deployed there is nothing to roll back. No
env knob and no un-parked experiment to advance: #2 replay-gated, #5 awaits Daniel.
FN unmeasured → FN-veto keeps the threshold hold standing. Trigger-side FP
suppression remains ruled out (exp #3/#4 concluded: motion knobs can't separate FP
from real animals). Nothing to change tonight; the notification-layer REVIEW gate is
doing its job.

## 2026-07-02 — KEEP/HOLD, no deploy (stock config)

**Batch 1236-1292, 57 triggers.** fp 55/57=0.965 (2 real animals = blackbirds
1260/1261). fp_human 3/3=1.0 (1290/1291/1292, all no_animal/unclassifiable garden
movement). fp_md 52/54=0.963. tier-2 2/2 (1260/1261). FN unmeasured. **3 human
labels — feedback drought stays broken (3 nights running).**

**Exp #5 (human-main-channel-leak) leak-watch: CLEAN.** Both `status=identified`
main-channel alerts were real birds, not humans: 1260 classified generic
`;;;;;;animal`, 1261 `aves;;;;;bird`; both frames show the same dark blackbird
foraging at the pond edge (lower-left). No homo-taxon leak → no new evidence forcing
the policy call. Exp #5 stays PARKED pending Daniel's product/privacy decision
(runs/0004 leak-watch log appended). REVIEW gate held: all 55 FP were
no_animal/unclassifiable → correctly 🔍 REVIEW-prefixed, none leaked to main.

**Decision: HOLD, no deploy.** active_experiment=null, deployed={} (stock). Volume
57 vs baseline 42 elevated (bright-summer daytime garden movement, hours 9-17) but
not an explosion; nothing deployed → nothing to roll back. No env knob and no
un-parked experiment to advance: #2 replay-gated, #5 awaits Daniel. FN unmeasured →
FN-veto keeps the threshold hold standing. Trigger-side FP suppression stays ruled
out (exp #3/#4 concluded). Notification-layer REVIEW gate doing its job.

- 2026-07-03 — Loop tick (batch 1293-1388, 96 triggers). FP 94/96 = 0.979 auto
  (fp_md), CI [0.927,0.994]; n_human = 0 (Daniel tapped no feedback today);
  fp_trustworthy true. Volume 96 vs baseline 42 — a bright-July garden-activity
  spike (hours 8-17, wind + sun + people in the garden), >2x baseline but NOT a
  deploy-driven explosion (deployed={} stock → nothing to roll back).
  **Exp #5 (human-main-channel-leak) leak-watch: NOT CLEAN — 2 real human leaks.**
  Both `identified` main-channel alerts tonight were humans, zero real-animal IDs:
  1362 (14:51 `homo species`) = person in yellow shorts walking the bed; 1388
  (17:42 `homo sapiens;human`) = bare arm/hand with a watering can. First recurrence
  with actual humans since the 06-30 audit (prior two nights' `identified` leaks were
  birds). Leak mechanism unchanged (`identified` bypasses `_REVIEW_STATUSES`, taxon-
  blind `is_review_detection`). Neither human-tapped → reconciled as animal, not in
  fp_count (per auto-labels-not-truth). Exp #5 fix is code-ready + minimal/reversible
  but stays PARKED — alerting-on-humans is Daniel's product/privacy call; flagged in
  tonight's verdict as forcing evidence, no unilateral deploy. (runs/0004 leak-watch
  log appended.)
  **Decision: HOLD, no deploy.** active_experiment=null, deployed={} stock. No env
  knob addresses today's issue (busy day + human leaks); #2 replay-gated, #5 awaits
  Daniel. FN unmeasured → FN-veto holds the threshold. Feedback 22/3/0 over the last
  3 days — only today at 0, so no feedback-starved freeze yet (watch tomorrow).

- 2026-07-04 — Loop tick (batch 1389-1464, 76 triggers). FP 71/76 = 0.934 auto
  (fp_md), CI [0.855,0.972]; n_human = 0 (no feedback tapped today); fp_trustworthy
  true. Volume 76 vs baseline 42 — another bright-July garden-activity day (hours
  08-19, wind + sun + garden use), ~1.8x baseline but NOT a deploy-driven explosion
  (deployed={} stock → nothing to roll back).
  **Exp #5 (human-main-channel-leak) leak-watch: NOT CLEAN — 1 human leak.** 5
  `identified` main-channel alerts tonight, visually adjudicated: 1389 (08:17
  `homo sapiens;human`, conf 0.966) = person in yellow shorts, bare legs/forearm
  crossing the bed → HUMAN LEAK; 1396/1399/1400 (11:08-11:29, aves/animal) = same
  real blackbird foraging lower-left over ~20 min; 1423 (15:02, animal) = real bird
  bathing at the water dish. So 1/5 identified = human, 4/5 = genuine wildlife.
  **Second consecutive night with a human leak** (07-03 had 1362+1388). Mechanism
  unchanged (`identified` bypasses `_REVIEW_STATUSES`, taxon-blind
  is_review_detection). Human not tapped → reconciled as animal, not in fp_count
  (auto fp = 71/76); true operational fp = 72/76. Exp #5 fix stays code-ready but
  PARKED pending Daniel's product/privacy call — flagged in verdict as accumulating
  forcing evidence (2 nights running). (runs/0004 leak-watch log appended.)
  **Decision: HOLD, no deploy.** active_experiment=null, deployed={} stock. No env
  knob addresses today's issue (busy day + human leak); #2 replay-gated/parked, #5
  awaits Daniel. FN unmeasured → FN-veto holds the threshold. **Feedback-starved
  watch:** last human label was 07-02; 07-03 + 07-04 both zero = 2-day gap. Freeze
  triggers at 3 consecutive zero days → one more quiet day (07-05) trips it, though
  with no active experiment it is a near-no-op (already holding stock). Noted in
  verdict.

## 2026-07-05 — HOLD (stock), leak-watch CLEAN, feedback drought BROKEN
- **Batch 1465-1519, 55 triggers, 53 FP (96.4% auto).** n_human=13, fp_human 12/13
  (=0.92); n_md=42, fp_md 41/42. FN unmeasured. Volume 1.3x baseline (42) — sunny
  garden, within normal, no collapse/explosion. Stock config (deployed={},
  active_experiment=null).
- **Feedback DROUGHT BROKEN.** 2 zero-label days (07-03, 07-04) had put us one quiet
  day from the 3-day feedback-starved freeze; today Daniel tapped **13 labels** (12
  false_positive on the morning no_animal/unclassifiable run 1465-1476, + 1 `animal`
  on 1478). Freeze does NOT trip; the watch resets. Daniel is engaged again.
- **Exp #5 leak-watch CLEAN** — breaks the 07-03/07-04 two-night human-leak streak.
  Only 2 `identified` main-channel alerts tonight, both `aves;;;;;bird`
  (`b1352069…`), both visually confirmed **blackbirds**: 1478 (10:17, human-tapped
  `animal`, on the water-dish rim) and 1515 (18:35, foraging in grass). No `homo`/
  `homo species` taxon anywhere in the batch. taxonomy_release.txt confirms
  `b1352069…`=bird, `990ae9dd…`=homo sapiens (absent), `f2efdae9…`=no-cv-result
  (the unclassifiable frames 1476/1481/1487/1493/1507, all correctly REVIEW-gated).
- **Decision: HOLD, no deploy.** No env knob addresses a clean high-FP sunny-garden
  day; #2 replay-gated/parked, #5 awaits Daniel's product/privacy call (no new
  forcing evidence tonight). FN unmeasured → FN-veto holds the threshold. Stock
  config unchanged. (runs/0004 leak-watch log appended.)

## 2026-07-06 — HOLD (stock), leak-watch CLEAN (2nd consecutive), busy sunny day
- **Batch 1520-1590, 71 triggers, 66 FP (92.96% auto).** n_human=1 (1 false_positive
  tap); n_md=70, fp_md 65/70. FN unmeasured. Volume 71 = 1.7x baseline (42) — sunny
  garden, within normal range, no collapse/explosion. Stock config (deployed={},
  active_experiment=null).
- **Exp #5 leak-watch CLEAN — 2nd consecutive clean night** (07-05 + 07-06), breaking
  further from the 07-03/07-04 human-leak streak. 5 `identified` main-channel alerts,
  ALL visually-confirmed real blackbirds, no `homo`/`homo species` taxon in the batch:
  1521 (10:17, `aves;;;;;bird` 0.671, blackbird on water-dish rim) + a 4-alert series
  1544/1545/1546/1547 (11:42-11:44, same blackbird foraging left grass border; the
  generic `1f689929…;;;;;;animal` taxon on 1545-47 is just lower classifier confidence
  on the same bird). tier-1 already labeled all 5 `animal`; no tier-2 change. No forcing
  evidence added; exp #5 stays code-ready + PARKED pending Daniel's product/privacy call.
- **Feedback watch:** 07-05 had 13 taps (drought broken), 07-06 has 1 human tap — not a
  zero-label day, so the 3-day feedback-starved freeze does NOT trip; watch stays reset.
- **Decision: HOLD, no deploy.** No env knob addresses a clean high-FP sunny-garden day;
  #2 replay-gated/parked, #5 awaits Daniel's product/privacy call (no new forcing evidence
  tonight). FN unmeasured → FN-veto holds the threshold. Stock config unchanged.
  (runs/0004 leak-watch log appended.)

## 2026-07-07 — HOLD (stock), leak-watch NOT CLEAN (2 human leaks), high-volume sunny day
- **Batch 1590-1706, 116 triggers, 111 FP (95.69% auto).** n_human=0 (no taps today);
  n_md=116, fp_md 111/116. FN unmeasured. Volume 116 = **2.8x baseline (42)** — busy
  sunny garden with people present (watering + pond tending); stock config so nothing to
  roll back, noted not actioned. active_experiment=null, deployed={}.
- **Exp #5 leak-watch NOT CLEAN — 2 real human leaks** (breaks the 07-05/07-06 clean
  streak; 3rd human-leak night in 5). 5 `identified` main-channel alerts (1590-1706):
  1592 (08:44 `aves` 0.886) + 1612 (11:59 `aves` 0.780) = birds (frames aged out, but
  `aves` never carries homo); 1656 (14:47 `;;;;;;animal` 0.609) = low-conf bird at right
  frame edge; **1633 (14:00 `homo;;homo species` 0.994) = HUMAN** (person in dark clothes
  watering pond with blue-nozzle hose, left third, bright daylight — frame unmistakable);
  **1694 (18:39 `homo;;homo species` 0.730) = HUMAN** (large motion-blurred person-mass
  filling left half close to lens, dusk, tending pond). Both leaked to MAIN channel with
  no REVIEW prefix (`identified` bypasses `_REVIEW_STATUSES`, taxon-blind is_review_detection).
  Neither tapped → reconciled animal, not in fp_count (auto fp 111/116; true operational
  fp 113/116 counting humans as non-wildlife). No tier-2 relabel (avoids self-poisoning).
- **Feedback watch:** 07-05 had 13 taps, 07-06 had 1, 07-07 has 0 — last 3 days are NOT
  all-zero, so the 3-day feedback-starved freeze does NOT trip. Watch active but unfrozen.
- **Decision: HOLD, no deploy.** exp #5 fix is code-ready but PARKED — alerting-on-humans
  is Daniel's product/privacy call, not an autonomous deploy; tonight's 2 leaks are
  continuing forcing evidence, flagged in the verdict. #2 replay-gated/parked. FN
  unmeasured → FN-veto holds the threshold. Stock config unchanged.

## 2026-07-08 — SHIPPED: human-suppression (exp #5) + blur-gate false-negative fix (exp #6) — loop baselines change, not an anomaly

- **Daniel made the product/privacy call on exp #5 (07-07):** SUPPRESS human alerts
  entirely (no Telegram, not REVIEW-tagged). Saved photos of HUMAN-status detections
  are kept 48h then purged; the DB row is kept as a metadata-only record. The shipped
  fix is a **MegaDetector person-gate (`human_detection_confidence` >= 0.3) OR
  `homo`-taxon check**, evaluated before the animal branch — broader than the
  originally-proposed taxon-only REVIEW-tag approach, because most human captures
  turned out to be blurry NO_ANIMAL/UNCLASSIFIABLE frames the classifier never
  confidently tags `homo` at all. See `runs/0004-human-main-channel-leak.md`
  ("Resolution" section, appended, leak-watch log kept intact).
- **Separately, a real false negative on 07-07 ~19:10** (Daniel watched a bird bathe
  at the pond; 17 motion triggers, 4 captured bursts, all discarded silently by the
  8.6-9.4 < 11.0 sharpness floor — zero DB rows, zero notification) forced a second
  fix: the blur gate no longer silently drops below-floor bursts. Every burst now
  gets species ID + a DB row; a blurry burst with an animal found still alerts, a
  blurry burst with no animal found is DB-logged but muted (not sent to Telegram),
  so REVIEW volume doesn't rise. New run doc: `runs/0005-blur-gate-false-negative.md`.
- **Both fixes shipped together on branch `fix/human-gate-blur-gate`** (5 commits:
  config + DB + purge + blur-gate-no-drop + blur-gate-notify-mute), merging Tasks 1-4
  (code) and this Task 5 (docs). New config: `SPECIES_HUMAN_DETECTION_CONFIDENCE`
  (0.3), `PERFORMANCE_SUPPRESS_HUMAN_ALERTS` (true), `PERFORMANCE_HUMAN_RETENTION_HOURS`
  (48). New `DetectionStatus.HUMAN`. Full description in `CLAUDE.md`.
- **LOOP: read this before flagging an anomaly.** Config/behavior changed today —
  the nightly loop's volume and rollback baselines assume "stock config" and must be
  re-read in light of both fixes:
  - **Telegram notification volume will DROP.** Humans are now fully suppressed
    (previously some leaked to main channel, e.g. 1633/1694 on 07-07) and blurry
    no-animal bursts are now muted instead of occasionally clearing the old
    sharpness floor and reaching REVIEW. Do not read a volume drop vs the 42/night
    baseline as a trigger-side collapse — check `n_human` / human-tagged DB rows
    and `below_sharpness_floor` rows before concluding motion detection broke.
  - **DB rows/day will RISE, roughly ~2x.** Below-floor bursts that used to vanish
    with zero trace (no DB row at all) are now logged every time. This is a
    measurement-completeness change, not trigger-volume growth — do not read a
    jump in `total_triggers` as an anomaly or as evidence the motion threshold
    needs retuning.
  - Neither shift is an FP or FN regression signal by itself. If `fp_rate` or
    `fn_rate` genuinely move, attribute using the new `detection_status=human` and
    `sharpness_info.below_sharpness_floor` fields before concluding a knob needs
    to change.
- **No env-lever deploy recorded in `state.json.deployed`** — both fixes are code
  defaults already in the running config (not env-var overrides), so `deployed={}`
  ("stock config") remains literally true even though behavior changed. This is
  exactly why this entry exists: `deployed={}` is no longer a reliable proxy for
  "nothing changed" starting today. `experiments/state.json` backlog entries #5 and
  #6 updated to `concluded`/`live` to match the run docs.

## 2026-07-09 — SHIPPED: observability columns, file logging, best-guess caption, dusk short-exposure bias (ADR-004 Tasks 1-4) — four loop-facing notes below

Branch `feat/observability-and-dusk` (Tasks 1-4, code) merged today. Four
independent, individually-reversible changes; the loop must read all four
notes below before attributing any metric shift to an anomaly.

**(a) The observability columns runs/0005 told you to use for attribution
are now real DB columns — starting today.** The `detections` table gained
five nullable columns via the existing migration mechanism: `sharpness_score`
(REAL), `below_sharpness_floor` (BOOLEAN), `person_confidence` (REAL),
`top_species_raw` (TEXT), `top_species_score` (REAL). **They are populated on
every detection logged from 2026-07-09 onward and NULL on every row before
that date — there is no backfill.** The 07-08 entry above told the loop to
"attribute using the new `detection_status=human` and
`sharpness_info.below_sharpness_floor` fields before concluding a knob needs
to change" — `sharpness_info.below_sharpness_floor` existed in memory/logs
at that point but was never persisted to a queryable column; as of today it
(and `sharpness_score`, `person_confidence`) is. Those attribution
instructions are now actually actionable via SQL, not just via reading log
lines. See `CLAUDE.md` ("Observability columns" bullet) and commits
`91356a6`/`83c9d69`/`3d7d52d` (Task 1), `86979b3`/`8de0b4f`/`30376a6` (Task 3,
adds `top_species_raw`/`top_species_score` + the "Best guess" caption line).

**(b) Dusk sharpness scores are expected to RISE starting today — this is
the intended effect of Task 4, not an anomaly.** `CameraConfig.ae_exposure_mode`
now defaults to `"short"` (env `CAMERA_AE_EXPOSURE_MODE`, `normal|short|long`),
biasing libcamera's auto-exposure toward shorter exposures at dusk/low light
— the direct fix for the mechanism `runs/0006-dusk-short-exposure.md`
diagnoses behind both the 07-07 silent-drop incident and the 07-08 19:33-19:35
marginal below-floor alerts (10.0-10.4 vs. the 11.0 floor). **If a future
tick sees dusk-hour `sharpness_score` values trending up and
`below_sharpness_floor` rows at dusk trending down, that is this fix working
as designed — do not flag it as a data anomaly or a sensor change.** Watch
for the opposite failure mode instead: a *midday* sharpness regression would
be forcing evidence the short-exposure bias trades away too much
brightness/gain even in good light. **Rollback lever:**
`CAMERA_AE_EXPOSURE_MODE=normal` + `sudo systemctl restart
wildlife-camera.service` — single env var, no schema change, no code
rollback needed. See `runs/0006-dusk-short-exposure.md`.

**(c) Verify the first 48h human-purge cycle on the first tick after
2026-07-10 ~14:49.** Detection id **1725** (2026-07-08 14:49,
`capture_20260708_144907_frame*.jpg`) is a `DetectionStatus.HUMAN` row from
before today's deploy, and is the earliest HUMAN row old enough to exercise
`PERFORMANCE_HUMAN_RETENTION_HOURS` (48h, shipped 07-08 per `runs/0004`
Resolution) end-to-end since that feature went live. **On the first loop tick
that runs at or after 2026-07-10 ~14:49, check: (1) the `capture_20260708_144907_frame*.jpg`
files are gone from disk (purged); (2) the DB row for id 1725 is still
present and intact (metadata-only, per design — timestamp/status/confidence
kept, only the image files deleted).** If the files are still present past
that time, or the DB row is missing/altered, that is a real purge-mechanism
bug worth a new run doc, not a one-off to silently ignore.

**(d) `deployed={}` still means "no env-lever override," not "no behavior
change" — same posture as 07-08, reaffirmed.** All four of today's changes
(observability columns, file logging, best-guess caption, AE short-exposure
bias) are **code defaults**, not env-var overrides Daniel opted into, so
`state.json.deployed` stays `{}` even though DB schema, logging destination,
notification captions, and camera exposure behavior all changed today. Keep
reading `experiments/runs/000{1..6}` and this JOURNAL, not just `deployed`,
to know what's actually different about the running system.

Separately (Task 2, `ee8cdcf`/`524b19d`): `configure_logging(config)` now
installs a `RotatingFileHandler` at `<log_dir>/wildlife.log` (5MB × 5
backups, INFO+) alongside the console handler, because journald history was
lost on the 2026-07-08 21:14 reboot and took forensic log lines with it.
`StorageConfig.log_dir` (env `STORAGE_LOG_DIR`, default `data/logs`);
`logs_dir` is now a property aliasing `log_dir`. Verify after restart:
`systemctl status wildlife-camera.service` is `active`, and
`data/logs/wildlife.log` shows INFO lines flowing plus an AE-mode log line
(`"Auto-exposure mode: short"` from `_apply_ae_exposure_mode`).

## 2026-07-09 (loop tick; covers the missed 07-08 night too)

**Window:** ids 1707–1786, 80 triggers over two nights (07-08 + 07-09). The
07-08 tick never ran (`last_tick_completed_day` was 07-07), so `loop.metrics`
stamped both nights under date 2026-07-09. Volume 40/night vs baseline 42 —
inside the collapse/explosion band.

**Headline FP rate is 37.7%, not 95%.** `fp_human_rate = 20/53` (CI 26–51%).
The `fp_rate=0.56` field mixes in 22 tier-1 auto-labels whose `fp_md_rate` is
1.0 *by construction*, so it is not the truth number (see memory: auto-labels
are not truth). This is the first night with enough human labels (53) to say
anything real — and it says the system is far better than the auto-label
headline has been claiming for weeks. Prior nights' ~95% figures were
tier-1 tautology, not measured performance.

**Notification gate (exp #1) validated on human labels for the first time.**
All 20 human-labelled FPs have status ∈ {no_animal (16), unclassifiable (4)} —
i.e. the 🔍 REVIEW prefix catches 20/20 of them. Main-channel (`identified`)
precision was 31/31 = 100%. Cost: 2 real animals demoted to REVIEW.

**FN, measured for the first time (2 of 33 human-`animal` rows ≈ 6%).**
`loop.metrics` still reports `fn_rate: "unmeasured"` because it only derives FN
from an `fn_audit` timelapse pass that is not implemented. But the
classification-FN signal is available *today* by joining human `animal` labels
against `no_animal`/`unclassifiable` status (as CLAUDE.md documents): ids **1718**
and **1733**. Tier-2 adjudication of the frames: **1718 is a confirmed FN** — a
blackbird sits plainly on the gravel by the water spout in
`capture_20260708_114905_frame3.jpg`, logged `no_animal`. 1733's best frame
shows no animal I can confirm; left unasserted. Note these two are *classification*
FNs (trigger fired, classifier missed); they are NOT *trigger* FNs (animal present,
no capture at all), which remain genuinely unmeasurable without a timelapse pass.
Do not conflate them — an FP experiment must not claim FN safety on this number.

**Exp #6 (dusk-short-exposure) ROLLED BACK the same day it shipped.**
Its success metric was "dusk `sharpness_score` rises above the 11.0 floor, no
midday regression." Measured offline on 537 saved frames (the DB columns only
begin 07-09, so the frames were the only pre-deploy record — and retention is at
cap, so this was the last tick that could do it):

- AE=short *is* live and working mechanically: `wildlife.log` logs
  `Auto-exposure mode: short` at both restarts, and matched-hour 19h luma fell
  68.9 → 52.0. Shorter exposure, darker frame.
- Dusk `sharpness_score` **fell** (19h 10.19 → 9.76); 6 of 12 post-deploy rows
  landed below the floor (17:30–18:49 at 7.1–9.1).
- The pre-registered midday-regression trigger fired (16h 19.18 → 15.07).

The structural point: `sharpness_score` is Laplacian variance, which scales with
frame contrast (≈ luminance²). `AeExposureMode=Short` lowers luminance by design.
**The fix mathematically lowers the number it was shipped to raise.** It was
doomed by construction, and no amount of additional dusk data would have shown
otherwise. Because it depresses `sharpness_score` globally, it makes the
`runs/0005` mute path (below-floor AND no animal found → no Telegram) strictly
more reachable — the silent-FN class 0005 exists to close. FN unmeasured +
plausible FN rise ⇒ **FN-veto ⇒ rollback**, per the guardrail contract.

Rolled back via the lever `runs/0006` itself names: `CAMERA_AE_EXPOSURE_MODE=normal`
appended to `.env` (backup `.env.bak.20260709`), `Config()` verified to read
`normal`. **`.env` is gitignored — this change is invisible to git and is recorded
here on purpose.** It did *not* go through `loop.deploy`: `guardrails.BOUNDS` only
holds numeric `(low, high)` ranges and rejects `CAMERA_AE_EXPOSURE_MODE` as "not a
tunable parameter", so `state.deployed` stays `{}` and no `deployed_config.env`
is rendered. `pending_restart_at=2026-07-10T03:00:00+02:00`; `wildlife-deploy.timer`
fires 03:30 and `apply_pending_deploy` restarts on any due stamp regardless of
whether a delta was rendered. Exp #6 reopened as backlog id 7 (`running`) to
collect the 17–19h AE=normal baseline that never existed.

**New backlog id 8 — `sharpness-floor-is-a-brightness-gate` (the real root cause).**
Unconfounded, 470 pre-deploy AE=normal frames across multiple days:
P(lap<11.0) = 0% at luma≥80, **71% at luma 60–80**, 100% at luma<40, 0% at 80–100
and 100–130. `min_sharpness_threshold=11.0` is operationally a **light-level gate**,
not a blur gate: at dusk nearly every burst is "below floor" however sharp it is.
That, not the AE mode, decides whether a dusk burst can be silently muted.
Corroborating: the confirmed-FN frame 1718 is uniformly soft across the *whole*
scene, foreground and background — that is focus/contrast, not motion blur.
Candidate fixes (all code, no env knob reaches this): brightness-normalized
sharpness (`lap/gray_var`) as the floor statistic; a luma-dependent floor; or drop
the mute path and lean on the REVIEW prefix. **This is the next experiment.**

**Verification duties.**
- Human/privacy gate (`runs/0004` leak-watch): 5 HUMAN-status rows (1725, 1742,
  1743, 1773, 1786), **zero** of them carry any feedback label — no keyboard was
  ever attached, so none was notified. Gate holding. id 1786 recorded
  `person_confidence=0.42`.
- Blur gate (`runs/0005`): id 1781 (17:36, no_animal, `below_sharpness_floor=1`)
  was DB-logged and muted, exactly as designed. It is also, by construction, an
  *unobservable* FN candidate — muted means never labelled.
- Observability columns went live with the 15:49 restart; first populated row is
  id 1775. Recomputing Laplacian variance from `image_path` reproduced stored
  `sharpness_score` within ±0.3 on 11 of 12 rows, so the column is trustworthy.
- id 1725 purge check is **not yet due** (07-10 ~14:49); frames still present, as
  expected. Next tick must check it.
- New 5-button feedback keyboard shipped today, but all 53 human labels this
  window use the legacy vocabulary (`animal`/`false_positive`). No
  `animal_wrong_id`/`person`/`cant_tell` yet — consistent with the sidecar not
  having been restarted, or simply with no new-keyboard message being labelled
  yet. Worth confirming next tick before reading anything into label mix.

---

## 2026-07-10 — tick (loop-day 2026-07-10)

**Metrics (07-10 window, 62 new triggers since watermark 1786).** Human-labeled
FP 11/25 = **44%** (CI 0.27–0.63), statistically flat vs 07-09's 37.7% (CIs
overlap). All 11 human-FP rows are NO_ANIMAL/UNCLASSIFIABLE status → 100%
REVIEW-tagged, **zero clean-alert FP leaked to the main channel**. md-auto FP
16/17, cant_tell=1 (excluded from denominators). FN still `unmeasured`.
No volume collapse/explosion. No env deploy.

**New 5-button keyboard is live and in use** (resolves last tick's open
question): `person` (1793,1806,1821), `cant_tell` (1792), `animal_wrong_id`
(1799,1813,1818) all appear as `source=human` labels this window. Sidecar was
restarted; the legacy-only label mix from 07-09 is gone.

**Exp #7 (dusk-short-exposure) → CONCLUDED.** Reopened purpose (AE=normal 17–19h
dusk baseline) fulfilled. AE=normal went live at the 03:00 CEST restart;
first dusk under it (07-10 17–18h) scored 6.9–9.4 — **indistinguishable** from
AE=short's 07-09 17–18h (7.8–8.6), both below the 11.0 floor. AE mode is not the
lever. AE=normal retained. See runs/0006 Conclusion.

**Exp #8 (sharpness-floor-is-a-brightness-gate) → promoted to active/running.**
runs/0007 written. The floor is a light-level gate (P(lap<11)=0% at luma≥80,
100% at luma<40), so the blur-gate MUTE path (`is_blurry_review`,
wildlife_system.py:657: below-floor AND no-animal → no Telegram) fires at dusk
as a function of darkness, silently dropping possible dark-frame animal misses
(unobservable FN). Mute path fired 4× this window (1787 morning, 1845/1846/1848
dusk); one below-floor *animal* (1847, 18:23) correctly alerted. **Recommended
fix: brightness-gate the mute** — add mean-gray luma to `sharpness_info`, only
mute when `luma ≥ ~70` (new `blur_mute_min_luma` knob); below that, send as
REVIEW. FN-reducing (FN-veto does not block), reversible (git revert + restart),
volume cost bounded to dark no-animal bursts (~3/night, all REVIEW-tagged).
**HELD tonight** pending Daniel's OK on the small REVIEW-volume increase (his
standing product lever) + TDD/subagent implementation; no fire forces it.
Alternatives (a) lap/gray_var and (c) drop-mute-entirely recorded, not chosen.

**Verification duties (all pass).**
- Human/privacy gate leak-watch: 24 HUMAN-status rows this corpus, **zero**
  carry any feedback label — none was ever notified. Gate holding.
- id 1725 HUMAN purge (due 07-08 14:49 + 48h): frame gone ✓. 1742/1743 frames
  also gone (rolled off by the ~100-burst storage cap, <48h). 1773 (07-09 15:32)
  and 1786 (07-09 19:18) frames retained — within 48h and recent. Correct.
- Blur/observability columns trustworthy (recompute matched stored values in
  prior tick); new rows populate all five columns.

---

## 2026-07-11 — tick (loop-day 2026-07-11)

**Metrics (07-11 window, 25 new triggers since watermark 1848).** Human-labeled
FP **1/7 = 14%** (CI 0.03–0.51; small n, statistically consistent with 07-10's
44%). fp_md 16/18. The one human FP (1863, no_animal) is REVIEW-tagged → **zero
clean-alert FP leaked to the main channel**. No volume collapse/explosion. No env
deploy. FN reported `unmeasured` by `loop.metrics`, but see below — it is no
longer zero-signal.

**First observable FNs.** Two human `animal_wrong_id` labels on review-status
rows: 1861 (16:05, no_animal, luma 99, sharpness 15.4) and 1862 (16:28,
unclassifiable, luma 88, sharpness 12.5). Both **above** the sharpness floor and
in good light → **classifier recall misses**, not blur-gate mutes. No env knob or
sharpness/luma change addresses classifier recall; recorded qualitatively (the
metrics join for animal-on-review-status FN is not implemented).

**Exp #8 (brightness-gate the blur mute) → HELD again, no deploy.** Adjudicated
all 7 mute-path firings (1864–1869 @ 16:46–17:55, 1873 @ 21:04): **all
true-negatives, no concealed animal**. Premise-revising finding: 6/7 fired in
*daylight* (luma 71–81), only 1873 was dark (luma 20); the frames are uniformly
**soft-focus**, so sub-floor scores this window are driven by focus softness, not
brightness. A `blur_mute_min_luma≈70` gate would have un-muted exactly 1 row
(1873, no animal) → ~nil live benefit, +1 REVIEW msg. Held pending (a) Daniel's
OK on REVIEW volume (no approval signal in state) and (b) lower demonstrated
urgency. New side-observation: soft focus may be depressing sharpness generally —
candidate focus check via `scripts/camera_preview.py`, orthogonal to AE (exp #7)
and the floor statistic. See runs/0007 Observations 2026-07-11.

**Verification duties (all pass).**
- Human/privacy gate: no HUMAN-status triggers this window; leak-watch = **0**
  HUMAN rows carry any feedback label (gate holding, none notified).
- HUMAN purge: latest HUMAN rows are 07-10, all within 48h, frames correctly
  retained; none past-48h lingering. Purge functioning.
- 5-button keyboard in active use (animal, animal_wrong_id, false_positive all
  present this window as source=human).

---

## 2026-07-12 — tick (loop-day 2026-07-12)

**Window:** 53 new triggers (ids 1874–1926, watermark 1873→1926). Status: 23
HUMAN / 15 no_animal / 5 unclassifiable / 10 identified. **Zero human feedback
labels** this window → FP ground truth **unmeasured**; `loop.metrics` fp_rate
0.67 is MegaDetector auto-only (n_md 30, n_human 0), not truth. FN unmeasured.
No env deploy, no code change. Volume 53 vs baseline 42 — no collapse/explosion.

**Dominant event: sustained human presence 17:40–18:27** — 23 HUMAN-status rows,
20/23 pconf ≥ 0.35 (person in red trousers visible in frames), all correctly
SUPPRESSED. Human/privacy gate leak-watch **0** (no HUMAN row carries feedback),
purge clean (**0** HUMAN frames past 48h on disk). Gate working exactly as
designed.

**Mute-path adjudication: 6 firings, 0 concealed animals** (2nd night running).
ids 1889/1892/1906/1925/1926 = empty soft-focus pond (TN); 1908 = the same HUMAN
(pconf 0.17 < 0.3, no `homo` taxon → slipped the human gate but muted by the blur
gate anyway; no leak, no animal). The blur-mute path hid no FN.

**Exp #8 (brightness-gate the blur mute) → HELD again.** No Daniel greenlight on
REVIEW volume; mute concealed 0 animals two nights running → live benefit ~nil;
tonight's below-floor firings again dominated by soft focus, not darkness, which
the luma-gate doesn't touch. No fire.

**Soft-focus is now a two-night pattern (07-11 + 07-12).** Every frame out of
focus, day and dusk; raw Laplacian 6–11 even at good luma. Escalating the
physical focus check (`scripts/camera_preview.py`) to the top actionable item —
soft focus depresses sharpness globally AND plausibly lowers classifier recall
(blurry animals missed), an FN driver no env/code lever reaches. Orthogonal to
AE (exp #7) and the floor statistic (exp #8). See runs/0007 Observations 2026-07-12.
