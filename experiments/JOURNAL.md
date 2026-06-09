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
