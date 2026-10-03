# Detection gates — authoritative reference

How a captured burst becomes a status, and which gates decide whether it reaches
Telegram. This is the detailed reference split out of `CLAUDE.md` (2026-10-03);
per-experiment narrative lives in `experiments/runs/*.md`, `experiments/JOURNAL.md`
and `experiments/LEARNINGS.md`.

**Universal invariants** (true of every mute gate below unless stated otherwise):

- Every burst is species-ID'd and DB-logged. A gate only skips the Telegram send.
- Exactly one suppression log per burst: the earliest-precedence gate wins, and every
  later gate's `is_*` flag is ANDed with `not <every earlier gate>`
  (`WildlifeSystem._process_and_notify_detection`).
- Gates fail **open** (never mute) on any exception, missing input, or a `0`/disabled
  setting. Exceptions are called out explicitly (exp #39 Phase 1).
- Mute-gate DB columns use one convention: `True`/`False` when the gate evaluated the
  row, `NULL` when the gate didn't apply (wrong status, disabled, or row predates the
  column). No backfill.
- "Review-class" = `DetectionStatus.NO_ANIMAL` or `UNCLASSIFIABLE`
  (`data_models._REVIEW_STATUSES`). Only review-class bursts get the 🔍 REVIEW prefix.
- Config is read from `.env` then `experiments/deployed_config.env` (the latter wins,
  for `MOTION_*`, `PERFORMANCE_*`, `SPECIES_*`). The loop writes deployed values only
  via `loop.deploy`, bounded by `src/loop/guardrails.py::BOUNDS`.

## 1. Notification precedence table

Order is the `if/elif` chain in `_process_and_notify_detection`. "Default" = code default
in `src/config.py`; **deployed** values are from `experiments/deployed_config.env`
(as of 2026-10-03) where they differ.

| # | Gate | What it does | Key setting(s) — default (deployed) | DB column | Log tag | Rollback lever | Origin |
|---|------|--------------|-------------------------------------|-----------|---------|----------------|--------|
| 1 | Human/Privacy | Suppress HUMAN-status bursts entirely | `SPECIES_HUMAN_DETECTION_CONFIDENCE` 0.3 (**0.5**); `PERFORMANCE_SUPPRESS_HUMAN_ALERTS` true | `detection_status='human'`, `person_confidence` | `[HUMAN-GATE]` | `PERFORMANCE_SUPPRESS_HUMAN_ALERTS=false` | runs/0004, 0008, 0012, 0016 |
| 1a | Burst human sweep (upstream) | Re-ID divergent sibling frames of a review-class burst; escalate to HUMAN | `PERFORMANCE_HUMAN_SWEEP_DIVERGENCE_THRESHOLD` 0.03; `PERFORMANCE_HUMAN_SWEEP_MAX_FRAMES` 2 (**4**) | (status becomes `human`) | `[HUMAN-SWEEP]` | either `=0` | runs/0015, 0017 |
| 2 | Human-Proximity (window OR density, demoted-band widening) | Mute review-class / unnamed-animal bursts near or amid human activity | `PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS` 120 (**240**); `..._HUMAN_DENSITY_WINDOW_SECONDS` 1800; `..._HUMAN_DENSITY_COUNT` 8; `..._HUMAN_DEMOTED_PERSON_FLOOR` 0.3; `..._HUMAN_DEMOTED_WINDOW_SECONDS` 1800 | `human_proximity_muted` | `[HUMAN-PROXIMITY]` | window `=0`; density count `=0`; demoted window `=0` | runs/0010, 0018, 0019 |
| 3 | Blur (luma-conditioned) | Mute below-sharpness-floor review-class bursts, only when bright enough that low sharpness means blur | `PERFORMANCE_MIN_SHARPNESS_THRESHOLD` 11.0; `PERFORMANCE_BLUR_MUTE_MIN_LUMA` 70 | `sharpness_score`, `below_sharpness_floor` | `[BLUR]` | no dedicated lever; `PERFORMANCE_BLUR_MUTE_MIN_LUMA=255` effectively disables, or `git revert 683f5f3` | runs/0005, 0007 |
| 4 | Confident-Blank | Mute review-class bursts whose raw top-1 is `blank` at ≥ threshold | `PERFORMANCE_BLANK_CONFIDENCE_MUTE_THRESHOLD` 0.92 | `blank_confidence_muted` | `[BLANK-CONF]` | `=0` (human only; loop bounds 0.87–1.0) | runs/0021 |
| 5 | Review Sampling | Send only a deterministic fraction of surviving review-class bursts | `PERFORMANCE_REVIEW_SAMPLE_RATE` 0.25 (**0.5**) | `review_sampled_out` | `[REVIEW-SAMPLE]` | rate `=1.0` | runs/0009 |
| 6 | Deferred REVIEW send (cancel-on-human) | Hold surviving review sends; cancel if a HUMAN burst lands within the hold | `PERFORMANCE_REVIEW_DEFER_SECONDS` 240 | `human_proximity_muted` (follow-up UPDATE) | `[REVIEW-DEFER]` | `=0` | runs/0010 |
| — | Send | MAIN (IDENTIFIED / ANIMAL_UNCERTAIN / ERROR) or 🔍 REVIEW (review-class) | `PERFORMANCE_REVIEW_PREFIX_ENABLED` true | — | — | — | runs/0001 |

Not a notification gate, but same family: **human-retention purge** (§13) and
`gate_would_suppress` (ADR-004 shadow column: `not animals_detected`, no routing effect;
the `[GATE-SHADOW]` log line was removed 2026-10-03).

Gate numbers 3, 6 and 7 of the earlier table (Unnamed-Animal Blank-Raw, Scene-Unchanged,
Animal-Proximity exemption) were retired 2026-10-03 — see §15. Sections §5, §8, §9a, §9b
no longer exist; numbering of the remaining sections is unchanged.

## 2. Status routing (`species_identifier.py`)

`SpeciesIdentifier._parse_*` assigns one `DetectionStatus` per identification, in this
order (first match wins):

1. **HUMAN** — the human/privacy gate (§3), evaluated *before* the animal branch, so a
   frame with both a person and a confident animal routes to HUMAN.
2. **NO_ANIMAL** — MegaDetector found no animal box ≥ `SPECIES_MIN_DETECTION_CONFIDENCE`
   (default **0.2**).
3. **UNCLASSIFIABLE** — ensemble label contains the `no cv result` sentinel (crop
   unreadable). Precedes the confidence check (score is ~0 here).
4. **NO_ANIMAL (blank routing, exp #13, commit `55234f1`, runs/0011)** — ensemble label
   is SpeciesNet's empty-frame verdict: last segment `blank` and every taxonomy segment
   empty (`_is_blank_prediction`). It is emitted at ~0.99, so it used to clear the
   threshold and fire a MAIN alert that bypassed every review-class gate.
   `animals_detected=False`; metadata kept so `top_species_raw` records it. A populated
   taxonomy ending in `blank` is unaffected. Rollback: `git revert 55234f1`.
5. **ANIMAL_UNCERTAIN** — ensemble score < `SPECIES_UNKNOWN_SPECIES_THRESHOLD` (0.5).
6. **IDENTIFIED** — otherwise. Includes the fully-generic `<uuid>;;;;;;animal` rollup
   ("MegaDetector boxed something, classifier can't name it"; its ensemble confidence
   *is* the box confidence) — see §6 and §7.

**ERROR** on pipeline failure; `identify_species` always returns a valid result.

Label helpers in `utils.py` (all treat sentinel segments `no cv result`/`blank` as empty,
per exp #23's lesson): `is_blank_label`, `is_unnamed_animal_label` (last segment
`animal`, all taxonomy empty — `aves;;;;;bird` does *not* match),
`is_named_animal_label` (not unnamed, not blank, no `homo` segment, at least one
non-empty non-sentinel segment), `extract_common_name`.

## 3. Human/Privacy Gate

**Triggers** (any one → `DetectionStatus.HUMAN`, `species_identifier.py`):

- MegaDetector person-box confidence ≥ `SPECIES_HUMAN_DETECTION_CONFIDENCE`.
- A `homo` segment in the ensemble label.
- **Raw-classifier homo leak** (exp #9, runs/0008): raw classifier top-1 has a `homo`
  segment AND the ensemble did *not* name a specific animal
  (`_is_specific_animal_taxon`: non-empty genus+species). Catches a person the
  ensemble rolled up to blank/generic/unclassifiable while the person box was
  sub-threshold. **Exp #23 (commit `479e0ac`, runs/0016)**: this trigger was inert from
  shipping until 2026-09-11 because `no cv result` in every segment read as a
  "specific animal"; `_strip_sentinel` now treats sentinels as empty.

**Threshold.** Code default 0.3; **deployed 0.5** since exp #14 (runs/0012, commit
`6d8bcc1` + env delta, live 2026-08-31). Measured: 32/33 HUMAN bursts below 0.5 were the
empty garden (sub-threshold MegaDetector noise; 38% of all triggers were being
suppressed as HUMAN); every burst ≥ 0.5 was a real person. Loop BOUNDS `(0.3, 0.7)` —
floored at the old default, capped so the loop can't gut the gate. The 0.3–0.5
"demoted band" is covered downstream by the demoted-band widening (§4).

**Effects.** `PERFORMANCE_SUPPRESS_HUMAN_ALERTS` (true): no Telegram at all (not even
REVIEW). Still DB-logged; HUMAN row metadata carries `person_confidence` and the raw
top-1 (exp #23) so a later tick can tell *which* trigger fired. Photos purged after
48h (§13). Each HUMAN classification updates `_last_human_detection_at` and
`_recent_human_detection_times` (anchors for §4 and §11).

`person_confidence` is recorded on every parsed result (0.0 when no person box), not
only on HUMAN rows.

### 3a. Burst human sweep (exp #21 commit `f73a8ae`, runs/0015; exp #24 commit `c5fe171`, runs/0017)

The human gate judges **one** frame per burst — the sharpest — and sharpness is
uncorrelated with whether a person is visible (burst 5119 leaked a child's face because
the only one of five frames without a person won selection by 1%). When the selected
frame comes back review-class, `WildlifeSystem._burst_human_sweep` measures each sibling's
divergence from it (fraction of differing pixels on a 240×135 grayscale downsample);
siblings ≥ `PERFORMANCE_HUMAN_SWEEP_DIVERGENCE_THRESHOLD` (0.03) are re-identified,
most-divergent first, up to `PERFORMANCE_HUMAN_SWEEP_MAX_FRAMES`, stopping at the first
HUMAN result, which replaces the burst's result (so every downstream human path applies).

- Threshold 0.03: over 142 review-class bursts, the two person bursts scored 0.169/0.215
  (ranks 1–2), busiest empty burst 0.082. Sweeps ~10% of review bursts, ~10s per frame.
- Exp #24: the raw >40-grey-level measure was blind at dusk (burst 5169, mean luma 11,
  scored 0.0005 though frames were unrelated); frames are now contrast-normalised before
  divergence. Same tick raised max frames **2 → 4 (deployed)**; BOUNDS `(0, 4)`.
- Can only route *more* bursts to HUMAN. Fails to the single-frame result on error.
- Rollback: either setting `=0`.

## 4. Human-Proximity Mute Gate

MegaDetector scores extreme close-up / motion-blurred partial bodies at ~0.02–0.15 person
confidence, so such bursts land `no_animal` and leak a person to REVIEW (ids 3544/3553/
3554, 76–108s after a HUMAN burst). Origin exp #11, runs/0010.

**Scope**: review-class bursts, **plus** (exp #26, commit `80d0c00`, runs/0018)
IDENTIFIED bursts whose label `is_unnamed_animal_label`. A person at close range produces
exactly that rollup (bursts 5222/5270 reached MAIN as "animal detected"); re-routing the
label was FN-vetoed (18/78 such rows are human-labelled animals), so the gate's scope
was widened instead — mutes 4/78, zero labelled animals.

**Mute if window OR density** (computed in `process_detection`, persisted on INSERT):

- **Window**: `0 <= burst_time − _last_human_detection_at <= W`, W =
  `PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS`. Code default 120; **deployed 240** since
  2026-07-28 (exp #11 extension; nearest human-labelled animal review row sits 329s after
  a HUMAN burst, so 240 keeps a 37% margin). BOUNDS `(0, 600)`.
- **Demoted-band widening** (exp #27, commit `3752a7b`, runs/0019): if the burst's *own*
  `person_confidence >= PERFORMANCE_HUMAN_DEMOTED_PERSON_FLOOR` (0.3), W becomes
  `max(W, PERFORMANCE_HUMAN_DEMOTED_WINDOW_SECONDS)` (1800). Burst 5305 (pc 0.436, 480s,
  density 5) missed both conditions; no global env value reaches it without a known FN
  (window ≥480 mutes id 1838 at 329s; count ≤5 mutes id 2011). Max pc over all 20
  human-labelled animal rows is 0.0789. Mutes 4/3103 rows, zero labelled animals
  (~1.3/month). Fails open on `None` pc. Rollback: demoted window `=0`.
- **Density** (exp #11 extension, 2026-07-28): ≥ `PERFORMANCE_HUMAN_DENSITY_COUNT` (8)
  HUMAN detections in the trailing `PERFORMANCE_HUMAN_DENSITY_WINDOW_SECONDS` (1800) —
  "the garden is occupied" (long gardening sessions produced leaks 432s/732s after the
  last human burst). List seeded at startup via
  `DatabaseManager.get_recent_human_detection_times`, pruned before every use.
  Rollback: count `=0` (window condition untouched).

Anchors are seeded at startup (`get_last_human_detection_time`). Log reason text is
`window` / `density` / `demoted-band window` (the last only when widening alone caused
the mute). Muted rows are purged on the 48h human policy (§13). Validation: the closest
of the 12 human-labelled animal review rows (since the gate went live 2026-07-08) is 329s
from a preceding HUMAN burst.

## 6. Blur Gate (luma-conditioned)

`min_sharpness_threshold` (11.0, Laplacian variance) no longer discards bursts: a
below-floor burst gets a real image path + `sharpness_info`, is species-ID'd and logged
(runs/0005, exp #6). An animal found in a below-floor burst still alerts (caption notes
`below_sharpness_floor`). A review-class below-floor burst is muted **only if** the best
frame's mean luma ≥ `PERFORMANCE_BLUR_MUTE_MIN_LUMA` (70; exp #8 commit `683f5f3`,
runs/0007, concluded keep 2026-07-21).

Why luma: Laplacian variance scales with brightness, so the floor is really a
light-level gate — P(below floor) is 100% at luma 0–40 and 0% above ~80. Muting on floor
alone silently dropped dusk animals (a dusk blackbird at luma 67.8). Missing luma → no
mute (FN-safe). Loop BOUNDS `(0, 255)`.

Related: `CAMERA_AE_EXPOSURE_MODE` (exp #7, runs/0006) tried to lift dusk sharpness via
`short` AE bias; it could not and was rolled back — `.env` sets `normal` (code default is
still `short`).

## 7. Confident-Blank Mute Gate (exp #29, commit `3c2856c`, runs/0021)

Mute a review-class burst when the classifier's **raw** top-1 (`top_species_raw`, before
geofence/rollup) `is_blank_label` with score ≥ `PERFORMANCE_BLANK_CONFIDENCE_MUTE_THRESHOLD`
(0.92).

- Evidence (182 review-class rows with blank raw top-1): the 5 human-labelled
  animal/animal_wrong_id rows score 0.6431–**0.8475**; 42 confirmed FPs median 0.9219,
  max 0.9825. 0.92 mutes 52/182 (29%), zero animal- and zero person-labelled (highest
  person row 0.9116). Margin 0.0725 = 3.6× the pre-registered `max(animal)+0.02` rule.
- This is the "different discriminator" exps #16/#17 called for: scene similarity
  (the since-retired Scene-Unchanged Gate) did *not* separate animal (0.7886–0.9202) from
  FP (0.5335–0.9675); classifier blank-confidence does.
- **Loop BOUNDS `(0.87, 1.0)`** = 0.8475 + 0.02: the loop can never deploy at/below the
  animal ceiling. Config allows `[0, 1]` so a *human* can set `0.0` = DISABLE
  (special-cased; a literal `>= 0.0` would mute everything).
- Persisted on INSERT; `NULL` when not review-class or disabled. Fails open to `False`.

## 9. Review Sampling Gate (runs/0009, exp #10)

Of review-class bursts surviving gates 1–4, only `PERFORMANCE_REVIEW_SAMPLE_RATE` are
sent. Deterministic per burst: `_review_sample_fraction(detection_id)` = sha256 →
[0,1); muted when fraction ≥ rate. `None` id → always sends; rate ≥ 1.0 never samples
out (rollback); ≤ 0.0 always does.

- Deployed 2026-07-26 at 0.25 by Daniel's call (≈1 real catch per 155 REVIEW pings);
  **raised to 0.5** on the first night (2026-07-26 tick) after an FN was found in a sampled-out burst.
- Persisted via follow-up `update_review_sampled_out` (needs `detection_id`).
- **Volume lever, not an FP lever**: `fp_rate` is label-conditioned and unaffected;
  `loop.metrics` counts `n_sampled_out` and keeps them out of "Not yet labelled".
  Fewer labels is the intended effect, not a feedback-starved signal.

## 10. Deferred REVIEW Send Gate — cancel-on-human (exp #11 extension, 2026-08-01)

Human-Proximity is backward-looking, so it can't mute the **leading edge** of a visit (a
close-up smear classified `no_animal`, first HUMAN burst seconds later: 3829/3867/3909 at
75/51/81s before; 3909 shows a face). A review-class send that survives every gate is
held: `_schedule_deferred_review_send` builds the annotated image synchronously (it
depends on `last_motion_frame`), then a background task sleeps
`PERFORMANCE_REVIEW_DEFER_SECONDS` (240) and cancels if
`burst_time < _last_human_detection_at <= burst_time + defer` — logs `[REVIEW-DEFER]`,
persists via `update_human_proximity_muted`. Tasks tracked in `_pending_review_tasks`,
cancelled on shutdown. MAIN alerts are never delayed. Fails open (sends) on error.
FN cost: nearest labelled animal review row is 1846s before the next HUMAN burst; cancels
~18% of otherwise-sent reviews. Rollback `=0` (send immediately).

## 11. Observability columns

`detections` columns beyond the original schema (`DatabaseManager._DETECTION_EXTRA_COLUMNS`),
all nullable, no backfill:

| Column | Meaning |
|---|---|
| `animals_detected`, `detection_count`, `max_detection_confidence` | MegaDetector summary |
| `contour_count`, `largest_contour_area`, `foreground_pixel_count`, `background_drift`, `hour_of_day` | motion diagnostics (ADR-004 Phase 1) |
| `gate_would_suppress` | shadow gate: `not animals_detected` (no routing effect) |
| `detection_status` | `DetectionStatus` value |
| `sharpness_score`, `below_sharpness_floor` | blur gate inputs (from 2026-07-09) |
| `person_confidence` | max person-box conf on every parsed result; NULL for unreadable-image errors |
| `top_species_raw`, `top_species_score` | classifier raw top-1 before geofence/rollup |
| `scene_similarity`, `scene_gate_muted` | retired (§15); column kept, NULL on new rows |
| `review_sampled_out` | §9 (follow-up UPDATE) |
| `human_proximity_muted` | §4 / §10 (also set on unnamed-animal IDENTIFIED rows) |
| `blank_confidence_muted` | §7 |
| `unnamed_animal_blank_muted` | retired (§15); column kept, NULL on new rows |

Sharpness luma is in `sharpness_info` but is not a DB column.

## 12. "Best guess" caption line (`WildlifeSystem._best_guess_line`)

When the ensemble label is a generic rollup (empty genus/species, e.g. `aves;;;;;bird`,
`;;;;;;animal`), the caption appends `Best guess: <common name> (NN%)`, even at low
confidence. **Exp #28 (commit `838e5f6`, runs/0020)**: prefers
`metadata['best_geofenced_species']` — the first species-level top-k candidate
(`return_top_k`=5) that SpeciesNet's geofence allows in DEU/NW
(`_find_best_geofenced_species`) — over the ungeofenced raw top-1, which named a Himalayan
thrush (0.42) on bursts where `common blackbird` (0.03–0.06) was in-region. Raw score is
the wrong selection rule; the classifier isn't region-aware. Falls back to the raw top-1;
suppresses guesses that repeat the rollup name. Caption only — no routing, no DB column.
Defensive: a malformed `top_classifier_prediction` → no line and NULL columns.

## 13. Retention and purge rules

- `PERFORMANCE_MAX_IMAGES` (300; raised from 100 on 2026-07-27, BOUNDS `(50, 500)`):
  oldest bursts deleted as units (~282MB per 100 bursts). Note `experiments/loop.md`
  still says "~100 bursts".
- `StorageManager.purge_human_bursts()` runs after every detection:
  - HUMAN rows: photos purged `PERFORMANCE_HUMAN_RETENTION_HOURS` (48) after capture;
    DB row kept metadata-only.
  - Review-class rows within ±`PERFORMANCE_HUMAN_RETENTION_PROXIMITY_SECONDS` (240) of a
    HUMAN detection, either direction (the purge runs ≥48h later, so it may look
    forward). Covers leading-edge leaks. BOUNDS `(0, 3600)`.
  - **Exp #35 (commit `0bc9ec9`, runs/0026)**: any row with `human_proximity_muted = 1`
    (review-class *or* unnamed-animal IDENTIFIED) is purge-eligible unconditionally —
    purge follows the gate's verdict, since density/demoted-band mutes can sit far
    from any single HUMAN row (5444 at 310s). Strict superset; ~1.4 rows/month.
  - Implemented in `DatabaseManager.get_human_adjacent_review_detections(cutoff,
    window_seconds)` with adjacency matched in Python (a correlated SQL subquery took
    3.9s/call). `window_seconds <= 0` short-circuits the whole extension (rollback).
- Timelapse FN-audit stream (`timelapse_writer.py`): every 20s, max 10000 files
  (`PERFORMANCE_ENABLE_TIMELAPSE`, `_TIMELAPSE_INTERVAL`, `_TIMELAPSE_MAX_FILES`); audited
  by `scripts/fn_audit_timelapse.py`.

## 14. Feedback labels (`feedback_protocol.py`, 2026-07-09 redesign)

5-button, 2-row keyboard → `detection_feedback.label`:

| Button | code | label |
|---|---|---|
| ✅ Animal | `a` | `animal` |
| 🐦 Animal, wrong ID | `wid` | `animal_wrong_id` |
| 👤 Human | `p` | `person` |
| ❌ Nothing there | `fp` | `false_positive` |
| 🤷 Can't tell | `ct` | `cant_tell` |

- `person` (not `human`) avoids colliding with `human` as a labeller tier
  (`source='human'` vs tier1/tier2 auto-labels).
- `ws` → `wrong_species` is legacy: still parsed and in `VALID_FEEDBACK_LABELS`, never
  shown. Pre-2026-07-09 `wrong_species` rows can mean human OR animal (check frames).
- `cant_tell` wins reconciliation in `loop/ingest.py`, blocking tier-1/2 auto-label
  backfill. `loop/metrics.py` excludes `cant_tell` **and** (exp #38, commit `568746d`,
  runs/0027) `person` from the fp_rate denominator and every per-tier bucket — a person
  trigger is neither a false alarm nor a wildlife detection. Report lines: "Can't tell",
  person count, "Not sent (review sampling)".
- No FN button: a false negative is a human `animal`/`animal_wrong_id` label on a
  review-class row, found at query time by joining `detections` × `detection_feedback`.
- Headline fp_rate uses human labels only; auto-labels are estimates, never truth.

## 15. Retired 2026-10-03

Retired by Daniel (branch `prune-gates-decide`); code, config fields, `guardrails.BOUNDS`
entries and tests removed. DB columns are append-only and stay (new rows leave them
NULL); the `PERFORMANCE_*` env vars may still be present in `.env` /
`experiments/deployed_config.env` and are ignored (`extra='ignore'`). Full history is in
`experiments/runs/` and git.

- **Scene-Unchanged Gate** (`src/scene_gate.py`, `[SCENE-GATE]`, `scene_similarity`,
  `scene_gate_muted`, `PERFORMANCE_SCENE_GATE_*`; runs/0009, 0022; introduced in
  `53e9bd6`) — 0 mutes since 2026-08-30; near-vacuous at T=0.982 and
  superseded as FP lever by Confident-Blank (§7). The "human-ruled ON" ruling is void.
  `src/loop/scene_watch.py` (camera re-aim watchdog) is separate and unaffected.
- **Unnamed-Animal Blank-Raw Mute Gate** (exp #32, `[UNNAMED-BLANK]`,
  `unnamed_animal_blank_muted`, `PERFORMANCE_UNNAMED_ANIMAL_BLANK_MUTE_THRESHOLD`;
  commit `f4d7730`, runs/0023) — fired once ever. (The Human-Proximity scope over
  `;;;;;;animal` IDENTIFIED bursts, §4, is kept.)
- **Animal-Proximity Review Exemption**, both halves (exp #33 backward, commit
  `ea652bc`, runs/0024; exp #39 forward `[ANIMAL-DEFER]`, commit `657a30c`, runs/0028;
  `PERFORMANCE_ANIMAL_PROXIMITY_WINDOW_SECONDS`, `_last_animal_detection_at`,
  `get_last_animal_detection_time`, `utils.is_named_animal_label`) — 0 / ~1 per two
  months. A sampled-out review burst is again dropped immediately (no annotated image,
  no background task); `_deferred_review_send` is the cancel-on-human phase only (§10).
- Dead code: `[GATE-SHADOW]` log line, `DatabaseManager.get_recent_review_detections`,
  `scripts/validate_scene_gate.py`.
