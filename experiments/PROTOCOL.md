# PROTOCOL — Autonomous Tuning Loop SOP

Read this FIRST every `/loop` tick. You (the judgment layer) reconstruct all
state from this git notebook; there is no hidden stage machine.

## When to spend tokens vs invoke Python
- Deterministic work (SQL reads, metric math, config writes, env render, restarts,
  Telegram sends) is done by `src/loop/*.py`. INVOKE them, read their JSON.
- Spend tokens only on: tier-2 adjudication of ambiguous crops, experiment design,
  self-audit, journaling.

## Use existing data before proposing new instrumentation

Before concluding that a question requires new logging, columns, or schema changes,
verify it cannot be answered from already-captured artifacts:

- **`detections` DB rows** — `id`, `timestamp`, `image_path`, `motion_area`,
  `contour_count`, `largest_contour_area`, `foreground_pixel_count`, `hour_of_day`,
  `gate_would_suppress`, `background_drift`, `detection_status`.
- **Saved frames on disk** — `detections.image_path` points to the actual captured
  JPGs under `data/images/`. These are analyzable offline with OpenCV (perceptual /
  average hashing, frame similarity, clustering, centroid tracking, etc.).
  Scene-recurrence and near-duplicate-frame questions in particular ARE answerable
  now: aHash each saved frame, order by timestamp, compute Hamming distances, and
  cluster — no new columns needed. Write and run the script in-tick
  (`uv run python ...` from repo root, `PYTHONPATH=src`; OpenCV/numpy installed).
- **`detection_feedback` labels** — append-only human/tier-2 ground truth.

**Retention caveat**: storage cleanup retains only the most recent bursts (~100 of
however many triggers exist in the DB). Image-based retro-analysis is time-boxed —
run it promptly rather than deferring, and do not assume older DB rows still have
frames on disk.

Only propose new logging/instrumentation when the needed signal is genuinely not
recoverable from the above. A throwaway offline script over existing data is cheaper
and faster than a schema migration.

## CLI invocation contract (MUST follow — wrong CWD breaks Config)
All `python -m loop.*` commands MUST be run from the **repo root**
(`/home/daniel/animal_tracker`) with `PYTHONPATH=src` so that `.env` is found by
`Config()`. Running from inside `src/` fails with a missing-token validation error.

```
PYTHONPATH=src uv run python -m loop.ingest --since-id <watermark>
PYTHONPATH=src uv run python -m loop.metrics [--state <path>] [--date <YYYY-MM-DD>]
PYTHONPATH=src uv run python -m loop.report --mode summary [--state <path>] [--no-send]
```

`loop.metrics` writes `last_metrics` into `state.json` automatically (flat shape
with top-level `date` + metric fields including `fp_ci` as a list). `loop.report`
reads it directly — no manual reshaping required between stages. Use `--no-send`
on `loop.report` to render without calling Telegram (safe for dry runs and testing).

## Daily cycle (one nightly run, resumable)
1. **Gate** — handled deterministically by `loop.nightgate` (runs before the LLM
   session). Checks: is it night? Is `state.json["last_tick_completed_day"]` !=
   `loop_day()`? If gated out → sends a heartbeat (once per loop-day) and stops.
   The LLM never sees a gated-out tick.
2. **Ingest** — `python -m loop.ingest --since-id <watermark>`; reconcile labels.
   Note: `loop.metrics` (step 4) now advances the watermark in `state.json`
   automatically — you do NOT hand-write the watermark.
3. **Label** — adjudicate ambiguous crops (tier-2); append to `gold/`. Never
   UPDATE/DELETE existing labels. **Checkpoint immediately after adjudicating**:
   `python -m loop.checkpoint --message "tick: tier-2 adjudication done"` — tier-2
   is the only token-expensive step; checkpointing means an interrupted tick never
   re-pays for it.
4. **Measure** — `python -m loop.metrics`; paired FP/FN + CIs → `metrics/daily.csv`.
   This also advances `state.json["watermark"]` to the new watermark automatically.
   Checkpoint after: `python -m loop.checkpoint --message "tick: metrics written"`.
5. **Check** — does the active experiment's prediction still hold (CI-based)? done?
6. **Self-audit (cadence)** — auto-labels vs the day's human labels; re-check past
   wins on the larger corpus; note confidence in `runs/`.
7. **Decide** — keep / rollback; if concluded, pick next from backlog / OFAT within
   bounds. Respect freeze + one-experiment-at-a-time + `paused`.
8. **Validate** — Layer A = `python -m loop.replay` (STUB → "skipped"). Layer B =
   bounds + predicted live effect. FN-veto: reject FP wins that worsen (or risk, if
   FN unmeasured) FN.
9. **Deploy** — `python -m loop.deploy --delta '{...}' --restart-at <pre-sunrise>`;
   writes state.json + renders env + stamps the restart.
10. **Record** — update `runs/NNNN-<slug>.md` (front matter + observations), append
    a `JOURNAL.md` line, update `state.json` pointers. Before reporting, set
    `state["nightly_verdict"]` to a ≤2-sentence plain-English verdict (no jargon,
    no CIs, no aHash) — e.g. `"High FP night, 40/42 garden movement. No change;
    FN unmeasured so threshold hold stands."` The report sends this as Telegram
    message 2. JOURNAL.md stays dense and git-only.
11. **Report** — `python -m loop.report --mode summary` (sends message 1 = metrics
    summary; message 2 = `state["nightly_verdict"]` if set); commit + push.
12. **Mark complete** — `python -m loop.endtick` stamps
    `state.json["last_tick_completed_day"] = loop_day()` so the rest of tonight's 2h
    ticks skip (one Opus session/night). Run this ONLY after a fully successful tick.
    It is the LLM's final explicit action, deliberately NOT a systemd `&& endtick`: if
    the tick is interrupted (usage limit, crash, hang) or any step failed, `endtick`
    never runs, the day stays unmarked, and the next tick RESUMES from committed state.
    The failure direction is benign (re-run), never "skip needed work".

**Checkpoint as you go** using `python -m loop.checkpoint --message <msg>` — this
stages `experiments/` and commits (never pushes). Use it after each expensive or
irreversible step (tier-2 labels after step 3; state.json after `loop.metrics` in
step 4), not only at step 11. The loop has **no conversation memory** across ticks:
committed git state + `state.json` ARE the resume point. Budget exhausted or
interrupted mid-run → the next tick reloads committed state and continues, never
repeating tier-2 adjudication already in `gold/` or re-ingesting below the stored
watermark (which `loop.metrics` now persists automatically).

## Change levers (env first, code allowed with cause)
- **Env-var delta via `loop.deploy` is the default lever** (bounded, rollback =
  restore `best_known_good`). Prefer it whenever a tunable parameter can plausibly
  achieve the goal.
- **Code changes ARE permitted** when no env knob reaches the root cause (e.g. a
  motion-detection algorithm fix). Record the justification in the active
  `runs/NNNN-<slug>.md`, keep it minimal/reversible (rollback = `git revert`), and
  **commit it separately with the experiment id in the message** (e.g.
  `fix(motion): exp #4 (mog2-recurrent-frames) — <what/why>`); note the SHA in the
  run file. A code change only takes effect on a **camera restart**, so stamp
  `pending_restart_at` (pre-sunrise window) just like an env deploy.
- All decision gates below apply equally to code and env changes.

## Autonomy: the gates below are the ONLY approval mechanism

There is NO human pre-approval ("greenlight") step in this protocol, and there
never was. If a change passes the guardrail gates, ship it yourself — this
explicitly includes changes that alter REVIEW volume, notification routing, or
the human-privacy gate. Daniel's controls are post-hoc, not pre-approval:
`/pause`, `/rollback`, `state.json.paused`, `git revert`. Holding an experiment
while "recommending greenlight in the verdict" is a protocol violation — decide
and act within the gates. (2026-07-17: Daniel explicitly reaffirmed this after
exp #8 was held 6 nights and backlog #9 held 4 nights waiting for approval the
protocol never required.)

## Guardrail contract (hard rules)
- BOUNDS in `src/loop/guardrails.py` are enforced by the system (config validators
  + deploy). Never propose out-of-range values.
- FN-veto: an FP win with an FN rise beyond CI is rejected; if FN is unmeasured and
  the change could raise FN, HOLD.
- Volume collapse/explosion vs baseline → rollback.
- Feedback-starved freeze: no human labels for 3 days → freeze, hold best_known_good.
- One active experiment at a time. Respect `state.json.paused`.

## Scene-gate ownership (2026-07-17, overridden 2026-07-26)

> **STATUS: ENABLED at T=0.97 since 2026-07-26 by human override.** The HOLD
> described in the next two paragraphs is HISTORY — read the "SUPERSEDED
> 2026-07-26" block below before acting on anything in this section.

The scene-unchanged gate (`src/scene_gate.py`, review-class bursts muted when
near-identical to a recent empty-scene reference) shipped **disabled**
(`scene_gate_enabled=False`, threshold placeholder `0.97`) because Task 5's
offline validation (`scripts/validate_scene_gate.py`) found zero human
`animal`/`animal_wrong_id`-labeled review-class rows with a frame still on
disk — the 17 such rows corpus-wide all predate the ~100-burst retention
window, and the 53 on-disk review-class frames that do survive are
daytime-only. Per the FN-veto acceptance rule, no threshold may be chosen
with zero counter-evidence. This is a normal HOLD state, not a pending
human approval — per the Autonomy section above, enabling it is the loop's
call to make once the evidence exists, same as any other lever.

**Enablement procedure** — on any tick where new human `animal`/
`animal_wrong_id` labels exist on review-class rows whose frames are still
on disk: re-run `PYTHONPATH=src uv run python scripts/validate_scene_gate.py`
from repo root. If the animal-labeled bucket is non-empty and yields a safe
threshold, deploy via `loop.deploy` with delta
`{"PERFORMANCE_SCENE_GATE_ENABLED": 1, "PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD": <T>}`
(`PERFORMANCE_SCENE_GATE_ENABLED` is in `BOUNDS` in `src/loop/guardrails.py`
purely so `loop.deploy` accepts it — it is a flag, not a range, and carries
no config field-validator).

**SUPERSEDED 2026-07-26 — human override, gate is now ENABLED at T=0.97.**
Daniel directed enabling the scene gate despite the empty animal-labeled
bucket, as an explicit accepted-risk decision to cut REVIEW volume (paired
with the review-sampling gate deployed the same day). This overrides the
FN-veto HOLD above and the threshold-selection rule below — **do not
re-derive the threshold, and do not disable the gate on the grounds that
the animal bucket is empty.** That condition is known, permanent for now
(all animal-labeled review rows corpus-wide predate image retention), and
has already been ruled on by a human.

Threshold rationale, for the record: a fresh validator re-run on 2026-07-26
scored 35 on-disk review-class frames (still all `unlabeled`; buckets
`human_animal`/`human_fp` both n=0), distribution min 0.7376 / median 0.9534
/ max 0.9793. `T=0.97` mutes 6/35 = 17% and sits just under the observed
ceiling, so only near-identical frames are muted. It is the pre-registered
placeholder, not an invented number.

The **post-enable monitoring duty below still applies in full** and is now
the primary safety net in place of the missing pre-validation: adjudicate
every `scene_gate_muted=1` burst each tick, and treat a concealed animal as
an FN-veto event exactly as described. Raising `T` remains in-bounds and is
the preferred response; disabling the gate is still permitted if no in-bounds
`T` would have prevented the mute — but only on such positive evidence of a
real muted animal, never on absence-of-evidence.

Caveat the loop must account for: the review-sampling gate deployed the same
day cuts human label supply (and therefore FN detection power) roughly 4x —
see "Review sampling" below. Do not read the resulting label scarcity as a
feedback-starved freeze unless genuinely zero human labels arrive for 3 days.

**Threshold-selection rule (SUPERSEDED — retained for history; see override
above before applying):**
`T = max(similarity over human animal-labeled rows) + 0.02` safety margin,
clamped to `BOUNDS["PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD"]` = `(0.80,
1.0)`. Raising the threshold is always the safe direction (mutes fewer
bursts), so when in doubt round up, not down. Before enabling, also re-check
the low-texture diagnostic the validation script prints: the corpus scored in
Task 5 was daytime-only, and flat/low-texture dusk frames can inflate
similarity scores in a way daytime frames don't exercise — if the corpus
re-run still lacks dusk/dark review-class frames, treat that as a coverage
gap and weight the threshold conservatively (or hold) rather than trusting
the number blindly.

**Post-enable monitoring (nightly duty, same shape as the blur-mute path):**
once `scene_gate_enabled=True`, adjudicate every `scene_gate_muted=1` burst
from the tick's ingest window for a concealed animal, exactly like the
blur-mute (`below_sharpness_floor`) path already gets adjudicated. A
concealed animal in a `scene_gate_muted=1` burst is an FN-veto event —
respond the same tick by raising `scene_gate_similarity_threshold` strictly
above that frame's recorded `scene_similarity` (within bounds), or by
disabling the gate (`PERFORMANCE_SCENE_GATE_ENABLED: 0`) if no in-bounds
threshold would have prevented the mute. Do not defer this to "next tick."

## Review sampling (2026-07-26, Daniel's call)

`PERFORMANCE_REVIEW_SAMPLE_RATE` (default `0.25`, in `BOUNDS`) sends only a
deterministic ~1/4 sample of review-class bursts that survive the mute gates
to Telegram. Precedence: Human > Blur > Scene > Sampling. Suppressed bursts
are still species-ID'd and DB-logged with `review_sampled_out=1`; nothing is
lost from the corpus, only from Daniel's inbox.

Motivation (measured 2026-07-12..26): 679 review-class bursts, ~44 REVIEW
messages/night, 128 human labels, and only **4** false negatives — roughly
one real catch per 155 REVIEW pings. Daniel judged that ratio not worth the
notification load.

Rules for the loop:
- This is a **notification-volume lever, not an FP lever.** It changes what
  Daniel sees, never what is captured or logged. Do not credit a sampling
  change with an `fp_rate` improvement — `fp_rate` is label-conditioned and
  unaffected by construction.
- Raising the rate back toward 1.0 is the correct response to genuine FN
  evidence (a real animal found in a sampled-out burst), and is in-bounds.
  It is NOT a valid response to merely having fewer labels — that shrinkage
  is the intended effect, not a malfunction.
- Sampled-out rows are unlabelled-because-unsent. Keep them out of the
  "not yet labelled" backlog line (`loop.report` already separates them);
  never treat them as unlabelled-because-Daniel-ignored-them.

## Anti-self-poisoning & self-skepticism
- Ground truth is append-only; never rewrite `detection_feedback`, `gold/`, or prior
  `runs/` observations.
- Treat your own auto-labels with suspicion; reconcile against human labels in the
  self-audit step before trusting a "win."
