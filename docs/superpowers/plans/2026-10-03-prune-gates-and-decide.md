# Plan: auth alert, gate pruning, decide() refactor, experiment cleanup (2026-10-03)

Spec: none separate — this plan is derived from the 2026-10-03 logic + results reviews
(summarised in the session) and Daniel's instruction: "send a telegram message if we have a
logout; clean up the loop experiments, none is actually being observed; retire and prune all
the stuff that isn't used; refactor the decide — at least the parts we still need".

Reference for current behaviour: `docs/detection-gates.md` (authoritative gate doc),
`src/wildlife_system.py` (`process_detection`, `_process_and_notify_detection`,
`_schedule_deferred_review_send`, `_deferred_review_send`).

## Global Constraints

- Test suite: `uv run pytest tests/ -v` must pass at the end of every task (run from repo root of the worktree).
- **Privacy is never weakened.** HUMAN status, human-proximity (window, density, demoted-band,
  and its scope over unnamed-animal `;;;;;;animal` IDENTIFIED bursts), burst human sweep, raw-homo
  leak trigger, deferred cancel-on-human, and the human/human-adjacent photo purge all stay.
  Behaviour of every KEPT gate must be preserved except the two bug fixes in Task 3.
- **DB schema is append-only**: never drop columns. Retired gates' columns stay (historical data);
  new rows simply leave them NULL. Existing DB at `data/detections.db` must keep working.
- Config: removed fields must not break startup if their env vars are still present
  (pydantic settings already use `extra='ignore'` — verify). Remove matching entries from
  `src/loop/guardrails.py::BOUNDS` and any loop code that references them.
- Every gate still fails open except where the code today deliberately fails closed (HUMAN), plus
  the new fail-closed behaviour in Task 3.
- Exactly one suppression log line per suppressed burst, with the existing tags for kept gates:
  `[HUMAN-GATE] [HUMAN-PROXIMITY] [BLUR] [BLANK-CONF] [REVIEW-SAMPLE] [REVIEW-DEFER]`.
- Keep `docs/detection-gates.md` and `CLAUDE.md` accurate for whatever each task changes.
- Do not touch `experiments/deployed_config.env` by hand (rendered by `loop.deploy`); Task 4 changes
  `state.json`'s `deployed` block and re-renders via the existing render path if one exists.

## Task 1: Telegram alert when the nightly Claude run fails (auth expiry especially)

Problem: `wildlife-loop.service` runs `loop.nightgate` then `claude -p ... --model claude-opus-5-5 < experiments/loop.md`.
When the OAuth session expires, claude prints `Failed to authenticate: OAuth session expired and could not be refreshed`
and exits non-zero; nothing tells Daniel (two outages: 27 nights in Aug, 5+ nights Sep 26–Oct 3; the
staleness dead-man's switch only fires days later).

Intent:
- New small module `src/loop/tickalert.py` (name may vary) invoked by the service **only when claude exits non-zero**,
  receiving the exit code and the captured claude output (e.g. tee to a per-run log file under `data/logs/` and pass its path).
- Classify: auth failure (match "Failed to authenticate", "OAuth", "not logged in", "/login" case-insensitively) vs other failure.
- Send a Telegram message via the existing `loop.report.send` path (same as nightgate alerts). Auth message must say
  plainly that the login expired and what to do: on the Pi run `claude` and `/login` (or `claude auth login`). Other
  failure message: exit code + last few lines of output (trim, no secrets).
- Dedupe: at most one alert per loop-day per kind, stamped in `state.json` (e.g. `last_auth_alert_loopday`,
  `last_tick_failure_alert_loopday`) using the same stamping helper style as nightgate's `_send_alert`.
- Best-effort: a failure to send/stamp is logged and never raises; the unit's exit status should still reflect the claude failure.
- Update `wildlife-loop.service` ExecStart accordingly (keep the documented constraint: prompt fed via shell stdin redirect,
  `--model claude-opus-5-5`, nightgate `|| exit 0`). Keep stdout/stderr also going to the journal.
- Unit tests for classification, dedupe, best-effort behaviour (mock the send).
- Do NOT install the unit into /etc/systemd (controller does that after merge).

## Task 2: Prune unused gates

Retire (code + config + BOUNDS + tests + docs), keeping DB columns:
1. **Scene-Unchanged Gate** decision path: stop computing `scene_similarity`/`scene_gate_muted` per burst, drop the
   `SceneReferenceSet` seeding/updates in `WildlifeSystem`, its config fields (`scene_gate_*`) and BOUNDS entries, and
   `[SCENE-GATE]` handling. 0 mutes since 2026-08-30. **Keep** whatever in `src/scene_gate.py` is still used by
   `src/loop/scene_watch.py` (camera re-aim watchdog) — only delete what becomes dead. Remove `scripts/validate_scene_gate.py`
   if it only served the retired gate. `DatabaseManager.get_recent_review_detections` may become dead — delete if so.
2. **Unnamed-Animal Blank-Raw gate** (exp #32, `[UNNAMED-BLANK]`, `unnamed_animal_blank_*`): 1 fire ever.
3. **Animal-Proximity Review Exemption**, both halves (exp #33 backward, exp #39 forward/Phase 1 of the deferred task,
   `[ANIMAL-PROXIMITY]`, `[ANIMAL-DEFER]`, `animal_proximity_window_seconds`, `_last_animal_detection_at`,
   `get_last_animal_detection_time`, `require_animal_proximity`). 0 / ~1-per-2-months fires. After removal a
   sampled-out review burst is dropped immediately again (no annotated image built, no background task).
   `utils.is_named_animal_label` may become dead — delete if unused.
4. Dead code: the `[GATE-SHADOW]` "sending anyway" log/comment in `wildlife_system.py` (~lines 448-454), and
   `src/loop/replay.py` if it is an unreferenced stub (check `experiments/loop.md`/PROTOCOL references first; if referenced, leave it).
Everything else stays (see Global Constraints privacy list; Blur, Confident-Blank, Review Sampling, Deferred cancel-on-human,
best-guess caption, observability columns).

Update `docs/detection-gates.md` (precedence table + sections: mark retired gates in a short "Retired 2026-10-03" section with
one line each and their last commit pointer, delete their long sections) and CLAUDE.md's precedence list/rulings
(the scene-gate "human-ruled ON" ruling becomes "retired by Daniel 2026-10-03").

## Task 3: decide() refactor of the kept gates + the two decision bugs

Intent: replace the duplicated flag-recombination (`process_detection` computes flags → `_process_and_notify_detection`
recombines them in `not a and not b ...` chains and a parallel `elif` log chain) with ONE ordered decision function.

Structure:
- A pure function (new module e.g. `src/notification_gate.py`) `decide(ctx) -> Decision` where Decision carries
  `action` (SEND / MUTE / DEFER), `channel` (MAIN / REVIEW), `gate` (which gate decided; None for a plain send), and a
  human-readable `reason` for the log line. Ordered gate list, first match wins:
  HUMAN → HUMAN-PROXIMITY (window | demoted-band | density; review-class or unnamed-animal IDENTIFIED) → BLUR (review-class,
  below floor, luma ≥ min) → BLANK-CONF (review-class) → REVIEW-SAMPLE (review-class) → review-class with defer>0 → DEFER;
  otherwise SEND. Context is a small dataclass of everything the gates read (status, label predicates, flags/scores, recent
  human event info, config values).
- Per-gate DB flags (`human_proximity_muted`, `below_sharpness_floor`, `blank_confidence_muted`, `review_sampled_out`) are still
  persisted with today's semantics — they're observability the loop's metrics read. Keep computing them; the Decision drives routing and logging.
- **Recent-events store (fixes review bug HIGH-1)**: replace the single `_last_human_detection_at` value used by the
  deferred cancel-on-human check with a store of all recent HUMAN-status timestamps (the density list already exists —
  unify them: one sorted container, pruned to the longest window needed, seeded from the DB at startup). The deferred
  check becomes "any HUMAN timestamp in (t, t + review_defer_seconds]" rather than "the latest one is in the window".
  Backward window/density/demoted checks read the same store. Regression test: human at t+60 then another at t+250 before
  wake → the burst must NOT send.
- **Fail-closed on post-classification errors (fixes review bug MEDIUM-2)**: if an exception occurs after species
  identification/DB insert in `process_detection`, the fallback must not send a photo for a burst that is review-class,
  unnamed-animal, or inside a human window/density (or for which any privacy flag was already computed True). Today only
  HUMAN fails closed and everything else becomes an ERROR photo send. Keep fail-open for genuinely unrelated cases
  (e.g. a named-animal IDENTIFIED burst still sends). Regression test included.
- Golden test: enumerate the relevant combinations of status × flags and assert `decide()` reproduces the previous precedence
  (post-Task-2) — written BEFORE the old chains are deleted, then the old chains are deleted.
- The deferred task becomes single-phase (Phase 1 is gone after Task 2): sleep, then re-check humans via the store.
- Note on clocks: in-memory timestamps are capture-time; leave the two-clock issue alone (out of scope), but use capture
  time consistently for the in-memory store.
- Update docs/detection-gates.md "decision flow" description to point to `decide()` as the single source of precedence.

## Task 4: Close out the experiment ledger

Intent: the loop's open experiments are not being observed; close them all honestly.
- `experiments/state.json` `backlog`: every item with status `running`/open/active → concluded with a one-line outcome:
  kept (gate still live after Tasks 2-3), retired (removed in Task 2, cite this plan), or parked. Clear
  `active_experiment_id`. Remove `deployed` keys for retired config fields (scene gate threshold etc.) so `loop.deploy`
  no longer renders them; re-render `experiments/deployed_config.env` via the existing `loop.deploy` render function if one is
  callable without restarting anything — otherwise just edit state.json and note it.
- `experiments/runs/*.md` front matter `status:` → `concluded` (or `retired`) with a one-line `outcome:`; do not rewrite bodies.
- Append a dated JOURNAL.md entry "2026-10-03 — human-led cleanup" summarising: loop down Sep 26–Oct 3 (OAuth),
  gates retired, decide() refactor, bugs fixed, and the review's key findings (FP rate of sent notifications
  unchanged ~0.6 vs ~0.67; FN unmeasured; headline fp_rate mostly auto-labels) as context for future ticks.
- Add a short PROTOCOL.md rule: at most ONE experiment open at a time (no `occupies_active_slot: false` side channel),
  and an experiment is not concluded "keep" on zero firings — zero firings after its observation window means retire.
- Do not change loop.md's tick workflow otherwise.
