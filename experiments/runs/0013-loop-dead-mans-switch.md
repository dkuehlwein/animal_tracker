---
id: 15
slug: loop-dead-mans-switch
status: running
validation: live          # shipped 2026-09-02 commit 2f469fa; loop-side code, live on the NEXT nightgate invocation (no camera restart needed)
occupies_active_slot: true   # exp #14 (runs/0012) concluded KEEP this tick; this takes the slot
hypothesis: "The loop cannot report its own death. loop.nightgate heartbeats only on GATED-OUT ticks, so a tick that passes the gate and then dies is completely silent, and a dead camera is silent too because nothing checks it. Two best-effort checks inside nightgate — loop staleness and camera liveness, each alerting once per loop-day — convert both silent-failure modes into a Telegram alert without adding any way for the gate itself to fail."
created: 2026-09-02
promoted_from: "backlog #15, opened 2026-08-30 after the 27-night OAuth outage (2026-08-04..29) was discovered only because Daniel happened to look. Second confirming incident 2026-09-01: camera manually stopped, down ~20 h, also silent."
confidence: high          # the failure modes are observed history, not hypothesis; the change is additive monitoring with no effect on detection or notification routing
---

## Why this is worth an experiment slot at all

This experiment does not reduce false positives or false negatives. It is
infrastructure: every other experiment in this notebook is only as trustworthy
as the loop's ability to notice it has stopped running.

Two incidents establish the need, and both are already recorded in this
notebook rather than assumed:

1. **2026-08-04 → 2026-08-29, 27 nights.** The Claude session's OAuth token
   expired. `wildlife-loop.timer` kept firing, `loop.nightgate` kept passing the
   gate (it was night, the day was not done), and the judgment session then died
   immediately. Because nightgate only heartbeats on a *skip*, a passing-then-dying
   tick emits nothing at all. The camera ran fine the whole time; 483 triggers
   accumulated unanalysed. Discovered by Daniel, not by the system.

2. **2026-09-01 19:13:59 → 2026-09-02 15:14:23, ~20 h.** `wildlife-camera.service`
   was stopped by an interactive `sudo systemctl stop` and never restarted. No
   photos were captured for most of a day. Nothing checks whether the camera is
   running, so again: no notification.

These are opposite failures — the analyst died with a healthy camera, then the
camera died with a healthy analyst — and neither was observable. The 09-01 tick
also simply never ran, which is a third mode discussed under Residual below.

## The change (commit `2f469fa`)

Two independent checks in `loop.nightgate`, run on **every** tick, before the
exit code is raised, and deliberately *outside* the proceed/skip branch — the
27-night outage was on ticks that PASSED the gate, so a check that only runs on
skips would have missed it entirely.

- **Loop staleness** — `days_behind(last_tick_completed_day, loop_day()) > 2`
  → alert. Stamped once per loop-day in `last_staleness_alert_loopday`.
  Threshold 2 days, not 1: a single missed night is normal (a quiet day, a
  usage limit, a reboot) and must not page.
- **Camera liveness** — `systemctl is-active wildlife-camera.service` != `active`
  → alert. Stamped once per loop-day in `last_camera_alert_loopday`. Only stdout
  is inspected; a non-zero return code is normal for an inactive unit.

Decision logic is three pure functions (`days_behind`, `should_alert_staleness`,
`should_alert_camera_down`) with no I/O, so the once-per-loop-day suppression and
the threshold boundaries are tested directly.

**Failure containment is the main design constraint.** The gate is the one
component whose death silences everything downstream, so making it do more work
is inherently risky. Both checks are wrapped in `try/except Exception` that logs
and continues; neither can change the gate's exit code, and neither can raise out
of `main()`. A broken alert path degrades to the old behaviour (silence), never
to a broken gate.

### Verification

- 550 tests pass (was 532; ~250 lines of new nightgate tests).
- Replayed against both real incidents: `days_behind("2026-08-04","2026-08-31")`
  = 27 → staleness alert fires; the 09-01 camera stop → camera alert fires.
- Live smoke test against a copy of the real `state.json` at 2026-09-03 00:0x:
  prints `proceed: night, run not done`, exit 0, **no alert stamped and no
  Telegram sent** — current state is exactly 2 days behind, and `> 2` is
  correctly false at the boundary. The gate is not spuriously noisy.

No env delta and no `pending_restart_at`: `wildlife-loop.service` invokes
`loop.nightgate` from the repo on each tick, so this is live on the next tick.
Camera-side config is untouched.

### Gates

FN-veto is not applicable — this touches no detection, classification, mute or
notification-routing path; it cannot conceal an animal. Not paused, not
feedback-starved (human labels on 08-28, 08-31, 09-02). Volume guardrail
unaffected. Exp #14 concluded this tick, so the one-experiment-at-a-time rule
is satisfied.

### Pre-registered predictions

| prediction | how it will be judged |
|---|---|
| Zero alerts on healthy nights | any alert on a night where the loop completed and the camera stayed up is a false page → raise thresholds |
| The gate never fails because of these checks | any nightgate traceback or non-0/1 exit → revert immediately |
| A real outage produces exactly one alert per loop-day, not per tick | check `last_*_alert_loopday` stamping on the next real incident |

Rollback: `git revert 2f469fa`. No restart needed.

## Residual gap — recorded so it is not mistaken for solved

The switch lives *inside* nightgate. If `wildlife-loop.timer` itself stops firing,
nothing runs and nothing alerts. That is not hypothetical: the **2026-09-01 tick
never ran at all**. This change covers "the tick ran and something was dead"; it
does not cover "the tick never ran." Closing that needs a watchdog off the Pi
(or at minimum a separate timer), which is a different piece of work and is not
claimed here.

## Incident — git object corruption, 2026-09-02 22:11 (recovered)

Not part of the experiment, but it happened during the tick that shipped it and a
future tick needs the record.

The Pi rebooted uncleanly at 22:11:20 (uptime confirms boot ~22:12;
`wildlife-camera.service` restarted at 22:11:20). The reboot landed **mid-commit**:
7 object files in `.git/objects` were left zero-length, including the commit object
`e0c99e30…` that `refs/heads/main` and `HEAD` both pointed at. Every `git` command
failed with `fatal: bad object HEAD`. The reflog's final entry was itself truncated.

Sequence of the crashed tick, reconstructed from the reflog: ingest → tier-2
adjudication (committed `3f60768`, 22:06:58) → metrics (committed `96c8b79`,
22:07:09) → **crash at 22:11 while committing the notebook + this code change**.
Lost: the notebook write, the report send, `endtick`. Not lost: everything through
metrics, because the protocol's checkpoint-as-you-go rule had already committed it.

Recovery, 2026-09-03 00:0x: backed up `.git` to `/tmp/git-backup-20260903-000115`,
quarantined the 7 empty objects to `/tmp/git-quarantine`, reset `refs/heads/main`
and the index to `96c8b79` (the last intact commit). `git fsck` is now clean —
only dangling objects, no corruption. The working tree was untouched by the crash,
so the uncommitted `nightgate.py` / test changes survived intact and are what
commit `2f469fa` contains.

**The checkpoint discipline is what made this a 10-minute recovery instead of a
lost night.** Tier-2 adjudication — the only token-expensive step — was already
on disk and was not re-paid for. Because `loop_day()` maps 00:0x back to
`2026-09-02`, this tick is a *resume* of the crashed one, not a new night: ingest
and metrics were verified still current (`MAX(id)` = 4912 = the stored watermark,
zero new rows) and deliberately not re-run.

Second reboot in two days (09-01 19:13, 09-02 22:11), both unexplained by any
log entry. Worth watching for a hardware/power fault; not actionable on two
data points, and explicitly not opened as an experiment.

## Night 1 — 2026-09-03 (running)

First night with the switch live. Window `id 4913..4920`, 8 triggers
(2026-09-03 14:57:39 → 18:14:11), the whole of the camera's active day.

### Did the switch behave? Yes — silently, which is the correct outcome

`wildlife-loop.timer` fired 7 times today (00:00, 02:00, 04:00, 06:00, 18:00,
20:00, 22:00). Every one of them ran the two new checks, on both branches:
`proceed: night, run not done` (00:00, 22:00), `skip: tonight's run already done`
(02/04/06:00), `skip: daytime` (18/20:00). Exit codes correct, no traceback.

- **Staleness check — correctly silent.** `last_tick_completed_day` = `2026-09-02`,
  loop-day `2026-09-03` → `days_behind` = 1, and `1 > 2` is false. No alert, no
  `last_staleness_alert_loopday` stamp in `state.json`. This is the "single missed
  night must not page" case the threshold-2 choice was made for, and it held.
- **Camera check — correctly silent.** `systemctl is-active wildlife-camera.service`
  = `active` all day (`ExecMainStartTimestamp` 2026-09-02 22:11:20, `NRestarts=0`).
  No alert, no `last_camera_alert_loopday` stamp.
- **Containment held.** The `except Exception` branches log
  `"nightgate: staleness alert failed"` / `"... camera liveness check failed"`;
  neither string appears anywhere in `journalctl -u wildlife-loop.service` today.
  No exception was swallowed — the checks ran clean, they did not merely fail quietly.

Prediction 1 ("zero alerts on healthy nights") **holds** for night 1. Predictions 2
and 3 are unexercised — 2 is a null result confirmed above, 3 needs a real outage
and cannot be forced without staging one, which is not worth doing.

**Known observability limit, not a defect:** a healthy check emits no log line at
all, so "ran clean" is inferred from (a) the correct gate reason printing, (b) the
absence of the two failure strings, and (c) `main()` reaching its exit normally.
That is sufficient evidence, but a future tick should not expect a positive
"checks ok" line to exist.

### Reboot watch — clean

The two unexplained reboots (09-01 19:13, 09-02 22:11) did **not** get a third.
`uptime -s` = 2026-09-02 22:11:59, 23 h 52 min up, camera `NRestarts=0`. Still not
actionable, still not an experiment; the counter simply did not advance.

### Standing duties, all discharged

Eight triggers: 4 HUMAN (4916, 4918, 4919, 4920), 3 `no_animal` (4913, 4914, 4917),
1 `unclassifiable` (4915). **Zero animals of any kind, for the second tick running.**

Every review-class burst had frames on disk and every one was adjudicated
(tier-2 labels appended for the three without a human label):

| id | time | disposition | adjudication |
|---|---|---|---|
| 4913 | 14:57:39 | **sent to REVIEW** | empty pond scene, 5 frames, no subject → FP. Daniel independently labelled it `false_positive` at 21:17, matching. |
| 4914 | 16:51:54 | review-sampled out | empty pond → `false_positive` (tier2) |
| 4915 | 16:59:46 | **deferral-cancelled** | frame 1 shows a dark motion-smeared human leg crossing the near field; frames 2–5 empty → `person` (tier2) |
| 4917 | 17:38:18 | review-sampled out | empty pond; the bright bottom-right blob is low-sun lens glare, fixed across frames, not a subject → `false_positive` (tier2) |

- `human_proximity_muted=1`: exactly one (4915) — adjudicated, **0 concealed animals**.
- `scene_gate_muted=1`: none tonight (scene gate evaluated but never fired).
- review-class rows with `person_confidence` in [0.30, 0.50) (exp #14 standing duty):
  **none** — tonight's review-class rows sit at pc 0.196–0.233.
- Person frames reaching REVIEW: **0**.

### Two findings worth carrying forward

**1. The deferral gate earned its keep again (exp #11, concluded/live).** 4915 is a
textbook leading-edge leak: a person's leg at close range, classified
`unclassifiable` (raw top-1 `blank` @ 0.91), pc 0.233 — below every threshold — and
the visit's first HUMAN burst (4916) did not land until **60 s later**. Backward
window, density and blur/scene/sampling are all blind to it by construction; only
`review_defer_seconds=240` caught it, logging
`[REVIEW-DEFER] ... (human detected 60s after burst, no animal found)` at 17:03:46.
Running total of confirmed leading-edge cancellations: 4184 (44 s), 4212 (215 s),
4915 (60 s). The 240 s window remains comfortably wide for all three.

**2. Independent confirmation that backlog #16 was right to be rejected.** 4918 is a
real person — light trousers, dark top, close range, unmistakable across all 5
frames — and its `person_confidence` is **0.214**. It is HUMAN *only* because the
homo-taxon arm fired; the person-confidence arm (threshold 0.5 since exp #14) is
nowhere near it. That is a fresh, out-of-sample instance of exactly what the
09-02 adjudication found: the taxon arm carries real people that the confidence
arm cannot see, so demoting or thresholding it leaks person photos into REVIEW.
No new experiment; recorded as corroboration so a later tick does not reopen it.

### Volume — low, but not a guardrail event

8 triggers is the second-lowest day since 08-14 (08-26 had 2). Checked before
treating it as a signal: no config was deployed tonight or since 08-31, the camera
ran the full day (sunrise transition 06:48:52, sunset 20:12:16, "8 detections
today"), and daily volume under the *identical* deployed config has ranged 8–86 in
the last four days alone. The distribution since 08-14 is 2, 4, 5, 7, 8, 11, 13,
14, 14, 18, 32, 40, 43, 45, 75, 86 — activity-driven, and the high days are
human-dense (08-30: 64/75 HUMAN; 09-01: 68/86). Not a volume collapse; no rollback
trigger. `baselines.volume_per_night = 192` in `state.json` is stale by an order of
magnitude and should not be read as the comparison point.

### Verdict — KEEP RUNNING

One clean night is not enough to conclude a monitoring change whose entire value
is realised on failure. Prediction 1 held, the containment design held, and the
gate is not noisy. Continue; conclude once there is either a real incident that
the switch catches (the strong evidence) or enough healthy nights that "zero false
pages" is established (the weak evidence).

## Night 2 — 2026-09-04 (running)

Window `id 4921..4968`, **48 triggers**, all on 2026-09-04 (10:12–18:03). Camera
`active`, `NRestarts=0`, `uptime -s` 2026-09-02 22:11:59 (no third unexplained
reboot). Nightgate proceeded normally; **no staleness alert and no camera alert**
(`last_staleness_alert_loopday` / `last_camera_alert_loopday` still absent from
`state.json`, i.e. never stamped). Loop was exactly 1 day behind at gate time, and
the threshold-2 rule correctly stayed silent. Prediction 1 ("zero alerts on healthy
nights") **holds for night 2**; prediction 2 (gate never fails) holds — exit 0, no
traceback.

### Standing duties — all discharged, all clean

All 48 bursts had frames on disk. 46 review-class bursts adjudicated (contact
sheets of every best frame, plus full 5-frame strips for the 13 bursts with
`person_confidence >= 0.13`): **every one is the empty pond scene**. Tier-2
`false_positive` appended for all 46.

- `human_proximity_muted=1`: **none**. The two HUMAN bursts (4945 12:36:07 pc 0.047,
  4946 12:36:40 pc 0.851 — both plainly real people, correctly suppressed) are
  followed by the next review-class burst 940 s later, well outside the 240 s
  window, and density was 2 (< 8). Correct no-op, not a miss.
- `scene_gate_muted=1`: **none** (see below — the gate could not fire at all).
- `below_sharpness_floor=1`: 1 row (4968, 18:03, sharpness 10.4) — empty scene,
  sampled out anyway.
- Review-class rows with `pc` in [0.30, 0.50) (exp #14 duty): 2 — 4958 (0.343) and
  4968 (0.322). Both adjudicated across all 5 frames: **empty**.
- **0 concealed animals. 0 person frames reached REVIEW.** 0 animals of any kind
  today: no IDENTIFIED row in the window.

### What actually drove 48 triggers

The pond's water feature was running: a visible stream from the nozzle in every
frame from ~10:12 onward, absent from 09-03's frames of the same scene. Moving
water in the central region plus sun/shadow drift, for eight hours. This is an
environmental transient, not a config regression — nothing has been deployed
since 08-31.

Checked whether any trigger-side lever could cut it, and **both are FN-vetoed by
measurement, not by assumption**:

- `MOTION_THRESHOLD` (800, BOUNDS 200–8000): real animals sit *on the floor*.
  Across 289 IDENTIFIED rows the minimum `motion_area` is **800**, and the human
  `animal`/`animal_wrong_id`-labelled rows run 800, 802, 803, 804, 805, 808, 810…
  Raising the threshold at all starts deleting confirmed animals immediately.
- `MOTION_MIN_CONTOUR_AREA` (50): tonight's water FPs have `largest_contour_area`
  166–19973 (mostly 500–1900), and confirmed animals have 92, 675, 692, 743, 744,
  757, 791, 800, 802… — the same range. Entangled, exactly as exps #3 and #4 found
  for ROI and MOG2 knobs. Third independent confirmation that this scene's motion
  features do not separate FP from animal.

### The scene gate is inert here — measured, and it now has FN evidence

This is the tick's real finding, opened as **backlog #17**.

| measurement | value |
|---|---|
| cross-burst similarity, 45 same-scene review bursts | min 0.664, median 0.887, **max 0.944** |
| within-burst (frame1 vs frame5, seconds apart, same scene) | median 0.968, max 0.990 |
| person filling the frame (4945/4946 vs preceding empty refs) | **0.474–0.739** |
| **real animal** (4516, 2026-08-16 IDENTIFIED) vs its preceding empty ref | **0.931** |

Three consequences, none of which were knowable before tonight:

1. `T = 0.97` is **unreachable** cross-burst in this scene — 45 chances, max 0.944.
   Corpus-wide the gate has muted 25 bursts since 2026-07-26 and **0 in the last
   5 days**. It is not cutting REVIEW volume; it is doing nothing.
2. The metric is dominated by **illumination/time drift**, not subject presence:
   a person filling the frame (0.474–0.739) *overlaps* the empty-scene band's low
   end (0.664). The score mostly encodes how recently the reference was taken.
3. The FN-veto on lowering `T` is no longer absence-of-evidence. Burst 4516 — the
   only animal burst with frames still on disk — scores **0.931** against the empty
   reference immediately before it, i.e. *inside* tonight's empty band and above
   its 60th percentile. Lowering `T` to 0.93 would have muted a real animal.
   **`T` stays 0.97.**

### Shipped this tick (instrumentation, not an experiment)

`obs(scene-gate)`, commit **f14ed0d**, restart-gated `pending_restart_at`
2026-09-05T03:25. `scene_similarity` is now measured and DB-logged for **every**
status, not only review-class. `scene_gate_muted` and the reference-set `add()`
stay review-class-only, so no burst changes routing and no animal/HUMAN frame can
become an "empty scene" reference — **FN-veto N/A by construction**. 550 tests pass.

Why a code change rather than reasoning from what exists: the missing datum is
"what does an animal-containing burst score against a recent empty reference?", and
the only large supply of animal-containing bursts is IDENTIFIED rows, whose
similarity was never computed. It cannot be recovered retroactively — tonight only
**1 of 40** IDENTIFIED bursts still had frames on disk. This does not take exp #15's
active slot: it has no detection/notification behaviour to validate, and cannot
confound a loop-side monitoring experiment.

### Verdict — KEEP RUNNING

Night 2 clean: no false pages, containment held, camera up. Same reasoning as
night 1 — a monitoring change is concluded on a caught incident or on enough
healthy nights, not on two.

---

## Night 3 — 2026-09-05 — CONCLUDED (KEEP, live)

Third consecutive clean night (09-03, 09-04, 09-05). The timer fired 7× today;
every tick ran both checks on both the proceed and the skip branch, exit codes
correct, and neither `except`-branch warning string
(`staleness alert failed` / `camera liveness check failed`) appears anywhere in
the journal since ship — the checks ran, they did not fail quietly.

**Both alarms correctly stayed silent, and both silences are the right answer:**

| check | live input tonight | expected | observed |
|---|---|---|---|
| staleness | `last_tick_completed_day` 2026-09-04, loop-day 2026-09-05 → `days_behind`=1 | silent (`1 > 2` false) | silent, no stamp written |
| camera | `systemctl is-active wildlife-camera.service` = `active` | silent | silent, no stamp written |

`state.json` carries neither `last_staleness_alert_loopday` nor
`last_camera_alert_loopday` — the stamps are only written on a send, so their
absence is direct evidence that zero alerts (and therefore zero **false** alerts)
have fired in three nights.

### Why three nights is enough, and what is NOT being claimed

The trigger conditions have never fired in production. That is the *correct*
outcome — nothing broke — but it means the conclusion rests on three legs, none
of which is "we watched it catch a real outage":

1. **Replay against both real incidents** (night 0): the 27-night OAuth outage and
   the ~20 h camera stop both fire. Recorded at ship time, not re-litigated here.
2. **The send path is independently proven in production.** `_send_alert` calls
   `report.send()` — the *same* function `_send_heartbeat` uses. The heartbeat
   delivered on each of 09-03/04/05 (`last_heartbeat_loopday` = 2026-09-05 today).
   So the one seam a pure unit test cannot cover — does a Telegram message
   actually leave this Pi — is exercised daily by a different caller.
3. **Composition is unit-tested, not just the pure functions.** `main()` is
   covered on the proceed path, the skip path, the do-not-resend path, and the
   failure-isolation path for both checks (`tests/test_loop_nightgate.py`,
   31 cases). The gate's exit code cannot change and `main()` cannot raise.

Precedent for concluding rare-event insurance unexercised: exp #9's raw-homo
trigger, concluded KEEP after six nights during which it never fired, on exactly
this reasoning — an unexercised guard whose *cost* is measured at zero and whose
*correctness* is established by replay is a keep, not an open question.

**Residual gap, restated unchanged (still not solved, still not claimed):** the
switch lives inside `nightgate`, so it cannot fire if `wildlife-loop.timer`
itself stops. That needs an off-Pi watchdog.

**One thing this experiment demonstrably did NOT catch**, worth recording against
its own hypothesis: the camera was **physically re-aimed** during the 2026-09-01
outage (see runs/0014). `systemctl is-active` was `active` throughout afterwards,
so the liveness check was correct and silent while the system's entire input
distribution changed. Service liveness is not scene liveness. That is a real
blind spot in the *monitoring* story, but not a defect in this experiment — it is
the opening finding of exp #18.

**Verdict: CONCLUDED — KEEP, live.** Rollback remains `git revert 2f469fa`, no
restart. Active slot released to exp #18.
