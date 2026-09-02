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
