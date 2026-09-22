---
id: 34
slug: deploy-stamp-tz-crash
status: concluded   # 2026-09-22: prediction met exactly, fix verified live in production
validation: live   # code change, commit 6d13364; 3 regression tests reproduce the exact production TypeError
occupies_active_slot: false  # loop-infrastructure repair, not a detection-behaviour change; exp #21 keeps the slot
hypothesis: "An offset-naive pending_restart_at crashes wildlife-deploy.service before it restarts the camera, so no experiment — code or env — can ever go live again until the stamp is coerced to local time on read."
created: 2026-09-21
promoted_from: "night of 2026-09-21: wildlife-deploy.service found in `failed` state, 16h after the 03:30 fire, with {\"error\": \"can't compare offset-naive and offset-aware datetimes\"}. Exp #33 (ea652bc) had not gone live."
confidence: high   # the failure is a deterministic TypeError reproduced in a unit test, not a statistical claim
delta: {}   # code change only; no env knob exists for this and none is wanted
commit: 6d13364
restart_at: 2026-09-22T03:25:00+02:00
opens: backlog #34 (deploy-stamp-tz-crash)
---

## Tonight

**The loop's deploy path has been broken since 03:30 this morning, and the
break was silent.** Everything below is secondary to that.

10 triggers, every one a false positive, zero animals and zero people. But the
day's real content is infrastructural.

### The deploy crash

`wildlife-deploy.service` fired at 03:30:07 as it does every night and exited
1 immediately:

```
Sep 21 03:30:07 raspberrypi uv[637129]: {"error": "can't compare offset-naive and offset-aware datetimes"}
Sep 21 03:30:07 raspberrypi systemd[1]: wildlife-deploy.service: Failed with result 'exit-code'.
```

`apply_pending_deploy.apply()` compared `pending_restart_at` against
`datetime.now().astimezone()`. `now` is always tz-aware. The stamp is not
always: `loop.deploy` writes whatever the caller passes, and for a **code-only
experiment there is no env delta, so `loop.deploy` never runs at all** and last
night's tick hand-wrote `"2026-09-21T03:25:00"` — no offset. Every prior night
back to at least 09-14 carried `+02:00` and applied cleanly, which is why this
had never fired before.

Consequences, in order of severity:

1. **Exp #33 (`animal-proximity-review-exemption`, ea652bc) never went live.**
   `wildlife-camera.service` has `ExecMainStartTimestamp=Sun 2026-09-20
   03:30:07` — it is still running exp #32's code (f4d7730). Exp #33 has had
   zero measurement nights, not one.
2. **The failure was self-perpetuating.** `pending_restart_at` is cleared only
   on the success path, after `restart_fn()` returns. A crash before that
   leaves the bad stamp in place, so 09-22, 09-23 and every night after would
   have raised the same TypeError on the same string. This was not a one-night
   blip; it was a permanent, silent halt of the loop's only actuator.
3. **Nothing in the loop would have noticed.** The nightly report reads
   `state.json`, not systemd. `pending_restart_at` staying non-null looks
   identical to "a deploy is stamped and pending". The loop would have gone on
   opening experiments, writing notebooks and reporting verdicts indefinitely
   while shipping nothing.

### The fix

`_as_aware()` parses the stamp and attaches the local timezone when it is
naive, on both sides of the comparison. Local is the correct reading, not a
guess: every stamp this loop writes is a local pre-sunrise wall-clock time
(~03:25 CEST), and `loop.deploy`'s own callers have always meant local.

Fixed at the **reader**, deliberately, rather than by teaching tonight's tick
to write an offset. The writer is sometimes `loop.deploy` and sometimes a
hand-written line in a tick like this one; only the reader is a single
chokepoint that every future stamp must pass through.

Three regression tests, written red-first — the first reproduces the exact
production `TypeError`; the second pins that coercion must not make a *future*
naive stamp fire early; the third covers the symmetric naive-`now` case. Full
suite: **718 passed**.

Re-stamped `pending_restart_at` to `2026-09-22T03:25:00+02:00` (offset present,
and before the 03:30 timer fire), which ships both this fix and — finally —
exp #33.

### Audit: was anything else lost?

No. A missed restart costs exactly one thing, the reload of
`experiments/deployed_config.env` and of `src/`. The deployed env has been
unchanged since exp #30 (09-18), so no env delta was stranded; the only
stranded artifact is exp #33's code. Detection, capture, species ID, DB logging
and Telegram all ran normally all day — the camera process was up the whole
time, just running yesterday's build.

## The day's detections

All 10 bursts (5396-5405, 10:54-17:26) adjudicated tier-2 as
`false_positive`; 0 human labels arrived today. Per-burst frame-differencing
against the burst median put the dominant motion blob in the **wind-blown
bamboo in the upper right** in 8 of 10, and on the yucca/wall at the top left
in the other 2 (5398, 5404). No animal, no person, in any frame of any burst.
This is a windy-day signature, not a new failure mode.

Gate behaviour was correct throughout:

- **5399** — blank-confidence muted (raw top-1 blank @0.941 ≥ 0.92). Frame
  inspected: empty pond, bamboo moving. Correct mute.
- **5405** — blur muted (sharpness 6.7 < 11.0), dusk shot at 17:26. Frame
  inspected: empty. Correct mute.
- **5396, 5398** — review-sampled out. Both inspected: empty. No loss.
- **Scene gate muted nothing** (0 of 10; similarities 0.80-0.89, all well under
  T=0.982). No `scene_gate_muted=1` burst exists tonight, so the post-enable
  monitoring duty is discharged vacuously.
- **Human/proximity/deferral gates** — no HUMAN-status burst today, so all
  inert by construction.

## Self-audit: the scene-gate threshold, re-checked on a larger corpus

Exp #30 raised `PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD` 0.97 → 0.982 on
09-18, the first time the animal bucket was non-empty. Re-measured tonight with
three more days of labels: **17** animal-labelled rows now carry a recorded
`scene_similarity`, max **0.9621** (id 5343). The protocol's pre-registered
rule, `max(animal) + 0.02`, gives **0.9821**. Deployed value: **0.982**.

The threshold still lands where the rule puts it. No change, and deliberately
no re-derivation — recorded as a confirmation that exp #30 holds on the
enlarged corpus, per protocol step 6.

Worth noting how much this has changed: the scene gate was enabled on
2026-07-26 by human override precisely because the animal bucket was empty and
believed permanently so. It is no longer empty — 23 animal-labelled bursts now
have frames on disk. The accepted-risk override has been retroactively earned.

## Rejected on measurement: an upper-right ROI exclusion

With 8 of 10 of tonight's triggers coming from the same bamboo corner, the
obvious move is to stop looking at that corner. Measured before proposing it,
and it does not survive.

Motion-blob centroids over all **248** labelled bursts with frames still on
disk (220 FP, 23 animal, 5 person):

| exclusion zone | FP removed | animals lost |
|---|---|---|
| x≥0.72, y≤0.30 | 73/220 (33%) | **3/23 (13%)** |
| x≥0.70, y≤0.35 | 76/220 (35%) | **4/23 (17%)** |
| x≥0.65, y≤0.40 | 88/220 (40%) | **4/23 (17%)** |
| x≥0.75, y≤0.25 | 57/220 (26%) | **3/23 (13%)** |

Every variant costs animals: 5125 (x=.896 y=.303), 5360 (.828/.125), 5362
(.848/.176), 5363 (.961/.238). Birds perch in the bamboo — the FP source and
the animals are the same bush. **FN-veto: rejected.**

This is a genuinely new test, not a re-run of exp #3: that experiment rejected
edge *bands* (left/right/top/bottom 15%) in June, on a corpus whose animal
bucket was 18 rows and a framing since changed. This rejects the upper-right
*corner* specifically, on the current framing, with a 23-row animal bucket. The
conclusion is the same and now rests on two independent measurements: in this
scene, FP motion and animal motion are not spatially separable. Filed as
backlog #35 so no future tick re-derives it.

## Prediction

Tomorrow's tick must find `wildlife-deploy.service` in `active (exited)`, not
`failed`, with `{"restarted": true, "reason": "applied deploy stamped
2026-09-22T03:25:00+02:00"}`, and `wildlife-camera.service` started
2026-09-22 ~03:30. If it is still `failed`, the coercion is wrong and the right
response is `git revert 6d13364` plus a hand-run of
`loop.apply_pending_deploy`.

**Standing check for every future tick:** read `systemctl is-failed
wildlife-deploy.service` before trusting that the previous tick's experiment is
live. Tonight that check was worth more than any tuning decision.

## Next

No new behaviour experiment opened. Exp #33 has had zero measurement nights
through no fault of its own; it ships tonight and gets its window. Opening
anything else now would confound it.


---

## Outcome (2026-09-22) — CONCLUDED, keep

The prediction was met in every particular.

```
Active: inactive (dead) since Tue 2026-09-22 03:30:03 CEST
uv[669234]: {"restarted": true, "reason": "applied deploy stamped 2026-09-22T03:25:00+02:00"}
wildlife-camera.service  Active: active (running) since Tue 2026-09-22 03:30:03 CEST
```

`wildlife-deploy.service` exited 0, the stamp was consumed, and the camera came
up on the new build at 03:30:03. `pending_restart_at` is back to `null` — the
self-perpetuating failure is cleared, not merely survived once.

Both stranded changes went live together: this fix (6d13364) and **exp #33**
(ea652bc), which today got the first measurement night it has ever had.

The reader-side `_as_aware()` coercion is the right chokepoint and stays. The
standing check this experiment introduced — read `systemctl is-failed
wildlife-deploy.service` before trusting that the previous tick shipped
anything — is now permanent tick procedure and was run first thing tonight.

Experiment closed. No follow-up.
