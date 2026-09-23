
## 2026-09-23 — exp #35 (human-proximity-purge-gap) opened + shipped 0bc9ec9

32 triggers, **zero Telegram messages** — the first fully-suppressed night. 29
person (28 HUMAN-status + 5444), 3 FP, 0 animals. All 32 adjudicated visually.
`fp_rate` 0.094 (3/32) is **not** an improvement on yesterday's 0.286: the
person bucket sits in the denominator and never in `fp_count`. Honest form:
3 non-person triggers, 3 false positives, 0 animals.

**Exp #32 (unnamed-animal-blank) recorded its first live mute, and it was
right.** 5413 (16:17) came back `;;;;;;animal` with raw top-1 `blank` @0.760 <
T=0.900 → `[UNNAMED-BLANK]`, suppressed. Frames are the upper-right bamboo stand
under a gust (largest inter-frame blob 13383px at x=1898-2028, y=0-248); no
animal. Four nights after shipping, the exact failure it was built for recurred
and was caught. Running total 1 mute / 1 correct.

**The finding.** The human-proximity *mute* gate fires on window OR density OR
demoted-band; the human-retention *purge* re-tested every candidate against the
narrow ±240s window alone — including rows the gate had already flagged
`human_proximity_muted=1`. So the system could decide "this is a person, do not
notify" and then decline to apply its own 48h human-photo policy to that same
burst. Burst 5444 tonight is the measured instance: a recognisable person at
close range (green fabric with cloth folds filling the left third,
`person_confidence` 0.063, far under the 0.5 primary gate), muted by the
**density** condition at 310s from the nearest HUMAN burst — 70s outside the
purge window — and therefore retained for the full `max_images` rotation.

Replayed over the whole corpus: **12 rows ever fell in this gap** (~1.4/month).
Five still had frames on disk, including 5074/5075/5076 at 15 days and 5154 at
11 days against a 48h policy. Four of those five are empty garden; 5444 is the
person. Realised leak rate inside the gap is therefore 1 in 5, not 5 in 5 —
stated plainly because the case for the fix does not rest on volume. It rests on
the fix being free: it deletes image files from bursts the system has *already*
judged to be people.

Shipped `0bc9ec9`, restart-gated 09-24T03:25+02:00 (tz-aware, per exp #34).
`human_proximity_muted=1` rows are now purge-eligible unconditionally; the
±window test is untouched for unmuted rows, which is the leading-edge case it
exists for (a burst before a visit's first HUMAN burst necessarily has the flag
False). Strictly a superset, image files only, DB rows kept, no new knob, no
routing or classification change. `PERFORMANCE_HUMAN_RETENTION_PROXIMITY_SECONDS=0`
still disables the whole extension. 5 tests; suite 723 passed. FN-veto N/A — no
notification behaviour changes.

Other gates: human/privacy 28 suppressions, all correct (spot-checked 5438/5442
at full res — both `person_confidence` 0.0, routed on the `homo` segment alone,
both plainly people at arm's length); human-proximity 3 mutes (5425/5432 window,
empty; 5444 density, the person); confident-blank evaluated 5425/5432 but yielded
precedence, one log each as designed; scene gate 0 of 3 (0.49-0.93 vs T=0.982);
exp #33 no firing opportunity for a third night — nothing survived to the
sampling gate, and there was no named-animal identification to exempt from.

Exp #33 keeps the active slot. Feedback clock reset: Daniel labelled 5411
(animal) at 10:36 today, so the 3-day starvation freeze did not trigger.

New backlog #38 (person-rows-dilute-fp-rate): `loop.metrics` puts person-labelled
rows in the `labeled_triggers` denominator but never in `fp_count`, so any
person-heavy night prints a flatteringly low `fp_rate` into `metrics/daily.csv`.
Tonight's 0.094 is such a row. A future tick could read that dip as an FP win.
Not changed tonight — one change per tick, and it is measurement plumbing rather
than a live gate.
