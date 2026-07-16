---
id: 8
slug: sharpness-floor-is-a-brightness-gate
status: running          # proposed | running | concluded | rolled_back | parked
validation: parked        # live | replay | parked  (design/data phase — not yet deployed)
hypothesis: "min_sharpness_threshold=11.0 is applied to raw Laplacian variance, a whole-frame contrast statistic that scales with scene brightness. So the 'blur gate' is operationally a LIGHT-LEVEL gate: at dusk nearly every burst is below-floor regardless of actual motion blur. Because the blur-gate MUTE path (below_sharpness_floor AND no-animal-found -> no Telegram) is therefore reachable mainly as a function of darkness, real dusk animals the classifier misses are silently dropped — the exact unobservable-FN class runs/0005 existed to close, reopened by the dusk data. Fix: stop letting darkness alone trigger the mute."
created: 2026-07-10
promoted_from: "backlog id 8 (filed 2026-07-09 at runs/0006 rollback); now the active experiment after exp #7 (AE mode) concluded — AE bias is not the lever, the floor statistic is"
confidence: high          # the brightness->floor relationship is a measured step function over 470+ frames across multiple days and both AE modes; the mechanism (Laplacian variance ~ luminance^2) is analytic
---

## Why this is now the active experiment

Exp #7 (`runs/0006`) tested whether biasing auto-exposure shorter would lift
dusk sharpness above the 11.0 floor. It cannot, and was rolled back; the
reopened AE=normal baseline (07-10, 17–18h sharpness 6.9–9.4) confirmed AE mode
is indistinguishable from AE=short at dusk. **The floor statistic, not the
exposure mode, decides whether a dusk burst is muted.** That is this experiment.

## The measured problem (unconfounded, from `runs/0006` + 07-10 data)

P(Laplacian variance < 11.0) as a function of frame luma, over 470 pre-deploy
AE=normal frames spanning multiple days:

| luma bin | P(lap < 11.0) | n |
|---|---|---|
| 0–40   | 100.0% | 100 |
| 40–60  | 0.0%   | 2 |
| 60–80  | 71.4%  | 28 |
| 80–100 | 0.0%   | 60 |
| 100–130| 0.0%   | 278 |

Above luma ~80 essentially nothing is below floor; below luma ~60 essentially
everything is. `min_sharpness_threshold` is a **light-level gate**. The floor
was intended to catch motion blur; it actually catches darkness.

## Why it matters (the FN this reopens)

The blur-gate mute path is `src/wildlife_system.py::_should_send_notification`
→ `is_blurry_review`: suppress Telegram when `below_sharpness_floor` AND the
status is review-class (no animal found). At dusk `below_sharpness_floor` is
true almost unconditionally, so any dusk burst the classifier fails to find an
animal in is silently dropped. If the classifier missed a real animal in a dark
frame, that is an **unobservable false negative** — muted means never labelled.

07-10 window: the mute path fired 4× (ids 1787 morning, 1845/1846/1848 dusk),
all below-floor no-animal/unclassifiable. One below-floor *animal* (1847,
18:23) alerted correctly — the gate does the right thing when an animal is
found; the risk is only in the no-animal branch, where "no animal" may be a
classifier miss on a dark frame rather than a true empty scene.

## Recommended fix (minimal, reversible, FN-reducing) — pending Daniel's OK on volume

**Brightness-gate the MUTE, not the notification.** Only let the mute path fire
when low sharpness plausibly means blur — i.e. in adequate light. When the
scene is dark (low luma), a below-floor score is explained by darkness, so the
burst should flow through as a normal 🔍 REVIEW notification instead of being
silently muted.

Concretely:
- Compute mean-gray **luma** for the selected best frame (cheap; already have
  the frame in `_capture_and_select_best_frame`) and add it to `sharpness_info`.
- Add `is_blurry_review`'s condition a `luma >= brightness_floor` term (a new
  config knob, e.g. `blur_mute_min_luma`, default ~70 per the step function
  above). Below that luma, do not mute.
- Keep the raw Laplacian floor as-is for the animal-alert path (unchanged) and
  for the DB `below_sharpness_floor` column (still recorded, still honest).

This is a **code change** (no env knob reaches it), so: commit separately with
the experiment id, note the SHA here, and stamp a pre-sunrise `pending_restart_at`
since it only takes effect on camera restart. Rollback = `git revert` + restart.

**FN-veto stance:** this change *lowers* FN risk (fewer dark bursts silently
muted), so the FN-veto does not block it. The only cost is a small REVIEW-volume
increase — bounded to the below-floor no-animal *dark* bursts (07-10: the 3 dusk
ones, 1845/1846/1848). All of it is 🔍 REVIEW-tagged, in-channel.

**Why HELD tonight, not shipped:** (1) REVIEW-channel volume is Daniel's
standing product lever (he rejected a 2nd channel to keep volume manageable), so
a change that raises it — even modestly — should get his explicit nod first;
(2) it needs luma plumbed into `sharpness_info` + a new validated config knob +
TDD, which per CLAUDE.md is subagent-driven work, not a rushed end-of-tick edit.
There is no fire forcing it tonight: FP rate is stable and 100% REVIEW-tagged
with zero main-channel leak. Flagged in tonight's verdict for Daniel's greenlight;
ready to implement next tick.

## Alternatives considered (recorded, not chosen)

- **(a) brightness-normalized sharpness** (`lap / gray_var`) as the floor
  statistic. Cleaner in theory but replaces a calibrated threshold with an
  uncalibrated one — needs a fresh distribution study before it can ship
  without risking either FN (too strict) or REVIEW spam (too loose). Higher
  risk than the luma-gate for the same FN benefit.
- **(c) drop the mute path entirely**, rely only on the REVIEW prefix. Simplest
  and maximally FN-safe, but raises REVIEW volume at *all* light levels, not
  just dark ones — every genuinely-blurry daylight no-animal burst would then
  notify too. The luma-gate is (c) restricted to the cases where the floor is
  actually lying, so it buys the same FN safety at a fraction of the volume cost.

## Observations — 2026-07-11 tick (still HELD)

Adjudicated all 7 mute-path firings this window (below-floor + review-class):
ids 1864,1865,1866,1867,1868,1869 (16:46–17:55) and 1873 (21:04). **All are
true-negatives** — inspected the frames, no hidden animal in any → the mute path
did not conceal a single FN tonight.

**The finding that revises this experiment's premise:** the below-floor mutes
this window are NOT a dusk-darkness phenomenon. 6 of the 7 fired in *daylight*
(luma 71–81 @ 16:46–17:55; sunset ~21:40), with only 1873 (luma 20, 21:04)
genuinely dark. The frames are uniformly **soft-focus** — the whole scene is out
of focus even at good luma — which is why raw Laplacian variance sits at 8–25
across the board, day and night. So this window's sub-floor scores are driven by
**focus softness, not brightness**.

Consequence for the proposed fix: a `blur_mute_min_luma≈70` gate would have
un-muted exactly **1 of 7** rows tonight (1873), and that one held no animal —
i.e. the luma-gate's live benefit this window was zero and its cost was one extra
REVIEW message. The brightness→floor step function (runs 0006) is still real over
the multi-day corpus, but tonight shows the mute path's *day-to-day* firing is
dominated by soft focus, which the luma gate does not touch. Two implications:
(1) the fix's urgency is lower than framed — it addresses a narrow dusk slice
that was empty tonight; (2) a soft-focus camera may be depressing sharpness
scores generally (worth checking focus via `scripts/camera_preview.py`), which
is orthogonal to both AE mode (exp #7) and the floor statistic.

**The real FN signal tonight is outside this experiment's scope.** Two observable
FNs — 1861 (16:05, no_animal, luma 99, sharpness 15.4, *above* floor) and 1862
(16:28, unclassifiable, luma 88, sharpness 12.5, above floor), both human-labeled
`animal_wrong_id`. These are **classifier recall misses in good-light, in-focus-
enough frames**, not blur-gate mutes. No env knob and neither the luma-gate nor
any sharpness change addresses classifier recall. `loop.metrics` still reports
`fn_rate: unmeasured` (it does not compute the animal-label-on-review-status join),
so these FNs are recorded here qualitatively; FN is no longer strictly zero-signal.

**Decision: HOLD again, no deploy.** (1) Still pending Daniel's OK on the
REVIEW-volume increase (his standing product lever); no approval signal in state.
(2) Tonight's evidence lowers the fix's expected live benefit to ~nil (mute path
concealed no animal; luma-gate would have flipped 1 empty dusk row). (3) FP is
well-controlled (human FP 1/7 this window). No fire forces the change. Next tick:
if Daniel greenlights, implement TDD/subagent-driven per the plan; separately
consider whether the soft-focus observation warrants a focus check.

## Observations — 2026-07-12 tick (still HELD; big human-presence night)

**Window:** 53 new triggers (ids 1874–1926, watermark 1873→1926). Status split:
23 HUMAN, 15 no_animal, 5 unclassifiable, 10 identified. **Zero human feedback
labels this window** — FP ground truth is unmeasured tonight; `loop.metrics`'s
fp_rate 0.67 is entirely MegaDetector auto-labels (n_md 30, n_human 0), not truth
(see memory: auto-labels are not truth).

**The night's dominant event is a sustained human presence, 17:40–18:27** — 23
HUMAN-status rows (ids 1891, 1895–1902, 1905, 1907, 1909–1921), 20 of 23 at
pconf ≥ 0.35, correctly SUPPRESSED (no Telegram). Verified frames: a person in
red trousers is plainly visible. The human/privacy gate did exactly its job.
Leak-watch: **0 HUMAN rows carry any feedback label** (none notified). Purge:
**0 HUMAN frames past 48h lingering**.

**Mute-path adjudication (this experiment's core check): 6 firings, 0 concealed
animals.** below-floor + review-class ids 1889 (17:05, no_animal, sharp 10.5),
1892 (17:51, unclassifiable, 7.4), 1906 (18:13, no_animal, 8.7), 1908 (18:15,
no_animal, 7.0), 1925 (19:31, unclassifiable, 6.5), 1926 (19:33, unclassifiable,
6.5). Inspected every frame: 1889/1892/1906/1925/1926 are the empty soft-focus
pond scene (true negatives); **1908 is a HUMAN** (same red-trouser person, foot
in frame) that scored pconf 0.17 < 0.3 and carried no `homo` taxon, so it slipped
the human gate but was still muted by the blur gate — muted either way, no
Telegram, no leak, and no *animal* concealed. So the mute path hid **no false
negative** for the second night running.

**Soft-focus persists (07-11 carryover, now stronger).** Every frame this window,
day and dusk, is out of focus; raw Laplacian sits 6–11 even at good luma. This is
the same soft-focus signature flagged 07-11 — it is now a two-night pattern, not a
one-off, and it depresses sharpness globally (independent of AE mode and of the
floor statistic). Still recommend a physical focus check via
`scripts/camera_preview.py`; escalating this from "worth a look" to the most
actionable non-held item on the board, since soft focus plausibly also lowers
classifier recall (blurry animals → missed) — a real FN driver the loop's env/code
levers cannot touch.

**Decision: HOLD exp #8 again.** Unchanged rationale: no Daniel greenlight on
REVIEW volume; mute path concealed 0 animals two nights running, so live benefit
remains ~nil; no fire. The luma-gate would have un-muted only dark dusk rows, and
tonight's below-floor firings are again dominated by soft focus (not darkness),
which the luma gate does not address.

## Observations — 2026-07-13 tick (exp #8 HELD 3rd night; PRIVACY LEAK found)

**Window:** 64 new triggers (ids 1927–1990, watermark 1926→1990). Status split:
22 human, 31 no_animal, 4 unclassifiable, 7 identified. **One human feedback
label** this window: id 1965 (17:19, `identified`, generic `;;;;;;animal` conf
0.53, raw top-1 bird) labelled `animal` → a correctly-alerted **true positive**,
not an FN. FP ground truth otherwise unmeasured (n_human 1).

**Second family-in-garden evening** (carryover from 07-12's human event): an
adult + a small child around the pond ~18:56–20:22. 22 HUMAN-status rows
correctly SUPPRESSED (pconf up to 0.94; e.g. 1967/1970/1982/1984/1986).

**Mute-path adjudication (this exp's core check): 3 firings, 0 concealed
animals** — below-floor + review-class ids 1975 (19:02, no_animal, sharp 9.3),
1987 (19:49, no_animal, 8.0), 1990 (20:22, unclassifiable, 9.7). Inspected every
frame: **1975 = an adult's bare legs/shorts** (person close to camera, pconf 0.24
< 0.3 → slipped the human gate, muted by blur gate anyway), **1987 = the small
child crouching** at the pond edge (pconf 0.12 → same), **1990 = empty dark pond**
(true negative, genuinely dark dusk — the exact case the luma-gate targets, and it
held no animal). So the mute path hid **no false negative for the 3rd night
running**; two of the three firings were humans muted (no leak, no animal).

**Exp #8 verdict unchanged: HOLD.** Mute path has concealed 0 animals across
3 nights; live benefit remains ~nil. Only 1990 (1 of 3) was dusk-darkness, held
no animal → luma-gate benefit again ~zero. This experiment is low-value; the
board's live defect is elsewhere (below).

---

## CROSS-CUTTING FINDING — human/privacy gate LEAK (id 1988), exp #5 leak-watch

**This is the first observed leak of exp #5's leak-watch and outranks exp #8.**

**id 1988 (19:50) leaked a photo of a person to the MAIN channel.** Details:
`detection_status = identified`, `species_name = 1f689929…;;;;;;animal` (generic
rollup, conf 0.72 → notifies as a real detection, NOT REVIEW-prefixed),
`top_species_raw = 990ae9dd…;mammalia;primates;hominidae;homo;sapiens;human`
(tss 0.59), `person_confidence = 0.10`. Frame inspected: unmistakably the same
adult (red/green clothing, close, motion-blurred). The "Best guess" caption line
would have rendered **"Best guess: human (59%)"** — a person's photo alerted to
Daniel labelled human.

**Root cause — both human-gate paths bypassed:**
1. MegaDetector person box scored 0.10 < `human_detection_confidence` 0.30 (person
   heavily motion-blurred / partially framed at close range → weak box).
2. The SpeciesNet **ensemble** rolled the classifier's homo-sapiens top-1 UP to a
   generic `;;;;;;animal` label that carries **no `homo` taxon segment**, so the
   ensemble-taxon homo check also missed it.

The gate checks person_confidence and the *ensemble* taxon. It does **not** check
the **raw classifier top-1** (`top_species_raw` / `metadata['top_classifier_prediction']`),
which here correctly said "homo sapiens human" at 0.59. That raw signal is the
missed lever.

**Data-validated, FN-safe fix (proposed — needs Daniel's OK; his strongest
product lever is human handling):** extend the human gate to also fire
`DetectionStatus.HUMAN` when the raw classifier top-1 taxon contains a `homo`
segment **and** the ensemble did NOT confidently identify a specific animal
(i.e. the ensemble label is a generic rollup / `blank` / `no cv result` /
review-class) — so a confident species ID is never overridden. Specificity check
over the **entire DB**: `top_species_raw ~ homo/human` occurs on exactly **2 rows,
both actual humans** (1852 unclassifiable-muted 07-12, 1988 leaked 07-13), **zero
real animals** → the override would have caught both leak/near-leak with **zero
observed false-suppression**. FN risk is therefore negligible on evidence (an
animal whose raw top-1 is confidently "human" AND whose ensemble can't ID it has
never occurred), but because it changes the **privacy-suppression gate** — Daniel's
single strongest product call — and FN is formally unmeasured, it is **HELD for his
explicit greenlight**, not auto-shipped tonight. Same discipline as exp #8's volume
lever. Severity note: the leak recipient is Daniel's own private channel of his own
garden/family, so exposure is "against design intent + annoying," not a third-party
breach — which is why a next-tick TDD/subagent ship (on greenlight) is proportionate
rather than an emergency unreviewed edit. Filed as **backlog id 9**
(`human-gate-raw-classifier-leak`). Also caught the 1852 near-miss retroactively.

## Observations — 2026-07-14 tick (HELD 4th night; FIRST concealed animal in mute path)

**Window:** 64 new triggers (ids 1991–2054, watermark 1990→2054). Status split:
27 human, 25 no_animal, 6 unclassifiable, 6 identified. Two human-presence events:
a morning one (07:04–07:47, ~16 HUMAN incl. a **child in a Ronaldo #7 shirt**) and
an evening one (17:23–20:32, ~11 HUMAN). Human feedback this window: **6 labels** —
1991 `false_positive`, 2007/2009 `person`, 2011 `animal_wrong_id`, 2017/2018
`animal`. `loop.metrics`: fp_rate 0.78 (fp_human 1/6; n_md 31 auto), FN unmeasured.

**THE FINDING — the mute path concealed a real animal for the first time in 4
nights.** id **2035 (17:15, unclassifiable, sharpness 8.54, below floor → MUTED)**
contains an unmistakable **blackbird foraging on the pond-edge rocks** (cropped +
2.2× enlarged to confirm; the same spot is empty rocks 26 min later at 2041/2042).
Raw classifier top-1 was `bird` @ 0.12 but the ensemble rolled it to
`unclassifiable`, and below-floor + review-class → the blur-gate mute fired → **no
Telegram.** This is exactly the FN class this experiment exists to close, observed
live for the first time (prior 3 nights: 0 concealed animals).

**Would the proposed luma-gate fix have caught it? YES (marginally).** 2035's luma
is **67.8**, just under the proposed `blur_mute_min_luma ≈ 70` threshold → the fix
would have un-muted it into a 🔍 REVIEW notification instead of silently dropping
it. First live evidence the luma-gate has non-zero benefit (07-11/07-12/07-13 it
would have flipped only empty dusk rows).

**But the net product harm tonight was nil** — the *same blackbird* was captured
again 3 min later at **id 2036 (17:18)** which the ensemble DID ID as `bird` @ 0.72
and alerted correctly to the main channel. So Daniel was notified of this bird; the
2035 mute is a "soft" FN (concealed-but-net-covered), not a bird missed for the
night.

**Cost/threshold nuance recorded for the ship.** A luma-gate at 70 would un-mute,
this window: 2035 (bird, good), 2041 (luma 64.1, empty → +1 REVIEW), 2043 (luma
65.2, empty → +1 REVIEW), 1992 (luma 39.8, the **child** → human into REVIEW,
mildly undesirable). 2042 (luma 80.3) stays muted. So the fix buys the 2035 catch
at a cost of ~2 empty + 1 human REVIEW message this window. Threshold choice (70)
directly trades FN-safety vs REVIEW volume — worth a small distribution check when
implementing, not a blind 70.

**Other mute-path firings adjudicated (below-floor + review-class): 1992, 2041,
2042, 2043.** 1992 = the child in the Ronaldo shirt (pconf 0.17 < 0.30, no homo
taxon → slipped human gate, muted by blur gate anyway; no leak, no animal). 2041 /
2042 / 2043 = empty soft-focus pond (true negatives). So of 5 mute-path firings:
**1 concealed a real animal (2035), 1 was a muted human (1992), 3 were empty.**

**Observable FN outside the mute path: id 2011 (07:59, no_animal, sharpness 12.99,
ABOVE floor)** labelled `animal_wrong_id` by Daniel — a classifier recall miss in
good light/in-focus-enough; above floor so it was REVIEW-notified (Daniel saw and
labelled it), not muted. Same class as the 07-11 recall FNs; no sharpness/env lever
touches classifier recall.

**Leak-watch (exp #5 / backlog #9): 0 main-channel leaks tonight.** All 6
`identified` rows are birds (aves/corvus), pconf ≤ 0.10, no `homo` in any
`top_species_raw`. Human gate correctly suppressed both human events. Two humans
(2007, 2009) slipped the gate into the 🔍 REVIEW channel (no_animal, pconf 0.06/0.14,
no raw species) — a known residual (blurry no-detection human frames), REVIEW-tagged
not main-channel, and NOT addressable by backlog #9 (they carry no top_species_raw).

**Soft-focus still present** (raw Laplacian 8–9 at luma 64–80). Physical focus check
via `scripts/camera_preview.py` remains the most actionable non-held item.

**Decision: HOLD exp #8 a 4th night — but urgency is now materially higher.** The
mute path has now demonstrably concealed a real animal (2035), and the proposed
luma-gate would have caught it — so the fix's expected benefit is no longer ~nil.
Still not shipped tonight because (1) it changes REVIEW volume, Daniel's standing
product lever → needs his greenlight; (2) it's a code change needing TDD/subagent
work + a threshold distribution check, not a rushed end-of-tick edit. **Escalating
in tonight's verdict: recommend greenlight to implement next tick.** Net harm
tonight stayed nil only because 2036 happened to re-catch the same bird — that is
luck, not the gate working.

## Observations — 2026-07-15 tick (HELD 5th night; quiet daytime-only window, mute path 0 concealed)

**Window:** 69 new triggers (ids 2055–2123, watermark 2054→2123). Status split:
34 human, 26 no_animal, 6 unclassifiable, 3 identified. Time span **08:03–17:48
only** — no deep-dusk frames this window (last trigger 17:48, still daylight;
prior nights ran to 20:xx). Two human-presence events: morning (08:03–08:20,
ids 2055–2064) and a large afternoon one (15:12–15:30, ids 2087–2111, ~24 HUMAN
rows, pconf up to 0.95) — all correctly SUPPRESSED (no Telegram).

**Human feedback this window: 4 labels — 3 `animal` (TPs) + 1 `person`.**
- 2073 (11:34, identified, raw bird 0.64) → `animal` = **true positive**, alerted.
- 2117 (17:03, identified, raw bird 0.58) → `animal` = **true positive**, alerted.
- 2118 (17:04, identified, raw bird 0.65) → `animal` = **true positive**, alerted.
- 2095 (15:17, no_animal, pconf 0.116, raw None) → `person` = a **blurry human that
  slipped the human gate** (pconf 0.116 < 0.30, no `homo` taxon) into the 🔍 REVIEW
  channel. Same known residual as 07-14's 2007/2009 — REVIEW-tagged, not main
  channel, and **NOT addressable by backlog #9** (no `top_species_raw`).
`loop.metrics`: fp_rate 0.886 but that is **MD-auto-driven** (n_md 31, fp_md 31/31);
**fp_human 0/4 = 0.0** — zero human-confirmed false positives tonight. FN unmeasured.

**Mute-path adjudication (exp #8 core check): 1 firing, 0 concealed animals.**
Only ids **2093** (15:16, human, sharp 10.71, below floor) and **2121** (17:37,
no_animal, sharp 10.87, below floor) fell below the 11.0 floor this window. 2093 is
HUMAN (suppressed as human, not a blur-mute concern). **2121 is the sole review-class
mute-path firing** — inspected the frame (luma 76.1, lap 10.9): **empty pond, no
animal → true negative.** So the mute path concealed **no false negative**. Note
2121's luma is **76.1 > proposed blur_mute_min_luma≈70**, so the luma-gate would NOT
have un-muted it anyway — a soft-focus borderline, not a darkness case.

**Sharpness minimum jumped to 10.7 (prior nights: 5.0–6.9) — but explained by the
window, not a focus fix.** 07-15 sharpness min=10.7 median=18.6 max=24.3 across 69
frames; only 2/69 below floor. The higher minimum is because **this window has no
deep-dusk frames** (ends 17:48); prior nights' 5–7 minima came from 19–21h dark-dusk
captures. The 2121 frame confirms the scene is **still visibly soft-focus** even at
good luma → soft focus persists; the sharpness-min rise is a sampling artifact of a
daytime-only window, NOT evidence the focus was corrected. (`scripts/camera_preview.py`
has uncommitted local edits — Daniel may be working on the focus tool — but production
frames remain soft.)

**Leak-watch (exp #5 / backlog #9): 0 main-channel leaks.** All 3 `identified` rows
are birds (raw bird 0.58–0.64), pconf ≤ 0.02, no `homo` in any `top_species_raw`.
Human gate correctly suppressed both human events; the one residual (2095) went to
🔍 REVIEW, not main, and carries no raw species → not a backlog-#9 case.

**Decision: HOLD exp #8 (5th night) and backlog #9 — no Daniel greenlight in state,
no deploy.** No greenlight signal arrived (state.paused=false, no note; only new
working-tree changes are `scripts/camera_preview.py` + `.claude/`, neither a code-ship
authorization). Both changes modify Daniel's product levers (REVIEW volume / privacy
gate) and FN is formally unmeasured → both require his explicit OK per protocol.
Nothing forced either tonight: mute path concealed 0 animals, 0 leaks, 0
human-confirmed FPs, 3 TP birds correctly alerted. Both held fixes remain the board's
top actionable items; recommending greenlight again in the verdict.

---

## 2026-07-16 — exp #8 HELD 6th night; dusk human cluster all-suppressed; NEW: blank→main FP pattern

**Window:** 86 triggers (ids 2124–2209, watermark 2123→2209), span **08:xx–18:56**,
with a large **dusk pond-work human cluster ~18:29–18:56**. Status breakdown: **33
human, 23 no_animal, 20 unclassifiable, 10 identified.**

**Human labels: 1 — 2184 `person`** (18:31, no_animal, sharp 11.4 above floor, pconf
0.034, raw None) → a blurry human that slipped the gate into 🔍 REVIEW (known residual;
REVIEW not main; **NOT backlog-#9-catchable** — pconf 0.034 ≪ 0.30, no `top_species_raw`).
`loop.metrics` fp_rate 0.849 is MD-auto (n_md 49, fp_md 42/49); **fp_human 0/1 = 0.0**
(person label not counted as an animal-FP). fp_claude 3/3 = my blank labels below. FN
unmeasured.

**NEW FINDING — "blank" predictions alert to the MAIN channel as `identified`.** Ids
**2139 / 2143 / 2158** (10:52, 11:33, 14:31) are `status=identified`,
`species_name=…;;;;;;blank`, conf **0.99**, pconf ≤0.10 — inspected all three frames:
**empty pond, no animal.** They reach MAIN (not REVIEW: `is_review_detection` excludes
IDENTIFIED) and render as `🚫 No animal (99%)` (`wildlife_system.py:483-485`). This is
**long-standing, not new**: 50+ `blank`-species `identified` rows since 2026-06-09
(~1/night), already baked into the measured baseline — Daniel has never flagged it, so
low-nuisance, but it IS main-channel volume for empty frames. Adjudicated all 3 as
**tier2 `false_positive`** (ground truth for a future experiment). **Backlog candidate
(not opened — one-experiment-at-a-time, exp #8 active):** route `blank`-label ensemble
predictions to NO_ANIMAL/REVIEW (or mute) instead of IDENTIFIED. Low FN risk (`blank`
means the classifier itself asserts empty), code change, restart-gated — HELD like the
others until it can be the sole active experiment and/or Daniel greenlights.

**Mute-path adjudication (exp #8 core check): 1 firing, 0 concealed animals — and it
muted a *person*.** 12 rows below the 11.0 floor tonight; 11 are HUMAN-status
(suppressed as human — human gate precedes the blur mute, no concealment risk). The
sole review-class mute-path firing is **2206** (18:52, no_animal, sharp 8.2, pconf
0.05): inspected → a **person bent over at the pond** (only legs/back visible, so
MegaDetector fired no person box → pconf 0.05, human gate missed it). The blur mute
suppressed it anyway → **beneficial** (person muted, 0 animals concealed).

**Leak-watch (exp #5 / backlog #9): 0 main-channel human leaks.** All 10 `identified`
rows: 7 birds/animal (pconf ≤0.10, no `homo`), 3 blank empty-pond (above). The dusk
human cluster (33 HUMAN) all correctly SUPPRESSED. Two persons slipped the human gate
as no_animal — **2184→REVIEW** (above floor) and **2206→muted** (below floor) — neither
reached MAIN, and **neither is backlog-#9-catchable** (both pconf ~0.03–0.05, raw None;
back-view/legs-only frames MegaDetector under-scores and the classifier never says
`homo`). This residual is a distinct failure mode from backlog #9's raw-classifier leak.

**Decision: HOLD exp #8 (6th night) and backlog #9 — no greenlight, no deploy, nothing
to roll back.** state.paused=false, no greenlight note arrived. Nothing forced a change:
mute path concealed 0 animals (and caught a person), 0 main-channel leaks, 0
human-confirmed FPs, 7 TP animal/bird alerts. New this tick: the blank→main-channel FP
pattern is now documented + tier2-labeled as a fresh backlog candidate. Recommending
Daniel greenlight the queued exp #8 / #9 fixes again in the verdict.
