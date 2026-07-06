---
id: 5
slug: human-main-channel-leak
status: proposed          # proposed | running | concluded | rolled_back | parked
validation: parked        # live | replay | parked — pending Daniel's product/privacy call
hypothesis: "Humans the classifier IDs as 'homo species' with detection_status=identified bypass the REVIEW prefix (which only fires on NO_ANIMAL/UNCLASSIFIABLE) and leak to the MAIN channel as if a real wildlife identification. Route human/homo identifications to REVIEW (or suppress) to stop main-channel human alerts. Zero FN risk to wildlife (humans are not target animals)."
param_delta: null         # no env lever — REVIEW routing is taxon-blind in code (is_review_detection / _REVIEW_STATUSES); a fix is a CODE change
predicted_effect: { fp_rate: "main-channel human leaks -> 0", fn_risk: "none for wildlife (suppressing human alerts cannot hide an animal)" }
created: 2026-06-30
decision: pending         # product/privacy dimension is Daniel's call (cf. no-second-channel precedent)
confidence: high          # leak mechanism confirmed in code; two visually-confirmed human leaks on disk
---

## Finding (2026-06-30 tick, batch 1143-1167)

Three animal-tier rows tonight:
- **1147** (10:20, `aves;...;bird`, conf 0.81) — genuine. Frame shows a dark
  blackbird (Amsel) on the ground bottom-left. Human tapped ✅ animal. Correctly
  surfaced to main channel. No correction.
- **1163** (19:28, `mammalia;primates;hominidae;homo;;homo species`, conf 0.82) —
  **real human**. Frame is an unmistakable close-up of a person (brown hair, blue
  shirt) bending right in front of the camera, motion-blurred.
- **1167** (19:48, same `homo species` rollup, conf 0.92) — **real human** at the
  right frame edge (dusk, motion blur).

## Leak mechanism (confirmed in code, not assumed)

`src/data_models.py`: `_REVIEW_STATUSES = {NO_ANIMAL, UNCLASSIFIABLE}` and
`is_review_detection(status)` is taxon-blind. `src/wildlife_system.py:446` only
prepends the `🔍 REVIEW` header when `is_review_detection(status)` is true.
A `homo species` classification produces `detection_status = identified`, which is
**not** a review status → no prefix → the photo lands in the MAIN channel exactly
like a real bird ID. So 1163/1167 alerted Daniel's main channel as wildlife.

This is the same class as the 06-27 leak audit (classifier-FP "animal" rollups
bypassing the REVIEW prefix). 06-28's humans did NOT leak because their status was
not `identified`; tonight's two were `identified` at high conf → they did.

## Metrics caveat (do not silently fold in)

tier-1 maps `identified` → `animal`, so 1163/1167 reconcile as **animal**, not FP.
With no human tap on them they are NOT in tonight's fp_count. Reconciled fp = 22/25;
true operational FP (humans are non-wildlife) = 24/25. Per the standing rule
(auto-labels are not truth, headline = human-only), the reported headline is
**fp_human 5/6 = 0.83**; the 2-human-leak undercount is footnoted, not relabelled
via tier-2 (avoids self-poisoning the reconciled series). See JOURNAL 2026-06-30.

## Proposed fix (code change, minimal/reversible) — PENDING DANIEL

Add the human/homo taxon to REVIEW routing (or suppress human alerts entirely).
Smallest change: in `is_review_detection` / the notification path, treat a `homo`/
`homo species` classification as review-eligible regardless of `detection_status`,
so human detections get the `🔍 REVIEW` prefix instead of a clean main-channel ID.

**Why this is parked, not deployed tonight:** whether a wildlife camera should
alert on humans at all is a product/privacy decision (REVIEW-tag vs suppress vs
leave) that is Daniel's to make — same posture as the no-second-channel call. The
mechanism and impact are settled; the policy choice is not. Flagged in tonight's
verdict. If Daniel says go, this becomes a running code experiment with the fix
committed separately (`fix(notify): exp #5 ...`) + a pre-sunrise restart stamp.

## Leak-watch log

- **2026-07-01**: CLEAN. 3 `identified` main-channel alerts (1179/1180/1196) all `aves;;;;;bird` — real blackbirds, no homo leak.
- **2026-07-02**: CLEAN. 2 `identified` main-channel alerts (1260 `;;;;;;animal`, 1261 `aves;;;;;bird`) both visually confirmed = same real blackbird foraging at pond edge (lower-left frame). No homo/human taxon leak. Exp #5 stays PARKED pending Daniel's product/privacy call; no new forcing evidence.
- **2026-07-03**: **NOT CLEAN — 2 real human leaks.** Both `identified` main-channel alerts tonight were humans, zero real-animal IDs:
  - **1362** (14:51, `mammalia;primates;hominidae;homo;;homo species`) — unmistakable person: bare legs, yellow shorts, walking through the garden bed in bright daylight. Leaked to main channel with no REVIEW prefix.
  - **1388** (17:42, `mammalia;primates;hominidae;homo;sapiens;human`) — human: bare arm/hand reaching in from the right holding a blue-nozzled watering can. Leaked to main channel.
  This is the **first recurrence with actual humans** since the 06-30 audit (1163/1167); the two prior nights' `identified` leaks were birds. The leak mechanism is exactly as documented above (`identified` bypasses `_REVIEW_STATUSES`). No human tap on either → not in fp_count (reconciled fp = 94/96 auto). Exp #5 fix remains **code-ready but PARKED pending Daniel's product/privacy call** — flagged prominently in tonight's verdict as forcing evidence. No unilateral deploy (alerting-on-humans is Daniel's policy decision).
- **2026-07-04**: **NOT CLEAN — 1 human leak (second consecutive night with a human leak).** 5 `identified` main-channel alerts tonight, visually adjudicated:
  - **1389** (08:17, `mammalia;primates;hominidae;homo;sapiens;human`, conf 0.966) — unmistakable person: yellow shorts, bare legs and forearm crossing the bed in bright morning light. Leaked to main channel with no REVIEW prefix. **HUMAN LEAK.**
  - **1396** (11:08, `aves;;;;;bird`, 0.766), **1399** (11:28, `aves;;;;;bird`, 0.799), **1400** (11:29, `;;;;;;animal`, 0.525) — same real **blackbird** foraging lower-left of the pond over ~20 min. Genuine wildlife, correctly on the main channel.
  - **1423** (15:02, `;;;;;;animal`, 0.666) — real bird **bathing** at the water dish (center). Genuine wildlife.
  So 1 of 5 identified = human leak, 4 = real birds. Mechanism unchanged (`identified` bypasses `_REVIEW_STATUSES`, taxon-blind `is_review_detection`). Human not tapped → reconciled as animal, not in fp_count (auto fp = 71/76); true operational fp = 72/76 (human is non-wildlife). Human leaks now on **two consecutive nights** (1362/1388 on 07-03, 1389 tonight). Exp #5 fix stays **code-ready but PARKED** pending Daniel's product/privacy call; flagged in tonight's verdict as accumulating forcing evidence. No unilateral deploy.
- **2026-07-05**: **CLEAN — no human leak** (breaks the 07-03/07-04 two-night human-leak streak). Only 2 `identified` main-channel alerts tonight, both `b1352069…aves;;;;;bird` and both visually confirmed **blackbirds** — no `homo`/`homo species` taxon appears anywhere in the batch (1465-1519):
  - **1478** (10:17, conf 0.837) — glossy black **blackbird** perched on the rim of the water dish (bottom-right). **Human-tapped `animal`** by Daniel → the one non-FP human label tonight. Correct main-channel wildlife ID.
  - **1515** (18:35, conf 0.768) — dark **blackbird** foraging in the grass (lower-left). Genuine wildlife, correctly on the main channel.
  No forcing evidence added tonight; exp #5 remains code-ready and PARKED pending Daniel's product/privacy call. Notable that human labels also resumed today (13 taps after a 2-day drought), so Daniel is engaged — the parked policy question is now the only blocker, not attention.
- **2026-07-06**: **CLEAN — no human leak** (second consecutive clean night). 5 `identified` main-channel alerts tonight (batch 1520-1590), all visually confirmed real birds, no `homo`/`homo species` taxon anywhere:
  - **1521** (10:17, `b1352069…aves;;;;;bird`, conf 0.671) — dark **blackbird** on the rim of the grey water dish (bottom-right). Genuine wildlife.
  - **1544** (11:42, `aves;;;;;bird`, 0.753), **1545** (11:43, `1f689929…;;;;;;animal`, 0.505), **1546** (11:44, `;;;;;;animal`, 0.725), **1547** (11:44, `;;;;;;animal`, 0.638) — same real **blackbird** foraging in the left grass border over ~2 min; the generic `animal` taxon on 1545-1547 is just lower classifier confidence on the same bird, frames unambiguous. Genuine wildlife, correctly on the main channel.
  So 5/5 identified = real birds, 0 human leaks. Two consecutive clean nights (07-05, 07-06). No forcing evidence added tonight; exp #5 stays code-ready and PARKED pending Daniel's product/privacy call.
