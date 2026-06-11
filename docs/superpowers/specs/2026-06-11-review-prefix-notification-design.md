# REVIEW-prefix notification flagging — design

**Date:** 2026-06-11
**Status:** approved (design)
**Related:** ADR-004 autonomous tuning loop; experiment #1 `notification-gate-live`
(`experiments/runs/0001-notification-gate-live.md`); ingest label-trust fix `8f3ff01`.

## Problem

The camera fires ~87 times/night and ~87% are false positives (wind-blown
vegetation, the swinging bird-feeder, moving sun-dapple). All triggers land in
one Telegram channel, burying the real animals.

The loop's strongest-evidenced fix is to route the junk-class triggers
(`NO_ANIMAL`, `UNCLASSIFIABLE`) to a separate "review" channel. That has been
**HOLD/infra-blocked** on provisioning a second Telegram channel, and a hard
*suppression* variant is **FN-vetoed** (false negatives are unmeasured, and 6 of
the `NO_ANIMAL` triggers were actually real animals the classifier missed).

## Decision

Ship the gate now as **labeling, not routing**: keep one channel, and prepend a
`🔍 REVIEW` header to the captions of review-class detections. Nothing is dropped
or moved, so it is **inherently FN-safe** (worst case: a real animal is merely
mis-labeled "review", still fully visible with its feedback buttons). This
removes both blockers — no second channel, no FN-veto — and makes the main feed
scannable today. It also serves as the visible validation step before any future
real channel split.

## Decisions locked (from brainstorming)

- **Review set / prefix predicate:** `review = detection_status ∈ {NO_ANIMAL, UNCLASSIFIABLE}`.
- **Control:** env toggle `REVIEW_PREFIX_ENABLED`, default `true`.
- **Prefix style:** a distinct header line, `🔍 REVIEW — likely false positive`,
  above the normal status caption.
- **No DB or labelling changes.** The feature is purely a caption render keyed off
  the in-memory `detection_status`. The `detection_feedback` table, the feedback
  buttons, and the existing `gate_would_suppress` column are all untouched.

## Components

### 1. Review predicate (for the prefix only)

A helper `is_review_detection(status) -> bool` returning
`status in {NO_ANIMAL, UNCLASSIFIABLE}`, used by the caption builder to decide
whether to prepend the header.

Placed so `wildlife_system.py` can import it (e.g. in `data_models.py` next to
`DetectionStatus`, or a small helper in that module). Operates on
`DetectionStatus` values.

**`gate_would_suppress` is NOT touched.** The existing shadow-gate column keeps
its current definition (`not animals_detected`). It and the new prefix predicate
answer different questions and are deliberately kept independent — no DB-semantics
change, nothing for the loop's metrics to reinterpret. (The loop already handles
`unclassifiable` correctly via the `8f3ff01` ingest fix, so there is no reason to
disturb the column.)

### 2. Config toggle

`REVIEW_PREFIX_ENABLED` env var → a boolean config field, **default `true`**,
living alongside `send_annotated_image` (same config dataclass /
`PERFORMANCE_SEND_ANNOTATED_IMAGE` neighborhood). Setting it `false` is an
instant off-switch with no code change or redeploy of logic.

### 3. Caption prefix

In `_build_caption` (`wildlife_system.py`): if the flag is enabled **and**
`is_review_detection(status)`, prepend the header line
`🔍 REVIEW — likely false positive` (own line, `\n`) above the existing caption
the method already builds per status. All non-review statuses (`IDENTIFIED`,
`ANIMAL_UNCERTAIN`) and the flag-off case render byte-for-byte as today.

`_build_caption` already receives the detection status and has `self.config`, so
no new plumbing into the send path is required.

### 4. Feedback buttons unchanged

Review-flagged items still get the same inline feedback buttons via
`send_detection_notification`, so human labeling continues feeding the loop —
essential, since this is the loop's measurement channel.

## Data flow

1. Motion → capture → SpeciesNet → `detection_status` (unchanged).
2. DB logging unchanged — including `gate_would_suppress` (still `not animals_detected`).
3. `_build_caption`: if `REVIEW_PREFIX_ENABLED` and `is_review_detection(status)`
   → prepend `🔍 REVIEW` header; else caption as today.
4. Send photo + caption + feedback buttons to the single configured channel
   (unchanged send path).

## Testing

- **Predicate:** `is_review_detection` true for `NO_ANIMAL` and `UNCLASSIFIABLE`,
  false for `IDENTIFIED`, `ANIMAL_UNCERTAIN`, `ERROR`.
- **Caption:** review status + flag on → header present; flag off → no header;
  non-review status (flag on) → never prefixed; existing caption body unchanged
  in all cases.
- Follow existing test style (`tests/`, pytest). Full suite stays green.

## Docs / notebook

- `CLAUDE.md`: document `REVIEW_PREFIX_ENABLED` in the config/env section.
- `experiments/runs/0001-notification-gate-live.md`: record the shipped
  same-channel-prefix variant, the commit SHA, and flip experiment #1's status
  to reflect it (concluded as labeling variant).
- `experiments/JOURNAL.md`: one-line entry.

## Out of scope (YAGNI)

- Second Telegram channel / message routing.
- A `guardrails.BOUNDS` entry or `loop.deploy` gating. `BOUNDS` governs *numeric
  parameter* auto-tuning the loop sweeps within safe ranges; this is a **code
  change**, not a tunable number, so it doesn't belong there. Reversal is
  `git revert` of the code commit (the loop's mechanism for code changes), with
  the `REVIEW_PREFIX_ENABLED=false` env flag as an extra instant off-switch.
- Any change to the DB schema, the `gate_would_suppress` column, or labelling.
- Any change to suppression behavior (still send everything).
