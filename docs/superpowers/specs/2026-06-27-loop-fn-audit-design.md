# Spec: Selection-FN / Classifier-FN audit from saved bursts (sub-project B1)

**Date:** 2026-06-27
**Status:** Approved (design), pending implementation plan
**Scope:** Sub-project **B1** of the loop-honesty initiative. Measures two
false-negative classes from the burst frames already saved on disk. Sub-project
**A** (per-tier FP split + plain-English report) is merged. **B2** (trigger-FN
timelapse audit) is a separate, later spec.

## Problem

The autonomous tuning loop currently measures false positives but is blind to
false negatives — animals it should have reported but didn't. Daniel observed a
concrete case: a bird passed the camera and Telegram showed nothing, with no way
to tell whether the bird was never captured or whether the system captured it and
sent the wrong frame.

There are three FN classes:

| Class | What happened | Catchable how |
|-------|---------------|---------------|
| **Trigger FN** | Motion never fired; no burst exists | Only via independent observation (timelapse) — **B2** |
| **Selection FN** | Burst contains the animal, but the frame we *sent* had no animal | **From disk now — B1** |
| **Classifier FN** | Animal *is* in the sent frame, but SpeciesNet reported no-animal/blank | **From disk now — B1** |

Selection-FN and classifier-FN are catchable today because the live system saves
**all** burst frames (`capture_<ts>_frame1..N.jpg`) but runs SpeciesNet on, and
logs, only the single selected frame. The other frames' evidence is thrown away.
B1 re-analyses that discarded evidence.

## Goals

- Quantify **selection-FN** and **classifier-FN** rates from saved bursts.
- Keep the honesty discipline from sub-project A: the **headline FN rate reflects
  human-confirmed labels only**; the automatic MegaDetector signal is reported on
  its own line as an auto/unverified estimate, never folded into the headline.
- Make the loop **aware** of the new signal: a new deterministic tick step, new
  metrics/report fields, and a new lever in the agent's protocol — so the agent
  can *decide* to change frame selection. B1 itself measures and informs; it does
  not touch the live capture path.

## Non-goals

- **No change to the live camera / frame-selection path.** Fixing frame selection
  is the loop agent's decision (informed by this measurement), implemented later
  under loop control — not in B1.
- **Trigger-FN** (no burst at all) — that is B2 and needs an independent timelapse.
- **No re-classification of burst siblings.** The audit runs the *detector only*
  (MegaDetector), not the full species classifier, to keep cost bounded.
- No change to how feedback is *collected* beyond adding two label codes to the
  existing keyboard.

## Definitions (per trigger / burst)

Let `sent` = the frame logged in `detections.image_path`; `siblings` = the other
`capture_<ts>_frameN.jpg` files in the same burst; "animal in frame X" = MegaDetector
returns ≥1 animal box on X above the existing detection-confidence threshold.

- **Selection-FN candidate** — animal in *some* sibling **and not** in `sent`.
  (We had the animal on disk and picked the wrong frame.)
- **Classifier-FN candidate** — animal in `sent` **and** the reported detection
  status was no-animal/blank. (We sent the right frame but the classifier/ensemble
  dropped it.)
- **OK** — neither condition holds.

A trigger can be at most one candidate class; selection-FN takes precedence if both
somehow apply (a bad-frame pick is the more actionable finding).

## Design

### B1.1 — Audit component (`src/loop/fn_audit.py`)

A new deterministic CLI step, `python -m loop.fn_audit`, run during the nightly
tick before `loop.metrics`. It reads the `detections` table directly (it does not
depend on `loop.ingest`'s reconciled rows), so its position relative to `ingest`
is not load-bearing — the data-flow diagram places it ahead of `ingest` so this
tick's audit rows are reconciled in the same tick. For each trigger inside the
**burst-retention window** (bursts are a rolling window; see Ephemerality):

1. Derive the burst prefix from `detections.image_path` (strip the `_frameN`
   suffix) and glob the sibling frames on disk. Skip triggers whose siblings have
   already been cleaned up (log + count as `unauditable`, do not guess).
2. Run **detector-only** MegaDetector inference (~3 s/frame) on the **non-sent**
   siblings. The sent frame's detector verdict is already known from the DB
   (`detection_status` / logged boxes); do not re-run it.
3. Classify the trigger as OK / selection-FN candidate / classifier-FN candidate
   per the definitions above. Record, per candidate, the **score** = the
   MegaDetector detector confidence on the missed frame (highest box). This is the
   ranking/strength signal (see "No human label" below). Classifier confidence is
   deliberately not computed (detector-only).
4. Write the **automatic** verdict to `detection_feedback` as an append-only row
   with `source='megadetector_audit'` and a label in the FN vocabulary (e.g.
   `selection_fn` / `classifier_fn`). Append-only is preserved — the audit never
   UPDATEs/DELETEs, consistent with ADR-004's anti-self-poisoning rule. Idempotency:
   re-running a tick must not double-write; `fn_audit` checks for an existing
   `megadetector_audit` row for that `detection_id` before inserting.

**Detector-only inference:** the production `SpeciesIdentifier` loads
`components='all'`. The audit should obtain a detector-only path (prefer SpeciesNet's
detector component if its API exposes one; otherwise run the full `predict()` and
read only the `detections` boxes). The implementer verifies the cheapest correct
option; the spec's requirement is "detector verdict per sibling without paying for
classification where avoidable."

### B1.2 — Human confirmation (reuse existing feedback path)

The automatic verdict is a *candidate*, never truth. At nightly report time the bot
surfaces up to **N = 5** candidates (configurable; cap exists to prevent flooding),
**ranked by detector confidence, highest first**. Each card shows the relevant
frame (the missed sibling for selection-FN; the sent frame for classifier-FN) and a
yes/no keyboard: *"Did we miss an animal here?"*.

Confirmations reuse the existing infrastructure unchanged end-to-end:

- `src/feedback_protocol.py` — add two wire codes mapping to canonical FN labels
  (e.g. `mf` → `missed_animal`, `nm` → `no_missed_animal`) and a
  `build_fn_confirm_keyboard(detection_id)` that emits them under the existing
  `CALLBACK_PREFIX`. Because the prefix is unchanged, the running
  `telegram_feedback.py` `CallbackQueryHandler` and `parse_callback_data` handle
  the taps with **no handler changes** and write
  `add_feedback(detection_id, label, source="human")`.
- The human row (`source='human'`, `missed_animal`/`no_missed_animal`) is the
  **truth** tier; the `megadetector_audit` row is the **auto** tier — exactly the
  two-tier structure `ingest`/`metrics` already apply to FP.

### B1.3 — Metrics (`src/loop/metrics.py`)

`compute_metrics(rows, fn_audit)` already takes an `fn_audit` argument, today a
minimal `{"missed": int, "animal_frames": int}` scaffold (the ADR-004 FN-veto) that
yields a single blended FN rate. B1 **extends this shape** to carry the per-tier,
per-class counts below, and reworks the FN-rate computation to be human-headline +
auto-estimate rather than one blended number — the same move sub-project A made for
FP. Preserve the FN-veto's consumers (or update them in step with the new shape).
Extend `compute_metrics` to emit FN fields alongside the existing FP fields,
partitioned by the same precedence (human > audit):

- **Human-confirmed FN** (the headline): over triggers with a human FN label,
  count `missed_animal`. Fields e.g. `fn_human_count`, `n_fn_human`,
  `fn_human_rate`, `fn_human_ci` (Wilson, reuse `wilson_ci`).
- **MegaDetector-audit FN** (auto/unverified estimate): over triggers the audit
  examined, count selection-FN + classifier-FN candidates, split so the two classes
  are separable (`fn_selection_count`, `fn_classifier_count`, `n_fn_audited`,
  rates). Plus `fn_unauditable` (siblings gone).
- Persist the new fields into `state.last_metrics` (flat) and add columns to
  `_CSV_FIELDS` / `_row_for_csv` so `metrics/daily.csv` stays complete. CIs as the
  existing CSV CI columns do.

**No human label (the common early/steady state):** when `n_fn_human == 0`, the
headline legitimately reads "0 confirmed / not yet verified" — we never fabricate
truth. The actionable read falls back to the **MegaDetector-audit estimate**, which
is always shown and is **ranked/weighted by detector confidence (highest first)**.
The strongest-confidence candidates are both what gets surfaced for confirmation and
what the agent should weight most when no human signal exists. The loop is therefore
never blind, while the headline stays honest.

### B1.4 — Report (`src/loop/report.py`)

`render_summary` gains a plain-English FN section, mirroring the FP section's style
(no CI/jargon in Telegram). Shape:

> 🔎 Missed animals (so far): you confirmed **1** missed.
> • Auto check (unverified): **3** likely misses — **2** wrong-frame, **1** classifier
> • Not yet confirmed: **2**

- Headline line = human-confirmed only. If zero confirmed, say so plainly
  ("none confirmed yet").
- The auto line is explicitly tagged **(unverified)** and separates wrong-frame
  (selection-FN) from classifier misses, because the two point at different fixes.
- Precise CIs/counts persist to `state.json` + CSV; not printed to Telegram.

### B1.5 — Loop awareness (`experiments/PROTOCOL.md`, `experiments/loop.md`)

- Add `loop.fn_audit` to the documented tick sequence (around Ingest, before
  Measure), including its checkpoint guidance, matching the existing step style.
- Extend the agent's **lever list** with frame-selection levers, conditioned on the
  split this audit produces:
  - **selection-FN dominates** → the saved bursts contain the animal but selection
    missed it ⇒ consider changing frame-selection strategy (e.g. select/identify
    across all burst frames, or revisit motion-aware selection), implemented under
    loop control via the normal deploy path.
  - **classifier-FN dominates** → the sent frame had the animal but the classifier
    dropped it ⇒ a detection-threshold / model concern, not a selection one.
- Make explicit that the **fix is the agent's decision**, not automatic, and that
  the audit's headline (human-confirmed) governs trust while the auto estimate
  guides attention when confirmations are sparse.

## Data flow

`detections` + saved burst frames on disk → **`loop.fn_audit`** (detector re-run on
siblings → `detection_feedback` `megadetector_audit` rows + ranked candidate list)
→ nightly cards (top-N by confidence) → human taps → `detection_feedback` `human`
rows → `loop.ingest` → `loop.metrics` (**now FP *and* FN, per-tier**) →
`state.last_metrics` + CSV → `loop.report` (**now with the FN section**) → Telegram.

## Ephemerality (important constraint)

Bursts are a rolling window: `StorageConfig.max_images` (~100) bursts, oldest deleted
as a unit by the live system's cleanup. Therefore:

- `loop.fn_audit` **must run every tick**, before cleanup removes that day's bursts.
- Only recent nights are auditable; **historical backfill is impossible**. Triggers
  whose siblings are already gone are counted as `unauditable`, never guessed.
- This bounds nightly cost: ~ (triggers/night × ~4 siblings × ~3 s) detector-only.

## Testing

- **fn_audit classification unit tests:** synthetic bursts (animal in sibling only →
  selection-FN; animal in sent + reported blank → classifier-FN; animal in sent +
  reported animal → OK; no animal anywhere → OK; siblings missing → unauditable),
  using a mock detector so no model download is needed. Assert precedence
  (selection-FN wins when both apply) and idempotency (re-run writes no duplicate
  `megadetector_audit` row).
- **feedback_protocol tests:** new FN codes round-trip through
  `build_fn_confirm_keyboard` → `parse_callback_data` → canonical label; existing FP
  codes unchanged.
- **metrics tests:** FN partition by human > audit; headline counts only human rows;
  `n_fn_human == 0` yields an empty/"not yet" headline while the audit estimate is
  still populated and confidence-ranked; selection vs classifier counts separated;
  CSV/`state` carry the new fields.
- **report tests:** FN section renders plain English with no "CI"/aHash substrings;
  zero-confirmed reads "none confirmed yet"; auto line tagged "(unverified)" and
  splits wrong-frame vs classifier; uses `--no-send`.
- Full `uv run pytest tests/ -v`; a dry-run tick (`fn_audit` over a temp copy of a
  few bursts → `metrics --state /tmp/...` → `report --no-send`) renders the new
  output without mutating real state or sending Telegram messages.

## Rollback

Additive: a new module, two new label codes, new (nullable) metrics/CSV columns,
and doc updates. No change to the live capture/selection path and no `loop.deploy`.
Revert via `git revert`; no camera restart. Already-written `detection_feedback`
rows are harmless (append-only) and ignored if the readers are reverted.
