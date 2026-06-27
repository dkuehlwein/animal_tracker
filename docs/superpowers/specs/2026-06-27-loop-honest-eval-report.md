# Spec: Honest evaluation — per-tier label split + plain-English nightly report

**Date:** 2026-06-27
**Status:** Approved (design), pending implementation plan
**Scope:** Sub-project **A** of the loop-honesty initiative. B1 (selection/classifier FN from saved bursts) and B2 (trigger-FN timelapse audit) are separate, later specs.

## Problem

The autonomous tuning loop judges itself on a blended label that treats a
MegaDetector guess as equal to a human's verdict. `ingest.py` reconciles labels
as `reconciled = human or tier2 or tier1` and `metrics.py` counts the FP rate over
that blend. On a night with few human labels, auto-labels dominate and the report
shows an alarming, misleading headline (e.g. "FP rate 95%") that is really a
labeling artifact — the code's own comment notes auto-labels are only ~29–36%
concordant with human labels.

Two consequences:
1. **Dishonest self-evaluation** — the loop's headline number is not ground truth.
2. **Unreadable nightly updates** — Telegram receives a stats summary plus the raw,
   jargon-dense `JOURNAL.md` entry (CIs, aHash cluster counts, "FN-veto/HOLD").

Daniel's instruction: *"Stop treating auto-labels as truth. They are a very
different class from human labels."* Claude (tier-2) is its **own** class, distinct
from both human and MegaDetector.

## Goals

- Headline FP rate reflects **human labels only**.
- The three label sources — **human**, **Claude (tier-2)**, **MegaDetector (tier-1)** —
  are reported as separate lines, each with its own count, never blended into the
  headline.
- The nightly Telegram update is **plain English**: the numbers as prose, plus a
  short plain-language verdict. The dense lab-notebook entry stays in git only.

## Non-goals

- Measuring false negatives (that is B1/B2).
- Changing the trigger, the species threshold, or any detection behavior.
- Changing how labels are *collected* (the Telegram feedback sidecar, the
  `detection_feedback` table). Only how they are *aggregated and reported*.

## Design

### A1 — Per-tier metrics (`src/loop/metrics.py`)

`compute_metrics` already receives ingested rows that carry `tier1`, `tier2`,
`human`, and the precedence-collapsed `reconciled_label` (see `ingest.py:112–136`).
Add a per-tier partition alongside the existing computation.

**Partition rule** — assign each labeled trigger to exactly one bucket by best
available source (same precedence already used for `reconciled_label`):

- a row with a non-None `human` label → **human** bucket
- else a row with a non-None `tier2` label → **Claude** bucket
- else a row with a non-None `tier1` label → **MegaDetector** bucket
- rows with no label in any tier are unlabeled (excluded, as today)

This is a true partition: buckets do not overlap and sum to `labeled_triggers`.
Rationale for partition-by-best-source rather than three overlapping views: it
answers "what is our best read on each trigger, and how trustworthy is that read,"
without double-counting a trigger that both Claude and a human labeled.

For each bucket compute: `fp_count`, `n` (bucket size), `fp_rate`, and a Wilson CI
(reuse the existing `wilson_ci`). Emit them under explicit keys, e.g.:

```
fp_human_rate, fp_human_count, n_human, fp_human_ci
fp_claude_rate, fp_claude_count, n_claude, fp_claude_ci
fp_md_rate,    fp_md_count,    n_md,    fp_md_ci
```

**Headline** = the human bucket. The existing blended `fp_rate` / `fp_count` /
`fp_ci` keys remain in the dict for continuity (history, CSV) but are no longer the
headline the report leads with.

**No sufficiency threshold / flag.** Each line shows its `n`; the count itself
signals trust. (Explicitly dropped at Daniel's request — do not add a
`human_label_sufficient` boolean.)

The existing error-rate trustworthiness check (`fp_trustworthy`,
`ERROR_RATE_UNTRUSTWORTHY_THRESHOLD`) is unchanged and still applies.

**Persistence:** the new per-tier fields are written into `state.last_metrics`
(flat, like the current metrics). The per-night CSV (`_CSV_FIELDS` / `_row_for_csv`)
gains columns for the three bucket counts and rates so the daily history stays
complete; CI columns for the buckets are optional (persist if cheap, since the
existing CSV already carries `fp_ci_low/high`).

### A2 — Plain-English report (`src/loop/report.py`)

Today `report.main` sends two Telegram messages: `render_summary` (stats) and the
latest `JOURNAL.md` entry (dense). Replace with two **plain** messages.

**Message 1 — total, then an explicit per-tier breakdown.** Lead with the total
images captured, then one line per tier giving *how many were labelled* and *how
many of those were false alarms*. No CI notation. The partition guarantees the three
labelled counts never double-count (a trigger you and Claude both labelled counts
once, under you).

> 🦊 Last night: **42** images captured.
> • You labelled **2** — **1** false alarm
> • Claude labelled **0**
> • MegaDetector (auto, unverified): **40** — **38** false alarms

Format notes:
- Every tier line shows its labelled count explicitly, including `0` (so it's clear
  Claude/you simply didn't label, vs. an error). A quiet all-zero night still lists
  the tiers.
- MegaDetector's line is explicitly tagged **(auto, unverified)** so its FP count is
  visibly a machine guess, not ground truth.
- Unlabelled remainder, if any (`total − sum of the three labelled counts`), gets a
  trailing line, e.g. "• Not yet labelled: 0".
- Precise CIs are persisted to `state.json` and the CSV; deliberately not printed to
  Telegram.

**Label provenance** (informs the wording, no real-time path needed): MegaDetector
labels are written automatically at capture; human labels arrive via Telegram
buttons; **Claude (tier-2) labels are written by the loop agent at night during the
tick**, when it adjudicates that day's ambiguous crops *before* `loop.metrics` runs
(`PROTOCOL.md:62`). Claude adjudicates selectively (ambiguous crops only), so
`n_claude` is expected to be small or zero on most nights — that is normal, not a
bug.

**Message 2 — the verdict, plain English.** A new, dedicated short field the loop
agent writes each night: `nightly_verdict` — ≤2 sentences, no jargon, no aHash/CI
terms. Example: *"Nothing changed tonight — I still can't measure missed animals,
so I won't touch the trigger."*

- Source of truth for `nightly_verdict`: written by the agent during the tick.
  Storage: a field in `state.json` (e.g. `state["nightly_verdict"]`) set alongside
  `last_metrics`, OR a one-line companion the report reads. (Plan to pick the
  simpler of the two during implementation; `state.json` is the default.)
- `report.py` reads `nightly_verdict` and sends it as message 2 **instead of** the
  raw JOURNAL entry.

**`JOURNAL.md` is unchanged and stays in git only.** It remains the dense technical
lab notebook for future agent sessions; it is simply no longer pushed to Telegram.
`latest_journal_entry()` is no longer used by the summary send path (keep the
function or remove it per implementation tidiness).

The `/loop` per-tick prompt (`experiments/loop.md`, step 5) and `PROTOCOL.md` are
updated so the agent knows to write `nightly_verdict` as part of writing the
notebook.

## Data flow (unchanged except where noted)

`detections` + `detection_feedback` → `loop.ingest` (rows with `tier1/tier2/human`)
→ `loop.metrics` (**now also per-tier buckets**) → `state.last_metrics` + CSV →
`loop.report` (**now plain prose + `nightly_verdict`**) → Telegram. The agent writes
`nightly_verdict` into state during the notebook step.

## Testing

- **metrics partition unit tests:** synthetic rows exercising each precedence path
  (human-only, Claude-only, MD-only, human+Claude on same row → human bucket,
  human+MD → human, Claude+MD → Claude), asserting buckets partition correctly,
  sum to `labeled_triggers`, and per-bucket `fp_rate`/`n` are right.
- **Artifact regression:** the historical "thin human labels, many MD labels" night
  (e.g. 2026-06-26: 5 human, ~37 auto) must yield a human headline driven only by
  the human rows, with the MD estimate on its own line — assert the headline is no
  longer the blended 95%.
- **report rendering tests:** `render_summary` produces the plain prose with no "CI"
  / aHash substrings; `n == 0` tiers collapse/omit; `nightly_verdict` present →
  message 2 is the verdict, absent → graceful fallback. Use `--no-send` (no Telegram
  credential needed) as the existing tests do.
- Run the full `uv run pytest tests/ -v` plus any `loop` tests; a dry-run tick
  (`ingest --since-id 0` → `metrics --state /tmp/...` → `report --no-send`) renders
  the new output without mutating real state.

## Rollback

Pure reporting/aggregation change; no detection behavior touched. Revert via
`git revert`. No camera restart and no `loop.deploy` involved.
