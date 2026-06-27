# Per-Tier Label Split + Plain-English Report — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the tuning loop report a human-truth FP headline plus a per-tier breakdown (human / Claude / MegaDetector), and send a plain-English nightly Telegram update instead of the dense JOURNAL blob.

**Architecture:** Pure aggregation + reporting change in `src/loop/`. `compute_metrics` gains a precedence-partitioned per-tier FP breakdown; `metrics.main` persists it to `state.json` + `daily.csv`; `report.render_summary` becomes plain prose; `report.main` sends a new agent-written `nightly_verdict` as message 2 in place of the JOURNAL entry. No detection/camera behavior changes.

**Tech Stack:** Python 3.13, pytest, UV. Run everything from repo root `/home/daniel/animal_tracker` with `PYTHONPATH=src`.

**Convention (project preference):** This plan describes intent, field names, signatures, and test assertions — it deliberately does **not** paste full function bodies. The implementer writes the code via TDD: failing test first, minimal implementation, green, commit.

## Global Constraints

- Run from repo root with `PYTHONPATH=src`; tests via `uv run pytest`.
- Headline FP = **human labels only**. Never blend tiers into the headline.
- Three label classes, partitioned by precedence **human > Claude (tier-2) > MegaDetector (tier-1)** — exactly one bucket per labeled trigger, no double-counting.
- **No** `human_label_sufficient` flag or any sufficiency threshold (explicitly dropped by Daniel). Show each tier's count; the count signals trust.
- Telegram output: plain English, **no CI notation, no aHash/jargon**. Precise CIs persist to `state.json`/CSV only.
- `JOURNAL.md` is unchanged and stays git-only; it is no longer sent to Telegram.
- Existing behaviors to preserve: paused banner, error-rate `fp_trustworthy` alert, the no-data-tick guard in `metrics.main`, `--no-send` dry-run renders without Telegram credentials.
- Field-name contract (used across tasks):
  `fp_human_rate, fp_human_count, n_human, fp_human_ci` /
  `fp_claude_rate, fp_claude_count, n_claude, fp_claude_ci` /
  `fp_md_rate, fp_md_count, n_md, fp_md_ci`.
  A bucket's FP test uses **that tier's own label value** == `"false_positive"`.

---

### Task 1: Per-tier FP partition in `compute_metrics`

**Files:**
- Modify: `src/loop/metrics.py` (`compute_metrics`, ~lines 48–91)
- Test: `tests/test_loop_metrics.py`

**Interfaces:**
- Consumes: ingest rows, each a dict with keys `human`, `tier2`, `tier1` (label string or `None`) plus existing `reconciled_label`, `detection_status` (see `ingest.py:128–136`).
- Produces: `compute_metrics(...)` return dict gains the 12 per-tier fields named in Global Constraints. Existing keys (`fp_rate`, `fp_count`, `fp_ci`, `labeled_triggers`, etc.) stay unchanged.

**Partition rule:** for each row, bucket = human if `row["human"] is not None`, else Claude if `row["tier2"] is not None`, else MegaDetector if `row["tier1"] is not None`, else unlabeled (excluded). Bucket FP count = rows in that bucket whose **own-tier label** == `"false_positive"`. Per-bucket CI via existing `wilson_ci(count, n)`.

- [ ] **Step 1: Write failing tests** in `tests/test_loop_metrics.py`:
  - `test_per_tier_partition_no_overlap_sums_to_labeled` — mixed rows; assert `n_human + n_claude + n_md == labeled_triggers`.
  - `test_per_tier_precedence_human_wins` — a row with both `human` and `tier2` set lands in human bucket only (n_claude unaffected).
  - `test_per_tier_precedence_claude_over_md` — row with `tier2` and `tier1` (no human) → Claude bucket.
  - `test_per_tier_fp_uses_own_tier_label` — a row where `human="animal"` but `tier1="false_positive"` counts as a human-bucket non-FP (tier1 ignored because human present).
  - `test_per_tier_artifact_regression` — 5 human rows (1 false_positive) + 37 md-only rows (35 false_positive): assert `fp_human_rate == 0.2`, `n_human == 5`, and `fp_md_rate` ≈ 0.95 on its own — i.e. headline is no longer the blended ~0.86.
- [ ] **Step 2: Run tests, verify they fail** — `PYTHONPATH=src uv run pytest tests/test_loop_metrics.py -k per_tier -v` → FAIL (missing keys).
- [ ] **Step 3: Implement** the partition in `compute_metrics` (a small helper that walks rows once, assigns each to a bucket, tallies count/n, computes rate + `wilson_ci`). Add the 12 fields to the return dict.
- [ ] **Step 4: Run tests, verify pass** — same command → PASS. Then full file: `PYTHONPATH=src uv run pytest tests/test_loop_metrics.py -v` (existing tests still green).
- [ ] **Step 5: Commit** — `git add src/loop/metrics.py tests/test_loop_metrics.py && git commit` with message `feat(loop): per-tier FP partition (human/Claude/MegaDetector) in compute_metrics`.

---

### Task 2: Persist per-tier fields to state.json + daily.csv

**Files:**
- Modify: `src/loop/metrics.py` (`_jsonable` ~135–139, `_CSV_FIELDS` ~94–98, `_row_for_csv` ~101–116)
- Test: `tests/test_loop_metrics.py`

**Interfaces:**
- Consumes: the per-tier fields from Task 1.
- Produces: `state["last_metrics"]` carries per-tier rates/counts/`n`/CIs (CIs as JSON lists, matching how `fp_ci` is serialized). `daily.csv` gains columns `n_human, fp_human_count, fp_human_rate, n_claude, fp_claude_count, fp_claude_rate, n_md, fp_md_count, fp_md_rate` (per-tier CI columns omitted — kept in state only, per spec "optional").

- [ ] **Step 1: Write failing tests:**
  - `test_jsonable_serializes_per_tier_ci_to_lists` — `_jsonable(m)` returns per-tier CIs as 2-element lists (JSON-round-trippable).
  - `test_csv_has_per_tier_columns` — `append_daily` then read back: header contains the 9 new columns and the row carries the right values.
  - `test_csv_append_backward_compat` — an existing CSV written with only the old `_CSV_FIELDS` can be re-read and re-written (old rows get blank new columns, no crash). (DictWriter writes `restval=""` for missing keys.)
- [ ] **Step 2: Run, verify fail** — `PYTHONPATH=src uv run pytest tests/test_loop_metrics.py -k "jsonable or csv" -v` → FAIL.
- [ ] **Step 3: Implement** — extend `_jsonable` to listify the three per-tier CIs; append the 9 names to `_CSV_FIELDS`; add them to the `_row_for_csv` dict.
- [ ] **Step 4: Run, verify pass** — same `-k` command → PASS; then whole file green.
- [ ] **Step 5: Commit** — `feat(loop): persist per-tier FP metrics to state.json and daily.csv`.

---

### Task 3: Plain-English `render_summary` (Telegram message 1)

**Files:**
- Modify: `src/loop/report.py` (`render_summary` ~67–111)
- Test: `tests/test_loop_report.py`

**Interfaces:**
- Consumes: flat `metrics` dict with `total_triggers` + per-tier `n_*`/`fp_*_count` from Tasks 1–2; `state` (for `paused`); `active_experiment`.
- Produces: a plain-text block — total line, one line per tier (count + FP count), a "not yet labelled" remainder line when `total − (n_human+n_claude+n_md) > 0`. No CI text. Preserves paused banner and the error-rate untrustworthy alert.

Target shape (wording may vary, assertions below are the contract):
```
🦊 Last night: 42 images captured.
• You labelled 2 — 1 false alarm
• Claude labelled 0
• MegaDetector (auto, unverified): 40 — 38 false alarms
```

- [ ] **Step 1: Write failing tests:**
  - `test_summary_per_tier_lines` — given a metrics dict, output contains the total and a line each for human/Claude/MegaDetector with correct counts.
  - `test_summary_md_line_marked_unverified` — MegaDetector line contains "auto" / "unverified".
  - `test_summary_no_ci_or_jargon` — output contains no "CI", no "aHash", no "%CI" substrings.
  - `test_summary_zero_tiers_still_listed` — a tier with `n==0` still renders its line ("Claude labelled 0").
  - `test_summary_remainder_line_when_unlabelled` — total 42, labelled sums to 40 → a "Not yet labelled: 2" line; and absent when remainder is 0.
  - `test_summary_paused_banner_preserved` and `test_summary_untrustworthy_alert_preserved` — update the existing equivalents to the new format (keep these behaviors).
  - `test_summary_backward_compat_missing_per_tier_keys` — old `last_metrics` lacking `n_*` keys renders without crashing (`.get` defaults to 0).
- [ ] **Step 2: Run, verify fail** — `PYTHONPATH=src uv run pytest tests/test_loop_report.py -k summary -v` → FAIL.
- [ ] **Step 3: Implement** — rewrite `render_summary` to the per-tier prose; drop the blended `fp_rate`/CI/FN lines from the Telegram text; keep paused banner + untrustworthy alert. Use `.get(key, 0)` for per-tier fields for backward-compat.
- [ ] **Step 4: Run, verify pass** — same `-k` command → PASS.
- [ ] **Step 5: Commit** — `feat(loop): plain-English per-tier render_summary, drop CI/jargon from Telegram`.

---

### Task 4: `nightly_verdict` as message 2; retire JOURNAL send; update agent docs

**Files:**
- Modify: `src/loop/report.py` (`main` summary branch ~160–194; the `latest_journal_entry` call + `--journal` plumbing)
- Modify: `experiments/loop.md` (step 4–5), `experiments/PROTOCOL.md` (notebook-writing step)
- Test: `tests/test_loop_report.py`

**Interfaces:**
- Consumes: `state["nightly_verdict"]` — a short plain-English string the loop agent writes during the tick (≤2 sentences, no jargon).
- Produces: summary mode sends message 1 (`render_summary`) then, **if** `nightly_verdict` is present and non-empty, message 2 = that verdict. If absent/empty, no second message (graceful). `latest_journal_entry` is no longer called by the summary path (leave the function + its unit tests in place; just stop using it here).

- [ ] **Step 1: Write/adjust failing tests:**
  - `test_no_send_summary_includes_nightly_verdict` — state has `nightly_verdict="..."`; `--no-send` JSON output exposes it as the second message (replace `journal_entry` key with `nightly_verdict`).
  - `test_no_send_summary_no_verdict_when_absent` — no `nightly_verdict` in state → second message null/omitted.
  - `test_real_send_summary_calls_send_twice_with_verdict` — monkeypatch `send`; with a verdict present, `send` called twice (summary + verdict).
  - `test_real_send_summary_sends_once_when_no_verdict` — no verdict → `send` called once.
  - Update/remove the old `*_journal_entry*` tests that asserted JOURNAL was the second message (the `latest_journal_entry` *function* tests stay).
- [ ] **Step 2: Run, verify fail** — `PYTHONPATH=src uv run pytest tests/test_loop_report.py -k "verdict or send_summary" -v` → FAIL.
- [ ] **Step 3: Implement** — in `report.main` summary branch, read `st.get("nightly_verdict")`; replace the journal-entry second-send with the verdict send; update the `--no-send` JSON payload key. Remove the now-dead journal-send block (keep `latest_journal_entry` defined).
- [ ] **Step 4: Update agent docs** — in `experiments/loop.md` (notebook step) and `experiments/PROTOCOL.md`: instruct the agent to set `state["nightly_verdict"]` to a ≤2-sentence plain-English verdict when writing the notebook, and note the report now sends that verdict (not the JOURNAL entry); JOURNAL stays git-only.
- [ ] **Step 5: Run, verify pass** — `PYTHONPATH=src uv run pytest tests/test_loop_report.py -v` → PASS (full file).
- [ ] **Step 6: Commit** — `feat(loop): send agent nightly_verdict to Telegram instead of JOURNAL blob; update loop docs`.

---

### Task 5: Full-suite gate + dry-run tick

**Files:** none (verification only)

- [ ] **Step 1: Full suite** — `PYTHONPATH=src uv run pytest tests/ -v`. Expected: all pass.
- [ ] **Step 2: Dry-run tick against a temp state** (no real state/DB mutation, no Telegram):
  ```bash
  cp experiments/state.json /tmp/state_test.json
  PYTHONPATH=src uv run python -m loop.metrics --state /tmp/state_test.json --csv /tmp/daily_test.csv
  PYTHONPATH=src uv run python -m loop.report --mode summary --state /tmp/state_test.json --no-send
  ```
  Expected: report renders the per-tier breakdown; if `/tmp/state_test.json` has no `nightly_verdict`, second message is null. (Note: a real overnight tick is a no-data tick — `metrics.main` returns `status: no_data` and preserves baseline; that path is unchanged.)
- [ ] **Step 3: Commit** any incidental fixes; otherwise done.

---

## Self-Review

- **Spec coverage:** A1 per-tier split → Tasks 1–2; A1 persistence (state+CSV) → Task 2; A2 message 1 plain prose → Task 3; A2 message 2 `nightly_verdict` + JOURNAL retired + agent-doc update → Task 4; testing/dry-run → Task 5. Provenance note is documentation captured in Task 4 doc step. No spec requirement left unmapped.
- **Placeholder scan:** every task has concrete files, named tests with assertions, and exact commands; no TBD/"handle edge cases".
- **Type/name consistency:** the 12 per-tier field names are fixed in Global Constraints and reused verbatim in Tasks 1–3; `nightly_verdict` (state key) is consistent across Task 4 and the docs.
- **Backward-compat:** explicit tests for old CSV rows (Task 2) and old `last_metrics` without per-tier keys (Task 3).
