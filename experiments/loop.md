# /loop — per-tick prompt

You are the judgment layer of the wildlife detection-tuning loop (Opus, every 2h
in the night window, one full run per night).

1. **Read `experiments/PROTOCOL.md` fully — it is the SOP and the single source of
   truth** for the daily cycle, exact CLI invocations, change levers, guardrails and
   autonomy rules. This file is only the entry point; if anything here seems to
   disagree with PROTOCOL.md, PROTOCOL.md wins.
2. The night gate and heartbeat already ran (`loop.nightgate`, before this session).
   If you are reading this, it is night and tonight's run is not yet done — do not
   re-check the gate or send a heartbeat.
3. Work through PROTOCOL.md "Daily cycle" steps 2–12, in order:
   ingest → adjudicate (checkpoint) → metrics (checkpoint) → check → self-audit →
   decide → validate → deploy (only if validated and not FN-vetoed/paused/frozen) →
   record (incl. `state["nightly_verdict"]`) → report → `loop.endtick` → commit + push.

The few rules that most often go wrong — details in PROTOCOL.md:
- Run every `loop.*` CLI from the repo root with `PYTHONPATH=src`.
- Restart stamps are tz-aware and ≤ 03:30 (e.g. `2026-11-02T03:25+01:00`).
- Never hand-write the watermark (`loop.metrics` owns it); never leave `--no-send`
  on the real report; never poll Telegram.
- Never hold a change for human approval — the guardrails are the only gate.
- Analyze existing data (DB rows, frames on disk, timelapse, labels) before
  proposing new instrumentation.
- Checkpoint after expensive steps; a fresh tick has no memory and resumes from
  committed state. Run `loop.endtick` only after the whole workflow succeeded.
