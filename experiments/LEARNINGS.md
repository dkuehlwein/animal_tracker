# Learnings

Distilled, firm conclusions (semantic memory) — only things proven on a real
corpus with CI support. Keep terse; link the run file that established each.

- **FP is dominated by the no_animal + unclassifiable status classes, and a
  same-channel REVIEW prefix flags ~99% of them with 0 FN.** Routing those two
  statuses (not just no_animal) catches 89/90 FP; the unprefixed stream is ~94%
  true animals; the 3 real animals MegaDetector misses still arrive fully (labeling
  ≠ suppression). **Accepted final design = same channel + 🔍 REVIEW prefix; a
  second Telegram channel is explicitly OUT** (Daniel, 2026-06-15: "I am not
  clicking on two channels"). Holds across old + new camera locations. (run 0001)
- **Motion features do not separate FP from animals**, so MOTION_THRESHOLD/ROI
  sensitivity tuning is FN-risky and futile as an FP lever; recurrence is REAL
  motion (swinging feeder / wind / sun-dapple), not static scenes MOG2 failed to
  absorb. No env knob in BOUNDS reaches the FP root cause. (run 0002, exp #4)
