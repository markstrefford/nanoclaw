---
id: s-open-model-cost-eval-t50-run-and-report
kind: task
project: nanoclaw
status: done
autonomy: attended
parent: s-open-model-cost-eval
created: 2026-06-05
updated: 2026-06-12
---

# t50 — Compare cost, call quality, decide go/no-go

## Outcome

Compare Kimi's real cost against the Sonnet baseline, take Mark's quality read,
and land a plain decision under a fixed rule.

## Acceptance

- A short writeup reports Iris's spend on Kimi over the trial window (t70-priced,
  Kimi-correct) against her recent Sonnet baseline spend, with raw token counts
  and the per-model rates shown so the cost claim is auditable. If t80 showed Kimi
  doesn't cache like Anthropic, that caveat sits beside the cost number.
- Mark's quality verdict over the window is recorded in one line: better / same /
  worse than Sonnet for how Iris is actually used.
- The decision follows a fixed rule, stated plainly:
  - clearly **worse** → revert to Sonnet, close the thread;
  - clearly **better** → Kimi is a switch candidate (the switch itself is a
    separate, out-of-scope decision);
  - **too close to call** → go cheapest (Kimi), since reverting is trivial.

## Test specification

No automated tests — analysis. The "test" is reproducibility: the trial window
dates, model ids, the Moonshot endpoint, the t70 rates, and Mark's verdict are all
recorded so the conclusion could be re-derived from the same `token_usage` data.

## Implementation notes

Grounded in what exists after the prior tasks:

- Cost: read per-model spend from the t70-priced path (`collectUsage` / `ncl cost`),
  scoped to the trial window and Iris's group. Baseline = Iris's own recent Sonnet
  spend (same agent, roughly comparable task mix). This is deliberately NOT a
  controlled probe — per the operator's call, a rough same-agent before/after is
  enough to drive the cheapest-wins-on-a-tie rule; reverting is cheap if wrong.
- Quality is the operator's direct read, not a scored rubric — by explicit
  decision. No probe, no blind scoring.
- Output is a short writeup filed alongside the story. On a "better" or
  "tie → Kimi" outcome, the production switch is a separate decision the story
  leaves out of scope.
- Reuse-from-v1 note: the v1 t50 mechanics (OPENAI_MODEL/base-url routing,
  getTokenUsageByContainer) do NOT apply — v2 reaches Kimi via the Anthropic-compat
  provider (t60) and prices via t70.

## Status

done — Verdict: GO (cheapest-wins, and clearly better-and-cheaper). Kimi ~1–2¢/turn
vs Sonnet $0.60–1.30; Mark's quality read = no drop. Closed 2026-06-12.
