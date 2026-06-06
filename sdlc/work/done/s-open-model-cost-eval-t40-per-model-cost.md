---
id: s-open-model-cost-eval-t40-per-model-cost
kind: task
project: nanoclaw
status: done
autonomy: attended
parent: s-open-model-cost-eval
created: 2026-06-05
updated: 2026-06-05
---

# t40 — Make cost reflect the model that was actually used

## Outcome

Make reported cost use the price of the model that actually ran, not Sonnet's price
for everything.

## Pulled forward (2026-06-05)

This task is being done out of sequence because the model-blind cost calc is a real
bug regardless of which models we ever bench. Split into two halves: the
**model-aware bugfix lands now** (this task); **adding specific challenger model
prices is deferred** until the model list is locked (a one-line table entry per
model, tracked in t50, not blocking).

## Acceptance

- `estimateCost` is model-aware: two usage summaries with identical token counts but
  different `model` values produce different costs that match each model's table
  entry. Same tokens no longer always cost the same.
- A model not in the pricing table falls back to the documented default and does not
  throw.
- Existing Claude cost numbers are unchanged.
- (Deferred, not in this task) challenger model price rows — added when the bench
  model list is final.

## Test specification

- Unit (new test against the stats cost logic): two usage summaries with identical
  token counts but different `model` values yield different costs that match each
  model's table entry.
- Unit: a model not in the table falls back to the documented default and does not
  throw.
- Boundary: zero tokens → zero cost for any model.
- Regression: the three existing Claude entries produce the same numbers they do
  today.

## Implementation notes

Two grounded defects in `src/stats-api.ts`:

1. **(this task)** `estimateCost()` (~line 46) hardcodes `const p = DEFAULT_PRICING`
   and never reads the model — so every model is priced as Sonnet. Fix: thread the
   `model` string in and look it up in `MODEL_PRICING`, falling back to
   `DEFAULT_PRICING` only on miss.
2. **(deferred to when the model list is final)** `MODEL_PRICING` (~lines 17-35) has
   no challenger entries. Add each bench model with its real published rates when
   chosen. The pricing key must EXACTLY match the model string the runner records —
   the OpenAI runner writes the `process.env`-derived model verbatim into
   `token_usage` (openai-runner.ts:471). Whatever access door is used (OpenRouter /
   Ollama Cloud / native), the recorded model id is whatever string was set as the
   model — so the price key must match that exact id.

Scope boundary that matters for this spike: the per-row data has `model` (db.ts
`token_usage` schema ~line 72), but the aggregate endpoints throw it away —
`getTokenUsageSummary` returns a single cross-model total, and `getTokenUsageByGroup`
aggregates per group with no model field. Fully fixing those aggregates to be
model-aware is larger than the spike needs. For THIS story, the minimum fix is:
make `estimateCost` model-aware and add Kimi pricing, and rely on the per-container
rows (`getTokenUsageByContainer`, which keeps `model`) for the actual comparison in
t50. Note in the code/comment that summary/by-group cost remains model-blind so a
later story can decide whether to fix it.

Pre-existing debt to NOT propagate: openai-runner.ts:471 interpolates the model name
straight into a `sqlite3` shell string. Kimi ids are safe, but don't copy that
pattern into the pricing/stats changes.

Pure host-side change (stats-api.ts, possibly a small db helper). No container
rebuild needed.

## Status

done — 2026-06-05. `estimateCost` made model-aware (exported, optional `model`,
falls back to default on miss/null); per-container stats endpoint now prices each
row by its own model. 6 unit tests added, full suite green (266). Verifier: pass.
Challenger price rows deferred to when the model list is locked (per t50).
