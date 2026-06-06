---
id: s-open-model-cost-eval-t70-per-model-pricing
kind: task
project: nanoclaw
status: done
autonomy: attended
parent: s-open-model-cost-eval
created: 2026-06-06
updated: 2026-06-06
---

# t70 — Price non-Claude models correctly in the cost path

## Outcome

Make the cost report tell the truth for a non-Claude model, so the trial's
spend numbers are real and not Anthropic-priced.

## Acceptance

- A turn run on Kimi is costed at Kimi's published rates, and a turn run on
  Claude is costed as it is today. The per-bot and on-demand (`ncl cost`) reports
  reflect the right price for whichever model produced the turn.
- The report no longer silently trusts the SDK's running cost total for a model
  the SDK did not price. Where a model has no rate in the table, the report says
  so rather than presenting a confident-but-wrong number. A Kimi row is NEVER
  costed via the SDK's (Anthropic-priced) cost series.
- The cost figure is auditable: the rate used per model is visible in the code
  (a single rate table), and the raw token counts behind each figure are still
  reported alongside the dollar amount.
- `ncl cost` shows the model id (or provider) per bot line, so during the trial a
  reader can see at a glance whether Iris's line is Kimi-priced or Sonnet-priced.
- Kimi's cache-token rates are pinned in the table from Moonshot docs. If a rate
  isn't published, use a documented assumption and flag it in the report rather
  than silently pricing cache at the input rate — cache pricing is what makes the
  "cheaper" claim true or false.

## Test specification

- A pure pricing function `priceTurn(model, {input, output, cacheRead,
  cacheCreation}) → usd` with a model→rate table. Unit-test (vitest, host tree):
  - Kimi K2.6 row: input $0.95/M, output $4.00/M → assert exact dollar maths on a
    known token vector.
  - Claude Sonnet 4.6 row: rates pinned from the claude-api reference at execute
    time → assert it reproduces the same order of magnitude the SDK reported on a
    real historical row (sanity, not exact).
  - Cache read/creation priced at their own rates per model (not at the input
    rate). Kimi cache rates are unknown at plan time — see gate below; the
    function must accept them as table fields, defaulting conservatively.
  - Unknown model id → function returns a sentinel (null) so the caller can mark
    the figure "unpriced", not silently zero or input-rate it.
- `collectUsage` test: given token_usage rows tagged `model='kimi-k2.6'`, the
  group total equals the sum of `priceTurn` over the rows — NOT the cumulative
  `cost_usd` walk. Assert a `kimi-k2.6` row never routes through the cumulative
  series (a typo'd id must surface as "unpriced", not silently inflate via the
  Anthropic-priced series). Given Claude rows, behaviour is unchanged from today.
- **Mixed-model window** (this is exactly what t80's flip produces): a single
  group's window containing both `claude` and `kimi-k2.6` rows totals to
  `priceTurn` summed per row at each row's own model rate — proving the accumulator
  replaced the per-group series walk, not sat beside it.
- Boundary: rows with a known model but null cache columns still price (cache
  terms contribute 0), and the existing "cumulative series resets on respawn"
  handling is irrelevant once per-row pricing replaces the series walk for priced
  models.

## Implementation notes

Grounded in `src/cost-report.ts` read at plan time:

- Today `collectUsage` (cost-report.ts:57) ignores `model` and reconstructs spend
  by walking the cumulative `cost_usd` series (lines 99–108) because the SDK
  reports `total_cost_usd` as a running total. That series is Anthropic-priced and
  wrong for Kimi. The `model` column IS recorded per row
  (`agent-runner/src/db/token-usage.ts`), and input/output/cache tokens are summed
  independently (lines 120–126) — so per-row pricing needs no new instrumentation.
- Add a rate table + `priceTurn` (new small module, e.g. `src/model-pricing.ts`).
  **Replace** the per-group cumulative-series accumulator (cost-report.ts:103-128)
  — don't leave it running beside the new path. For rows whose `model` is in the
  table, sum `priceTurn` per row; for unknown models only, fall back to the
  cumulative-series walk and flag the group "approx/unpriced". Kimi rows are in the
  table, so they never reach the fallback. Add the model id to `BotTotals` and the
  per-bot line so a Kimi bot reads as Kimi.
- Kimi cache rates: `priceTurn`'s table fields for cache-read/cache-creation must
  be real, not defaulted-to-input. Source from Moonshot docs; if unpublished, use a
  stated assumption and flag it — the cost claim is cache-dominated, so an unpriced
  cache term silently decides the verdict otherwise.
- Update the footnote in `formatReport` (line 171): when every bot in the window
  is priced from the table, drop "assumes API pricing; approximate"; when any bot
  is unpriced, keep a caveat naming which.
- `ncl cost` (`src/cli/commands/cost.ts`) reuses these builders unchanged — it
  inherits the fix for free.
- This is a port of the v1 t40 pricing work, re-grounded on the v2 outbound.db
  `token_usage` schema rather than v1's `getTokenUsageByContainer`.

## Status

active
