---
id: s-open-model-cost-eval-t80-trial
kind: task
project: nanoclaw
status: active
autonomy: attended
parent: s-open-model-cost-eval
created: 2026-06-06
updated: 2026-06-06
---

# t80 — Flip Iris to Kimi and run the multi-day trial

## Outcome

Run Iris on Kimi K2.6 live for a couple of days, after a quick check that its
cost number is real — then judge it by use.

## Acceptance

- Iris's group is switched to Kimi K2.6 (per-group config), runs on real traffic
  for the trial window (a couple of days), and can be switched back to Sonnet in
  seconds with no rebuild.
- Before the window is trusted for cost, a one-glance sanity check passes: Kimi
  turns record cache reads, so the large system prompt isn't billed as fresh input
  every turn. If it isn't caching, that's noted beside the cost number — it's the
  one thing that could make "cheaper per token" not mean "cheaper per task."
- Over the window, spend is readable through the t70 per-model pricing (Kimi-
  correct), and Mark forms his own quality read from using it.

## Test specification

No automated tests — live operational task. Verifiable steps:

- **First-turn cost sanity:** after Iris's first Kimi turn, query that session's
  `outbound.db` `token_usage` and confirm `model='kimi-k2.6'`, sane in/out tokens,
  and non-trivial `cache_read_tokens`. No caching → record it; it changes how the
  cost result is read, but does not block the trial.
- Reversibility: flipping the group back to `provider='claude'` restores Sonnet on
  the next turn with no rebuild.
- Spend is queryable mid-trial via `ncl cost`, reading through t70 pricing.

## Implementation notes

Grounded in code read at plan time:

- Flip = update Iris's `container_configs` to `provider='kimi'`,
  `model='kimi-k2.6'` (depends on t60). Reverse = set back to `claude`. Per-group,
  reversible, no rebuild — `resolveProviderName` picks it up on next spawn.
- The cache fields already flow end-to-end: provider `TurnUsage` →
  `recordTokenUsage` writes `cache_read_tokens` / `cache_creation_tokens`. The
  sanity check is reading those rows, not new code. It exists because NanoClaw's
  per-turn input is dominated by a large cached system prompt + skills — the
  "cheaper" claim only holds if Kimi caches comparably.
- Once the first live Kimi turn is verified, delete the Moonshot key from the v1
  `nanoclaw/.env` (it now lives only in the OneCLI vault).
- Trial group is Iris/Ayah (personal), chosen over North (bigger, business-linked
  blast radius) and over a sandbox (too slow, and without real Gmail/Calendar
  integrations the comparison is meaningless). Internal keys still read `iris`
  though the display name is Ayah (grep `iris`).

## Status

active
