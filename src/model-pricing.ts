/**
 * Per-model token pricing.
 *
 * The container records raw token counts per turn into `token_usage`
 * (input/output/cache-read/cache-creation) plus the SDK's `cost_usd`. That
 * `cost_usd` is Anthropic-priced — correct for Claude, WRONG for a model served
 * through an Anthropic-compatible endpoint (e.g. Kimi via Moonshot), where the
 * SDK has no idea what the provider actually charges. This module prices a turn
 * from the raw token counts and a per-model rate table, so the cost report tells
 * the truth regardless of which model produced the turn.
 *
 * Rates are USD per single token (published $/M ÷ 1e6). Sources:
 *   - claude-sonnet-4-6: input $3 / output $15 per M; cache-write 1.25× input,
 *     cache-read 0.1× input (Anthropic prompt-cache multipliers).
 *   - kimi-k2.6 (Moonshot): input $0.95 / output $4 per M; cache-read (cache
 *     hit) $0.16/M published. Moonshot publishes no separate cache-WRITE
 *     premium, so cache-creation is assumed at the input rate and the model is
 *     flagged `exact: false` — the report surfaces that the Kimi cost carries a
 *     cache-write assumption.
 */

export interface ModelRates {
  /** USD per input token. */
  input: number;
  /** USD per output token. */
  output: number;
  /** USD per cache-creation (write) token. */
  cacheCreation: number;
  /** USD per cache-read token. */
  cacheRead: number;
  /** True if every rate is published; false if any is a documented assumption. */
  exact: boolean;
}

export interface TurnTokens {
  input: number;
  output: number;
  cacheRead: number;
  cacheCreation: number;
}

const M = 1_000_000;
const perM = (n: number): number => n / M;

/**
 * Per-token rates, matched against `token_usage.model` by case-insensitive
 * prefix so date-suffixed ids (`claude-sonnet-4-6-20251101`) resolve to the
 * base entry. Order does not matter — prefixes are disjoint.
 */
const RATES: ReadonlyArray<{ prefix: string; rates: ModelRates }> = [
  {
    prefix: 'claude-sonnet-4-6',
    rates: { input: perM(3), output: perM(15), cacheCreation: perM(3.75), cacheRead: perM(0.3), exact: true },
  },
  {
    prefix: 'kimi-k2.6',
    rates: { input: perM(0.95), output: perM(4), cacheCreation: perM(0.95), cacheRead: perM(0.16), exact: false },
  },
];

/** Rates for `model`, or null if the model has no entry (→ caller marks it unpriced). */
export function getModelRates(model: string | null | undefined): ModelRates | null {
  if (!model) return null;
  const m = model.toLowerCase();
  for (const { prefix, rates } of RATES) {
    if (m.startsWith(prefix)) return rates;
  }
  return null;
}

/**
 * USD cost of one turn for `model`, or null if the model has no rate entry.
 * Null (not zero, not input-rate) so the caller can mark the figure "unpriced"
 * rather than present a confident-but-wrong number — a Kimi row must never be
 * silently costed by the Anthropic-priced SDK total.
 */
export function priceTurn(model: string | null | undefined, t: TurnTokens): number | null {
  const r = getModelRates(model);
  if (!r) return null;
  return t.input * r.input + t.output * r.output + t.cacheRead * r.cacheRead + t.cacheCreation * r.cacheCreation;
}
