import { describe, it, expect } from 'vitest';

import { getModelRates, priceTurn } from './model-pricing.js';

describe('priceTurn', () => {
  it('prices a Kimi K2.6 turn at Moonshot rates', () => {
    // 1M input, 1M output, 0 cache → $0.95 + $4.00
    expect(priceTurn('kimi-k2.6', { input: 1_000_000, output: 1_000_000, cacheRead: 0, cacheCreation: 0 })).toBeCloseTo(
      4.95,
      6,
    );
    // cache-read priced at $0.16/M, cache-write at the (assumed) input rate $0.95/M
    expect(priceTurn('kimi-k2.6', { input: 0, output: 0, cacheRead: 1_000_000, cacheCreation: 1_000_000 })).toBeCloseTo(
      1.11,
      6,
    );
  });

  it('prices a Claude Sonnet 4.6 turn at Anthropic rates, incl. cache multipliers', () => {
    // 1M input + 1M output → $3 + $15
    expect(
      priceTurn('claude-sonnet-4-6', { input: 1_000_000, output: 1_000_000, cacheRead: 0, cacheCreation: 0 }),
    ).toBeCloseTo(18, 6);
    // cache-read 0.1× input = $0.30/M, cache-write 1.25× input = $3.75/M
    expect(
      priceTurn('claude-sonnet-4-6', { input: 0, output: 0, cacheRead: 1_000_000, cacheCreation: 1_000_000 }),
    ).toBeCloseTo(4.05, 6);
  });

  it('matches date-suffixed model ids by prefix', () => {
    expect(getModelRates('claude-sonnet-4-6-20251101')).not.toBeNull();
    expect(getModelRates('Kimi-K2.6')).not.toBeNull(); // case-insensitive
  });

  it('returns null for an unknown or missing model (never input-rates or zero-prices it)', () => {
    expect(priceTurn(null, { input: 1_000_000, output: 0, cacheRead: 0, cacheCreation: 0 })).toBeNull();
    expect(priceTurn('gpt-5', { input: 1_000_000, output: 0, cacheRead: 0, cacheCreation: 0 })).toBeNull();
    expect(priceTurn(undefined, { input: 1, output: 1, cacheRead: 1, cacheCreation: 1 })).toBeNull();
  });

  it('flags Kimi as inexact (cache-write assumption) and Sonnet as exact', () => {
    expect(getModelRates('kimi-k2.6')?.exact).toBe(false);
    expect(getModelRates('claude-sonnet-4-6')?.exact).toBe(true);
  });
});
