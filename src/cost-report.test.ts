import { describe, it, expect } from 'vitest';

import { priceRows, type UsageRow } from './cost-report.js';

const row = (p: Partial<UsageRow>): UsageRow => ({
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheCreation: 0,
  cost: 0,
  turns: 0,
  model: null,
  ...p,
});

describe('priceRows', () => {
  it('prices Kimi rows from the rate table, never from the (Anthropic) cost_usd series', () => {
    // cost_usd is a fat Anthropic-priced number; it must be ignored for kimi.
    const rows = [row({ model: 'kimi-k2.6', input: 1_000_000, output: 1_000_000, cost: 999 })];
    const { cost, unpriced } = priceRows(rows);
    expect(cost).toBeCloseTo(4.95, 6); // $0.95 + $4.00 — NOT 999
    expect(unpriced).toBe(false);
  });

  it('leaves Claude pricing intact and exact', () => {
    const rows = [row({ model: 'claude-sonnet-4-6', input: 1_000_000, output: 1_000_000, cost: 0.001 })];
    const { cost, assumed, unpriced } = priceRows(rows);
    expect(cost).toBeCloseTo(18, 6);
    expect(assumed).toBe(false);
    expect(unpriced).toBe(false);
  });

  it('handles a mixed-model window (mid-window flip) — each row at its own rate', () => {
    const rows = [
      row({ model: 'claude-sonnet-4-6', input: 1_000_000 }), // $3
      row({ model: 'kimi-k2.6', output: 1_000_000 }), // $4
    ];
    const { cost, models, assumed } = priceRows(rows);
    expect(cost).toBeCloseTo(7, 6);
    expect(models).toEqual(new Set(['claude-sonnet-4-6', 'kimi-k2.6']));
    expect(assumed).toBe(true); // Kimi cache-write assumption flag rides along
  });

  it('flags unpriced and falls back to the cumulative cost_usd series for unknown models', () => {
    const rows = [
      row({ model: 'mystery-model', cost: 0.5 }),
      row({ model: 'mystery-model', cost: 0.9 }), // climbs → +0.4
      row({ model: 'mystery-model', cost: 0.2 }), // drop → new run, +0.2
    ];
    const { cost, unpriced } = priceRows(rows);
    expect(cost).toBeCloseTo(0.5 + 0.4 + 0.2, 6);
    expect(unpriced).toBe(true);
  });
});
