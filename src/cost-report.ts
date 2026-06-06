/**
 * Daily token + cost report.
 *
 * The container records per-turn usage into each session's outbound.db
 * (`token_usage` table). This module aggregates those rows across all sessions,
 * grouped by agent group ("bot"), and pushes a plain-text summary to a
 * configured chat target each morning — computed entirely host-side so the
 * report itself costs no agent tokens.
 *
 * Config (env):
 *   COST_REPORT_TARGET   "<channelType>|<platformId>[|<threadId>]" — where to
 *                        send. If unset, the schedule is disabled and the
 *                        report is only available via scripts/cost-report.ts.
 *   COST_REPORT_HOUR     local hour (0–23) to send the daily report. Default 7.
 */
import Database from 'better-sqlite3';

import { COST_REPORT_HOUR, COST_REPORT_TARGET } from './config.js';
import { getDb } from './db/connection.js';
import { getModelRates, priceTurn } from './model-pricing.js';
import { outboundDbPath } from './session-manager.js';
import { getDeliveryAdapter } from './delivery.js';
import { log } from './log.js';

interface BotTotals {
  name: string;
  input: number;
  output: number;
  cacheRead: number;
  cacheCreation: number;
  cost: number;
  turns: number;
  calls: number;
  /** Distinct model ids seen in this group's window (for the per-bot label). */
  models: Set<string>;
  /** True if any row's model had no rate entry → cost fell back to SDK pricing. */
  unpriced: boolean;
  /** True if any priced row used a documented-assumption rate (e.g. Kimi cache-write). */
  assumed: boolean;
}

interface SessionRef {
  session_id: string;
  agent_group_id: string;
  group_name: string;
}

/** All sessions with their owning agent group's display name. */
function listSessions(): SessionRef[] {
  return getDb()
    .prepare(
      `SELECT s.id AS session_id, s.agent_group_id AS agent_group_id,
              COALESCE(g.name, s.agent_group_id) AS group_name
       FROM sessions s
       LEFT JOIN agent_groups g ON g.id = s.agent_group_id`,
    )
    .all() as SessionRef[];
}

/** One `token_usage` row as read for pricing. */
export interface UsageRow {
  input: number;
  output: number;
  cacheRead: number;
  cacheCreation: number;
  /** SDK `total_cost_usd` (Anthropic-priced cumulative) — used only as the unknown-model fallback. */
  cost: number;
  turns: number;
  model: string | null;
}

export interface PricedWindow {
  cost: number;
  unpriced: boolean;
  assumed: boolean;
  models: Set<string>;
}

/**
 * Price an ordered window of rows. Each row is costed by its own model from the
 * rate table; rows whose model has no entry fall back to the cumulative
 * `cost_usd` series (add the increment while it climbs; treat a drop as a new
 * container run). `prevCost` tracks across ALL rows so the fallback series stays
 * continuous even when priced (Kimi/Claude) and unpriced rows interleave — which
 * is exactly what a mid-window provider flip produces.
 */
export function priceRows(rows: UsageRow[]): PricedWindow {
  let cost = 0;
  let prevCost: number | null = null;
  let unpriced = false;
  let assumed = false;
  const models = new Set<string>();
  for (const r of rows) {
    const rates = getModelRates(r.model);
    if (rates) {
      cost += priceTurn(r.model, {
        input: r.input,
        output: r.output,
        cacheRead: r.cacheRead,
        cacheCreation: r.cacheCreation,
      })!;
      if (!rates.exact) assumed = true;
    } else {
      cost += prevCost === null || r.cost < prevCost ? r.cost : r.cost - prevCost;
      unpriced = true;
    }
    prevCost = r.cost;
    if (r.model) models.add(r.model);
  }
  return { cost, unpriced, assumed, models };
}

/**
 * Sum token_usage rows in [sinceIso, untilIso) across every session, grouped by
 * agent group. Missing DBs / tables (older sessions) are skipped silently.
 */
export function collectUsage(sinceIso: string, untilIso: string): BotTotals[] {
  const byGroup = new Map<string, BotTotals>();

  for (const s of listSessions()) {
    let db: Database.Database;
    try {
      db = new Database(outboundDbPath(s.agent_group_id, s.session_id), { readonly: true, fileMustExist: true });
    } catch {
      continue; // outbound.db not created yet
    }
    try {
      db.pragma('busy_timeout = 2000');
      // Token/turn fields are per-result increments → SUM is correct.
      // Cost is priced per-row from the model rate table (model-pricing.ts), NOT
      // from cost_usd: the SDK reports `total_cost_usd` as an Anthropic-priced
      // running total, which is wrong for a non-Claude model reached through an
      // Anthropic-compatible endpoint (Kimi via Moonshot). Rows whose model has
      // no rate entry fall back to the cumulative cost_usd series (the old
      // behaviour) and flag the group "unpriced".
      const rows = db
        .prepare(
          `SELECT
             COALESCE(input_tokens, 0)          AS input,
             COALESCE(output_tokens, 0)         AS output,
             COALESCE(cache_read_tokens, 0)     AS cacheRead,
             COALESCE(cache_creation_tokens, 0) AS cacheCreation,
             COALESCE(cost_usd, 0)              AS cost,
             COALESCE(num_turns, 0)             AS turns,
             model                              AS model
           FROM token_usage
           WHERE ts >= ? AND ts < ?
           ORDER BY ts`,
        )
        .all(sinceIso, untilIso) as Array<{
        input: number;
        output: number;
        cacheRead: number;
        cacheCreation: number;
        cost: number;
        turns: number;
        model: string | null;
      }>;

      if (rows.length === 0) continue;

      const { cost, unpriced, assumed, models } = priceRows(rows);

      const acc = byGroup.get(s.group_name) ?? {
        name: s.group_name,
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheCreation: 0,
        cost: 0,
        turns: 0,
        calls: 0,
        models: new Set<string>(),
        unpriced: false,
        assumed: false,
      };
      for (const r of rows) {
        acc.input += r.input;
        acc.output += r.output;
        acc.cacheRead += r.cacheRead;
        acc.cacheCreation += r.cacheCreation;
        acc.turns += r.turns;
      }
      acc.cost += cost;
      acc.calls += rows.length;
      acc.unpriced = acc.unpriced || unpriced;
      acc.assumed = acc.assumed || assumed;
      for (const m of models) acc.models.add(m);
      byGroup.set(s.group_name, acc);
    } catch {
      // token_usage table absent on this (pre-feature) session DB — skip.
    } finally {
      db.close();
    }
  }

  return [...byGroup.values()].sort((a, b) => b.cost - a.cost);
}

function humanTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`;
  return String(n);
}

/** Short model label for a bot line — e.g. "kimi-k2.6", or "" when unknown. */
function labelModels(models: Set<string>): string {
  const ids = [...models].filter(Boolean);
  if (ids.length === 0) return '';
  if (ids.length === 1) return ids[0];
  return `${ids.length} models`;
}

/**
 * Footnote that is honest about how the cost was derived:
 *  - all rows priced from the rate table, every rate published → no footnote.
 *  - some priced via a documented-assumption rate (Kimi cache-write) → flag it.
 *  - some rows had no rate entry → fell back to model-reported (Anthropic) cost.
 */
function pricingFootnote(totals: BotTotals[]): string | null {
  const unpriced = totals.filter((t) => t.unpriced).map((t) => t.name);
  const assumed = totals.some((t) => t.assumed);
  if (unpriced.length === 0 && !assumed) return null;
  const parts: string[] = [];
  if (assumed) parts.push('Kimi cache-write priced at input rate (Moonshot publishes no cache-write rate)');
  if (unpriced.length > 0) parts.push(`${unpriced.join(', ')} uses model-reported pricing (approximate)`);
  return `_${parts.join('; ')}._`;
}

/** Format a collected window into a Telegram-friendly plain-text report. */
export function formatReport(
  label: string,
  totals: BotTotals[],
  opts: { footnote?: boolean; icon?: string } = {},
): string {
  const icon = opts.icon ?? '📊';
  if (totals.length === 0) {
    return `${icon} NanoClaw usage — ${label}\n\nNo agent activity recorded.`;
  }

  const lines = [`${icon} NanoClaw usage — ${label}`, ''];
  let cost = 0;
  let turns = 0;
  for (const t of totals) {
    cost += t.cost;
    turns += t.turns;
    const model = labelModels(t.models);
    lines.push(
      `*${t.name}*${model ? ` (${model})` : ''} — $${t.cost.toFixed(2)} · ${humanTokens(t.input)} in / ${humanTokens(
        t.output,
      )} out · ${t.turns} turns`,
    );
  }
  lines.push(
    '',
    `*Total* — $${cost.toFixed(2)} · ${turns} turns across ${totals.length} bot${totals.length === 1 ? '' : 's'}`,
  );
  if (opts.footnote !== false) {
    const note = pricingFootnote(totals);
    if (note) lines.push('', note);
  }
  return lines.join('\n');
}

/** Local-midnight boundary for `daysAgo` days back (0 = today's midnight). */
function midnight(daysAgo: number): Date {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() - daysAgo);
  return d;
}

/** Local midnight on the 1st of the current calendar month. */
function monthStart(): Date {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  d.setDate(1);
  return d;
}

/** Yesterday's full-day report (the daily-push window). */
export function buildYesterdayReport(): string {
  const since = midnight(1);
  const until = midnight(0);
  return formatReport('yesterday', collectUsage(since.toISOString(), until.toISOString()));
}

/** Calendar-month-to-date report. */
export function buildMonthToDateReport(): string {
  const since = monthStart();
  const until = new Date();
  const monthName = since.toLocaleString('en-GB', { month: 'long' });
  return formatReport(`${monthName} so far`, collectUsage(since.toISOString(), until.toISOString()), { icon: '📅' });
}

/** The morning push: yesterday's breakdown + calendar-month-to-date. */
export function buildDailyPush(): string {
  const since = midnight(1);
  const until = midnight(0);
  const yesterday = formatReport('yesterday', collectUsage(since.toISOString(), until.toISOString()), {
    footnote: false,
  });
  return `${yesterday}\n\n${buildMonthToDateReport()}`;
}

/** Today-so-far report (handy for the on-demand script). */
export function buildTodayReport(): string {
  const since = midnight(0);
  const until = new Date();
  return formatReport('today so far', collectUsage(since.toISOString(), until.toISOString()));
}

/** Last N days, inclusive of today so far. */
export function buildRangeReport(days: number): string {
  const since = midnight(days - 1);
  const until = new Date();
  return formatReport(`last ${days} days`, collectUsage(since.toISOString(), until.toISOString()));
}

/** Send yesterday's report to COST_REPORT_TARGET via the delivery adapter. */
export async function sendDailyCostReport(): Promise<void> {
  const report = buildDailyPush();
  const target = COST_REPORT_TARGET;
  if (!target) {
    log.info(`Cost report (COST_REPORT_TARGET unset — not sending)\n${report}`);
    return;
  }
  const [channelType, platformId, threadId] = target.split('|');
  if (!channelType || !platformId) {
    log.warn('Cost report: COST_REPORT_TARGET malformed (want "<channelType>|<platformId>[|<threadId>]")');
    return;
  }
  const adapter = getDeliveryAdapter();
  if (!adapter) {
    log.warn('Cost report: no delivery adapter set — skipping');
    return;
  }
  try {
    // The delivery adapter JSON.parses `content` into the message object the
    // channel bridge expects (`{ text }`). Passing raw text made it throw
    // "not valid JSON" on the report's leading emoji, so the report never sent.
    await adapter.deliver(channelType, platformId, threadId ?? null, 'chat-sdk', JSON.stringify({ text: report }));
    log.info('Cost report sent');
  } catch (err) {
    log.error('Cost report delivery failed', { err });
  }
}

let scheduled = false;

/** Start the once-daily report schedule. No-op if COST_REPORT_TARGET is unset. */
export function startCostReportSchedule(): void {
  if (scheduled) return;
  if (!COST_REPORT_TARGET) {
    log.info('Cost report schedule disabled (COST_REPORT_TARGET unset)');
    return;
  }
  scheduled = true;
  scheduleNext();
}

function scheduleNext(): void {
  const hour = Math.min(23, Math.max(0, Number(COST_REPORT_HOUR) || 7));
  const now = new Date();
  const next = new Date(now);
  next.setHours(hour, 0, 0, 0);
  if (next <= now) next.setDate(next.getDate() + 1);
  const ms = next.getTime() - now.getTime();
  log.info(`Cost report scheduled for ${next.toISOString()}`);
  setTimeout(() => {
    void sendDailyCostReport().finally(scheduleNext);
  }, ms);
}
