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
      const row = db
        .prepare(
          `SELECT
             COALESCE(SUM(input_tokens), 0)          AS input,
             COALESCE(SUM(output_tokens), 0)         AS output,
             COALESCE(SUM(cache_read_tokens), 0)     AS cacheRead,
             COALESCE(SUM(cache_creation_tokens), 0) AS cacheCreation,
             COALESCE(SUM(cost_usd), 0)              AS cost,
             COALESCE(SUM(num_turns), 0)             AS turns,
             COUNT(*)                                AS calls
           FROM token_usage
           WHERE ts >= ? AND ts < ?`,
        )
        .get(sinceIso, untilIso) as Omit<BotTotals, 'name'> | undefined;

      if (!row || row.calls === 0) continue;

      const acc = byGroup.get(s.group_name) ?? {
        name: s.group_name,
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheCreation: 0,
        cost: 0,
        turns: 0,
        calls: 0,
      };
      acc.input += row.input;
      acc.output += row.output;
      acc.cacheRead += row.cacheRead;
      acc.cacheCreation += row.cacheCreation;
      acc.cost += row.cost;
      acc.turns += row.turns;
      acc.calls += row.calls;
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
    lines.push(
      `*${t.name}* — $${t.cost.toFixed(2)} · ${humanTokens(t.input)} in / ${humanTokens(t.output)} out · ${t.turns} turns`,
    );
  }
  lines.push(
    '',
    `*Total* — $${cost.toFixed(2)} · ${turns} turns across ${totals.length} bot${totals.length === 1 ? '' : 's'}`,
  );
  if (opts.footnote !== false) lines.push('', '_$ assumes API pricing; approximate._');
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
    await adapter.deliver(channelType, platformId, threadId ?? null, 'text', report);
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
