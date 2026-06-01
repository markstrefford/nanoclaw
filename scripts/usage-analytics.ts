#!/usr/bin/env tsx
/**
 * Per-turn usage analytics export for the model-mix study.
 *
 * Walks every session's outbound.db `token_usage` table and emits one CSV row
 * per turn with the analytics columns (bot, source, model, tokens, cost, tool
 * calls, outcome, escalation, and — when LOG_TEXT_FOR_ANALYTICS was on — the
 * raw inbound text). Computed entirely host-side; no agent tokens spent.
 *
 *   pnpm exec tsx scripts/usage-analytics.ts                 # all turns, CSV to stdout
 *   pnpm exec tsx scripts/usage-analytics.ts 7               # last 7 days
 *   pnpm exec tsx scripts/usage-analytics.ts 7 > week.csv    # save for spreadsheet
 *
 * `cost_usd` is the SDK's running per-container total (not a per-turn delta);
 * use the daily cost report for spend. Token counts here ARE per-turn.
 */
import path from 'node:path';

import Database from 'better-sqlite3';

import { DATA_DIR } from '../src/config.js';
import { getDb, initDb } from '../src/db/connection.js';
import { outboundDbPath } from '../src/session-manager.js';

initDb(path.join(DATA_DIR, 'v2.db'));

const days = /^\d+$/.test(process.argv[2] ?? '') ? Number(process.argv[2]) : null;
const sinceIso = days ? new Date(Date.now() - days * 86400_000).toISOString() : null;

interface SessionRef {
  session_id: string;
  agent_group_id: string;
  group_name: string;
}

const sessions = getDb()
  .prepare(
    `SELECT s.id AS session_id, s.agent_group_id AS agent_group_id,
            COALESCE(g.name, s.agent_group_id) AS group_name
     FROM sessions s
     LEFT JOIN agent_groups g ON g.id = s.agent_group_id`,
  )
  .all() as SessionRef[];

interface Row {
  ts: string;
  model: string | null;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_creation_tokens: number;
  num_turns: number;
  trigger_kind: string | null;
  channel_type: string | null;
  message_count: number | null;
  tool_calls: number | null;
  outcome: string | null;
  escalated: number | null;
  message_text: string | null;
}

const COLUMNS = [
  'ts',
  'bot',
  'trigger_kind',
  'channel_type',
  'model',
  'input_tokens',
  'output_tokens',
  'cache_read_tokens',
  'cache_creation_tokens',
  'num_turns',
  'message_count',
  'tool_calls',
  'outcome',
  'escalated',
  'message_text',
] as const;

function csv(v: unknown): string {
  const s = v == null ? '' : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

const out: string[] = [COLUMNS.join(',')];

for (const s of sessions) {
  let db: Database.Database;
  try {
    db = new Database(outboundDbPath(s.agent_group_id, s.session_id), { readonly: true, fileMustExist: true });
  } catch {
    continue; // outbound.db not created yet
  }
  try {
    db.pragma('busy_timeout = 2000');
    // Older session DBs lack the analytics columns; SELECT * tolerates that and
    // missing fields read as undefined → blank cells.
    const where = sinceIso ? 'WHERE ts >= ?' : '';
    const rows = db
      .prepare(`SELECT * FROM token_usage ${where} ORDER BY ts`)
      .all(...(sinceIso ? [sinceIso] : [])) as Row[];
    for (const r of rows) {
      out.push(
        [
          r.ts,
          s.group_name,
          r.trigger_kind,
          r.channel_type,
          r.model,
          r.input_tokens,
          r.output_tokens,
          r.cache_read_tokens,
          r.cache_creation_tokens,
          r.num_turns,
          r.message_count,
          r.tool_calls,
          r.outcome,
          r.escalated,
          r.message_text,
        ]
          .map(csv)
          .join(','),
      );
    }
  } catch {
    // token_usage table absent on this (pre-feature) session DB — skip.
  } finally {
    db.close();
  }
}

process.stdout.write(out.join('\n') + '\n');
