#!/usr/bin/env tsx
/**
 * On-demand token/cost report. Reads the same per-session token_usage data the
 * daily push uses and prints it to stdout — no agent tokens spent.
 *
 *   pnpm exec tsx scripts/cost-report.ts            # yesterday + today so far
 *   pnpm exec tsx scripts/cost-report.ts today
 *   pnpm exec tsx scripts/cost-report.ts yesterday
 *   pnpm exec tsx scripts/cost-report.ts 7          # last 7 days
 */
import path from 'node:path';

import { DATA_DIR } from '../src/config.js';
import { initDb } from '../src/db/connection.js';
import {
  buildTodayReport,
  buildYesterdayReport,
  buildRangeReport,
  buildMonthToDateReport,
} from '../src/cost-report.js';

// Open the central DB (read-only use here) before any getDb() call.
initDb(path.join(DATA_DIR, 'v2.db'));

const arg = process.argv[2];

if (!arg) {
  console.log(buildYesterdayReport());
  console.log('\n');
  console.log(buildTodayReport());
  console.log('\n');
  console.log(buildMonthToDateReport());
} else if (arg === 'today') {
  console.log(buildTodayReport());
} else if (arg === 'yesterday') {
  console.log(buildYesterdayReport());
} else if (arg === 'month') {
  console.log(buildMonthToDateReport());
} else if (/^\d+$/.test(arg)) {
  console.log(buildRangeReport(Number(arg)));
} else {
  console.error(`Unknown arg "${arg}". Use: today | yesterday | month | <N days> | (none)`);
  process.exit(1);
}
