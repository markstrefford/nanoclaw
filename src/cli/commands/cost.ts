/**
 * `ncl cost [today|yesterday|month|<N>]` — report API token spend on demand.
 *
 * Resourceless (like `help`), so it bypasses the group-scope resource
 * allowlist and works for any agent regardless of cli_scope — cost figures
 * are non-sensitive aggregate spend. Read-only (`access: 'open'`), computed
 * host-side from each session's token_usage, so producing the report itself
 * spends no agent tokens. Reuses the same builders as the daily push.
 *
 *   ncl cost              → today so far
 *   ncl cost today        → today so far
 *   ncl cost yesterday    → full prior day
 *   ncl cost month        → month-to-date
 *   ncl cost 7            → trailing 7 days
 */
import { buildMonthToDateReport, buildRangeReport, buildTodayReport, buildYesterdayReport } from '../../cost-report.js';
import { register } from '../registry.js';

register<{ period: string }, string>({
  name: 'cost',
  description: 'Report API token spend (today | yesterday | month | <N days>). Default: today.',
  access: 'open',
  parseArgs: (raw) => {
    // `ncl cost today` arrives as command "cost-today" → dispatch trims to
    // "cost" with id="today"; bare `ncl cost` has no period → today.
    const period = String(raw.id ?? raw.period ?? raw.range ?? 'today')
      .toLowerCase()
      .trim();
    return { period };
  },
  handler: async ({ period }) => {
    if (period === 'yesterday') return buildYesterdayReport();
    if (period === 'month' || period === 'mtd') return buildMonthToDateReport();
    const n = Number(period);
    if (Number.isInteger(n) && n > 0) return buildRangeReport(n);
    return buildTodayReport();
  },
});
