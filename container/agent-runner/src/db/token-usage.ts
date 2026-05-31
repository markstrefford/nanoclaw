import { getOutboundDb } from './connection.js';
import type { TurnUsage } from '../providers/types.js';

/**
 * Record one turn's token + cost usage into outbound.db. Called from the
 * poll-loop when the provider yields a result event carrying usage. The host
 * reads these rows across all sessions to build the daily cost report.
 *
 * bun:sqlite requires the `$` prefix in both the SQL and the JS keys.
 */
export function recordTokenUsage(u: TurnUsage): void {
  getOutboundDb()
    .prepare(
      `INSERT INTO token_usage
         (ts, model, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, cost_usd, num_turns)
       VALUES ($ts, $model, $input, $output, $cacheRead, $cacheCreation, $cost, $turns)`,
    )
    .run({
      $ts: new Date().toISOString(),
      $model: u.model ?? null,
      $input: u.inputTokens,
      $output: u.outputTokens,
      $cacheRead: u.cacheReadTokens,
      $cacheCreation: u.cacheCreationTokens,
      $cost: u.costUsd,
      $turns: u.numTurns,
    });
}
