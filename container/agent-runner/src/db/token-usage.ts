import { getOutboundDb } from './connection.js';
import type { TurnUsage, TurnContext } from '../providers/types.js';

/**
 * Record one turn's token + cost usage into outbound.db. Called from the
 * poll-loop when the provider yields a result event carrying usage. The host
 * reads these rows across all sessions to build the daily cost report and the
 * model-mix analytics export.
 *
 * `ctx` carries the per-turn analytics fields the provider can't see (the
 * inbound batch + outcome). All analytics columns are nullable, so older
 * readers and pre-feature rows are unaffected.
 *
 * bun:sqlite requires the `$` prefix in both the SQL and the JS keys.
 */
export function recordTokenUsage(u: TurnUsage, ctx: TurnContext = {}): void {
  getOutboundDb()
    .prepare(
      `INSERT INTO token_usage
         (ts, model, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, cost_usd, num_turns,
          trigger_kind, channel_type, message_count, tool_calls, outcome, escalated, message_text)
       VALUES ($ts, $model, $input, $output, $cacheRead, $cacheCreation, $cost, $turns,
          $triggerKind, $channelType, $messageCount, $toolCalls, $outcome, $escalated, $messageText)`,
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
      $triggerKind: ctx.triggerKind ?? null,
      $channelType: ctx.channelType ?? null,
      $messageCount: ctx.messageCount ?? null,
      $toolCalls: u.toolCalls ?? null,
      $outcome: ctx.outcome ?? null,
      $escalated: u.escalated == null ? null : u.escalated ? 1 : 0,
      $messageText: ctx.messageText ?? null,
    });
}
