/**
 * OneCLI manual-approval handler — NEUTRALIZED for native credential proxy.
 *
 * This install uses the built-in credential proxy (`src/credential-proxy.ts`)
 * instead of the OneCLI gateway, so there is no gateway to long-poll for
 * credential approvals. The SDK-backed request handler has been removed; the
 * module retains its public API as no-ops so the approvals response handler
 * and module wiring compile unchanged, and keeps the startup sweep so any
 * `onecli_credential` rows left over from a prior OneCLI-backed run get cleared.
 *
 * To re-enable OneCLI: restore `@onecli-sh/sdk`, the `configureManualApproval`
 * handler, and `handleRequest`/`buildQuestion` (see git history).
 */
import { getPendingApprovalsByAction, deletePendingApproval, updatePendingApprovalStatus } from '../../db/sessions.js';
import type { ChannelDeliveryAdapter } from '../../delivery.js';
import { log } from '../../log.js';
import type { PendingApproval } from '../../types.js';

export const ONECLI_ACTION = 'onecli_credential';

type Decision = 'approve' | 'deny';

interface PendingState {
  resolve: (decision: Decision) => void;
  timer: NodeJS.Timeout;
}

const pending = new Map<string, PendingState>();
let started = false;
let adapterRef: ChannelDeliveryAdapter | null = null;

/** Called from the approvals response handler when a card button is clicked. */
export function resolveOneCLIApproval(approvalId: string, selectedOption: string): boolean {
  const state = pending.get(approvalId);
  if (!state) return false;
  pending.delete(approvalId);
  clearTimeout(state.timer);

  const decision: Decision = selectedOption === 'approve' ? 'approve' : 'deny';
  updatePendingApprovalStatus(approvalId, decision === 'approve' ? 'approved' : 'rejected');
  deletePendingApproval(approvalId);

  state.resolve(decision);
  log.info('OneCLI approval resolved', { approvalId, decision });
  return true;
}

export function startOneCLIApprovalHandler(deliveryAdapter: ChannelDeliveryAdapter): void {
  if (started) return;
  started = true;
  adapterRef = deliveryAdapter;

  // No OneCLI gateway in native-proxy mode — nothing to long-poll. Just clear
  // any rows left over from a previous OneCLI-backed process.
  sweepStaleApprovals().catch((err) => log.error('OneCLI approval sweep failed', { err }));
}

export function stopOneCLIApprovalHandler(): void {
  started = false;
  for (const state of pending.values()) {
    clearTimeout(state.timer);
  }
  pending.clear();
  adapterRef = null;
}

async function editCardExpired(row: PendingApproval, reason: string): Promise<void> {
  if (!adapterRef || !row.platform_message_id || !row.channel_type || !row.platform_id) return;
  try {
    await adapterRef.deliver(
      row.channel_type,
      row.platform_id,
      null,
      'chat-sdk',
      JSON.stringify({
        operation: 'edit',
        messageId: row.platform_message_id,
        text: `Expired (${reason})`,
      }),
    );
  } catch (err) {
    log.warn('Failed to edit expired OneCLI approval card', { approvalId: row.approval_id, err });
  }
}

async function sweepStaleApprovals(): Promise<void> {
  const rows = getPendingApprovalsByAction(ONECLI_ACTION);
  if (rows.length === 0) return;
  log.info('Sweeping stale OneCLI approvals from previous process', { count: rows.length });
  for (const row of rows) {
    await editCardExpired(row, 'host restarted');
    deletePendingApproval(row.approval_id);
  }
}
