/**
 * Live progress-status line for a turn.
 *
 * The problem: during a long turn the user sees a "typing…" indicator but no
 * content for tens of seconds, then the full answer lands at once. The agent
 * itself can't reliably narrate this — the model has no sense of its own
 * wall-clock latency. So the runner drives a deterministic status line:
 *
 *   1. After APPEAR_AFTER_MS of silence in a turn, post one status message
 *      ("⏳ Working on it… (7s)") to the current conversation.
 *   2. As tool activity arrives (fed via noteProgress), edit that same message
 *      in place to show what the agent is doing ("⏳ Searching the vault…").
 *   3. When the turn produces its result, delete the status message — the real
 *      answer takes its place, leaving no clutter.
 *
 * All three operations are ordinary outbound rows (kind 'progress'); the host
 * delivers, edits, and deletes them through the normal channel adapter. Editing
 * and deleting need the delivered message's platform id, which the host writes
 * to the delivered table ~1s after the initial post — we resolve it lazily via
 * getMessageIdBySeq and simply wait if it isn't ready yet.
 *
 * Disabled when the session has no user-facing routing (agent-to-agent or
 * internal sessions) — there's no human waiting on a typing indicator there.
 */
import { writeMessageOut, getMessageIdBySeq } from './db/messages-out.js';
import { getSessionRouting } from './db/session-routing.js';

const APPEAR_AFTER_MS = 6000; // silent stretch before the status line appears
const EDIT_THROTTLE_MS = 5000; // minimum gap between in-place edits

interface Routing {
  channel_type: string;
  platform_id: string;
  thread_id: string | null;
}

function genId(): string {
  return `prog-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Resolve a posted status row's delivered platform id, or null if the host
 * hasn't delivered it yet. Before delivery getMessageIdBySeq falls back to the
 * internal "prog-"/"msg-" id, which the platform won't accept for edit/delete.
 */
function resolveDelivered(seq: number): string | null {
  const pid = getMessageIdBySeq(seq);
  if (pid && !pid.startsWith('prog-') && !pid.startsWith('msg-')) return pid;
  return null;
}

function writeProgress(routing: Routing, content: Record<string, unknown>): number {
  return writeMessageOut({
    id: genId(),
    kind: 'progress',
    platform_id: routing.platform_id,
    channel_type: routing.channel_type,
    thread_id: routing.thread_id,
    content: JSON.stringify(content),
  });
}

/**
 * Delete a posted status line. Self-contained (takes a snapshot of routing +
 * seq) so it can be fire-and-forget without racing a follow-up turn that
 * resets the TurnProgress instance. Waits briefly for delivery if the status
 * was posted only moments before the turn ended.
 */
async function deleteStatus(routing: Routing, seq: number): Promise<void> {
  let pid = resolveDelivered(seq);
  for (let i = 0; i < 20 && !pid; i++) {
    await sleep(100);
    pid = resolveDelivered(seq);
  }
  if (!pid) return; // couldn't resolve delivery — rare; leave it rather than guess
  writeProgress(routing, { operation: 'delete', messageId: pid });
}

export class TurnProgress {
  private routing: Routing | null = null;
  private turnStart = 0;
  private statusSeq: number | null = null;
  private platformId: string | null = null;
  private currentActivity: string | null = null;
  private lastText = '';
  private lastEditAt = 0;
  private active = false;

  /** Begin tracking a fresh turn. Resolves the destination once, up front. */
  start(): void {
    this.turnStart = Date.now();
    this.statusSeq = null;
    this.platformId = null;
    this.currentActivity = null;
    this.lastText = '';
    this.lastEditAt = 0;

    const s = getSessionRouting();
    if (s.channel_type && s.platform_id && s.channel_type !== 'agent') {
      this.routing = { channel_type: s.channel_type, platform_id: s.platform_id, thread_id: s.thread_id };
      this.active = true;
    } else {
      this.routing = null;
      this.active = false;
    }
  }

  /** Record the latest tool activity label (drives the status text). */
  noteProgress(message: string): void {
    if (message) this.currentActivity = message;
  }

  private statusText(): string {
    const elapsed = Math.round((Date.now() - this.turnStart) / 1000);
    const label = this.currentActivity ?? 'Working on it';
    return `⏳ ${label}… (${elapsed}s)`;
  }

  /** Called periodically while the turn runs. Posts/edits the status line. */
  tick(): void {
    if (!this.active || !this.routing) return;

    if (this.statusSeq === null) {
      if (Date.now() - this.turnStart < APPEAR_AFTER_MS) return;
      const text = this.statusText();
      this.statusSeq = writeProgress(this.routing, { text });
      this.lastText = text;
      this.lastEditAt = Date.now();
      return;
    }

    // Posted — edit in place once the host has delivered it (so we have a
    // platform id) and the throttle window has elapsed.
    if (!this.platformId) this.platformId = resolveDelivered(this.statusSeq);
    if (!this.platformId) return;
    if (Date.now() - this.lastEditAt < EDIT_THROTTLE_MS) return;
    const text = this.statusText();
    if (text === this.lastText) return;
    writeProgress(this.routing, { operation: 'edit', messageId: this.platformId, text });
    this.lastText = text;
    this.lastEditAt = Date.now();
  }

  /**
   * Turn finished — delete the status line so the real answer stands alone.
   * Snapshots its state and hands off, so it's safe to fire-and-forget even if
   * a follow-up turn calls start() immediately after. Idempotent.
   */
  finish(): Promise<void> {
    const seq = this.statusSeq;
    const routing = this.routing;
    this.active = false;
    this.statusSeq = null;
    this.platformId = null;
    if (seq === null || !routing) return Promise.resolve();
    return deleteStatus(routing, seq);
  }
}
