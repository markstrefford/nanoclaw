/**
 * Container-runtime watch — degraded-mode tracking for a missing Docker daemon.
 *
 * The host does not need a container runtime to run. Channel adapters connect,
 * inbound messages route and queue in the session DBs, and `wakeContainer`
 * already returns false rather than throwing when a spawn fails. What was
 * missing was a way to notice, report, and recover without a restart:
 * startup used to exit on a failed `docker info`, launchd relaunched it, and
 * the circuit breaker backed the next attempt off to a 15-minute ceiling with
 * no upper bound — so a daemon that didn't come back after a reboot produced
 * an indefinite silent outage.
 *
 * This module polls the runtime (once at startup, then on each 60s host-sweep
 * tick), DMs an owner on the down and up transitions, and runs orphan cleanup
 * on recovery. Queued messages drain on their own once spawning works again.
 */
import { cleanupOrphans, isContainerRuntimeUp } from './container-runtime.js';
import { log } from './log.js';
import { notifyOwnerGlobal } from './modules/approvals/primitive.js';

/** Re-alert cadence while the runtime stays down, so a long outage isn't one scrolled-past message. */
const REALERT_INTERVAL_MS = 6 * 60 * 60 * 1000;

export type RuntimeTransition = 'unchanged' | 'went-down' | 'came-up' | 'still-down';

interface WatchState {
  /** null before the first probe. */
  lastUp: boolean | null;
  downSinceMs: number;
  lastAlertMs: number;
}

const state: WatchState = { lastUp: null, downSinceMs: 0, lastAlertMs: 0 };

/**
 * Classify a probe result against the previous one. Pure — the alerting and
 * recovery side effects hang off the returned transition.
 *
 * A first probe that finds the runtime down counts as 'went-down': the host
 * has just started into a degraded state and the owner should hear about it.
 */
export function decideRuntimeTransition(
  prev: boolean | null,
  now: boolean,
  nowMs: number,
  lastAlertMs: number,
): RuntimeTransition {
  if (now) return prev === false ? 'came-up' : 'unchanged';
  if (prev !== false) return 'went-down';
  return nowMs - lastAlertMs >= REALERT_INTERVAL_MS ? 'still-down' : 'unchanged';
}

/** Human-readable elapsed time for the recovery message. */
export function formatDuration(ms: number): string {
  const mins = Math.round(ms / 60_000);
  if (mins < 60) return `${mins} min`;
  const hours = ms / 3_600_000;
  if (hours < 24) return `${hours.toFixed(1)} h`;
  return `${(hours / 24).toFixed(1)} days`;
}

/** True when the last probe found no container runtime. */
export function isRuntimeDown(): boolean {
  return state.lastUp === false;
}

/**
 * Probe the runtime and act on any transition. Returns the current state.
 * Best-effort throughout — never throws, so a sweep tick can call it directly.
 */
export async function checkContainerRuntime(): Promise<boolean> {
  const up = isContainerRuntimeUp();
  const nowMs = Date.now();
  const transition = decideRuntimeTransition(state.lastUp, up, nowMs, state.lastAlertMs);

  if (transition === 'went-down') {
    state.downSinceMs = nowMs;
    state.lastAlertMs = nowMs;
    log.error('Container runtime is down — agents cannot run, messages will queue');
    void notifyOwnerGlobal(
      `🔌 Docker isn't running, so no agent can start. I'm still receiving your messages — ` +
        `they'll queue and run as soon as it's back. Start Docker and I'll pick up within a minute; ` +
        `no restart needed.`,
    );
  } else if (transition === 'still-down') {
    state.lastAlertMs = nowMs;
    const downFor = formatDuration(nowMs - state.downSinceMs);
    log.error('Container runtime still down', { downFor });
    void notifyOwnerGlobal(`🔌 Docker has now been down for ${downFor}. Messages are still queuing.`);
  } else if (transition === 'came-up') {
    const downFor = formatDuration(nowMs - state.downSinceMs);
    log.info('Container runtime recovered', { downFor });
    cleanupOrphans();
    void notifyOwnerGlobal(`✅ Docker is back after ${downFor}. Working through anything that queued.`);
  }

  state.lastUp = up;
  return up;
}

export function _resetRuntimeWatchForTesting(): void {
  state.lastUp = null;
  state.downSinceMs = 0;
  state.lastAlertMs = 0;
}
