/**
 * Telegram channel adapter (v2) — uses Chat SDK bridge, with a pairing
 * interceptor wrapped around onInbound to verify chat ownership before
 * registration. See telegram-pairing.ts for the why.
 */
import { createTelegramAdapter } from '@chat-adapter/telegram';

import { readEnvFile } from '../env.js';
import { log } from '../log.js';
import { createMessagingGroup, getMessagingGroupByPlatform, updateMessagingGroup } from '../db/messaging-groups.js';
import { grantRole, hasAnyOwner } from '../modules/permissions/db/user-roles.js';
import { upsertUser } from '../modules/permissions/db/users.js';
import { createChatSdkBridge, type ReplyContext } from './chat-sdk-bridge.js';
import { sanitizeTelegramLegacyMarkdown } from './telegram-markdown-sanitize.js';
import { registerChannelAdapter } from './channel-registry.js';
import type { ChannelAdapter, ChannelSetup, InboundMessage } from './adapter.js';
import { tryConsume } from './telegram-pairing.js';

/**
 * Retry a one-shot operation that can fail on transient network errors at
 * cold-start (DNS hiccups, brief upstream outages). Exponential backoff capped
 * at 5 attempts — if the network is truly down we surface it instead of
 * hanging the service indefinitely.
 */
async function withRetry<T>(fn: () => Promise<T>, label: string, maxAttempts = 5): Promise<T> {
  let lastErr: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (attempt === maxAttempts) break;
      const delay = Math.min(16000, 1000 * 2 ** (attempt - 1));
      log.warn('Telegram setup failed, retrying', { label, attempt, delayMs: delay, err });
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  throw lastErr;
}

/**
 * Health signal for the polling loop, populated by an instrumented logger that
 * watches the adapter's own "polling request failed" warnings. The adapter's
 * polling loop runs `while (pollingActive)` and never exits on network errors —
 * it logs, backs off (capped at 30s), and retries forever. So a wedged poller
 * looks identical to a healthy one from the outside (`isPolling === true`); the
 * only observable difference is its failure stream. We track the most recent
 * failure and the adapter's own `consecutiveFailures` count to tell "currently
 * failing" from "healthy and idle".
 */
interface PollHealth {
  /** ms epoch of the most recent "polling request failed" log (0 = none seen). */
  lastFailureAt: number;
  /** consecutiveFailures value from that log (the adapter resets it to 0 on any success). */
  lastFailureCount: number;
}

/** Structural mirror of chat-sdk's Logger — avoids importing the transitive `chat` package. */
interface PollLogger {
  child(prefix: string): PollLogger;
  debug(message: string, ...args: unknown[]): void;
  info(message: string, ...args: unknown[]): void;
  warn(message: string, ...args: unknown[]): void;
  error(message: string, ...args: unknown[]): void;
}

/**
 * A logger that preserves the adapter's existing console diagnostics while
 * tapping the polling-failure stream to populate `health`. Child loggers share
 * the same health closure so the signal is captured no matter how the adapter
 * routes the log.
 */
function makePollLogger(prefix: string, health: PollHealth): PollLogger {
  const emit = (level: 'debug' | 'info' | 'warn' | 'error', message: string, args: unknown[]): void => {
    if (level === 'warn' && message === 'Telegram polling request failed') {
      const payload = args[0] as { consecutiveFailures?: number } | undefined;
      health.lastFailureAt = Date.now();
      health.lastFailureCount =
        typeof payload?.consecutiveFailures === 'number' ? payload.consecutiveFailures : health.lastFailureCount + 1;
    }
    (console[level] ?? console.log)(`[${prefix}] ${message}`, ...args);
  };
  return {
    child: (sub: string) => makePollLogger(`${prefix}:${sub}`, health),
    debug: (m, ...a) => emit('debug', m, a),
    info: (m, ...a) => emit('info', m, a),
    warn: (m, ...a) => emit('warn', m, a),
    error: (m, ...a) => emit('error', m, a),
  };
}

/**
 * Independent reachability probe. A successful HTTP response (any status — even
 * 401) means we reached Telegram and the underlying socket pool is healthy; a
 * thrown fetch means the network itself is down. Shares Node's undici pool with
 * the poll loop, so a healthy probe implies a fresh `startPolling()` will also
 * get a working socket.
 */
async function telegramReachable(token: string): Promise<boolean> {
  try {
    await fetch(`https://api.telegram.org/bot${token}/getMe`, { signal: AbortSignal.timeout(8000) });
    return true;
  } catch {
    return false;
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function extractReplyContext(raw: Record<string, any>): ReplyContext | null {
  if (!raw.reply_to_message) return null;
  const reply = raw.reply_to_message;
  return {
    text: reply.text || reply.caption || '',
    sender: reply.from?.first_name || reply.from?.username || 'Unknown',
  };
}

/** Look up the bot username via Telegram getMe. Cached after first call. */
async function fetchBotUsername(token: string): Promise<string | null> {
  try {
    const res = await fetch(`https://api.telegram.org/bot${token}/getMe`);
    const json = (await res.json()) as { ok: boolean; result?: { username?: string } };
    return json.ok ? (json.result?.username ?? null) : null;
  } catch (err) {
    log.warn('Telegram getMe failed', { err });
    return null;
  }
}

function isGroupPlatformId(platformId: string): boolean {
  // platformId is "telegram:<chatId>". Negative chat IDs are groups/channels.
  const id = platformId.split(':').pop() ?? '';
  return id.startsWith('-');
}

interface InboundFields {
  text: string;
  authorUserId: string | null;
}

function readInboundFields(message: InboundMessage): InboundFields {
  if (message.kind !== 'chat-sdk' || !message.content || typeof message.content !== 'object') {
    return { text: '', authorUserId: null };
  }
  const c = message.content as { text?: string; author?: { userId?: string } };
  return { text: c.text ?? '', authorUserId: c.author?.userId ?? null };
}

/**
 * Build an onInbound interceptor that consumes pairing codes before they
 * reach the router. On match: records the chat + its paired user, promotes
 * the user to owner if the instance has no owner yet, and short-circuits.
 * On miss: forwards to the host.
 */
/**
 * Send a one-shot confirmation back to the paired chat. Best-effort — failures
 * are logged but never propagated, so a Telegram outage can't undo a successful
 * pairing or trigger the interceptor's fail-open path.
 */
async function sendPairingConfirmation(token: string, platformId: string): Promise<void> {
  const chatId = platformId.split(':').slice(1).join(':');
  if (!chatId) return;
  try {
    const res = await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        chat_id: chatId,
        text: 'Pairing success! Head back to the NanoClaw installer to finish setup.',
      }),
    });
    if (!res.ok) {
      log.warn('Telegram pairing confirmation non-OK', { status: res.status });
    }
  } catch (err) {
    log.warn('Telegram pairing confirmation failed', { err });
  }
}

function createPairingInterceptor(
  botUsernamePromise: Promise<string | null>,
  hostOnInbound: ChannelSetup['onInbound'],
  token: string,
  channelType: string,
): ChannelSetup['onInbound'] {
  return async (platformId, threadId, message) => {
    try {
      const botUsername = await botUsernamePromise;
      if (!botUsername) {
        hostOnInbound(platformId, threadId, message);
        return;
      }
      const { text, authorUserId } = readInboundFields(message);
      if (!text) {
        hostOnInbound(platformId, threadId, message);
        return;
      }
      const consumed = await tryConsume({
        text,
        botUsername,
        platformId,
        isGroup: isGroupPlatformId(platformId),
        adminUserId: authorUserId,
      });
      if (!consumed) {
        hostOnInbound(platformId, threadId, message);
        return;
      }
      // Pairing matched — record the chat and short-circuit so the
      // code-bearing message never reaches an agent. Privilege is now a
      // property of the paired user, not the chat: upsert the user, and if
      // this instance has no owner yet, promote them to owner.
      const existing = getMessagingGroupByPlatform(channelType, platformId);
      if (existing) {
        updateMessagingGroup(existing.id, {
          is_group: consumed.consumed!.isGroup ? 1 : 0,
        });
      } else {
        createMessagingGroup({
          id: `mg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          channel_type: channelType,
          platform_id: platformId,
          name: consumed.consumed!.name,
          is_group: consumed.consumed!.isGroup ? 1 : 0,
          unknown_sender_policy: 'strict',
          created_at: new Date().toISOString(),
        });
      }

      const pairedUserId = `telegram:${consumed.consumed!.adminUserId}`;
      upsertUser({
        id: pairedUserId,
        kind: 'telegram',
        display_name: null,
        created_at: new Date().toISOString(),
      });

      let promotedToOwner = false;
      if (!hasAnyOwner()) {
        grantRole({
          user_id: pairedUserId,
          role: 'owner',
          agent_group_id: null,
          granted_by: null,
          granted_at: new Date().toISOString(),
        });
        promotedToOwner = true;
      }

      log.info('Telegram pairing accepted — chat registered', {
        platformId,
        pairedUser: pairedUserId,
        promotedToOwner,
        intent: consumed.intent,
      });

      await sendPairingConfirmation(token, platformId);
    } catch (err) {
      log.error('Telegram pairing interceptor error', { err });
      // Fail open: pass through so a pairing bug doesn't break normal traffic.
      hostOnInbound(platformId, threadId, message);
    }
  };
}

function registerTelegramBot(channelType: string, envVar: string): void {
  registerChannelAdapter(channelType, {
    factory: () => {
      const env = readEnvFile([envVar]);
      if (!env[envVar]) return null;
      const token = env[envVar];
      const health: PollHealth = { lastFailureAt: 0, lastFailureCount: 0 };
      const telegramAdapter = createTelegramAdapter({
        botToken: token,
        mode: 'polling',
        logger: makePollLogger(`chat-sdk:${channelType}`, health),
      });
      const bridge = createChatSdkBridge({
        adapter: telegramAdapter,
        concurrency: 'concurrent',
        extractReplyContext,
        supportsThreads: false,
        transformOutboundText: sanitizeTelegramLegacyMarkdown,
        maxTextLength: 4000,
      });

      const botUsernamePromise = fetchBotUsername(token);

      // Poll watchdog. Two distinct failure modes, both leaving the bot silent
      // until the host is restarted by hand:
      //
      //   1. Loop exited — `isPolling === false`. startPolling() (idempotent,
      //      a no-op while active) revives it.
      //   2. Loop alive but wedged — `isPolling === true`, yet getUpdates keeps
      //      failing even after connectivity returns (observed after a long
      //      laptop-offline window: the poll loop retried for 15+ min past the
      //      network coming back). Here startPolling() does NOTHING because the
      //      loop is still "active", so the original re-arm-only watchdog could
      //      not recover it. The fix is a hard cycle — stopPolling() then
      //      startPolling() — to tear down the stale loop and build a fresh one.
      //
      // We only hard-cycle when the poller is *currently failing* (recent
      // failures, per the instrumented logger) AND Telegram is independently
      // reachable — otherwise we'd churn a healthy idle poller or thrash during
      // a real outage we can't fix anyway. Lives entirely on our side, so it
      // survives @chat-adapter/telegram updates.
      const WATCHDOG_INTERVAL_MS = 30_000;
      const WEDGE_RECENT_MS = 90_000; // backoff caps at 30s, so a failing loop logs at least this often
      const WEDGE_FAILURE_THRESHOLD = 3; // ignore one-off blips; require sustained failure
      let pollWatchdog: ReturnType<typeof setInterval> | null = null;

      const runWatchdogTick = async (): Promise<void> => {
        try {
          if (!telegramAdapter.isPolling) {
            log.warn('Telegram poller not active — restarting', { channelType });
            await telegramAdapter.startPolling();
            return;
          }
          const sinceFailureMs = Date.now() - health.lastFailureAt;
          const wedged =
            health.lastFailureAt > 0 &&
            sinceFailureMs < WEDGE_RECENT_MS &&
            health.lastFailureCount >= WEDGE_FAILURE_THRESHOLD;
          if (!wedged) return;
          if (!(await telegramReachable(token))) return; // real outage — nothing to fix
          log.warn('Telegram poller wedged while reachable — hard cycling', {
            channelType,
            consecutiveFailures: health.lastFailureCount,
            sinceFailureMs,
          });
          await telegramAdapter.stopPolling();
          await telegramAdapter.startPolling();
          health.lastFailureAt = 0;
          health.lastFailureCount = 0;
        } catch (err) {
          log.warn('Telegram poll watchdog tick failed', { channelType, err });
        }
      };

      const wrapped: ChannelAdapter = {
        ...bridge,
        channelType,
        resolveChannelName: async (platformId: string) => {
          const chatId = platformId.split(':').slice(1).join(':');
          if (!chatId) return null;
          try {
            const res = await fetch(`https://api.telegram.org/bot${token}/getChat`, {
              method: 'POST',
              headers: { 'content-type': 'application/json' },
              body: JSON.stringify({ chat_id: chatId }),
            });
            const data = (await res.json()) as { ok?: boolean; result?: { title?: string } };
            return data.ok ? (data.result?.title ?? null) : null;
          } catch {
            return null;
          }
        },
        async setup(hostConfig: ChannelSetup) {
          const intercepted: ChannelSetup = {
            ...hostConfig,
            onInbound: createPairingInterceptor(botUsernamePromise, hostConfig.onInbound, token, channelType),
          };
          await withRetry(() => bridge.setup(intercepted), 'bridge.setup');
          if (!pollWatchdog) {
            pollWatchdog = setInterval(() => {
              void runWatchdogTick();
            }, WATCHDOG_INTERVAL_MS);
            pollWatchdog.unref?.();
          }
        },
        async teardown() {
          if (pollWatchdog) {
            clearInterval(pollWatchdog);
            pollWatchdog = null;
          }
          await bridge.teardown();
        },
      };
      return wrapped;
    },
  });
}

registerTelegramBot('telegram', 'TELEGRAM_BOT_TOKEN');
registerTelegramBot('telegram_iris', 'TELEGRAM_BOT_TOKEN_IRIS');
