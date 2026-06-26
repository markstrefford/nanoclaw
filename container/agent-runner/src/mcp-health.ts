/**
 * MCP startup health-gate.
 *
 * The Claude Agent SDK initializes every configured MCP server before the first
 * turn and waits on each one's stdio handshake. If a single server hangs on
 * startup (a cold-starting backend, a credential proxy that stalls, a wedged
 * dependency) the WHOLE agent hangs — no heartbeat, no reply — until the host
 * kills the container at the 30-min ceiling. One flaky tool takes the whole bot
 * down, silently. (Seen with VOSS CRM and a gateway-routed Gmail server.)
 *
 * This gate probes each server the same way the SDK will — spawn it, run the
 * `initialize` + `tools/list` handshake — but bounded by a timeout. Servers that
 * answer in time are passed to the SDK; servers that hang or crash are DROPPED
 * (the agent runs with its remaining tools) and reported so the agent can tell
 * the user "that tool is down" instead of going silent.
 *
 * The probe spawns a throwaway instance and kills it; the SDK spawns its own.
 * stdio MCP servers are independent processes, so a probe-then-real double spawn
 * is harmless.
 */
import { spawn } from 'node:child_process';

export interface McpServerSpec {
  command: string;
  args: string[];
  env: Record<string, string>;
}

export interface HealthGateResult {
  healthy: Record<string, McpServerSpec>;
  dropped: Array<{ name: string; reason: string }>;
}

const DEFAULT_TIMEOUT_MS = Number(process.env.MCP_HEALTH_TIMEOUT_MS) || 20_000;

const INIT_REQUEST =
  JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'initialize',
    params: {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'mcp-health-gate', version: '1' },
    },
  }) + '\n';

const INITIALIZED_NOTIFICATION = JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }) + '\n';
const TOOLS_LIST_REQUEST = JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/list' }) + '\n';

/**
 * Probe a single MCP server: spawn it, complete initialize + tools/list within
 * the timeout. Resolves {ok:false, reason} on hang/crash/spawn-failure — never
 * rejects.
 */
function probe(spec: McpServerSpec, timeoutMs: number): Promise<{ ok: boolean; reason: string }> {
  return new Promise((resolve) => {
    let settled = false;
    let initialized = false;
    let buf = '';

    let child: ReturnType<typeof spawn>;
    try {
      child = spawn(spec.command, spec.args, {
        env: { ...process.env, ...spec.env },
        stdio: ['pipe', 'pipe', 'pipe'],
      });
    } catch (err) {
      resolve({ ok: false, reason: `spawn failed: ${err instanceof Error ? err.message : String(err)}` });
      return;
    }

    const finish = (ok: boolean, reason: string) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try {
        child.kill('SIGKILL');
      } catch {
        /* already gone */
      }
      resolve({ ok, reason });
    };

    const timer = setTimeout(
      () => finish(false, `no ${initialized ? 'tools/list' : 'initialize'} response within ${timeoutMs}ms (hung on startup)`),
      timeoutMs,
    );

    child.stdout?.on('data', (d: Buffer) => {
      buf += d.toString();
      let nl: number;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        let msg: { id?: number; result?: unknown; error?: unknown };
        try {
          msg = JSON.parse(line);
        } catch {
          continue; // non-JSON log line on stdout — ignore
        }
        if (msg.id === 1 && (msg.result || msg.error)) {
          if (msg.error) {
            finish(false, `initialize errored: ${JSON.stringify(msg.error).slice(0, 140)}`);
            return;
          }
          // Initialized OK — proceed to tools/list, matching the SDK's flow.
          initialized = true;
          try {
            child.stdin?.write(INITIALIZED_NOTIFICATION);
            child.stdin?.write(TOOLS_LIST_REQUEST);
          } catch {
            finish(false, 'stdin closed after initialize');
            return;
          }
        } else if (msg.id === 2) {
          finish(!msg.error, msg.error ? `tools/list errored: ${JSON.stringify(msg.error).slice(0, 140)}` : 'ok');
          return;
        }
      }
    });

    child.on('error', (err: Error) => finish(false, `process error: ${err.message}`));
    child.on('exit', (code: number | null) => finish(false, `exited before responding (code ${code ?? 'null'})`));

    try {
      child.stdin?.write(INIT_REQUEST);
    } catch {
      finish(false, 'stdin write failed at startup');
    }
  });
}

/**
 * Probe all servers concurrently; return the healthy subset plus a list of
 * dropped servers with reasons. `skip` names are trusted and passed through
 * without probing (e.g. the in-process nanoclaw tool server).
 */
export async function healthGateMcpServers(
  servers: Record<string, McpServerSpec>,
  opts: { timeoutMs?: number; skip?: string[]; log?: (m: string) => void } = {},
): Promise<HealthGateResult> {
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const skip = new Set(opts.skip ?? []);
  const log = opts.log ?? (() => {});

  const healthy: Record<string, McpServerSpec> = {};
  const dropped: Array<{ name: string; reason: string }> = [];

  const results = await Promise.all(
    Object.entries(servers).map(async ([name, spec]) => {
      if (skip.has(name)) return { name, ok: true, reason: 'trusted' };
      const r = await probe(spec, timeoutMs);
      return { name, ...r };
    }),
  );

  for (const r of results) {
    if (r.ok) {
      healthy[r.name] = servers[r.name];
      if (r.reason !== 'ok' && r.reason !== 'trusted') log(`MCP server "${r.name}" healthy (${r.reason})`);
    } else {
      dropped.push({ name: r.name, reason: r.reason });
      log(`MCP server "${r.name}" DROPPED at startup — ${r.reason}`);
    }
  }

  return { healthy, dropped };
}
