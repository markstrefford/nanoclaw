/**
 * NanoClaw Agent Runner v2
 *
 * Runs inside a container. All IO goes through the session DB.
 * No stdin, no stdout markers, no IPC files.
 *
 * Config is read from /workspace/agent/container.json (mounted RO).
 * Only TZ and OneCLI networking vars come from env.
 *
 * Mount structure:
 *   /workspace/
 *     inbound.db        ← host-owned session DB (container reads only)
 *     outbound.db       ← container-owned session DB
 *     .heartbeat        ← container touches for liveness detection
 *     outbox/           ← outbound files
 *     agent/            ← agent group folder (CLAUDE.md, container.json, working files)
 *       container.json  ← per-group config (RO nested mount)
 *     global/           ← shared global memory (RO)
 *   /app/src/           ← shared agent-runner source (RO)
 *   /app/skills/        ← shared skills (RO)
 *   /home/node/.claude/ ← Claude SDK state + skill symlinks (RW)
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { loadConfig } from './config.js';
import { healthGateMcpServers } from './mcp-health.js';
import { buildSystemPromptAddendum } from './destinations.js';
// Providers barrel — each enabled provider self-registers on import.
// Provider skills append imports to providers/index.ts.
import './providers/index.js';
import { createProvider, type ProviderName } from './providers/factory.js';
import { runPollLoop } from './poll-loop.js';

function log(msg: string): void {
  console.error(`[agent-runner] ${msg}`);
}

const CWD = '/workspace/agent';

async function main(): Promise<void> {
  const config = loadConfig();
  const providerName = config.provider.toLowerCase() as ProviderName;

  log(`Starting v2 agent-runner (provider: ${providerName})`);

  // Runtime-generated system-prompt addendum: agent identity (name) plus
  // the live destinations map. Everything else (capabilities, per-module
  // instructions, per-channel formatting) is loaded by Claude Code from
  // /workspace/agent/CLAUDE.md — the composed entry imports the shared
  // base (/app/CLAUDE.md) and each enabled module's fragment. Per-group
  // memory lives in /workspace/agent/CLAUDE.local.md (auto-loaded).
  let instructions = buildSystemPromptAddendum(config.assistantName || undefined);

  // Discover additional directories mounted at /workspace/extra/*
  const additionalDirectories: string[] = [];
  const extraBase = '/workspace/extra';
  if (fs.existsSync(extraBase)) {
    for (const entry of fs.readdirSync(extraBase)) {
      const fullPath = path.join(extraBase, entry);
      if (fs.statSync(fullPath).isDirectory()) {
        additionalDirectories.push(fullPath);
      }
    }
    if (additionalDirectories.length > 0) {
      log(`Additional directories: ${additionalDirectories.join(', ')}`);
    }
  }

  // MCP server path — bun runs TS directly; no tsc build step in-image.
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const mcpServerPath = path.join(__dirname, 'mcp-tools', 'index.ts');

  // Build MCP servers config: nanoclaw built-in + any from container.json
  const mcpServers: Record<string, { command: string; args: string[]; env: Record<string, string> }> = {
    nanoclaw: {
      command: 'bun',
      args: ['run', mcpServerPath],
      env: {},
    },
  };

  for (const [name, serverConfig] of Object.entries(config.mcpServers)) {
    mcpServers[name] = serverConfig;
    log(`Additional MCP server: ${name} (${serverConfig.command})`);
  }

  // Startup health-gate: probe each MCP server (initialize + tools/list, bounded)
  // before handing it to the SDK. A server that hangs on startup would otherwise
  // wedge the entire agent silently until the 30-min ceiling. Drop the unhealthy
  // ones, keep the rest, and tell the agent which are down so it can say so
  // instead of going quiet. The in-process `nanoclaw` server is trusted.
  const gate = await healthGateMcpServers(mcpServers, { skip: ['nanoclaw'], log });
  if (gate.dropped.length > 0) {
    const lines = gate.dropped.map((d) => `- ${d.name}: ${d.reason}`).join('\n');
    log(`Dropped ${gate.dropped.length} unhealthy MCP server(s) so the agent stays responsive:\n${lines}`);
    instructions +=
      `\n\n# Tools unavailable this session\n` +
      `These tool servers failed to start and are NOT available right now:\n${lines}\n` +
      `If the user asks for something that needs one of them, tell them that tool is currently down ` +
      `(it will be retried automatically next time) — do not pretend to use it or stall.`;
  }
  // MCP startup is non-blocking by default: the SDK builds the turn-1 tool list
  // from whichever servers have connected by then, and a slower one is simply
  // absent for the whole turn — the agent sees no error, just a missing tool,
  // and reports the integration as broken. (caldav-mcp logs into iCloud before
  // it connects its stdio transport, so it loses this race every time.)
  // `alwaysLoad` makes the SDK block on those servers instead, capped at 5s.
  const healthyMcpServers: typeof gate.healthy = { nanoclaw: mcpServers.nanoclaw, ...gate.healthy };
  if (gate.slow.length > 0) {
    const names = gate.slow.map((s) => `${s.name} (${s.handshakeMs}ms)`).join(', ');
    log(`Blocking startup on slow MCP server(s) so their tools exist on turn 1: ${names}`);
    for (const { name } of gate.slow) {
      healthyMcpServers[name] = { ...healthyMcpServers[name], alwaysLoad: true };
    }
    // Belt and braces: anything past the SDK's 5s cap still misses turn 1.
    instructions +=
      `\n\n# Tool servers that start slowly\n` +
      `These are healthy but take a few seconds to connect: ${gate.slow.map((s) => s.name).join(', ')}.\n` +
      `If you need one of their tools and cannot find it in your tool list, call WaitForMcpServers and retry. ` +
      `Never report one of these as missing, offline, or broken without having waited.`;
  }

  const provider = createProvider(providerName, {
    assistantName: config.assistantName || undefined,
    mcpServers: healthyMcpServers,
    env: { ...process.env },
    additionalDirectories: additionalDirectories.length > 0 ? additionalDirectories : undefined,
    model: config.model,
    effort: config.effort,
  });

  await runPollLoop({
    provider,
    providerName,
    cwd: CWD,
    systemContext: { instructions },
  });
}

main().catch((err) => {
  log(`Fatal error: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
