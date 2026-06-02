/**
 * Container config types and materialization.
 *
 * Source of truth is the `container_configs` table in the central DB.
 * This module provides:
 *   - Type definitions for the file shape (read by the container runner)
 *   - `materializeContainerJson()` — writes `groups/<folder>/container.json`
 *     from the DB at spawn time
 *   - `configFromDb()` — builds a `ContainerConfig` from a DB row + agent group
 */
import fs from 'fs';
import path from 'path';

import { GROUPS_DIR, LOG_TEXT_FOR_ANALYTICS, USE_CUSTOM_GMAIL } from './config.js';
import { getContainerConfig } from './db/container-configs.js';
import { getAgentGroup } from './db/agent-groups.js';
import type { AgentGroup, ContainerConfigRow } from './types.js';

export interface McpServerConfig {
  command: string;
  args?: string[];
  env?: Record<string, string>;
  instructions?: string;
}

export interface AdditionalMountConfig {
  hostPath: string;
  containerPath: string;
  readonly?: boolean;
}

/** Shape of the materialized `container.json` file read by the container runner. */
export interface ContainerConfig {
  mcpServers: Record<string, McpServerConfig>;
  packages: { apt: string[]; npm: string[] };
  imageTag?: string;
  additionalMounts: AdditionalMountConfig[];
  skills: string[] | 'all';
  provider?: string;
  groupName?: string;
  assistantName?: string;
  agentGroupId?: string;
  maxMessagesPerPrompt?: number;
  model?: string;
  effort?: string;
  /** Global model-mix study flag — stamped from LOG_TEXT_FOR_ANALYTICS. */
  logTextForAnalytics?: boolean;
}

// Proxy env vars to blank when a server opts out of the gateway.
const PROXY_VARS = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'];

/**
 * Gateway bypass for direct-to-provider MCP servers.
 *
 * The OneCLI gateway (HTTPS_PROXY injected container-wide) substitutes its
 * connected Google account on Gmail API calls — which overrides a group's own
 * personal credentials and serves the wrong mailbox. A server signals "reach
 * the provider directly with my own credentials" by declaring NO_PROXY in its
 * env. NO_PROXY alone is NOT honoured by the Node HTTP clients in use
 * (undici / gaxios), so when USE_CUSTOM_GMAIL is enabled we blank the proxy
 * vars for those servers — the only setting that actually bypasses the gateway.
 *
 * Servers without NO_PROXY (e.g. North's gateway-routed business Gmail) are
 * untouched. Gating on USE_CUSTOM_GMAIL makes this globally switchable.
 */
function applyGatewayBypass(mcpServers: Record<string, McpServerConfig>): void {
  if (!USE_CUSTOM_GMAIL) return;
  for (const server of Object.values(mcpServers)) {
    const env = server.env;
    if (env && (env.NO_PROXY || env.no_proxy)) {
      for (const v of PROXY_VARS) env[v] = '';
    }
  }
}

/** Build a `ContainerConfig` from a DB row + agent group identity. */
export function configFromDb(row: ContainerConfigRow, group: AgentGroup): ContainerConfig {
  const mcpServers = JSON.parse(row.mcp_servers) as Record<string, McpServerConfig>;
  applyGatewayBypass(mcpServers);
  return {
    mcpServers,
    packages: {
      apt: JSON.parse(row.packages_apt) as string[],
      npm: JSON.parse(row.packages_npm) as string[],
    },
    imageTag: row.image_tag ?? undefined,
    additionalMounts: JSON.parse(row.additional_mounts) as AdditionalMountConfig[],
    skills: JSON.parse(row.skills) as string[] | 'all',
    provider: row.provider ?? undefined,
    groupName: group.name,
    assistantName: row.assistant_name ?? group.name,
    agentGroupId: group.id,
    maxMessagesPerPrompt: row.max_messages_per_prompt ?? undefined,
    model: row.model ?? undefined,
    effort: row.effort ?? undefined,
    logTextForAnalytics: LOG_TEXT_FOR_ANALYTICS || undefined,
  };
}

/**
 * Materialize `container.json` from the DB. Called at spawn time so the
 * container always sees fresh config. Returns the `ContainerConfig` for
 * use by the caller (buildMounts, buildContainerArgs, etc.).
 */
export function materializeContainerJson(agentGroupId: string): ContainerConfig {
  const group = getAgentGroup(agentGroupId);
  if (!group) throw new Error(`Agent group not found: ${agentGroupId}`);

  const row = getContainerConfig(agentGroupId);
  if (!row) throw new Error(`Container config not found for agent group: ${agentGroupId}`);

  const config = configFromDb(row, group);

  const p = path.join(GROUPS_DIR, group.folder, 'container.json');
  const dir = path.dirname(p);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(p, JSON.stringify(config, null, 2) + '\n');

  return config;
}
