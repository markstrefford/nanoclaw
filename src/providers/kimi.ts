/**
 * Kimi (Moonshot) provider container config.
 *
 * Kimi K2.6 is reached through Moonshot's Anthropic-compatible endpoint, so the
 * container runs the full Claude Agent SDK with the complete toolset — there is
 * no tool handicap versus the native Claude path. This file is the host-side
 * half: it points the SDK at Moonshot and primes the auth header.
 *
 * The real Moonshot key never enters the container. An OneCLI generic secret
 * (host-pattern = the Moonshot host, header-name = Authorization, value-format
 * "Bearer {value}") rewrites the Authorization header on the wire. The container
 * only needs:
 *   - ANTHROPIC_BASE_URL — so the SDK calls Moonshot, not api.anthropic.com
 *   - ANTHROPIC_AUTH_TOKEN=placeholder — so the SDK emits an
 *     Authorization: Bearer header for OneCLI to overwrite
 *
 * The container-side half (registering `kimi` in the agent-runner's provider
 * registry, reusing ClaudeProvider) lives at
 * container/agent-runner/src/providers/kimi.ts.
 */
import { readEnvFile } from '../env.js';
import { registerProviderContainerConfig } from './provider-container-registry.js';

/** Moonshot's Anthropic-compatible base URL. Overridable via KIMI_BASE_URL. */
const DEFAULT_KIMI_BASE_URL = 'https://api.moonshot.ai/anthropic';

registerProviderContainerConfig('kimi', () => {
  const dotenv = readEnvFile(['KIMI_BASE_URL']);
  const baseUrl = dotenv.KIMI_BASE_URL || DEFAULT_KIMI_BASE_URL;
  return {
    env: {
      ANTHROPIC_BASE_URL: baseUrl,
      ANTHROPIC_AUTH_TOKEN: 'placeholder',
    },
  };
});
