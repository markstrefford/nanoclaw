/**
 * Kimi (Moonshot) provider — container-side registration.
 *
 * Kimi K2.6 is served over Moonshot's Anthropic-compatible endpoint, so it runs
 * through the exact same Claude Agent SDK path as the native Claude provider —
 * full toolset, no special-casing. The only differences (base URL + auth header)
 * are injected as env by the host-side provider config (`src/providers/kimi.ts`)
 * and the OneCLI gateway, both of which ClaudeProvider already reads from
 * `options.env`. So `kimi` is just ClaudeProvider under a different name.
 *
 * Without this registration the container throws `Unknown provider: kimi` on its
 * first poll (provider-registry.ts), before ever reaching Moonshot.
 */
import { ClaudeProvider } from './claude.js';
import { registerProvider } from './provider-registry.js';

registerProvider('kimi', (opts) => new ClaudeProvider(opts));
