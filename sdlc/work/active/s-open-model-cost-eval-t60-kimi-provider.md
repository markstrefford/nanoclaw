---
id: s-open-model-cost-eval-t60-kimi-provider
kind: task
project: nanoclaw
status: active
autonomy: attended
parent: s-open-model-cost-eval
created: 2026-06-06
updated: 2026-06-06
---

# t60 — Reach Kimi K2.6 through the v2 runner

## Outcome

Make Kimi K2.6 a selectable provider for one agent group, with the Moonshot key
held in the vault and never in the container.

## Acceptance

- A group can be pointed at Kimi K2.6 and its container talks to Moonshot's
  Anthropic-compatible endpoint with the full agent toolset — no tool handicap
  versus the Claude path. This requires BOTH provider registries to know `kimi`:
  the host-side one (spawn env) and the container-side one (the SDK provider the
  container actually instantiates).
- The Moonshot API key lives only in the OneCLI vault; it is not in `.env`, not
  in the container env, and not visible in chat context. (Removing the v1 copy of
  the key is deferred to t80, after a live Kimi turn is verified — t60 cannot
  itself confirm a live turn end-to-end on the trial group.)
- Selecting Kimi is per-group and reversible: flipping the group back to Claude
  needs no image rebuild, only a config change.

## Test specification

- **Host registry:** `src/providers/kimi.ts` registers a provider container config
  named `kimi` returning env that sets `ANTHROPIC_BASE_URL` to the Moonshot
  Anthropic endpoint plus `ANTHROPIC_AUTH_TOKEN=placeholder` (mirroring
  `claude.ts`). Assert the host registry resolves `kimi` to a config fn after
  `import './kimi.js'` is added to `src/providers/index.ts`.
- **Container registry (load-bearing — without it the container throws
  `Unknown provider: kimi` before reaching Moonshot):** the container-side factory
  `createProvider('kimi', …)` resolves to a working provider. Extend the existing
  `factory.test.ts` (bun:test) to assert `createProvider('kimi')` returns a
  `ClaudeProvider` instance (kimi reuses the Anthropic-compat path), alongside the
  existing claude/mock/bogus cases.
- Registering `kimi` twice throws (both registries dedupe) — the import must appear
  exactly once in each barrel.
- Manual/host check (no automated harness): with a group's
  `container_configs.provider='kimi'` and `model='kimi-k2.6'`, a spawned container
  reaches Moonshot and returns a normal assistant turn. A `401` here means the
  OneCLI secret/host-pattern or agent secret-mode is wrong, not the provider file.

## Implementation notes

Grounded in code read at plan time. There are TWO provider registries and both
must register `kimi`:

- **Host side** — `src/providers/kimi.ts` is a near-copy of
  `src/providers/claude.ts`. The claude provider only contributes env when
  `ANTHROPIC_BASE_URL` is set in `.env`; for kimi, hardcode/default the Moonshot
  Anthropic endpoint `https://api.moonshot.ai/anthropic` (verify the exact host
  against current Moonshot docs at execute — a wrong host means OneCLI won't
  rewrite the header and every call 401s) and set `ANTHROPIC_AUTH_TOKEN=placeholder`
  so the SDK emits an `Authorization: Bearer` header for OneCLI to overwrite. The
  host contribution sets `ANTHROPIC_BASE_URL` in the container env, which is what
  actually steers the SDK. Register via `registerProviderContainerConfig('kimi', …)`
  and append `import './kimi.js';` to `src/providers/index.ts`.
- **Container side** — the container instantiates the SDK provider via its own
  registry (`container/agent-runner/src/providers/provider-registry.ts`), which
  throws `Unknown provider: kimi` if not registered (`index.ts:89` →
  `factory.ts` → `getProviderFactory`). `ClaudeProvider` already reads
  `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` from env and passes `model` straight
  to the SDK, and Moonshot is Anthropic-compatible — so register `kimi` to REUSE
  `ClaudeProvider` rather than forking a class: add
  `registerProvider('kimi', (opts) => new ClaudeProvider(opts))` (in `claude.ts`
  beside the existing claude registration, or a tiny `kimi.ts`) and one barrel
  import in `container/agent-runner/src/providers/index.ts`. `ProviderName` is
  `string` (open set) — no type union to extend.
- **No image rebuild needed:** the agent-runner source is mounted read-only into
  the container at spawn (`src/container-runner.ts` mounts `container/agent-runner/src`
  at `/app/src`), so the new container-side registration is picked up on the next
  spawn — this is what makes the t80 flip reversible without a rebuild.
- Model selection is already wired: `container_configs.provider` →
  `resolveProviderName` (`container-runner.ts:230`); `container_configs.model` →
  `agent-runner/src/config.ts:53` → `index.ts:94` → the SDK `model` option.
  Set `model='kimi-k2.6'`.
- OneCLI secret: create a generic secret with host-pattern matching the
  configured Moonshot host, header-name `Authorization`, value-format
  `Bearer {value}`, value = the Moonshot key currently in v1 `nanoclaw/.env`.
  Deleting the v1 `.env` copy happens in t80, after the first live Kimi turn is
  verified — not here (t60 alone can't confirm an end-to-end turn).
- Gotcha to expect (documented in CLAUDE.md): an auto-created agent starts in
  `selective` secret mode, so the Moonshot secret may need explicit assignment
  (`onecli agents set-secret-mode --mode all`, or `set-secrets`) before the call
  stops returning `401`. No container restart needed after flipping mode.

## Status

active
