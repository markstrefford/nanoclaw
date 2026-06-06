# STATE

_Snapshot — regenerated 2026-06-05. Present-tense cursor, overwritten each session._

## ⚠️ Major pivot this session: v1 → v2

The Kimi trial moves to the LIVE install, **`/Users/mark/Development/nanoclaw-v2`**.
The `nanoclaw` (v1) tree where this session's code landed is NOT the running
assistant. All v1 work (t10/t30/t40 + the Moonshot key in v1/.env) is now REFERENCE,
not the deliverable. Confirmed: live process is `com.nanoclaw-v2-*` from nanoclaw-v2.

## Active focus

Build Kimi K2.6 as a **v2 provider**, trial it on the **Ayah/Iris** group.

## Why v2 is better than the v1 plan (decided 2026-06-05)

- v2 runs the full Claude Agent SDK. Moonshot has an Anthropic-compatible endpoint
  (`https://api.moonshot.ai/anthropic`, `ANTHROPIC_MODEL=kimi-k2.6`), so Kimi runs
  with the FULL toolset — the v1 tool handicap (t10) DISAPPEARS. True apples-to-apples.
- Key lives in the OneCLI vault (header rewrite on the wire), NOT in .env — fixes the
  secret-hygiene wrinkle. Mirror of `src/providers/claude.ts`.
- Per-group: `container_configs.provider`/`model` — trial Kimi on ONE group (Iris),
  rest stay on Claude. No global switch.

## v2 build shape (the new story — to plan/build)

1. `src/providers/kimi.ts` — mirror claude.ts, point ANTHROPIC_BASE_URL at Moonshot's
   anthropic endpoint, set model kimi-k2.6; add `import './kimi.js'` to providers/index.ts.
2. OneCLI secret for the Moonshot key (host=api.moonshot.ai, header=Authorization,
   value-format "Bearer {value}").
3. Cost: v2 cost-report reads SDK `total_cost_usd` = Anthropic-priced = WRONG for Kimi.
   Price recorded token usage with Kimi rates ourselves (port of v1 t40). THE real work.
4. Flip Iris's container_config provider/model to Kimi to start the trial. Reversible.

## Reference (v1, done — do not run as the trial)

- v1 t10/t30/t40 committed on branch `feat/token-tracking` in nanoclaw. Kimi K2.6
  pricing = $0.95/M in, $4.00/M out. Answers upstream discussion #948.
- Moonshot key currently in v1/.env (gitignored, 600). Will be re-homed to the OneCLI
  vault for v2; remove from v1/.env once v2 is live.

## Open questions

See OPEN.md. Key one: how v2 records per-turn token counts for a non-Claude model
(needed to price Kimi) — confirm at v2 plan time.

## Next

Plan the v2 Kimi-provider story against nanoclaw-v2 code, then build. Session is long
— recommend /compact (or fresh session in nanoclaw-v2) before the build.
