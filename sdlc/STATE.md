# State — last updated 2026-06-08

**Active focus:** s-open-model-cost-eval — Kimi K2.6 trial is **LIVE on BOTH groups (Ayah/Iris + North)**, accruing. Letting it run a couple of days before the verdict.

**Status — both groups on Kimi (Ayah deployed 2026-06-06 ~17:21; North flipped 2026-06-08):**
- `container_configs` for Ayah (f1399cd4-9b7d-4325-8935-49cd9b8a3e34): `provider=kimi, model=kimi-k2.6`.
- `container_configs` for North (ag-1780055476622-jk9cck): `provider=kimi, model=kimi-k2.6`. North's OneCLI agent already `secretMode: all` — no rebuild/restart needed (config is a live per-spawn DB read; running host already serves Kimi). Takes effect on North's next message.
- Reached via Moonshot's Anthropic-compatible endpoint (`https://api.moonshot.ai/anthropic`) → full Claude Agent SDK, full toolset, NO tool handicap.
- Moonshot key is a OneCLI vault secret ("Moonshot Kimi", host api.moonshot.ai, Bearer); Ayah's OneCLI agent is `secretMode: all`. Key is NOT in .env on v2.
- Root cause of the earlier outage: live host was running stale pre-Kimi compiled code. Fix = `pnpm run build` + restart the slug-labelled service (`com.nanoclaw-v2-8a163795`). Container agent-runner src is bind-mounted, so no image rebuild was actually required.

**Verified working:** Ayah replies in 4–18s, no errors/401. Prompt/context caching WORKS through Moonshot — cache_read climbs across turns (512 → 10k → 43k). Moonshot reports cache-read but cache_creation=0 (no read/write split); cost report prices cache-write at input rate and flags it inexact — so Kimi cost is conservative, not undercounted. Early signal: Kimi ~1–2¢/turn vs Sonnet $0.60–1.30/turn (7-turn sample — needs the couple of days).

**Cost truth:** use `ncl cost`. Note Ayah's day total is mostly PRE-flip Sonnet usage; isolate pure-Kimi days for the real comparison.

**Done:** t10/t30/t40 (v1 reference) · t60 provider · t70 pricing · t80 go-live (deployed).

**Loose ends (not blocking, for a later session):**
- `feat/kimi-provider` is only *running* on the live host — nothing merged to the main v2 branch yet.
- Contribution back to upstream discussion #948 (how to run Kimi via Moonshot) — queued until the trial proves out.
- Optional: drop the 30-min idle ceiling to ~10 min (operator said leave it for now).

**Open questions:** see OPEN.md.
