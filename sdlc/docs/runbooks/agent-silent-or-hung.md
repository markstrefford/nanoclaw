---
id: rb-agent-silent-or-hung
kind: runbook
project: nanoclaw
sources: [raw/nanoclaw-outage-gmail-calendar-regression.md]
verified-on: 2026-06-26
created: 2026-06-26
updated: 2026-06-26
---

# Agent silent or hung — causes & safety nets

When a bot stops replying — total silence, or a "⏳ working on it…" spinner that
never resolves, or a daily briefing that never arrives — work down this list.
It's ordered by how often each was the cause in practice.

## Diagnose in this order

1. **Model account balance.** The most common cause of a multi-bot, multi-day
   blackout is the model provider running out of credit (e.g. Kimi/Moonshot
   "insufficient balance"). As of 2026-06-26 this now posts a Telegram alert the
   moment it happens — if you see a "billing/quota error, top it up" message,
   that's it. Top up; both bots recover on the next message. The provider's own
   status page being green does **not** rule this out — an out-of-balance account
   still returns errors while the service is "operational".

2. **Look inside the running container.** `docker logs <container-name>` works on
   a **running** container even though we spawn with `--rm` (the flag only wipes
   it on *exit*). Don't guess from the host log — read the container. Tell-tales:
   - `WARNING: agent output had no <message to="..."> blocks — nothing was sent`
     → the agent is replying but its messages are being dropped (see safety nets).
   - `MCP server "X" DROPPED at startup` → a tool failed to start; the agent ran
     without it on purpose.
   - model/auth/quota errors → provider-side.

3. **Ceiling-kills = a wedge.** In the host log, `Killing container … reason="absolute-ceiling"`
   means the agent produced no heartbeat for 30 min. The host now DMs the owner
   when this happens (throttled). Repeated ceiling-kills for one session = a
   wedge that re-triggers each spawn.

4. **After any image rebuild, recycle running containers.** A rebuild does not
   touch already-running containers — they keep serving the old image until they
   exit. `docker kill` the running ones so the next spawn picks up the new image.
   Confirm with `docker ps --format '{{.Names}} {{.Image}} {{.Status}}'`.

## Safety nets now in place (shipped 2026-06-26)

These exist specifically so the failures above are *visible and bounded* instead
of silent 30-minute hangs:

- **Budget/quota → instant alert + fail-fast.** A non-retryable model error
  (out of balance, quota, auth) is surfaced in-channel immediately and the turn
  ends; retryable errors are bounded so a persistent one also fails fast rather
  than retry-storming to the ceiling. (`poll-loop.ts`, commit 42a1eec)
- **Host hang-watchdog.** The host DMs the owner when a container is ceiling-killed.
  (`host-sweep.ts` + `approvals/primitive.ts notifyOwner`, commit d4112ad)
- **MCP startup health-gate.** Each MCP tool server is probed (initialize +
  tools/list, bounded ~20s) before being handed to the agent SDK. One that hangs
  or crashes is dropped — the agent runs with its remaining tools and is told
  which are down — so a single flaky tool can no longer wedge the whole agent.
  (`mcp-health.ts`, commit 816c222)
- **`<message>` without `to=` is delivered, not dropped.** Some models emit
  replies as `<message>…</message>` without the destination attribute; these now
  route to the originating channel instead of being dropped into a silent
  retry loop. (`poll-loop.ts dispatchResultText`, commit 4c04e2e)

## Known failure modes seen on 2026-06-26

- **Kimi out of balance** → silent multi-day blackout of both bots. (Fixed: top-up + the budget alert above.)
- **North dropped every reply** → it emitted `<message>` without `to=`; the runner dropped + nudged + looped forever. (Fixed by the routing change above.)
- **Self-inflicted regression — do not repeat:** forcing `google-auth-library@10.x`
  to "fix" a Gmail token refresh **breaks `googleapis@129`** (it removed
  `DefaultTransporter`), crashing `gmail-mcp` *and* `@cocal/google-calendar-mcp`
  on startup. The original "Premature close" refresh error was transient (network
  thrash during the outage), not a deterministic version bug. Leave the pinned
  versions alone unless a fix is verified to **boot** both servers, not just refresh.
- **Calendar tool confusion:** with two calendar servers, give the Google one a
  distinct MCP key (`google-calendar`, not the generic `calendar`) so the model
  doesn't default Google for iCloud requests.
