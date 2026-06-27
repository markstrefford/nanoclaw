Title: Review all NanoClaw v2 reliability issues — consolidate the "agent goes silent" failure surface

Intent: A standing review of every reliability/availability issue we've hit on NanoClaw v2 specifically. The goal is one consolidated view of the "agent goes silent / unresponsive" failure surface, which now has at least FOUR independent root causes, each with its own detection + safety net, currently scattered across runbook entries, OPEN items, commit messages, and memory notes. Decide what's still open, what's systemic, and what hardening is worth doing next.

Why now: 2026-06-27 — North went fully unresponsive and Ayah was only producing host-side daily cost reports. Root cause was a NEW failure mode: the Telegram inbound poller wedges after a long laptop-offline window (10:45–20:15) and does NOT self-heal when the network returns — the @chat-adapter/telegram poll loop runs `while(pollingActive)` and retries getUpdates forever without exiting, so the old re-arm-only watchdog (startPolling() is a no-op while the loop is "active") could not recover it. Required a manual service restart. This is the same SYMPTOM (agent silent) as the 2026-06-26 model/MCP/container outage but a completely different LAYER (inbound transport, not model/tool/container). That's the signal that we need a consolidated reliability review, not another point fix.

Already-fixed today (do not re-open, just record): src/channels/telegram.ts now runs a hard-cycle watchdog — an instrumented logger taps the adapter's own "polling request failed" stream; when failures are sustained AND Telegram is independently reachable (getMe probe), it does stopPolling()->startPolling() to rebuild a fresh loop. Handles both loop-exited and loop-alive-but-wedged. Built, 382 tests pass, deployed on v2-main. NOT yet committed (awaiting Mark's go).

The four+ known root causes of "agent silent" to consolidate:
1. Model provider billing/quota (Kimi/Moonshot out-of-balance) — multi-day blackout. Safety net: instant in-channel alert + fail-fast (commit 42a1eec).
2. MCP tool server wedge at startup — one flaky tool hung the whole agent. Safety net: MCP startup health-gate drops it (commit 816c222).
3. Container hang → 30-min absolute-ceiling SIGKILL, re-triggers each spawn; message claimed-then-silent. Safety net: host hang-watchdog DMs owner (commit d4112ad). STILL THE MAIN OPEN ITEM — the fix bounds/alerts but doesn't prevent the wedge.
4. Inbound transport (Telegram poller) wedge after offline window — fixed today (this session).
   Plus: <message> without to= drop-loop (commit 4c04e2e); google-auth-library self-inflicted regression; dual-calendar tool confusion.

Seed sources for whoever runs the review:
- sdlc/docs/runbooks/agent-silent-or-hung.md (the 2026-06-26 outage compile)
- sdlc/OPEN.md (items tagged rb-agent-silent-or-hung: iCloud CalDAV flakiness, VOSS CRM cold-start, Gmail refresh transient)
- git log on v2-main (recent reliability commits: 42a1eec, d4112ad, 816c222, 4c04e2e, 5bb0697)
- Claude memory incident notes: telegram-poll-backoff-dead-windows, second-telegram-bot-identity-namespace, date-pin-plaintext-fix, mac-overnight-sleep-blockers, onecli-credential-path
- logs/nanoclaw.error.log + logs/nanoclaw.log

Scope: v2 ONLY (origin/v2-main; v1 is a separate divergent line, do not include).

Suggested shape when compiled: either expand rb-agent-silent-or-hung into a full reliability-surface map, or a "v2 reliability hardening" epic/strategy. Recommend executing in a FRESH session for an unbiased full-history sweep (this capture was written mid-incident).
