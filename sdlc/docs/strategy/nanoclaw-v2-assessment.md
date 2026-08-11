---
id: nanoclaw-v2-assessment
kind: strategy
project: nanoclaw
sources: [raw/v2-reliability-issues-review.md]
verified-on: 2026-06-27
created: 2026-06-27
updated: 2026-06-27
---

# NanoClaw v2 — reliability & capability assessment

**Purpose.** Mark is not convinced NanoClaw v2 is the right platform to keep
building on. There are a couple of alternative options on the table. This
document is the **measuring stick**: an honest, evidence-grounded record of what
has actually gone wrong with v2, why, what we did about it, and how v2 compares
to v1 — so any alternative can be scored against a real track record instead of
a hunch. It is not a fix plan; it is the baseline a decision is made against.

The headline finding up front, because it reframes everything below: **v2's core
reliability failures are not caused by the model provider, the network, or the
difficulty of the requests. They are a regression — v1 had self-protection
mechanisms that the ground-up rewrite dropped.**

---

## 1. The failure surface — what has actually gone wrong

Every incident below presents as the same symptom to the user: **the agent goes
silent.** Total silence, or a "working on it…" spinner that never resolves, or a
daily briefing that never arrives. That single symptom has at least five
independent causes at five different layers — which is itself the signal that
the reliability was never designed as a whole.

| Failure | Layer | Evidence |
|---|---|---|
| Model account ran out of balance (Kimi) — multi-day, silent blackout of both bots | Billing | 2026-06-26 outage |
| One flaky MCP tool server hung the whole agent at startup | Tooling | 2026-06-26 outage |
| **Container wedges → killed at 30-min ceiling → respawns into the same wedge, forever** | **Execution** | **355 ceiling-kills in the current log, from only 4 sessions; two sessions looped 240× and 112×** |
| Inbound poller (Telegram) stayed wedged after a laptop-offline window, didn't self-heal | Transport | 2026-06-27, ~1,968 poll failures logged |
| Replies emitted without a destination were silently dropped into a retry loop | Routing | 2026-06-26 (North dropped every reply) |
| Self-inflicted: forcing a credential-library version crashed Gmail + Calendar tools | Dependency | 2026-06-26 |
| Calendar tool confusion / iCloud CalDAV reachability flaky | Tooling | Open |

The third row is the dominant, still-live failure and the one this assessment is
built around.

---

## 2. The underlying cause

### The proven root (the crash-loop)

The 355 ceiling-kills came from **only four sessions**, and two of them account
for 352. Those two are also **the only two oversized sessions on the system** —
2.8 MB and 1.6 MB. Every healthy session is under 100 KB and has never looped.

The mechanism, confirmed against the code and the data:

1. v2 lets a session's accumulated history **grow without bound.**
2. On every container spawn, that entire history is replayed into the model.
3. A multi-megabyte replay makes even a trivial request ("add 10 images", "what's
   on my calendar") slow enough to blow past the 30-minute liveness ceiling.
4. The container is killed — but the session is left active, so it **respawns
   straight back into the same oversized replay** and stalls again.
5. Nothing breaks the loop. The only safety net throttles the *owner
   notification*, not the respawn.

This explains every observation precisely:
- A trivial task hangs because the cost is the replay of a month of accumulated
  context, not the task.
- It keeps hanging even after clearing the inbound messages, because the bloat is
  in the accumulated session, not the pending message.
- It is not the provider: over the last 90 days Kimi ran at **99.92%** uptime —
  higher than Claude's own surfaces (claude.ai 99.28%, Claude API 99.51%,
  Claude Code 99.37%). Every provider has routine sub-1% blips; the defect is
  that v2 cannot absorb them.

### The general property gap

Underneath the specific bug sits the real issue, and it is the thing Mark
actually wants: **the agent cannot observe or heal itself.** It has no internal
sense of its own liveness, so it cannot tell "thinking" from "stalled"; failure
is silent by default; work is monolithic and not resumable; nothing survives a
container exit to learn from; and the only recovery is the host bluntly killing
and respawning. A reliable agent senses a stall in seconds, says so, recovers
what it can (retry, rotate, route around a dead tool, resume from a checkpoint),
and escalates the rest with a precise ask. v2 does none of this.

---

## 3. What we have done about it

Every fix shipped to date is a **point-patch on one specific stall source.** None
addresses the underlying root.

| Fix | What it patches | Commit |
|---|---|---|
| Budget/quota → instant alert + fail-fast | Billing blackout | 42a1eec |
| Host hang-watchdog DMs the owner on a ceiling-kill | *Notification only* — not the loop | d4112ad |
| MCP startup health-gate drops a hung tool | Tool wedge **at startup only** | 816c222 |
| `<message>` without a destination is delivered, not dropped | Routing loop | 4c04e2e |
| 30-second timeout around CRM fetches | One tool, hand-wrapped | 44eb64b |
| Telegram poller hard-cycle watchdog | Transport wedge | a9dc4be (after dbe7a03 was insufficient) |
| Date pinned as a plain in-message line | Briefings dated wrong | bf5bc72 |

The pattern is the indictment: reliability accreted as one patch per incident.
The general defect — a turn can stall with no bound, and a session can respawn
into the same stall with no circuit breaker — was never addressed, so each new
stall source reappears as a fresh "agent silent" incident.

---

## 4. What v2 LOST relative to v1 (the regressions)

v1 had a coherent self-protection design. The ground-up rewrite dropped it.
Mark's own v1 commit (`ef36ad8`, April 2026) names the exact failure v2 now
exhibits:

> *"Container failures were causing runaway retries (each new message reset the
> retry counter), and a 16 MB session file meant every container spawn replayed
> hundreds of conversation turns through Claude."*

Two mechanisms from that commit are **absent in v2**:

| v1 had | What it did | v2 status |
|---|---|---|
| **Circuit breaker** | After 5 failed retries, the group enters a 5-min cooldown — no new containers spawn, messages queue for later | **Dropped** → the 240× respawn loop |
| **Auto session rotation** | A session over 2 MB is rotated to a fresh one before spawn | **Dropped** → unbounded bloat, oversized replay, the stall itself |

The net effect of losing both: the self-healing behaviour Mark remembers v1
doing once — noticing it was in trouble and protecting itself — does not exist in
v2. This is the single most important line in this document. The failure is not
that v2 was never reliable; it is that **v2 is less resilient than the system it
replaced.**

---

## 5. What v2 GAINED relative to v1 (and the cost of each)

The rewrite was not gratuitous — it bought real things. But each gain carries a
cost that belongs in the comparison.

| v2 gained | Benefit | Cost / trade-off |
|---|---|---|
| **OneCLI credential vault** | Secrets live in a vault and are injected per request — never in env vars or chat context. Genuinely more secure than v1's `.env`. | **One OAuth connection per provider, globally.** Cannot hold two Gmail accounts (business + personal). The personal agent has to bypass the vault with mounted tokens. A real capability ceiling, not just inconvenience. |
| **Per-session Docker containers** | Strong isolation between agents and sessions | Far more failure surface — container wedges, cross-mount DB-corruption bugs, image-rebuild races, 30-min ceiling kills. Much of this assessment's failure surface exists *because* of containerisation. |
| **Explicit entity model** (users → groups → sessions) | Multi-agent, scoped roles, explicit routing | More concepts, more wiring, more places to misconfigure |
| **Two-DB session split** | One writer per file, no cross-mount lock contention | More moving parts; the inbound/outbound split is its own source of subtle bugs |
| **Channels & providers as installable skills** | Clean, pinned, idempotent installs | Lives across sibling branches that must be kept in sync |
| **Multi-provider / Kimi** | Large cost saving on model spend | Orthogonal to reliability (see §2 — provider uptime is not the problem) |

The honest summary: **v2 traded resilience and multi-account capability for
isolation and credential security.** Whether that trade was worth it is the
decision this document exists to inform.

---

## 6. The measuring stick — criteria for any alternative

Derived directly from the issues above. Score each option against these:

1. **Resilience to routine transients** — does a sub-1% provider/tool blip cost a
   retry, or a 30-minute outage?
2. **Self-observation & self-recovery** — can it tell it's stalled, say so, and
   recover without a human?
3. **Bounded, resumable work** — does it ever silently skip what was asked?
4. **Credential security *and* multi-account** — can it be secure *and* hold two
   Gmail accounts? (v2 gives the first, not the second.)
5. **Operational burden** — how much does it cost to keep running (host,
   always-on, patching)?
6. **Isolation vs. simplicity** — is the failure surface worth the isolation it
   buys?
7. **Capability** — channels, tools, multi-agent, scheduling actually working.

---

## 7. Options under consideration

*To be completed with Mark's candidate options. Each scored against §6.*

| Option | Resilience | Self-heal | Multi-account | Op burden | Isolation | Capability | Notes |
|---|---|---|---|---|---|---|---|
| **Stay on v2, restore v1's guards** | | | | | | | Recover circuit breaker + session rotation + build self-observe layer |
| *(option B — TBD)* | | | | | | | |
| *(option C — TBD)* | | | | | | | |

---

## Appendix — evidence

- Ceiling-kills: 355 total, 4 distinct sessions; `sess-…54822j` 240×,
  `sess-…05lxe1` 112×. Both still `status=active` a month on (created 2026-05-29 /
  2026-05-30).
- Oversized sessions: `…05lxe1` 2.8 MB, `…54822j` 1.6 MB; all others < 100 KB.
- Telegram poll failures: ~1,968 in the current error log.
- Provider uptime (90 days): Kimi 99.92%, claude.ai 99.28%, Claude API 99.51%,
  Claude Code 99.37%.
- v1 self-protection: commit `ef36ad8` (circuit breaker + auto session rotation).
- v2 patches: `42a1eec`, `d4112ad`, `816c222`, `4c04e2e`, `44eb64b`, `a9dc4be`,
  `dbe7a03`, `bf5bc72`.
- Related: runbook `rb-agent-silent-or-hung`; OPEN.md items tagged the same.
