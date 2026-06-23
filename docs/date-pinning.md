# Date/time pinning (how every bot knows what "today" is)

## Problem

Daily briefings (and any "today"/"now" reasoning — calendar, email, scheduling)
were dating themselves **a day behind**: on Tuesday 23 June the morning briefing
was headed "Monday 22 June". This happened on **all** bots, not just one.

Two distinct causes were ruled in/out during diagnosis:

1. **Container clock skew** — *ruled OUT*. The Docker VM/container clock matches
   the host (verify: `docker run --rm busybox date -u` vs `date -u`). A
   laptop-hosted setup can drift after macOS sleep, so re-check this first if the
   symptom returns.
2. **Model ignoring the context header** — *the actual cause*. The per-turn
   prompt carried the date only as an XML attribute
   (`<context timezone="…" now="…" />`). That attribute was *delivered
   correctly* but **non-Claude models (e.g. Kimi) silently ignored it** and
   guessed the date from elsewhere (often the previous turn), landing a day off.

## Fix

`container/agent-runner/src/formatter.ts` → `formatMessages()` now pins the date
**twice**:

- the machine attribute `now="…"` (unchanged, for tooling/Claude), **and**
- a plain visible sentence the model cannot skim past:

  ```
  Current date and time: Tue, Jun 23, 2026, 7:26 AM (Europe/London). Treat this as "now"/"today"; never infer the date from message history or your own clock.
  ```

This is a single shared-formatter change, so it covers **every bot, every
message kind (chat, task/scheduled briefing, webhook, system), and every
schedule**. The value comes from `formatNowLocal(TIMEZONE)` (timezone resolved
from the container's `TZ`), so it is always the real local wall-clock time.

Regression guard: `formatter.test.ts` asserts both the attribute *and* the plain
sentence are present.

## Why this persists across restarts, host reboots, and rebuilds

- **Container restart** — the agent-runner source is bind-mounted read-only into
  the container (`-v …/container/agent-runner/src:/app/src:ro`,
  `src/container-runner.ts`), and the container runs `bun run /app/src/index.ts`
  directly. The next spawn reads the file fresh. No image rebuild needed.
- **macOS (host) restart** — the fix is a file on disk in the repo, committed to
  git. A reboot changes nothing.
- **New image builds** — the source is bind-mounted, not baked into the image, so
  `./container/build.sh` neither adds nor removes it. Even if the build process
  later bakes `src/` in, the committed file is the source of truth.

The only thing that could lose it is an uncommitted working-tree edit being
discarded — so it is **committed**, not left as a local change.

## If the symptom returns

1. Check clock skew (cause #1 above).
2. Confirm the plain line is actually in the prompt — `formatter.test.ts` should
   be green; if a refactor dropped the line, the test fails.
3. If a *new* provider/model still ignores even the plain line, strengthen the
   wording or move the date into the task prompt body itself
   (`formatTaskMessage`).
