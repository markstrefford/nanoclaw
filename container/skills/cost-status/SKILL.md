---
name: cost-status
description: >-
  Report API token spend / cost on demand. Use whenever the user asks what
  they've spent, today's or this month's cost, token usage, "how much is this
  costing", the run-rate, or similar. Run `ncl cost` and relay the figures.
---

# Cost / spend status

When the user asks about cost, spend, token usage, or run-rate, get the live
figures from the host with the `ncl cost` command — don't estimate or guess.

```bash
ncl cost            # today so far (default)
ncl cost today      # today so far
ncl cost yesterday  # the full prior day
ncl cost month      # month-to-date
ncl cost 7          # trailing 7 days
```

It returns a ready-made per-bot breakdown (cost, tokens, turns) plus a total.
Relay the relevant numbers conversationally — you don't need to paste the whole
block verbatim; pull out what the user asked for.

## Important caveats — state them when relevant

- The `$` figures **assume list API pricing and are approximate**. For the
  authoritative bill, the Anthropic console is the source of truth.
- This report covers **only this NanoClaw install's bots** (their per-session
  token usage). It does not include other API keys or usage outside the bots.
- Computing this costs **no tokens** — it's read from the host, so the user can
  ask as often as they like.
