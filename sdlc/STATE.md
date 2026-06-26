# State — last updated 2026-06-26

**Active focus:** none — the 2026-06-26 reliability incident is resolved and tidied up.

**Last completed:** Bot-outage incident (both Ayah + North). Root causes: Kimi/Moonshot ran out of balance (silent multi-day blackout) and North dropped every reply (un-attributed `<message>` looping). Both bots restored and verified live. Five fixes shipped to v2-main — budget→Telegram alert + fail-fast (42a1eec), host hang-watchdog (d4112ad), MCP startup health-gate (816c222), `<message>`-without-`to=` routing (4c04e2e), date-pin as plain line (bf5bc72) — plus calendar key disambiguation and revert of a bad google-auth-library override. Recorded in runbook `rb-agent-silent-or-hung`.

**Next:** operator's call. Three open threads in OPEN.md: Ayah's iCloud/CalDAV reachability (only real remaining bug), whether North keeps the VOSS CRM tool, and a watch-note on the (transient) Gmail refresh error. None blocking.

**Blockers:** none.

**Open questions:** 8 — see OPEN.md.
