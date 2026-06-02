# Updates

## 2026-06-02 — Ayah personal Gmail (direct-to-Google) + `USE_CUSTOM_GMAIL` toggle

### Problem
Ayah (personal assistant) read the **business** mailbox (`mark@reimagined.industries`) instead of
the personal one (`mark.strefford@gmail.com`), despite being wired to the correct personal
credentials. Had worked previously, then broke.

### Root cause
Agent containers route all traffic through the **OneCLI gateway** (`HTTPS_PROXY` +
`NODE_USE_ENV_PROXY=1`, injected by `@onecli-sh/sdk`). Once OneCLI held a connected **business
Gmail** account, the gateway **substituted that account on every Gmail API call** — overriding
Ayah's personal token on the wire. Same credentials file: through the proxy → business; direct →
personal. Not a credential/OAuth problem (re-authenticating would not have helped).

`NO_PROXY` did **not** fix it — neither `undici` (`fetch`) nor `gaxios`/`google-auth-library`
honoured it under `NODE_USE_ENV_PROXY=1`. The only thing that bypasses the gateway is setting the
proxy URL vars to **empty** (`HTTP_PROXY`/`HTTPS_PROXY`/`http_proxy`/`https_proxy` = `""`).

### Fix (durable, toggleable)
A server signals "reach the provider directly with my own credentials" by declaring `NO_PROXY` in
its MCP env. When **`USE_CUSTOM_GMAIL=true`**, `configFromDb()` blanks that server's proxy vars at
materialization, so it bypasses the gateway. Servers without `NO_PROXY` (North's business Gmail,
iCloud CalDAV) are untouched and stay on the gateway.

- `src/config.ts` — new `USE_CUSTOM_GMAIL` flag (read from `.env`, like `LOG_TEXT_FOR_ANALYTICS`).
- `src/container-config.ts` — `applyGatewayBypass()` in `configFromDb()`; empties proxy vars for
  `NO_PROXY`-marked servers when the flag is on.
- `.env` — `USE_CUSTOM_GMAIL=true`.
- DB (`container_configs`, agent group `f1399cd4-…` / Ayah) — `gmail` + `calendar` declare the
  `NO_PROXY` marker (the code supplies the actual bypass).

### Toggle
- **On:** `USE_CUSTOM_GMAIL=true` in `.env`, restart the service.
- **Off:** set `false` (or remove) + restart → everything routes back through the OneCLI gateway.
- Takes effect on each agent's **next fresh container** (config is materialized from the DB at
  spawn; a live container keeps its config until recycled).

### Verify
Ask Ayah to *search her inbox* (e.g. "list my recent emails") — that uses `mcp__gmail__*`, now
direct → personal. Note: an ad-hoc `Bash` curl from the agent still goes through the gateway
(→ business), so "what account am I on?" self-checks are misleading; trust the email tools.

### Restart
`launchctl kickstart -k gui/$(id -u)/com.nanoclaw-v2-8a163795`
