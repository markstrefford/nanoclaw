# Re-auth Gmail under production to retire the year-5138 expiry hack

**Captured:** 2026-06-08 · **Priority:** low · **Status:** parked

## What
Ayah/Iris's Gmail OAuth token (`~/.gmail-mcp/credentials.json`) is dated 2026-05-29 —
*before* the consent screen (`iris-personal-497911`) was published to production on
2026-06-07. It's currently surviving on a hardcoded `expiry_date=99999999999999`
(year-5138) hack that masks the access-token expiry check, rather than a clean
production-minted token.

## Why it matters (and why it's not urgent)
Gmail works today, so no user impact. But it's the *only* Google credential still
relying on the hack. Until it's re-minted under production, it's a latent risk: if
that hack is ever cleared or the refresh token (possibly Testing-era) is rejected,
Gmail breaks the same way Calendar did. Re-authing once puts Gmail on the same
permanent footing as Calendar (now fixed) — no expiry, no hack.

## How (when we get to it)
Same interactive one-liner used for Calendar on 2026-06-08, but with Gmail paths/scope:
- `GMAIL_OAUTH_PATH=~/.gmail-mcp/gcp-oauth.keys.json`
- `GMAIL_CREDENTIALS_PATH=~/.gmail-mcp/credentials.json`
- `GMAIL_ACCOUNT=mark.strefford@gmail.com`
- scope = gmail-mcp default (`gmail.settings.basic gmail.modify`)
- Gmail uses **flat** format — NO `{normal:…}` wrap needed (that wrap is Calendar-only).
Mark runs it (browser login). Verify the new token has no `refresh_token_expires_in`.
Full procedure + rationale: memory `reference_google_oauth_token_expiry`.
