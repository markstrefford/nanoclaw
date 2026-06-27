#!/usr/bin/env python3
"""Re-auth a gmail-mcp / google-mcp credential, optionally forcing a specific
Google account.

For groups using the direct-to-Google path (`USE_CUSTOM_GMAIL=true`, see
docs/UPDATES.md), the MCP server authenticates from a local OAuth credential
file. When that token expires (and especially when the host has several Google
accounts signed in), the stock `auth` flow auto-opens the *default* browser
account, which is easy to get wrong. This helper instead:

  - builds the OAuth URL with `authuser`/`login_hint` pinned to the account you
    pass, so it can't land on the wrong account regardless of browser default;
  - prints the URL (and best-effort opens it) and runs a localhost listener;
  - exchanges the code and writes the credentials file in the format the MCP
    server expects (backing up any existing one).

No installation-specific data is hardcoded — all paths and the account come from
flags or env, so this file is safe to commit.

Usage (paths/account supplied at runtime, never stored in the repo):

    GMAIL_OAUTH_PATH=/path/to/gcp-oauth.keys.json \
    GMAIL_CREDENTIALS_PATH=/path/to/credentials.json \
    GMAIL_ACCOUNT=you@example.com \
    python3 scripts/gmail-reauth.py

Flags override env: --keys, --creds, --account, --scopes, --port.
"""
import argparse, json, os, sys, time, urllib.error, urllib.parse, urllib.request, webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

# gmail-mcp's default scope set; override with --scopes / GMAIL_SCOPES if needed.
DEFAULT_SCOPES = "https://www.googleapis.com/auth/gmail.settings.basic https://www.googleapis.com/auth/gmail.modify"

p = argparse.ArgumentParser(description="Re-auth a gmail-mcp OAuth credential, pinning the account.")
p.add_argument("--keys", default=os.environ.get("GMAIL_OAUTH_PATH"), help="Path to gcp-oauth.keys.json (or $GMAIL_OAUTH_PATH).")
p.add_argument("--creds", default=os.environ.get("GMAIL_CREDENTIALS_PATH"), help="Output credentials.json (or $GMAIL_CREDENTIALS_PATH).")
p.add_argument("--account", default=os.environ.get("GMAIL_ACCOUNT"), help="Force this Google account via authuser/login_hint (or $GMAIL_ACCOUNT). Optional.")
p.add_argument("--scopes", default=os.environ.get("GMAIL_SCOPES", DEFAULT_SCOPES), help="Space-separated OAuth scopes.")
p.add_argument("--port", type=int, default=int(os.environ.get("GMAIL_REAUTH_PORT", "3000")), help="Loopback redirect port (default 3000).")
args = p.parse_args()

if not args.keys or not args.creds:
    p.error("--keys/$GMAIL_OAUTH_PATH and --creds/$GMAIL_CREDENTIALS_PATH are required.")

keys = json.load(open(args.keys))
node = keys.get("installed") or keys.get("web") or keys
client_id = node["client_id"]
client_secret = node["client_secret"]
token_uri = node.get("token_uri", "https://oauth2.googleapis.com/token")
redirect = f"http://localhost:{args.port}"

auth_params = {
    "client_id": client_id,
    "redirect_uri": redirect,
    "response_type": "code",
    "scope": args.scopes,
    "access_type": "offline",
    "prompt": "consent",
}
if args.account:
    auth_params["authuser"] = args.account
    auth_params["login_hint"] = args.account
auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(auth_params)

print("\n" + "=" * 72)
print("Open this URL" + (f" (forces account {args.account})" if args.account else "") + ":\n")
print(auth_url)
print(f"\nWaiting for the redirect on {redirect} ...")
print("=" * 72 + "\n")
try:
    webbrowser.open(auth_url)
except Exception:
    pass

result = {}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        if "code" in params:
            result["code"] = params["code"][0]
            self.wfile.write(b"<h2>Done. Close this tab and return to the terminal.</h2>")
        else:
            result["error"] = params.get("error", ["unknown"])[0]
            self.wfile.write(("<h2>Auth error: " + result["error"] + "</h2>").encode())

    def log_message(self, *a):
        pass

try:
    srv = HTTPServer(("localhost", args.port), Handler)
except OSError as e:
    sys.exit(f"ERROR: could not bind {redirect} ({e}). Free the port or pass --port.")

while "code" not in result and "error" not in result:
    srv.handle_request()

if "error" in result:
    sys.exit("Auth failed: " + result["error"])

data = urllib.parse.urlencode({
    "code": result["code"],
    "client_id": client_id,
    "client_secret": client_secret,
    "redirect_uri": redirect,
    "grant_type": "authorization_code",
}).encode()
try:
    resp = urllib.request.urlopen(urllib.request.Request(token_uri, data=data), timeout=30)
    tok = json.load(resp)
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print("Token exchange FAILED — HTTP", e.code)
    print(body[:400])
    if "invalid_client" in body:
        print("\ninvalid_client → the gcp-oauth.keys.json is for a deleted/unknown OAuth client.")
        print("Recreate a Desktop OAuth client in the right project, download the JSON, replace --keys.")
    sys.exit(1)

if not tok.get("refresh_token"):
    print("WARNING: no refresh_token returned (Google omits it if the app was already authorized).")
    print("Revoke at https://myaccount.google.com/permissions then re-run.")

out = {
    "access_token": tok["access_token"],
    "refresh_token": tok.get("refresh_token"),
    "scope": tok.get("scope", args.scopes),
    "token_type": tok.get("token_type", "Bearer"),
    "expiry_date": int(time.time() * 1000) + int(tok.get("expires_in", 3600)) * 1000,
}
if "refresh_token_expires_in" in tok:
    out["refresh_token_expires_in"] = tok["refresh_token_expires_in"]

if os.path.exists(args.creds):
    os.rename(args.creds, args.creds + ".bak-" + time.strftime("%Y%m%d-%H%M%S"))
with open(args.creds, "w") as f:
    json.dump(out, f)
os.chmod(args.creds, 0o600)
print("\nSUCCESS — wrote", args.creds)
print("refresh_token present:", bool(out["refresh_token"]))
