---
name: link-clipper
description: >-
  Resolve and archive URLs the user sends in. Use whenever an incoming message
  contains a URL (http/https link) — a web article, YouTube/video link, or an
  Instagram/LinkedIn/X/TikTok share. Fetch what the link actually contains,
  save it as a note in the vault's raw/links/ folder, and respond on the
  content. Not every URL is scrapable — degrade gracefully, never fail silently.
---

# Link clipper

When a message you receive contains a URL, treat the link as something to
**resolve into content**, not just a string to acknowledge. Extract what it
points to, file it into the vault, and answer based on what you found.

## Where it lands

The Obsidian vault is mounted at `/workspace/extra/obsidian-vault`. Save each
resolved link as a Markdown note in `raw/links/` there:

```
/workspace/extra/obsidian-vault/raw/links/<YYYY-MM-DD_HHMM>-<short-title>.md
```

Use a sortable timestamp prefix and a short kebab-case slug of the title.
Frontmatter, then the content:

```markdown
---
url: https://www.youtube.com/watch?v=...
source: youtube            # youtube | web | instagram | linkedin | x | tiktok | other
title: <resolved title>
author: <channel / site / handle, if known>
fetched: 2026-05-31T18:30:00Z
status: ok                 # ok | partial | link-only
---

# <title>

<the extracted article text, or the full transcript, or a summary>
```

## How to resolve — pick by source

Detect the kind of URL first, then use the matching method. **Order matters:
try the cheap/clean method before the heavy one.**

### Plain web pages / articles
Use your **WebFetch** tool. It returns readable page content. Save the article
text (or a faithful summary of a long one) to the note. `source: web`.

### YouTube (and other video links)
WebFetch is useless here — the page is a JS shell. Use **`yt-dlp`** (installed
in the container) to pull the **transcript** and metadata. No download of the
video itself.

```bash
# Metadata as JSON (title, uploader, duration, description)
yt-dlp -J --no-warnings "<url>"

# Auto-captions / subtitles → a .vtt/.srt file you then read and clean up
yt-dlp --skip-download --write-auto-subs --write-subs --sub-langs "en.*" \
       --convert-subs srt -o "/tmp/clip.%(ext)s" "<url>"
```

Read the produced subtitle file, strip the timestamps, and save the clean
transcript as the note body. `source: youtube`. If there are **no captions at
all**, save the metadata + description and set `status: partial`.

### Instagram / TikTok / X / LinkedIn (walled platforms)
These are login-walled and anti-scraping. Try **`yt-dlp -J`** first — it
resolves captions/metadata for many Instagram, TikTok and X posts. If that
returns useful text (caption, title, uploader), save it (`status: ok` or
`partial`). LinkedIn and private posts usually **can't** be extracted — when
everything fails, **do not invent content**:

- Save a `link-only` note: the URL, whatever title/preview you have, and a line
  noting it couldn't be auto-extracted.
- Tell the user plainly in your reply, e.g. *"That LinkedIn post is behind a
  login wall — I've saved the link to `raw/links/` but couldn't pull the text.
  Paste it in and I'll work with it."*

## Rules

- **Never fabricate** a transcript or article body. If you didn't actually
  fetch it, say so and store a `link-only` note.
- **One note per link.** Multiple links in one message → one note each.
- Clean up `/tmp` artifacts (subtitle files) after reading them.
- Keep your chat reply about the *content* (the answer, the summary, the
  takeaway) — not a play-by-play of which tool you ran.
- The vault is shared with the other agent and synced; write clean notes.
