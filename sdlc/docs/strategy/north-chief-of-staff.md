---
id: north-chief-of-staff
kind: strategy
project: nanoclaw
sources: ["Obsidian/10x Mark/STRATEGIC_WORKFLOW.md v2.1"]
verified-on: 2026-08-11
created: 2026-08-11
updated: 2026-08-11
status: draft — proposed amendment, not yet applied to the vault
---

# North as Chief of Staff

**Purpose.** Turn North from an agent that answers questions into the executor of
`STRATEGIC_WORKFLOW.md`. This document is a proposed **v2.2 amendment** to that
file. It is written as a change proposal so it can be read and argued with before
anything lands in the vault, which stays canonical.

The headline, because it reframes the rest: **the chief of staff spec already
exists.** `STRATEGIC_WORKFLOW.md` v2.1 (10 May 2026) defines the themes, the
card/task model, the Now List cap, the morning brief, the weekly rhythm, state
discipline, and the line between what stays human and what runs without
approval. It contains the sentence *"Claude is the interface. The operator does
not browse the files manually to do work."* That is a chief of staff job
description. It was never given an executor. Nothing below invents a second
system; it names the executor and closes three gaps that stopped the existing
one working.

## Why the system stalled

The evidence, as of 11 August 2026:

| Symptom | Evidence |
|---|---|
| Vault out of date | `work/02 - active/` holds 13 cards against the doc's own maximum of 6 |
| Commitments rotting | North's `open-actions.md` still lists items from the week of 2 June as open |
| Scheduled work dead | Every recurring task last fired 25–26 June; re-armed manually on 10 August |
| North blind to the diary | `container.json` wires Gmail, VOSS and Obsidian only — no calendar, despite `AGENT.md` advertising `mcp__calendar__*` |

The common cause is that **v2.1 is session-triggered**. "End of session: Claude
updates Current State." "Operator asks Claude for the weekly review." Every state
update depends on Mark opening a conversation. On the days he doesn't, nothing
updates and the record drifts from reality. Bringing the vault up to date by hand
fixes today and rots again within weeks.

The correction is an inversion: **North maintains the vault; Mark doesn't.** The
vault stops being reference material North reads and becomes the artefact North
is accountable for.

## The role

A chief of staff owns the operator's **attention and follow-through**. Not the
work — Mark does that. Not the decisions — Mark makes those. The system that
determines what gets attention, and the guarantee that nothing agreed quietly
dies.

Six duties. Three are already specified in v2.1; three are new.

| Duty | Meaning | v2.1 |
|---|---|---|
| **Brief** | Mark never starts a day or walks into a meeting cold | Yes — morning brief |
| **Prioritise** | Hold the Now List to 3–7; when it overflows, force the trade-off rather than absorb it | Yes — hard rules 2 and 3 |
| **Mirror** | Say the uncomfortable thing. *"You've done nothing on Health this week."* | Yes — scheduling notes |
| **Capture commitments** | Every promise made by Mark or to Mark, wherever it was made | **New** — the brief reads calendar, VOSS and cards, never the inbox |
| **Close out on a clock** | Update state daily whether or not a session happened | **New** — "end of session" only fires if Mark shows up |
| **Detect stall** | Surface active work that has quietly stopped moving | **New** — `hold/` is flagged after 2 weeks; nothing watches `active/` |
| **Close the meeting gap** | Ask what came out of any call it has no record of | **New** — see *Meeting coverage* |

## Proposed amendments to STRATEGIC_WORKFLOW.md

### 1. New section — "Who runs this"

> This document is executed by **North**, the chief of staff agent. North owns
> attention and follow-through: what reaches the operator, and whether what was
> agreed actually happened. North does not do the work and does not make the
> decisions — those remain with the operator, per *What stays human*.
>
> The operator is not required to open a session for this document to run. The
> daily and weekly rhythm below is scheduled and executes without being asked.

### 2. Amend "Daily rhythm" — add the inbox to the morning brief

Add to *Claude reads*:

> - Gmail (work account) for anything received since the last brief that creates,
>   changes or discharges a commitment

Add to *Claude surfaces*:

> - Anything in the inbox that needs a decision today, with a draft reply held
>   for approval where a reply is the obvious next step

### 3. Amend "Daily rhythm" — replace "End of session" with a scheduled close-out

The existing *End of session* block stays, but stops being the only trigger.
Add:

> **Daily close-out (scheduled, end of working day):**
> Runs whether or not a session happened that day. North:
> - Updates Current State on any card that moved, from email, calendar and chat
>   evidence rather than from memory
> - Appends History entries for anything completed
> - Records new open loops picked up during the day
> - Asks what came out of any meeting it has no record of (see *Meeting
>   coverage*), batched into the same message rather than pinging on each one
> - Reports what it changed, in one message. If nothing moved, it says so —
>   silence is not a status

### 4. New hard rule — stall detection

> 13. **Anything in `active/` with no movement for 14 days gets flagged.** Not
>     killed, not parked — surfaced, with the question of whether it should move
>     to `hold/` with a blocker or be dropped. The operator decides. Stall that
>     nobody names becomes a backlog nobody trusts.

### 5. Amend "What runs without further approval"

Add:

> - Drafting replies to email and holding them for approval. North never sends.
> - Running the daily close-out and reporting what changed
> - Flagging stalled items in `active/`

*What stays human* is unchanged. Compilation outcomes, the Now List, new cards or
tasks, and moves to `done/` or `hold/` all remain the operator's call.

### 6. Amend "References"

Add:

> - Gmail (work account — commitments in and out)

### 7. Migration history

> - **v2.2 (2026-08-11):** Named North as the executor. Added the inbox as a
>   commitment source in the morning brief, a scheduled daily close-out that runs
>   without a session, stall detection on `active/`, and the meeting-gap question
>   for calls with no transcript. Extended *what runs without approval* to cover
>   drafting replies (never sending).

## What this requires technically

Design decisions already taken, recorded here so the build is unambiguous.

| Item | Decision |
|---|---|
| Work Google account | `mark@reimagined.industries` — Gmail **and** Calendar |
| Credential path | Local OAuth, mirroring Ayah's setup: `~/.north-gmail-mcp` and `~/.north-calendar-mcp`, gateway bypass, mounted read-write |
| Why not OneCLI | OneCLI supports a single Gmail account. Ayah already holds the personal one, so at least one agent must run on local credentials regardless. Running one agent on each path doubles the failure modes for no benefit |
| Google Cloud project | Work account in its own project, separate from Ayah's `iris-personal-497911` |
| OAuth publishing status | **Must be Production, not Testing.** Testing mode caps refresh tokens at 7 days — the cause of Ayah's weekly Gmail drop-out. Left unfixed, work email dies weekly |
| Re-auth tooling | `scripts/gmail-reauth.py` already takes `--account` and `--scopes` and pins `login_hint`; works unchanged for a second account |
| Vault access | Already mounted read-write via `mcpvault` |
| Model | North currently runs Kimi k2.6, chosen on cost. Chief of staff is a judgement role — triage, escalation, tone in drafts. Recommend Claude for this workload, with the cost trade made knowingly |

### Gmail status — resolved 2026-08-11

Both agents read their Gmail fine. So despite OneCLI's vault holding no Google
credential (its secrets are Kimi, VOSS, iCloud CalDAV, Anthropic), North's
`~/.gmail-mcp/credentials.json` carries a working refresh token and `gmail-mcp`
renews directly against Google. The `onecli-managed` client id and the year-5138
`expiry_date` in that file are cosmetic — North is already on the local-OAuth
path in practice.

**Consequence: there is no Gmail migration.** The only missing piece is
Calendar for `mark@reimagined.industries`. Scope shrinks to adding
`~/.north-calendar-mcp` and wiring it into `container.json`. The OAuth
publishing-status warning above still applies to whichever Google Cloud project
issues that calendar grant.

## Meeting coverage — assume it is permanently partial

There is no tool that covers the whole surface, and waiting for one would block
the design indefinitely. Mark runs his own calls through Google Meet, which
transcribes; clients send Zoom invites; client work happens on Teams and
sometimes on a client laptop where nothing of ours can run. Otter and its
equivalents solve the tooling question but not the client question.

So coverage is a spectrum, not a switch:

| Call type | Transcript | Route |
|---|---|---|
| Mark hosts on Google Meet | Yes | Lands in Drive — readable if Drive is wired |
| Client hosts on Zoom / Teams | Sometimes, not ours | Assume none |
| Client laptop | Never | Assume none |

The design consequence is that **a missing transcript must be a tracked gap, not
a silent one.** A real chief of staff isn't in the room either — they ask
afterwards. North does the same:

> **New duty — close the meeting gap.** For any calendar event that ended with
> no transcript and produced no follow-up email, North asks one question in the
> daily close-out: *"You had [event] with [attendees] at [time] — anything come
> out of it?"* A one-line answer converts an unrecorded meeting into captured
> commitments. Unanswered by the next close-out, it is logged as unconfirmed
> rather than dropped.

This makes transcripts an optimisation rather than a dependency: where one
exists, North reads it and doesn't ask; where one doesn't, North asks. Wiring
Google Drive is then worth doing for the Meet leg, but nothing blocks on it.

## Consequences

- Mark stops maintaining the vault. If North is wrong about state, that is a bug
  in North, not a chore for Mark.
- The daily close-out will generate one message a day that did not exist before.
  It is the price of the record staying true.
- Stall detection will surface uncomfortable things — the 13 active cards against
  a limit of 6 is the first conversation it will force.
- Drafts North writes that Mark doesn't use are wasted tokens. That is the
  accepted cost of the Draft tier over Observe.
