---
id: s-open-model-cost-eval-t10-tool-parity
kind: task
project: nanoclaw
status: done
autonomy: attended
parent: s-open-model-cost-eval
created: 2026-06-05
updated: 2026-06-05
---

# t10 — Decide the footing: tool parity between the two models, or a recorded handicap

## Outcome

Decide up front whether Kimi and Sonnet are judged with the same tools, or record
the gap plainly if we accept it — before the benchmark is even written.

## Acceptance

- A decision is recorded: either Kimi runs with tool capability equivalent to the
  Sonnet baseline, or the capability gap is documented as an explicit caveat that
  travels with the benchmark results.
- The decision states which runner path Kimi uses and which tools are therefore
  available to it.
- The decision is written down where the benchmark spec (t20) and the final report
  (t50) can both reference it.

## Why this is first

The benchmark scenario can't be finalised until this is settled: if the benchmark
needs web research but Kimi runs on a path without web tools, either the benchmark
or the runner choice has to change. Deciding footing first stops t20 being rewritten.

## Test specification

If the chosen option requires a code change (routing Kimi through the fuller
toolset), that change gets unit coverage proportionate to its size — exercise the
runner-selection / proxy-translation path for a non-Claude model. If the decision is
"accept and document the handicap," there is no code and no test; the output is the
recorded caveat.

## Implementation notes

The confound is grounded in the code. Two runner paths exist:

- **OpenAI runner** (`container/agent-runner/src/openai-runner.ts`): a custom loop
  exposing only 5 builtin tools — Bash/Read/Write/Glob/Grep (`BUILTIN_TOOLS`,
  ~line 361) — plus MCP servers. NO WebSearch/WebFetch/Edit/Task/Teams/Skill. A
  Kimi model id auto-selects this path today (`index.ts:720-721`: any model not
  starting with `claude-`).
- **Claude Agent SDK** path: the full toolset (`index.ts` allowedTools, ~457-477),
  incl. WebSearch/WebFetch. The credential proxy contains an Anthropic→OpenAI
  translator (`src/openai-translator.ts`) meant for "when ANTHROPIC_BASE_URL points
  to an OpenAI-compatible endpoint" — i.e. a route to run a non-Claude model through
  the full SDK with translation.

Options, with cost flagged honestly:
- **(a) Route Kimi through the SDK + proxy translation** — apples-to-apples tools,
  but `openai-translator.ts` has never run in production under real tool calls. This
  is a genuine sub-feature to validate, NOT small wiring. Highest fidelity, highest
  effort/risk.
- **(b) Restrict the benchmark to tools both paths share** (no web research) and
  document that limitation.
- **(c) Run Kimi on the openai-runner as-is** and record the tool handicap as a
  caveat on the result.

For a tight spike, (b) or (c) keeps scope small; (a) is the only one that gives a
true like-for-like on a research-heavy task. Decide with Mark before writing t20.

## Decision (2026-06-05) — option (c): openai-runner as-is, handicap recorded

Mark chose the simplest path: run Kimi on the existing OpenAI-compat runner, accept
the reduced toolset, and dogfood it live for a couple of days rather than wire the
SDK+translator path now. Rationale: full-parity wiring is expensive in time and is
the bespoke/fragile change that creates upgrade-retrofit pain; the lean path is
small, additive, and upstreamable (ties to an open ticket — possible contribution).
If the trial gives confidence Kimi is good enough, full tool/MCP wiring is a later,
deliberate piece — not a prerequisite.

### Recorded handicap (travels with t50 results)

Kimi on this path HAS: Bash, Read, Write, Glob, Grep + MCP servers (nanoclaw
messaging; Gmail/Calendar/Notion for the main group).
Kimi on this path LACKS (vs Sonnet's full SDK): WebSearch, WebFetch, Edit, Skills,
agent teams/subagents, TodoWrite, ToolSearch, NotebookEdit.

Consequence for grading: research-heavy tasks are weakened by the missing web tools
(Bash+curl is a partial substitute), NOT necessarily by the model. Do not penalise
Kimi on research-dependent benchmark items as if it were a pure model gap.

### Trial shape

Rather than only a single scripted benchmark, the primary signal is a couple of days
of real assistant use on a Kimi API key, with cost read from the per-model stats
(now correct after t40) and quality judged by Mark's lived experience. t20's scripted
benchmark remains useful as a repeatable secondary check.

## Status

done — 2026-06-05. Footing decided (option c). Tool subset + handicap recorded above
for t20/t50 to reference. No code in this task.
