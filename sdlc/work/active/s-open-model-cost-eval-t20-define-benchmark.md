---
id: s-open-model-cost-eval-t20-define-benchmark
kind: task
project: nanoclaw
status: active
autonomy: attended
parent: s-open-model-cost-eval
created: 2026-06-05
updated: 2026-06-05
---

# t20 — Define the house benchmark task and how a run is graded

## Outcome

Define the one end-to-end task NanoClaw judges models against, and how a run is
scored good or bad.

## Acceptance

- A short written benchmark spec exists (in `sdlc/docs/` or alongside the story)
  describing: the single prompt/scenario, what tools the agent is expected to use,
  what a correct/useful result looks like, and a pass/fail (or scored) rubric a
  human can apply by reading the result.
- The scenario stays inside the tool footing decided in t10 — it must not require a
  tool one model can't reach unless t10 chose tool parity.
- The scenario still exercises the multi-step shape the story cares about: some
  research, at least one tool call, a memory/file write, and a user-facing reply.
- The rubric is concrete enough that the same run graded by Mark twice lands the
  same verdict.

## Test specification

No automated tests. This task produces a document, not code. Done = the spec reads
clearly enough that t50 can run it against two models without re-inventing the task
or the grading.

## Implementation notes

Pure authoring task, no code paths touched. Ground the scenario in how NanoClaw
actually runs: a single group prompt that flows through the agent runner
(`container/agent-runner/src/index.ts`) — research, a couple of file/Bash
operations, a write into the group folder, and a final reply. Keep it one
self-contained prompt so it can be replayed identically per model. Depends on t10:
if t10 chose option (b) or (c), the openai-runner path has no WebSearch/WebFetch, so
either keep "research" to fetch-by-Bash/known-URLs or drop web research from the
scenario and note it.

## Status

active
