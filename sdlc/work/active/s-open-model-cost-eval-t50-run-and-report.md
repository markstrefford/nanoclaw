---
id: s-open-model-cost-eval-t50-run-and-report
kind: task
project: nanoclaw
status: active
autonomy: attended
parent: s-open-model-cost-eval
created: 2026-06-05
updated: 2026-06-05
---

# t50 — Run Kimi vs Sonnet on the benchmark and report cost-per-task + verdict

## Outcome

Run both models on the benchmark, then report cost per successful task and a quality
verdict, ending in a plain go/no-go on a model switch.

## Acceptance

- Kimi K2.6 and Sonnet 4.6 each run the t20 benchmark through the real agent path,
  on the tool footing fixed in t10.
- For each run: the result is graded against the t20 rubric, and the run's token
  cost is read from the per-model-correct cost path (after t40).
- A short written comparison reports **total cost per successful task** (not
  $/token) and the quality grades side by side, then states a recommendation:
  parity-or-better-and-cheaper (worth a switch decision) or not (close the thread).

## Test specification

No new automated tests — this is an execution-and-analysis task. The "test" is that
the numbers are reproducible: the benchmark prompt, the model ids, the base URL, and
the cost-attribution method (group + time window) are all recorded so the run could
be repeated.

## Implementation notes

Mechanics grounded in what exists:

- Run selection: set the Kimi run via `OPENAI_MODEL` + `OPENAI_API_KEY` +
  `OPENAI_BASE_URL` (t30) in `.env`; the runner auto-routes (index.ts:720). Run the
  Sonnet baseline on the normal Claude path.
- Cost attribution: `token_usage` rows are keyed by `container_name`,
  `group_folder`, and `recorded_at` (db.ts schema ~line 68) — there is no per-run
  id. Read per-model cost from the **per-container** stats path
  (`getTokenUsageByContainer`, which retains `model`), NOT the summary or by-group
  endpoints — those aggregate the model away and stay model-blind after t40. Scope
  each run with a dedicated throwaway group folder and/or a recorded time window.
- "Successful task" = passes the t20 rubric. Cost-per-successful-task = run cost /
  number of passing runs (run the scenario a few times per model if a single run is
  too noisy to grade).
- Output is a written comparison (file alongside the benchmark spec), not code.
  Keep the raw token numbers and the per-model price source in the writeup so the
  cost claim is auditable. If t10 chose option (b)/(c), restate the tool handicap
  beside the verdict so the result isn't over-read.

## Status

active
