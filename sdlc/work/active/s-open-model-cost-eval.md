---
id: s-open-model-cost-eval
kind: story
project: nanoclaw
status: active
autonomy: attended
sources: [raw/eval-open-source-models-kimi.md]
created: 2026-06-05
updated: 2026-06-05
---

# Use-case eval harness + Kimi K2.6 vs Sonnet 4.6 cost trial

## Executive summary

Build a small, repeatable way to judge whether a cheaper model holds the **same
value** as Claude Sonnet 4.6 on real NanoClaw agent work — then use it to answer
one question: can Kimi K2.6 (Moonshot) do what Sonnet does, for less money?

Why now: the token-tracking work on this branch is about to make per-message spend
visible, so the cost side of the trade-off finally has numbers behind it. The
prompt is an external claim (Ferndesk's "Fern bench" found Kimi K2.6 beat every
model they tried at ~6× cheaper than Sonnet) that we want to test against *our*
workload rather than trust on a public benchmark.

What this does NOT change: the production runner stays Sonnet. This story produces
an eval and a measured comparison, not a model switch. Switching is a separate
decision that this story's results would inform.

## The bar (load-bearing)

Success is **parity-or-better quality at lower cost**, not "cheaper." A model that
is cheaper but visibly worse is a **fail** for this story. This is explicitly the
opposite of the abandoned GPT-5.4 nano effort, which was floor-shopping — cheapest
possible model, accept reduced value. The intent does not carry over; only the
non-Anthropic-runner plumbing might.

## Outcome

- A defined "house benchmark": one representative end-to-end NanoClaw agent task
  (research + tool calls + memory write + reply) with a way to grade a run for
  correctness and usefulness.
- A measured head-to-head of Kimi K2.6 against the Sonnet 4.6 baseline on that
  benchmark, reported as **total cost per successful task** (not $/token) alongside
  a quality verdict.
- A clear recommendation: parity-or-better-and-cheaper (worth a switch decision),
  or not (close the thread).

## Out of scope / carry-forward

- Actually switching the production runner — separate decision, gated on results.
- Broad multi-model league tables. Baseline is Sonnet 4.6; Kimi K2.6 is the
  challenger. At most one additional challenger, and only if the harness makes it
  near-free to include.
- A permanent/automated eval suite or CI gate. This is a one-task spike harness;
  generalising it is future work if the first run proves valuable.

## Standing gate

Touches the container runner path — a known, non-trivial integration surface, the
same one the earlier non-Anthropic (OpenAI) runner work explored. How Kimi is
reached (Moonshot API vs OpenAI-compat endpoint vs local) is an open question to
resolve at Plan time against the actual runner code. That earlier work's intent
(floor-shopping for cheapness) must not leak into this story. No downstream
consumers depend on this story's output yet.

## Acceptance

- There is a single agreed task that stands in as NanoClaw's house benchmark, with
  a stated way to judge a run good or bad.
- Kimi K2.6 and Sonnet 4.6 each run that task through the real agent path (not a
  toy harness), and each run is graded for correctness/usefulness (human-assessed
  is fine — automation is not required).
- The comparison reports cost per successful task alongside those quality grades,
  leading to a plain go/no-go recommendation on pursuing a model switch.

## Open questions (for Plan)

- Which single task becomes the house benchmark, and who/what grades it.
- How Kimi K2.6 is reached through the container runner, and how much of the
  abandoned OpenAI-runner work is reusable.
- Whether to include any third model, or keep it strictly Kimi vs Sonnet.

### Detailed implementation
*Populated at Plan time. Source material does not ground this level of detail.*

## Status

Compiled 2026-06-05 from raw/eval-open-source-models-kimi.md. Awaiting Plan.
