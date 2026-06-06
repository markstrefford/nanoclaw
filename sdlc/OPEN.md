# OPEN — deferred judgment queue

Append-only. One line per question, tagged with the artefact it belongs to.

- 2026-06-05 Stats cost endpoints round `estimated_cost_usd` to 2dp — hides sub-cent costs on small runs; drop the rounding or use higher precision? [s-open-model-cost-eval-t40]
- 2026-06-05 No integration test on the /api/stats/containers endpoint shape (only the cost function is unit-tested); add one? [s-open-model-cost-eval-t40]
- 2026-06-05 Access layer for the model bench not locked — OpenRouter vs Ollama Cloud (both OpenAI-compatible, reversible config); pick when starting the trial. [s-open-model-cost-eval]
- 2026-06-05 Document OPENAI_BASE_URL path-stripping in .env example/README: runner appends /v1/chat/completions as an absolute path, so a base with a custom sub-path is silently discarded. [s-open-model-cost-eval-t30]
- 2026-06-05 Contribution-back: discussion #948 (github.com/nanocoai/nanoclaw/discussions/948) asks exactly how to run Kimi K2.5 via Moonshot — unanswered. Our t30+t40 changes answer it; consider a PR/reply once the trial proves it works. [s-open-model-cost-eval]
