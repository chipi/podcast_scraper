# ADR-057: AutoResearch Thin Harness with Credential Isolation

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md)

## Context & Problem Statement

The pipeline's LLM prompts and ML inference parameters were set by hand and never
systematically optimized. An automated "ratchet loop" — where a coding agent mutates
one target, scores the result, and commits or reverts — can run 100 variants overnight
versus 5–10 manually. However, this loop must reuse existing evaluation infrastructure
(not fork it), isolate its API spend from production keys, and operate on dedicated
branches to avoid polluting `main`.

The key architectural question: is autoresearch a second pipeline, or a thin control
layer on top of existing code?

## Decision

AutoResearch is a **thin control layer**, not a second pipeline.

1. **Three-file contract**: Each optimization track (prompt tuning, ML params) consists
   of `program.md` (human-only instructions), mutable target(s) (agent edits within an
   allowlist), and `eval/score.py` (immutable scoring harness). The agent reads
   `program.md`, edits only allowlisted files, runs `score.py`, and commits or reverts.
2. **Reuse existing eval infrastructure**: `score.py` is a thin orchestrator that
   imports from `podcast_scraper.evaluation` and shells out to
   `scripts/eval/experiment/run_experiment.py`. No forked ROUGE, WER, prediction I/O, or
   experiment layout under `autoresearch/`.
3. **`score.py` is immutable**: The agent must not edit the scoring script — same
   principle as autoresearch's immutable `prepare.py`. This prevents the optimizer from
   gaming its own metric.
4. **Credential isolation via `AUTORESEARCH_*` env vars**: The harness reads
   `AUTORESEARCH_JUDGE_OPENAI_API_KEY`, `AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY`, etc.
   Bare `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` are not read unless
   `AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1` (default off). This separates spend and
   blast radius.
5. **Dedicated `autoresearch/<tag>` branches**: Each run operates on a date-tagged
   branch (e.g. `autoresearch/20260401-prompts`). No human feature work mixes onto
   these branches. Results merged to `main` only via reviewed PR.
6. **`results.tsv` as append-only log**: Every experiment records `experiment_id`,
   `score`, `delta`, `status` (kept/reverted/crash/contested), judge model IDs, rubric
   hash, and eval dataset ref.

## Rationale

- **Thin layer avoids drift**: If autoresearch forks scoring logic, it will inevitably
  diverge from `podcast_scraper.evaluation`. Reusing the same code means autoresearch
  improvements apply to production evaluation and vice versa.
- **Credential isolation**: Overnight loops consuming API credits must not silently
  drain production keys. Dedicated vars + small-budget cloud projects make spend
  visible and capped.
- **Immutable `score.py`**: The optimizer must not be able to "improve" its score by
  changing the metric. This is a fundamental integrity constraint.
- **Branch isolation**: `git reset --hard HEAD` after a failed experiment never drops
  uncommitted human changes because the branch is agent-only.

## Alternatives Considered

1. **Hyperparameter tuning libraries (Optuna, Ray Tune)**: Rejected for prompt track;
   these optimize numerical search spaces, not open-ended text. May complement the ML
   param track later.
2. **Single judge LLM**: Rejected; single-judge scoring introduces provider bias.
   Dual-judge with divergence detection is more robust (>0.15 divergence → contested).
3. **Shared production API keys**: Rejected; no spend separation, risk of accidental
   overlap with manual runs, cannot set independent budget caps.
4. **Run on cloud GPU**: Deferred; Apple Silicon MPS is sufficient for Track B
   inference parameter tuning at current eval set size.

## Consequences

- **Positive**: Systematic overnight optimization. Reuses existing eval code (no fork
  drift). Spend is isolated and capped. Results are auditable via `results.tsv` and
  git history.
- **Negative**: Requires dedicated API keys and env var setup. Agent-only branches add
  git management overhead. Eval harness must be robust enough to run unattended.
- **Neutral**: `autoresearch/` directory is added to the project root. Branches follow
  `autoresearch/<tag>` naming convention.

## Implementation Notes

- **Directory**: `autoresearch/prompt_tuning/` (Track A), `autoresearch/ml_params/`
  (Track B)
- **Pattern**: Agent reads `program.md` → edits allowlisted `.j2` files or
  `config/autoresearch/ml_params.yaml` → runs `score.py` → commits or reverts →
  appends to `results.tsv`
- **Cost controls**: `AUTORESEARCH_EVAL_N` (default 10 episodes, smoke 5), 50
  experiment hard cap, two-stage judge filter
- **Relationship to RFC-015**: Autoresearch run outputs go under `data/eval/runs/` (or
  `data/eval/autoresearch_runs/`), reusing the existing experiment layout convention

## References

- [RFC-057: AutoResearch Optimization Loop](../rfc/RFC-057-autoresearch-optimization-loop.md)
- [ADR-011: Secure Credential Injection](ADR-011-secure-credential-injection.md)
- [ADR-013: Standalone Experiment Configuration](ADR-013-standalone-experiment-configuration.md)
- [RFC-015: AI Experiment Pipeline](../rfc/RFC-015-ai-experiment-pipeline.md)
