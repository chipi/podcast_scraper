# Autoresearch — Track A (prompt tuning)

Human-maintained instructions for the coding agent. **Do not** let the agent edit this file during a run.

## Goal

Improve allowlisted summarization prompts (`.j2`) for the OpenAI experiment config
`data/eval/configs/autoresearch_prompt_openai_smoke_v1.yaml`, measured by
`autoresearch/prompt_tuning/eval/score.py` (ROUGE vs silver + dual LLM judges).

## Allowlisted mutable paths (v1)

- `src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2`
- Optionally add a provider override under `src/podcast_scraper/prompts/openai/summarization/` **if**
  you create that file explicitly (same logical name as in experiment YAML).

## Immutable (never agent-edit)

- `autoresearch/prompt_tuning/eval/score.py`
- `autoresearch/prompt_tuning/eval/rubric.md`
- `autoresearch/prompt_tuning/eval/judge_config.yaml`
- `data/eval/**` inputs (sources, datasets, references, baselines)

## Loop

1. Hypothesis: one sentence.
2. Edit **only** allowlisted `.j2` files.
3. From repo root (same as `make autoresearch-score`):

   ```bash
   make autoresearch-score
   ```

   Or directly:

   ```bash
   python autoresearch/prompt_tuning/eval/score.py
   ```

   Capture the single float printed to stdout (higher is better).

4. If score improves vs last kept commit: `git add` (allowlisted files only) and
   `git commit -m "[autoresearch] exp-<N>: <hypothesis>"`.
5. Else: `git reset --hard HEAD`.
6. Append a row to `autoresearch/prompt_tuning/results.tsv` (see header).

Use branch `autoresearch/<tag>`; do not commit on `main` directly.

## Environment

The score script loads **`.env`** from the repo root, then **`.env.autoresearch`** if it exists (overrides). Put keys in either file; both are gitignored.

- `AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY` — summarization calls inside eval runner
- `AUTORESEARCH_JUDGE_OPENAI_API_KEY` / `AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY` — judges
- `AUTORESEARCH_EVAL_N` — episodes (default 5; smoke dataset has 5 max)
- Optional: `AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1` for local dev (uses `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` from `.env`)

Dry judge-free rescoring (needs existing run output):

```bash
make autoresearch-score DRY_RUN=1
```

## Stop

Cap experiments per session in this file (e.g. 50) and stop when reached.
