# Autoresearch — Track A (paragraph summaries)

Human-maintained instructions for the coding agent. **Do not** let the agent edit this file during a run.

This is the **paragraph** research line. For **JSON bullet** tuning, use `program.md` instead.

## Goal

Improve allowlisted **paragraph** summarization prompts (`.j2`) for the OpenAI experiment config
`data/eval/configs/autoresearch_prompt_openai_smoke_paragraph_v1.yaml`, measured by
`autoresearch/prompt_tuning/eval/score.py` (ROUGE vs silver + dual LLM judges).

Use **reference** `silver_gpt4o_smoke_v1` (paragraph silver, `long_v1` shape).

## Allowlisted mutable paths (paragraph line)

- `src/podcast_scraper/prompts/openai/summarization/long_v1.j2`
- `src/podcast_scraper/prompts/openai/summarization/system_v1.j2` (optional; only if hypothesis targets system prompt)

## Immutable (never agent-edit)

Same as `program.md`: `eval/score.py`, `eval/rubric.md`, `eval/judge_config.yaml`, `data/eval/**` inputs.

## Loop

Same steps as `program.md`: hypothesis → edit allowlisted `.j2` only → `make autoresearch-score` with
`CONFIG` + `REFERENCE` (see `autoresearch/README.md` § Two research lines) → ratchet → `results.tsv`.

## Environment

Same env vars as `program.md` (`AUTORESEARCH_*`, `.env` / `.env.autoresearch`).

## Stop

Cap experiments per session in this file (e.g. 50) and stop when reached.
