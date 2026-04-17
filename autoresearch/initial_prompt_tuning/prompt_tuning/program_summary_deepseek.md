# Autoresearch — Track A (paragraph summaries, DeepSeek)

Human-maintained instructions for the coding agent. **Do not** let the agent edit this file during a run.

This is the **DeepSeek paragraph summary** research line. For OpenAI, see `program_summary.md`.

## Goal

Improve allowlisted **paragraph** summarization prompts (`.j2`) for the DeepSeek experiment config
`data/eval/configs/autoresearch_prompt_deepseek_smoke_paragraph_v1.yaml`, measured by
`autoresearch/prompt_tuning/eval/score.py` (ROUGE vs silver + dual LLM judges).

Use **reference** `silver_gpt4o_smoke_v1` (GPT-4o prose reference, reused across all providers).

**Provider details:**

- Backend: `deepseek`, model `deepseek-chat`
- Env key required: `DEEPSEEK_API_KEY` (loaded from `.env`)

**Product intent:** Summaries should help a reader grasp the episode — main ideas, accurate
facts, and clear prose — without reading the full transcript. Operationalised through
paragraph summaries scored by ROUGE vs a prose silver reference and dual LLM judges using
`eval/rubric.md`.

---

## Allowlisted mutable paths

Edit in this order — exhaust user prompt hypotheses before touching the system prompt:

1. `src/podcast_scraper/prompts/deepseek/summarization/long_v1.j2` — **user prompt** (primary)
2. `src/podcast_scraper/prompts/deepseek/summarization/system_v1.j2` — **system prompt** (secondary;
   only if user prompt gains have plateaued after ≥5 experiments)

## Immutable (never agent-edit)

- `autoresearch/prompt_tuning/eval/score.py`
- `autoresearch/prompt_tuning/eval/rubric.md`
- `autoresearch/prompt_tuning/eval/judge_config.yaml`
- `data/eval/**` inputs (sources, datasets, references, baselines)
- `autoresearch/prompt_tuning/program_summary_deepseek.md` (this file)

---

## Loop

1. **Hypothesis:** one sentence stating which dimension you expect to move and why.
2. Edit **only** allowlisted `.j2` files (user prompt first; system prompt only when
   user prompt plateau is confirmed).
3. From repo root:

   ```bash
   make autoresearch-score \
     CONFIG=data/eval/configs/autoresearch_prompt_deepseek_smoke_paragraph_v1.yaml \
     REFERENCE=silver_gpt4o_smoke_v1
   ```

   Capture the single float printed to stdout (higher is better).

4. If score improves >1% vs last kept commit:
   `git add` (allowlisted files only) and
   `git commit -m "[autoresearch-deepseek] exp-<N>: <hypothesis>"`.
5. Else: `git checkout HEAD -- <edited file(s)>`.
6. Append a row to `autoresearch/prompt_tuning/results_summary_deepseek.tsv` (see header).
7. **NEVER pause to ask the human a question.** If uncertain, make a conservative
   choice, log it in the notes column, and continue.

Use branch `autoresearch`; do not commit on `main` directly.

---

## Stop

- **Hard stop:** 15 experiments per session.
- **Early stop:** 3 consecutive experiments each with ≤1% improvement → stop, write summary.
- On early stop or hard stop: write a summary to
  `autoresearch/prompt_tuning/summary_deepseek_<YYYY-MM-DD>.md` covering:
  - Total experiments run
  - Best score vs baseline (absolute + % improvement)
  - What helped and what hurt
  - Suggested next directions
  - Any anomalies

---

## Environment

The score script loads **`.env`** from the repo root, then **`.env.autoresearch`** if it
exists (overrides). Put keys in either file; both are gitignored.

- `DEEPSEEK_API_KEY` — summarization calls (DeepSeek provider)
- `AUTORESEARCH_JUDGE_OPENAI_API_KEY` / `AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY` — judges
- `AUTORESEARCH_EVAL_N` — episodes (default 5; smoke dataset has 5 max)
- Optional: `AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1` for local dev

---

## Priority directions (seeded from OpenAI wins)

The following changes **won on OpenAI** (+1% threshold each). Try these first — high
probability of transferring to DeepSeek:

1. **Vocabulary alignment** (+45.6% on OpenAI): "Prefer the transcript's own terminology
   and phrasing where it aids clarity." This was the single biggest gain.
2. **Thesis sentence** (+2.0%): "Begin the first paragraph with a sentence that states the
   episode's central thesis or main topic."
3. **Anchor in specifics** (+2.1%): "Anchor each paragraph in specific claims, data points,
   or named entities from the transcript."
4. **Remove generic focus line** (+0.8%): Remove "Focus on key decisions, arguments, and
   lessons learned" — it conflicts with the more specific instructions above.

After these, explore DeepSeek-specific directions:

- **DeepSeek's strengths:** DeepSeek-chat is a strong reasoning model; it may respond well
  to explicit reasoning cues like "first identify the main topic, then cover supporting
  arguments" — though this risks verbose or non-prose output.
- **Tighter negative constraints:** DeepSeek may add editorial interjections; "Do not
  editorialize or add commentary not implied by the transcript" may help fidelity.
- **System prompt:** Test a minimal system prompt vs the current after ≥5 user experiments.
