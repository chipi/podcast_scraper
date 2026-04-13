# Autoresearch — Track A (paragraph summaries)

Human-maintained instructions for the coding agent. **Do not** let the agent edit this file during a run.

This is the **paragraph summary** research line. For **JSON bullet** tuning, use `program_summary_bullets.md` instead.

## Goal

Improve allowlisted **paragraph** summarization prompts (`.j2`) for the OpenAI experiment config
`data/eval/configs/autoresearch_prompt_openai_smoke_paragraph_v1.yaml`, measured by
`autoresearch/prompt_tuning/eval/score.py` (ROUGE vs silver + dual LLM judges).

Use **reference** `silver_gpt4o_smoke_v1` (paragraph silver, `long_v1` shape).

**Product intent:** Summaries should help a reader grasp the episode — main ideas, accurate
facts, and clear prose — without reading the full transcript. Operationalised through
paragraph summaries scored by ROUGE vs a prose silver reference and dual LLM judges using
`eval/rubric.md`.

**Dimensions to improve (inform hypotheses):**

| Dimension | Plain language | How the loop sees it |
| --- | --- | --- |
| **Coverage** | Core themes and beats from the transcript appear. | ROUGE overlap with silver; judges penalize missing main points. |
| **Fidelity** | No contradictions or invented facts vs. transcript. | Judges penalize hallucinations; ROUGE indirectly rewards alignment. |
| **Conciseness** | Clear, tight prose; no padding or repetition. | Judges; length bounds from experiment `params`. |
| **Format contract** | Valid paragraph structure per template. | Invalid or malformed output fails downstream. |

---

## Allowlisted mutable paths

Edit in this order — exhaust user prompt hypotheses before touching the system prompt:

1. `src/podcast_scraper/prompts/openai/summarization/long_v1.j2` — **user prompt** (primary)
2. `src/podcast_scraper/prompts/openai/summarization/system_v1.j2` — **system prompt** (secondary;
   only if user prompt gains have plateaued after ≥5 experiments)

## Immutable (never agent-edit)

- `autoresearch/prompt_tuning/eval/score.py`
- `autoresearch/prompt_tuning/eval/rubric.md`
- `autoresearch/prompt_tuning/eval/judge_config.yaml`
- `data/eval/**` inputs (sources, datasets, references, baselines)
- `autoresearch/prompt_tuning/program_summary.md` (this file)

---

## Loop

1. **Hypothesis:** one sentence stating which dimension you expect to move and why.
2. Edit **only** allowlisted `.j2` files (user prompt first; system prompt only when
   user prompt plateau is confirmed).
3. From repo root:

   ```bash
   make autoresearch-score \
     CONFIG=data/eval/configs/autoresearch_prompt_openai_smoke_paragraph_v1.yaml \
     REFERENCE=silver_gpt4o_smoke_v1
   ```

   Capture the single float printed to stdout (higher is better).

4. If score improves >1% vs last kept commit:
   `git add` (allowlisted files only) and
   `git commit -m "[autoresearch-summary] exp-<N>: <hypothesis>"`.
5. Else: `git checkout HEAD -- <edited file(s)>`.
6. Append a row to `autoresearch/prompt_tuning/results_summary.tsv` (see header).
7. **NEVER pause to ask the human a question.** If uncertain, make a conservative
   choice, log it in the notes column, and continue.

Use branch `autoresearch`; do not commit on `main` directly.

---

## Stop

- **Hard stop:** 15 experiments per session.
- **Early stop:** 3 consecutive experiments each with ≤1% improvement → stop, write summary.
- On early stop or hard stop: write a summary to
  `autoresearch/prompt_tuning/summary_summary_<YYYY-MM-DD>.md` covering:
  - Total experiments run
  - Best score vs baseline (absolute + % improvement)
  - What helped and what hurt
  - Suggested next directions
  - Any anomalies

---

## Environment

The score script loads **`.env`** from the repo root, then **`.env.autoresearch`** if it
exists (overrides). Put keys in either file; both are gitignored.

- `AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY` — summarization calls
- `AUTORESEARCH_JUDGE_OPENAI_API_KEY` / `AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY` — judges
- `AUTORESEARCH_EVAL_N` — episodes (default 5; smoke dataset has 5 max)
- Optional: `AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1` for local dev

---

## Promising directions to explore

Seeded from what we know about the paragraph format and the silver reference (GPT-4o prose):

- **User prompt — coverage guidance:** explicit instruction to cover central topic, key
  arguments, and conclusions across paragraphs
- **User prompt — paragraph count:** fix at 3–4 vs the current open-ended range to match
  silver reference style
- **User prompt — vocabulary alignment:** "prefer the transcript's own terminology and
  phrasing" (this won on the bullet line — worth testing here)
- **User prompt — opening sentence:** require the first sentence to state the episode's
  central thesis
- **System prompt — persona:** more specific about audience and purpose
  (e.g. "busy professional who needs the gist in 60 seconds")
- **System prompt — output contract:** add format constraints to reinforce paragraph structure

Explore user prompt directions first. Move to system prompt only after ≥5 user prompt
experiments have been run.
