# Autoresearch — Track A (prompt tuning)

Human-maintained instructions for the coding agent. **Do not** let the agent edit this file during a run.

## Goal

Improve allowlisted **summary-bullet** prompts (`bullets_json_v1.j2` — JSON bullet output)
for the OpenAI experiment config `data/eval/configs/autoresearch_prompt_openai_smoke_v1.yaml`,
measured by `autoresearch/prompt_tuning/eval/score.py` (ROUGE vs silver + dual LLM judges).

**Naming:** The experiment’s `prompts.user` value is `shared/summarization/bullets_json_v1`
(RFC-017), i.e. `prompts/shared/summarization/bullets_json_v1.j2`. The middle segment
`summarization/` is the pipeline task folder, not “paragraph vs bullets”—**`bullets_json_v1`**
identifies the bullet format.

## Research objectives (what “better summary” means)

**Product intent:** Summaries should help a reader grasp the episode—main ideas, accurate
facts, and clear bullets—without reading the full transcript. Track A v1 **operationalizes**
that through one task: **JSON bullet summaries** from the allowlisted template, scored by
ROUGE vs a silver reference plus dual judges using `eval/rubric.md`.

**Dimensions to improve (inform hypotheses; rubric aligns with 1–3):**

| Dimension | Plain language | How the loop sees it (v1) |
| --- | --- | --- |
| **Coverage** | Core themes and beats from the transcript appear. | ROUGE overlap with silver; judges penalize missing main points. |
| **Fidelity** | No contradictions or invented facts vs. transcript. | Judges penalize hallucinations; ROUGE indirectly rewards alignment with reference. |
| **Conciseness** | Tight bullets; little fluff or repetition. | Judges; length bounds come from experiment `params` (not agent-edited in v1). |
| **Format contract** | Valid JSON bullet structure per template. | Invalid output fails downstream; prompt edits should preserve the schema. |

**How to use this in the loop:** Each experiment’s one-line **hypothesis** should say *which*
dimension(s) you expect to move (e.g. “reduce hedge words to improve conciseness without
dropping coverage”). The stdout **scalar is a proxy**, not the full product—use `metrics.json`,
`metrics_report.md`, and spot-checking `predictions.jsonl` when the scalar is flat or
contested.

**Out of scope unless a human expands the allowlist or adds a new experiment YAML:**
different backend/model, non-bullet summary shapes, editing `data/eval/**`, preprocessing
profiles, or GI/KG prompts. Those are valid **research follow-ups**; they need their own
track or RFC-level scope, not ad-hoc edits in this run.

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
