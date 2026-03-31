# Autoresearch (RFC-057)

Thin automation layer on top of `scripts/eval/run_experiment.py` and
`src/podcast_scraper/evaluation/`.

- **Track A (v1):** `prompt_tuning/` — prompt edits + `eval/score.py` + dual judges.
- **Track B (future):** ML inference params (`config/autoresearch/ml_params.yaml`).

**Full design:** `docs/rfc/RFC-057-autoresearch-optimization-loop.md`  
**Human loop rules (source of truth for caps / policy):** `autoresearch/prompt_tuning/program.md`

---

## Claude Code brief (point the agent here)

Use this section as the **single onboarding context** for Claude Code (or any coding agent). Work from the **repository root** unless a path below is absolute.

### Mission

Iteratively **improve summary-bullet quality** (JSON bullet output from the
`bullets_json_v1` template, not a separate “paragraph summary” prompt) for the eval
experiment `data/eval/configs/autoresearch_prompt_openai_smoke_v1.yaml` by editing **only**
allowlisted Jinja templates. After each edit, run the score harness and **ratchet**:
commit if the scalar **improves**, otherwise `git reset --hard HEAD`.

**Research framing (what “better summary” means, dimensions, proxy vs product):**
`autoresearch/prompt_tuning/program.md` → section *Research objectives*. Use it when
writing hypotheses and when the scalar disagrees with spot checks.

### Score command (from repo root)

```bash
make autoresearch-score
```

Equivalent:

```bash
python autoresearch/prompt_tuning/eval/score.py
```

**Output:** one **float on stdout** (last line) — **higher is better**.

**Logs:** stderr (look for `rougeL_f1`, `judge_mean`, `contested`, `final`).

**Cost / speed:** set `AUTORESEARCH_EVAL_N=1` in the environment or `.env` for a **one-episode** smoke loop while experimenting.

**Dry re-score (no new LLM summarization, no judges):** only after a full run already exists:

```bash
make autoresearch-score DRY_RUN=1
```

Useful to re-check ROUGE from existing `predictions.jsonl`; **not** sufficient to evaluate a **new** prompt — run full `make autoresearch-score` after template changes.

### Scoring semantics (short)

- Metric combines **ROUGE-L vs silver** (`silver_gpt4o_smoke_v1`) with **two LLM judges** (OpenAI + Anthropic), per `autoresearch/prompt_tuning/eval/rubric.md` and `judge_config.yaml`.
- If judges **disagree strongly** (contested), the harness falls back to **ROUGE-only** for the final scalar — check stderr for `contested=True`.

### Allowlisted files you MAY edit

| Path | Role |
| --- | --- |
| `src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2` | **On-disk** template for summary bullets (this is what `prompts.user: shared/summarization/bullets_json_v1` loads) |

To use a provider-specific override later, switch the experiment YAML to
`openai/summarization/bullets_json_v1` and add `prompts/openai/summarization/bullets_json_v1.j2`
(RFC-017: provider path wins over shared).

The `summarization/` folder is the shared **task** namespace; **`bullets_json_v1`** pins the
bullet-JSON contract.

Do **not** edit other prompts or configs unless `program.md` is updated by a human to expand the allowlist.

### Files you must NOT edit

| Path | Reason |
| --- | --- |
| `autoresearch/prompt_tuning/eval/score.py` | Immutable harness |
| `autoresearch/prompt_tuning/eval/rubric.md` | Immutable during a run |
| `autoresearch/prompt_tuning/eval/judge_config.yaml` | Pinned judge models (human-only between runs) |
| `data/eval/**` | Gold inputs, references, datasets — read-only |

### Environment (local)

The score CLI loads **`.env`** then **`.env.autoresearch`** (optional overrides) from the repo root. Required for a **full** run (summarization + judges), unless documented fallbacks apply:

- `AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY` — OpenAI for `run_experiment` summarization
- `AUTORESEARCH_JUDGE_OPENAI_API_KEY` — judge A
- `AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY` — judge B
- Optional: `AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1` — use `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` when dedicated vars unset (local dev only)

See `config/examples/.env.example` for the full list.

### Prerequisites

- Materialized transcripts: `data/eval/materialized/curated_5feeds_smoke_v1/` (if missing: `make dataset-materialize DATASET_ID=curated_5feeds_smoke_v1`).

### Git workflow

1. Use a dedicated branch, e.g. `autoresearch/<short-tag>` — **not** `main`.
2. Before each experiment: state a **one-sentence hypothesis** (in chat or `results.tsv` notes).
3. After `make autoresearch-score`:
   - If scalar **improved** vs the last kept commit: `git add` **only** allowlisted files → `git commit -m "[autoresearch] exp-<N>: <hypothesis>"`.
   - Else: `git reset --hard HEAD` (branch should be agent-only; no mixed human WIP).
4. Append a row to `autoresearch/prompt_tuning/results.tsv` (header already in file).

### Stop condition

Respect the **maximum experiments per session** written in `autoresearch/prompt_tuning/program.md` by the human (e.g. 50). Stop when reached.

### Where outputs land

- Run artifacts: `data/eval/runs/autoresearch_prompt_openai_smoke_v1/` (`predictions.jsonl`, `metrics.json`, `metrics_report.md`).
