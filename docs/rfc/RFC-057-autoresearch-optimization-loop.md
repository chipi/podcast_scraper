# RFC-057: AutoResearch Optimization Loop — Prompt & ML Parameter Tuning

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team
- **Related PRDs**: *(none yet — candidate for PRD-020)*
- **Related RFCs** (reference — analogous patterns):
  - `docs/rfc/RFC-017-prompt-management.md` — Jinja prompt store, shared vs per-provider overrides (**mutable prompt targets must align with this**)
  - `docs/rfc/RFC-015-ai-experiment-pipeline.md` — experiment configs, `data/eval/` runs layout
  - `docs/rfc/RFC-053-adaptive-summarization-routing.md` — summarization pipeline (primary optimization target)
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md` — provider abstraction patterns
  - `docs/rfc/RFC-055-knowledge-graph-layer-core.md` — eval dataset conventions
- **Related Documents**:
  - `data/eval/` — Gold-standard eval dataset (episodes with verified transcripts and summaries)
  - `scripts/eval/run_experiment.py` — experiment runner (predictions → `data/eval/runs/<run_id>/`)
  - `src/podcast_scraper/evaluation/` — scoring (`scorer.py`, `gi_scorer.py`, `kg_scorer.py`, `eval_gi_kg_runtime.py`, experiment config)
  - `src/podcast_scraper/prompts/` — versioned `.j2` templates (see `prompts/shared/README.md`)
  - `docs/api/CONFIGURATION.md` — `summary_prompt_params`, summarization / Whisper fields
  - `docs/architecture/ARCHITECTURE.md` — Provider system and module boundaries
  - `docs/architecture/TESTING_STRATEGY.md` — Existing test pyramid

## Abstract

This RFC proposes an **AutoResearch optimization loop** for `podcast_scraper`, adapting Karpathy's [autoresearch pattern](https://github.com/karpathy/autoresearch) to two distinct optimization targets:

1. **Prompt template tuning** — Iteratively improve LLM provider prompts (summarization, speaker detection, transcription) using expensive frontier models as judges.
2. **ML inference parameter tuning** — Iteratively improve local model inference parameters (Whisper, BART/LED) scored against the same eval dataset.

Both tracks share a common **ratchet loop**: an AI coding agent mutates one target file (or allowlisted set of files), runs a scored eval, commits improvements via git or reverts failures, and repeats autonomously. The human defines the eval harness and walks away.

## Problem Statement

The pipeline's LLM prompts and ML inference parameters were set by hand and have never been systematically optimized. Without an automated loop:

- **Prompt quality** degrades silently across provider updates and new podcast domains.
- **ML inference params** (beam search width, length penalty, silence thresholds) are static defaults, not tuned to this project's content.
- **Manual iteration** is too slow — a human can reasonably run 5–10 prompt variants; an agent can run 100 overnight.

**Use cases:**

1. **Overnight prompt tuning**: Agent rewrites summarization/speaker-detection templates, judges score each variant, best changes committed under `src/podcast_scraper/prompts/` (per RFC-017).
2. **ML param sweep**: Agent mutates Whisper / local summarization inference settings, runs local eval on Apple Silicon (MPS), keeps improvements.
3. **Provider comparison**: Run the loop independently per provider (OpenAI, Gemini, local) to surface which provider + prompt combination scores best on the eval set.

## Goals

1. **Define the three-file contract** (`program.md`, mutable target(s), immutable eval harness) for each optimization track.
2. **Define the eval harness** — scoring script structure, judge LLM selection, rubric format, reproducibility, and cost controls.
3. **Define cost guardrails** — experiment caps, two-stage filtering, episode subset sizing (smoke vs default).
4. **Define the mutable targets** — exact paths and allowlists aligned with the prompt store and `Config`.
5. **Reuse existing eval plumbing** where possible — avoid parallel scoring implementations that drift from `podcast_scraper.evaluation`.
6. **Define git conventions** for autoresearch branches to avoid polluting `main`.
7. **Isolate autoresearch spend** — dedicated credentials and guardrails so overnight loops cannot silently drain production keys (see §Credentials and cost isolation).

## Reuse-First Thin Automation Layer (v1 Non-Negotiable)

Autoresearch is **not** a second product pipeline. It is a **thin control loop** on top of existing code:

- **`autoresearch/<track>/eval/score.py`** only **orchestrates**: CLI / env → invoke **`scripts/eval/run_experiment.py`** and/or **`podcast_scraper.evaluation`** (`ExperimentConfig`, `scorer.score_run`, comparators) → optional judge calls → single scalar on stdout. **Do not** reimplement ROUGE, WER, prediction I/O, or experiment layout under `autoresearch/` except minimal glue.
- **New Python** under `autoresearch/` should stay small (orchestrator, judge HTTP/SDK calls, rubric loading). If logic is generally useful, add it under **`src/podcast_scraper/evaluation/`** (or shared utils) and **import** it from `score.py` so production and autoresearch share one implementation.
- The **coding agent** + **`program.md`** replace the human clicking; they do not replace **`data/eval/`**, **`run_experiment.py`**, or **`Config`**.

## Relationship to Existing Code (Implementation Alignment)

### Prompts (Track A)

- Templates are **Jinja2 (`.j2`)** files loaded via `podcast_scraper.prompts.store`, not free-standing YAML prompt blobs under `config/prompts/`.
- Resolution order: provider-specific path under `src/podcast_scraper/prompts/<provider>/...` overrides `src/podcast_scraper/prompts/shared/...` for the same logical template name (see `prompts/shared/README.md`).
- **Mutable targets** for a given run are an **allowlist** in `autoresearch/prompt_tuning/program.md`, e.g. one or more of:
  - `src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2` (bundled autoresearch experiment YAML uses logical name `shared/summarization/bullets_json_v1`)
  - `src/podcast_scraper/prompts/openai/summarization/bullets_json_v1.j2` (optional provider override; switch YAML to `openai/summarization/bullets_json_v1` to prefer it)
  - Other `.j2` paths for speaker detection / cleaning as scoped by the run.
- **Optional secondary target**: structured knobs already supported by config (e.g. `summary_prompt_params` in YAML config) if the run optimizes parameters passed into templates — document those keys in `program.md` and keep them consistent with `docs/api/CONFIGURATION.md`.

### Eval & scoring

- **Preferred**: `autoresearch/<track>/eval/score.py` is a **thin orchestrator** run from the **repository root** (e.g. `python autoresearch/prompt_tuning/eval/score.py`) that:
  - Imports from `podcast_scraper.evaluation.*` and, where applicable, shells out to or reuses patterns from `scripts/eval/run_experiment.py`.
  - Reads episodes from `data/eval/` and writes/reads predictions under `data/eval/runs/` (or a dedicated `data/eval/autoresearch_runs/` subtree if isolation is needed — decide once in implementation; either way, **never mutate golden inputs** under the curated eval corpus paths).
- **GIL/KG autoresearch (future)**: If the loop later optimizes insight or graph prompts, reuse `eval_gi_kg_runtime.py`, `gi_scorer.py`, and `kg_scorer.py` rather than duplicating metrics.
- **Summarization-only Track A (v1)**: May compose `scorer.score_run` / experiment YAML contracts already used by `scripts/eval/`; extend only when the rubric requires judge LLM logic not present in `scorer.py`.

### ML parameters (Track B)

- Today, many inference knobs live on **`Config`** (e.g. Whisper model/device fields, `map_inference_params` / `reduce_inference_params` dicts — see `src/podcast_scraper/config.py`).
- **Mutable target (v1)**: a single YAML file, e.g. `config/autoresearch/ml_params.yaml`, whose schema is documented in the harness and **maps to `Config` fields** (or to a small loader that builds a partial `Config` overlay). The agent must not edit `config.py` directly.
- If the project later introduces a first-class `config/ml_params.yaml` consumed by the app, align this file with that schema or merge them in one RFC follow-up.

## Credentials and Cost Isolation

**You do not strictly need a second “pair” of accounts.** What you need is **clear separation of spend and blast radius**. Practical stack (combine as needed):

1. **Dedicated environment variables (required for v1 harness)**  
   `score.py` (and any judge helper it imports) should read **autoresearch-prefixed** vars for **every LLM call the harness makes** — at minimum the **judges** and any **cheap filter** model, e.g.:
   - `AUTORESEARCH_JUDGE_OPENAI_API_KEY`
   - `AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY`  
   Optional, if you want summarization-under-test billed separately from day-to-day CLI:
   - `AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY` (or per-provider equivalents)  
   **Do not** read bare `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in the harness **unless** an explicit opt-in exists, e.g. `AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1` for local dev only (default **off**). Document all vars in `config/examples/.env.example`.

2. **Second API keys vs same account**  
   - **Same org, new API key** (many providers allow multiple keys): point autoresearch vars at those keys — dashboards often aggregate by key or project.  
   - **Stronger isolation**: separate **project / workspace** (OpenAI, Anthropic, etc.) with its own keys and **monthly budget / alerts** — still two keys, not necessarily two billing identities.  
   - **Same key as production** only for quick hacks; you lose cost separation and risk accidental overlap with manual runs.

3. **Provider-side controls**  
   Set **budgets, alerts, and hard caps** in the cloud console for the project tied to autoresearch keys.

4. **Harness-side controls** (already in this RFC)  
   `AUTORESEARCH_EVAL_N`, experiment cap in `program.md`, two-stage judge filter, `--dry-run`, and optional **max judge calls / max USD per invocation** implemented once in `score.py` (fail closed).

**Summary:** Prefer **autoresearch-dedicated env vars** + **keys tied to a small-budget cloud project**; a literal “second pair” of keys is the usual way to achieve that, not a separate requirement from (1) and (2).

## Constraints & Assumptions

**Constraints:**

- **Must not** modify curated gold inputs under `data/eval/` during a run — eval inputs are immutable ground truth.
- **Must not** allow the agent to edit the scoring script (`eval/score.py` or equivalent) — same principle as autoresearch's immutable `prepare.py`.
- **Must** run within a fixed per-experiment budget (API cost cap for prompt track; wall-clock time cap for ML track).
- **Must** operate on a dedicated `autoresearch/<tag>` branch, never directly on `main`.
- Prompt track **must** be runnable without GPU (API-only, laptop or CI).
- ML track **must** be runnable on Apple Silicon via MPS (no CUDA required).

**Assumptions:**

- `data/eval/` contains a representative set of ≥10 episodes with verified transcripts and reference summaries.
- Claude Code (or equivalent coding agent) is the loop driver, pointed at `program.md`.
- Autoresearch LLM calls use **dedicated** `AUTORESEARCH_*` API keys (see §Credentials and cost isolation), documented in `config/examples/.env.example`.
- Per-experiment cost for prompt track is ~$0.10–0.20 at 10 eval episodes; total overnight run ~$15–30 (scales with episode count and judge models).

## Design & Implementation (High Level)

### 1. Three-File Contract

Each optimization track instantiates the same contract:

| File | Role | Editable by agent? |
| --- | --- | --- |
| `autoresearch/<track>/program.md` | Agent instructions, loop rules, stopping criteria, **allowlisted mutable paths** | Human only |
| Mutable target(s) | `.j2` templates and/or `config/autoresearch/ml_params.yaml` (see §Relationship to Existing Code) | Agent only (within allowlist) |
| `autoresearch/<track>/eval/score.py` | Immutable scoring harness | Never |

The agent reads `program.md`, mutates only allowlisted targets, runs `score.py`, reads the score, commits or reverts, logs to `results.tsv`, and repeats.

### 2. Track A — Prompt Template Tuning

**Mutable target**: Allowlisted `src/podcast_scraper/prompts/**/*.j2` files (and optionally config keys documented in `program.md`), per §Relationship to Existing Code.

**Eval harness** (`score.py`) flow:

1. Load `data/eval/` episodes (transcripts + reference summaries) per existing eval conventions.
2. Run the pipeline or summarization provider using current templates (e.g. via experiment config + `run_experiment.py` or direct provider calls — choose the smallest integration that stays representative).
3. Call judge LLM with a fixed rubric (see §Rubric) for each output.
4. Aggregate scores → single float (0.0–1.0). Higher is better.
5. Print score to stdout. Exit non-zero on failure.

**Judge selection**: Two judges to reduce bias (e.g. OpenAI and Anthropic flagship chat models) with averaged scores. **Pin API model identifiers** (e.g. `gpt-4o-2024-08-06`, `claude-3-5-sonnet-20241022`) in `autoresearch/prompt_tuning/eval/judge_config.yaml` (human-only, immutable during a run) and log the exact strings used on each experiment row in `results.tsv`. If scores diverge by >0.15, log as `contested` and treat as no-improvement (conservative).

**Reproducibility**: Same pinned models + rubric version + `data/eval/` dataset id (commit SHA or manifest hash) should be recorded per run so scores are comparable across days.

**Cost controls**:

- **Default eval subset: 10 episodes** (configurable via `AUTORESEARCH_EVAL_N`). **Smoke / first loop: 5 episodes** — same mechanism, smaller N; see Rollout.
- Hard cap: **50 experiments** per run (set in `program.md`; agent must stop).
- Two-stage filter: run cheap judge first; only call expensive judges if cheap score improves over baseline by ≥0.02.
- Pre-cache all transcript inputs before the loop starts — only prompt outputs vary per experiment.

### 3. Track B — ML Inference Parameter Tuning

**Mutable target**: `config/autoresearch/ml_params.yaml` (maps to `Config` inference-related fields), per §Relationship to Existing Code.

**Eval harness** (`score.py`) flow:

1. Load `data/eval/` episodes (raw audio + reference transcripts for Whisper; transcripts + reference summaries for BART/LED).
2. Run inference with current params on MPS.
3. Score: Word Error Rate (WER) for transcription; ROUGE-L + optional LLM judge for summarization.
4. Combine into a **single scalar** using a **fixed formula** documented in `program.md` and implemented once in `score.py`, e.g.  
   `score = w_wer * (1 - norm_wer) + w_rouge * norm_rouge + w_judge * norm_judge`  
   with weights summing to 1 and normalization chosen so each term is in [0, 1] (document `norm_wer` / baseline WER for division). **Constraint pass (recommended)**: reject any candidate that regresses WER vs. baseline by more than a small ε even if ROUGE improves — encode as hard filter or penalty term.
5. Print score to stdout.

**Episode subsets**: Use **5 episodes** for smoke / speed validation (same `AUTORESEARCH_EVAL_N` as Track A). Scale to **10+** once wall-clock per experiment is acceptable.

**Time controls**:

- Fixed wall-clock budget per experiment: **10 minutes** on Apple Silicon (vs. 5 min in original autoresearch on H100 — adjusted for MPS).
- Kill and mark as failure if exceeded.

### 4. Rubric (Prompt Track Judge)

The scoring rubric is defined in `autoresearch/prompt_tuning/eval/rubric.md` and passed verbatim to the judge LLM. It must not change during a run. Example dimensions (finalize before first run):

| Dimension | Weight | Description |
| --- | --- | --- |
| Topic coverage | 30% | Does the summary mention the main topics discussed? |
| Factual accuracy | 30% | No hallucinated names, dates, or claims vs. transcript |
| Speaker attribution | 20% | Speakers correctly identified where detectable |
| Conciseness | 20% | No padding, appropriate length for episode duration |

Rubric changes require a new `autoresearch/<tag>` branch — never mid-run.

### 5. Git Conventions

- Branch: `autoresearch/<tag>` where `<tag>` is date-based (e.g. `20260401-prompts`, `20260401-ml`). **Branch is agent-only until the PR**: do not mix human feature work on the same branch, so `git reset --hard HEAD` after a failed experiment never drops uncommitted human changes.
- `results.tsv` committed per experiment. **Minimum columns**: `experiment_id`, `score`, `delta`, `status` (`kept` / `reverted` / `crash` / `contested`), `notes`, `judge_a_model`, `judge_b_model`, `rubric_hash`, `eval_dataset_ref` (git SHA or manifest id).
- Successful experiments: `git commit` with message `[autoresearch] exp-<N>: <one-line hypothesis>`.
- Failed experiments: `git reset --hard HEAD` — no commit.
- After a run: open a PR from the branch for human review before merging to `main`.

### 6. Directory Layout

```text
autoresearch/
  prompt_tuning/
    program.md                 # Agent instructions (Track A) + allowlisted paths
    eval/
      score.py                 # Immutable harness (thin wrapper → podcast_scraper.evaluation)
      rubric.md                # Judge rubric (immutable during run)
      judge_config.yaml        # Pinned judge model IDs (human-only, immutable during run)
    results.tsv
  ml_params/
    program.md                 # Agent instructions (Track B) + scalar formula / weights
    eval/
      score.py                 # Immutable harness
    results.tsv
src/podcast_scraper/prompts/
  shared/ ...                  # Shared .j2 (mutable only if allowlisted)
  <provider>/ ...              # Provider overrides (typical Track A target)
config/
  autoresearch/
    ml_params.yaml             # Mutable target (Track B)
data/
  eval/                        # Immutable gold dataset (inputs)
  eval/runs/                   # Predictions / run outputs (existing convention; see RFC-015)
```

## Cost Model

| Phase | Track | Estimated Cost |
| --- | --- | --- |
| Eval harness development (manual iteration) | Prompt | $20–40 one-time |
| Single overnight run (50 experiments, 10 episodes) | Prompt | $15–30 |
| Additional runs | Prompt | $15–30 each |
| Single overnight run (50 experiments, 5 episodes) | ML | ~$0 (local compute only) |

**Total budget to reach stable optimized prompts**: ~$80–120 across 3–4 runs.

## Testing Strategy

- **Harness tests**: Unit test `score.py` with fixture outputs to verify scoring logic before any agent run; place tests under `tests/unit/...` following the existing layout (e.g. `tests/unit/autoresearch/` or next to evaluation tests).
- **Dry-run mode**: `score.py --dry-run` prints score for current baseline without calling judge — validates setup is working.
- **Baseline lock**: First experiment always runs against unmodified target to establish `baseline_score` in `results.tsv`. Agent must not proceed if baseline run fails.
- **Regression guard**: If final merged prompts regress on the existing `tests/` suite, revert before merging to `main`.

## Rollout

1. Implement Track A (prompt tuning) first — lower cost, faster feedback loop.
2. Add `judge_config.yaml` with pinned models; document them in `results.tsv`.
3. Validate eval harness independently with `--dry-run` before first agent run.
4. Run Track A on a **5-episode** subset (`AUTORESEARCH_EVAL_N=5`), review `results.tsv`, confirm loop is behaving.
5. Scale to **10-episode** default for overnight run.
6. Implement Track B (ML params) after Track A prompts are stable.
7. Document both tracks in `docs/guides/AUTORESEARCH_GUIDE.md`.

## Alternatives Considered

1. **Manual prompt iteration** — Rejected: too slow, not reproducible, no structured logging of what was tried.
2. **Hyperparameter tuning libraries (Optuna, Ray Tune)** — Rejected for prompt track: these optimize numerical search spaces, not open-ended text. For ML params, may revisit as a complement to the agent loop.
3. **Single judge LLM** — Rejected: single-judge scoring introduces provider bias; dual-judge with divergence detection is more robust.
4. **Run on cloud GPU** — Deferred: Apple Silicon (MPS) is sufficient for Track B inference param tuning. Revisit if eval set grows beyond ~20 episodes or weight fine-tuning is added.
5. **YAML prompt files under `config/prompts/`** — Rejected for Track A v1: conflicts with RFC-017 and the existing `.j2` prompt store; would require a parallel loading path.

## Track A v1 Implementation Checklist

Use this as the **minimal** first PR scope; extend only after one green agent smoke run.

**Decisions (record in `program.md` or team notes)**

- [ ] **Run output root**: reuse `data/eval/runs/` vs `data/eval/autoresearch_runs/` (pick one; avoid touching gold inputs).
- [ ] **Summarization keys**: production keys for candidate summaries vs `AUTORESEARCH_EXPERIMENT_*` only (see §Credentials).
- [ ] **First provider + template**: e.g. OpenAI + one allowlisted `.j2` path only.

**Repository layout**

- [ ] Add `autoresearch/prompt_tuning/program.md` (allowlist, experiment cap, how to run `score.py`, git rules).
- [ ] Add `autoresearch/prompt_tuning/eval/score.py` — thin orchestrator; **calls existing** `run_experiment.py` / `podcast_scraper.evaluation` (no forked metrics).
- [ ] Add `autoresearch/prompt_tuning/eval/rubric.md` and `judge_config.yaml` (pinned model IDs).
- [ ] Add `autoresearch/prompt_tuning/results.tsv` (seed header row) or document append-only creation on first run.
- [ ] Add `config/examples/.env.example` entries for all `AUTORESEARCH_*` vars used by the harness.

**Behavior**

- [ ] `score.py` implements `--dry-run` (no judge calls; exercise experiment + metric path).
- [ ] Baseline row required before agent continues (per Testing Strategy).
- [ ] `AUTORESEARCH_EVAL_N` respected; default smoke = 5, documented default full = 10.
- [ ] Judge calls use **only** `AUTORESEARCH_JUDGE_*` keys unless `AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1`.
- [ ] Each result line logs `judge_*_model`, `rubric_hash`, `eval_dataset_ref`.

**Quality gates**

- [ ] Unit tests for scoring aggregation / contested logic with fixtures (no live API).
- [ ] `make lint` / tests green for new modules.
- [ ] Optional: `docs/guides/AUTORESEARCH_GUIDE.md` stub pointing at this RFC (full guide can follow Rollout step 7).

## References

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — Original pattern
- [autoresearch-mlx fork](https://github.com/trevin-creator/autoresearch-mlx) — Apple Silicon adaptation
- [RFC-017: Prompt Management](RFC-017-prompt-management.md)
- [RFC-015: AI Experiment Pipeline](RFC-015-ai-experiment-pipeline.md)
- [RFC-053: Adaptive Summarization Routing](RFC-053-adaptive-summarization-routing.md)
- [RFC-049: GIL Core](RFC-049-grounded-insight-layer-core.md)
- `data/eval/` — Gold eval dataset
