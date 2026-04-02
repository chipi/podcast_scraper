# LLM standard defaults: eval, monitoring, evolution (note)

Working note for a **single versioned “standard” LLM stack** aligned with eval runs,
ongoing monitoring, and intentional upgrades—similar in spirit to promoted ML
`ModeConfiguration` / `summary_mode_id`, but for API providers (OpenAI, Gemini, etc.).

## Goal

- One **named default** (conceptually: e.g. `llm_openai_standard_v1`) that defines the
  usual transcription + speaker + summary + **hybrid transcript cleaning** models
  (and related knobs).
- **Eval configs** remain the contract: what you score against is explicit and
  reproducible (`config_sha256`, git SHA, `run_manifest.json`, `metrics.json`).
- **Monitor** a small set of headline metrics (cost/tokens, failures, optional quality
  scores from eval) before bumping the standard.
- **Evolve** by versioning (`_v2`, …) and keeping the previous baseline for regression
  comparison—avoid silent drift.

## Current codebase (relevant pieces)

- **Local ML summarization:** `ModelRegistry` + `ModeConfiguration` (RFC-044);
  `summary_mode_id` and `summary_mode_precedence` merge promoted map/reduce/tokenize
  defaults in `MLProvider`. See `src/podcast_scraper/providers/ml/model_registry.py`.
- **LLM providers:** Per-field `Config` (`openai_summary_model`, `openai_cleaning_model`,
  …) and defaults in `config_constants.py` / `config.py`. No `summary_mode_id` merge
  for OpenAI-style paths today.
- **Manual / eval YAMLs:** e.g. `config/manual/manual_planet_money_openai_*.yaml`,
  `data/eval/configs/*.yaml`.
- **Artifacts:** Per-run `metrics.json`, episode metadata `processing.config_snapshot`
  (includes summarization provider info; hybrid cleaning models attached when
  strategy is `hybrid` / `llm`).

## Possible direction (not implemented)

Introduce an **optional** layer parallel to ML modes—not by overloading
`ModeConfiguration`:

- **`llm_profile_id`** (name TBD) in YAML: resolves to a bundle of provider fields
  for the chosen stack (or a single primary provider), with precedence similar to
  `summary_mode_precedence` (`profile` wins vs `config` wins vs explicit overrides).
- **Registry location:** e.g. small Python table or `config/profiles/llm/*.yaml`
  versioned in git; documented in `docs/api/CONFIGURATION.md`.
- **Eval:** Reference the same profile id in eval configs so “standard” in app ==
  “standard” under test.

## Near-term without new code

- Treat one **canonical YAML** (manual or eval) as **the** standard; document its
  path and bump a **version comment** at the top when anything changes.
- Rely on existing **config hash + metrics** for before/after comparisons.

## Related WIP / docs

- `docs/wip/manual-test-plan-gi-kg.md` (manual runs).
- `docs/api/CONFIGURATION.md` (field reference).

## Open questions

- Should a profile pin **only OpenAI** or **full stack** (transcribe + speaker +
  summary provider mix)?
- How to align **prod** vs **test** defaults (`config_constants`) with a named
  profile without duplicating three sources of truth.
