# Multi-provider rollout + per-provider Track A plan (#698 follow-up)

> **Status: implementation done; matrix runs pending budget approval per provider.**
> All 7 providers (Gemini, OpenAI, Anthropic, Mistral, DeepSeek, Grok, Ollama)
> now expose ``extract_quotes_bundled`` / ``score_entailment_bundled`` using
> shared prompts (``providers/common/bundled_prompts.py``). The dispatch in
> ``gi/pipeline.py`` is provider-agnostic — adding bundled methods on a
> provider automatically enables matrix participation.

## Phase A — per-provider matrix runs (all use shared prompts)

Same 4-cell pattern as the Gemini champion run, anchored on the same silver
reference (``silver_sonnet46_gi_multiquote_benchmark_v2``) and the same
dataset (``curated_5feeds_benchmark_v2``, 5 episodes).

### Cells per provider

For each of OpenAI, Anthropic, Mistral, DeepSeek, Grok, Ollama:

1. ``baseline_staged_<provider>``: ``gil_evidence_quote_mode: staged``,
   ``gil_evidence_nli_mode: staged``
2. ``bundled_a_only_<provider>``: ``quote_mode: bundled``,
   ``nli_mode: staged``
3. ``bundled_b_only_<provider>``: ``quote_mode: staged``,
   ``nli_mode: bundled``
4. ``bundled_ab_<provider>``: both bundled

= **24 cell runs total** across 6 providers (Gemini already done).

### Cost estimates per provider (4 cells × 5 episodes)

| Provider | Model | Est. cost |
| --- | --- | --- |
| OpenAI | gpt-4o-mini | $1-3 |
| Anthropic | claude-haiku-4-5 | $3-5 |
| Mistral | mistral-small-latest | $1-2 |
| DeepSeek | deepseek-chat | $1-2 |
| Grok | grok-3-fast | $2-3 |
| Ollama | mistral-small3.2 (local) | $0 (slower wall-clock) |

**Total estimated cost: ~$10-15 across cloud providers.**

### Experiment YAML pattern

For each provider, clone ``experiments/baseline_staged.yaml`` and override
``backend.type`` + ``backend.model`` + ``params.gil_evidence_*_mode``. Naming:
``experiments/<cell>_<provider>.yaml`` (e.g. ``baseline_staged_openai.yaml``).

### Champion gates (per provider)

Same as Gemini matrix:

- GI insight coverage vs silver ≥ 75% (≥ 5pp drop from staged baseline acceptable)
- GIL cost reduction ≥ 30% vs staged
- Bundled fallback rate ≤ 20%
- Latency reduction ≥ 30% vs staged

### Scoring

Same two scorers as Gemini matrix:

```bash
# Quality vs silver
PYTHONPATH=. .venv/bin/python autoresearch/gil_evidence_bundling/eval/score_gi_vs_silver.py \
    --run-id <run_id> \
    --silver silver_sonnet46_gi_multiquote_benchmark_v2 \
    --dataset curated_5feeds_benchmark_v2

# Internal cost / latency / fallback-rate
PYTHONPATH=. .venv/bin/python autoresearch/gil_evidence_bundling/eval/score.py \
    --baseline data/eval/runs/baseline_staged_<provider>_v1 \
    --variant  data/eval/runs/<variant>_<provider>_v1
```

Append to ``results.tsv`` with one row per cell. Champion cell per
provider = highest scalar with all gates passing.

---

## Phase B — per-provider Track A autoresearch (prompt tuning)

**Trigger condition:** if any provider's matrix shows:

- Bundled fallback rate > 10% (parsing fragility)
- GI coverage drop > 5pp absolute vs that provider's staged baseline
- Quote count notably below silver's median

Then that provider needs **dedicated prompts** instead of the shared
defaults. Track A is the existing RFC-073 framework for that.

### Track A loop per provider (cost: ~$3-8 per round)

Mirrors ``autoresearch/bundled_prompt_tuning/`` (the OpenAI / Anthropic
summarization Track A). For #698 GIL evidence bundling:

1. **Branch:** ``feat/698-prompt-tuning-<provider>``.
2. **Working files** (mutable, agent-edited):
   - ``src/podcast_scraper/prompts/<provider>/evidence/extract_quotes_bundled/v2.j2``
   - ``src/podcast_scraper/prompts/<provider>/evidence/score_entailment_bundled/v2.j2``
3. **Override mechanism (~5 LOC per provider):** modify each provider's
   ``extract_quotes_bundled`` / ``score_entailment_bundled`` to check for
   provider-specific Jinja template via ``render_prompt(...)``; fall back
   to the shared prompt when no template exists.
4. **program.md** describes the round's hypotheses (e.g. "Anthropic models
   ignore JSON-only constraints; add explicit 'Do not include
   explanations' line").
5. **eval/score.py:** subprocess harness running the provider's matrix
   cell once per round + emitting blended scalar (silver coverage +
   grounded rate + fallback rate). Same shape as
   ``autoresearch/bundled_prompt_tuning/eval/score.py``.
6. **Result accept rule:** +5pp absolute on either coverage OR grounded
   rate vs the previous round's prompt, no regression on the other.

### Estimated cost for Track A per provider

- 3-5 rounds typical (per RFC-073 Track A history)
- ~5 episodes × 4 cells × $0.50-2 per round = $10-40 per provider
- Only run on providers that fail Phase A gates

If all 6 providers passed Phase A (no fallback issues), skip Phase B
entirely and ship as-is.

---

## Phase C — combined results writeup

After Phase A (and any required Phase B):

1. Append all matrix rows to ``autoresearch/gil_evidence_bundling/results.tsv``.
2. Build a per-provider champion table in
   ``docs/guides/eval-reports/EVAL_GIL_BUNDLING_2026_05.md``:
   | Provider | Coverage Δ | Grounded Δ | Latency Δ | Fallback rate | Champion mode |
3. Update ``program.md`` with per-provider conclusions.
4. Recommend per-profile defaults in a follow-up PR (not in #711):
   - ``cloud_thin.yaml`` (Gemini): champion mode from Gemini matrix
   - ``cloud_balanced.yaml`` (Gemini): same
   - Per-provider profile flips when those profiles exist (currently most
     non-Gemini paths use staged via the cloud-API pipeline image)

---

## Out of scope for this PR (#711)

- Per-provider experiment YAMLs (would balloon scaffold; cleaner to add
  them as a separate "matrix execution" PR)
- Provider-specific Jinja templates (Phase B follow-up only)
- Default profile flips (always a separate decision)
- Track A prompt-tuning rounds (only if Phase A reveals a problem)

---

## Phase D — execution order (when operator approves the budget)

Recommended sequence (cheap providers first to validate the pattern):

1. DeepSeek (~$1-2) — cheapest, validates the OpenAI-compat pattern
2. Mistral (~$1-2)
3. OpenAI (~$1-3) — most-deployed alternative to Gemini
4. Grok (~$2-3)
5. Anthropic (~$3-5) — most expensive but high-trust silver-matching
6. Ollama (~$0, slow) — local sanity check

If a cheap provider's bundled cells show >20% fallback rate, fix the
prompt or add provider-specific template before continuing to expensive
providers.

---

## Resume after merge

PR #711 ships:

- 7-provider implementation (this commit, ``0c7c2775``)
- Gemini matrix results (already in ``results.tsv``)

Next up (separate operator-driven action):

1. Merge #711.
2. Spawn ``feat/698-multi-provider-matrix`` branch.
3. Add ``experiments/<cell>_<provider>.yaml`` for the 24 cells.
4. Run them sequentially (DeepSeek first per execution order above).
5. Score each, append to ``results.tsv``, commit.
6. Decide on Phase B per provider based on results.
