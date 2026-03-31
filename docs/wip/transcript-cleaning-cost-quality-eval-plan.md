# Transcript cleaning: cost, quality, and debuggability (WIP)

## Context

For LLM summarization providers we support `transcript_cleaning_strategy`: `pattern`, `llm`, and `hybrid` (default). Hybrid is meant to add **semantic** cleaning only when pattern-based pass looks insufficient—extra API traffic and cost when that leg runs.

A reasonable alternative is to **avoid a separate cleaning call** and instead strengthen **summarization** instructions (“ignore sponsors, ads, housekeeping; focus on substantive content”). That saves calls and latency but keeps **full noisy text** in the summarization context, which can still affect bullets, quotes, and chunking behavior.

We should treat this as an **empirical** tradeoff, not a one-line product decision.

## Dimensions to balance

| Dimension | Pattern-only / prompt-only | Separate LLM cleaning (hybrid/llm) |
| --------- | -------------------------- | ----------------------------------- |
| **Cost** | Fewer calls; same transcript tokens on summarize | Extra call(s) when LLM cleaning runs; possible token reduction if transcript shortens |
| **Quality** | Depends on model + prompt; junk still in context | Can improve fidelity of text **before** summarize |
| **Debuggability** | Harder to inspect “what the model saw” vs raw RSS/transcribe | Saved `.cleaned` transcript + clear separation of objectives |

Goal: find a **default** and **documented knobs** that match our feeds and budgets, with evidence from runs.

## Proposed experiment arms

Run the **same frozen dataset** (same `dataset_id`, same transcripts) under fixed model settings; vary only cleaning / prompt behavior.

1. **Arm A — Minimal cleaning traffic:** `transcript_cleaning_strategy: pattern` only. No hybrid LLM cleaning leg.
2. **Arm B — Current default:** `transcript_cleaning_strategy: hybrid` (today’s behavior).
3. **Arm C — Prompt emphasis (no structural code change if possible):** Same as Arm A **plus** an explicit summarization instruction variant that stresses ignoring non-substantive segments (may require a dedicated `summary_prompt` or a small prompt-template variant for the experiment—define in eval config notes).

Optional later arm:

4. **Arm D — Full LLM clean:** `transcript_cleaning_strategy: llm` for comparison upper bound on cleaning cost.

## What to measure

- **Cost / traffic:** `llm_cleaning_calls`, cleaning vs summarization token totals from run metrics; wall time where relevant.
- **Quality:** Use the **summarization** path in `data/eval` (same task type, same references) so scores are comparable across arms—automated metrics plus optional human spot-check on a small slice.
- **Stability:** Failure rate, empty summaries, schema degradation counts if applicable.

## Role of `data/eval`

The eval layout is built for **comparable runs** on immutable datasets and configs (`data/eval/README.md`). This experiment fits there:

- One **experiment config per arm** (or one config with a documented parameter matrix), same `dataset_id`.
- Outputs under `data/eval/runs/` (or materialized baselines) with run IDs and metrics JSON for side-by-side comparison.
- Avoid changing transcripts in `sources/`; keep **inputs identical** so differences attribute to cleaning/prompt only.

## Opinion (working)

Searching for balance **via controlled experiments** is the right approach. Prompt-only cleanup is often “good enough” for cost-sensitive pipelines; hybrid earns its keep when we can show **measurable** quality or robustness gains that justify extra tokens. Using `data/eval` keeps the comparison **scientific** (same episodes, same references, frozen configs) instead of anecdotal A/B on one manual run.

## Next steps (when prioritized)

1. Pick a **small frozen dataset** (e.g. existing curated smoke/benchmark) and one LLM summary provider config.
2. Add eval config YAML entries (or documented CLI/config overrides) for Arms A–C.
3. Run `materialize` / experiment runner per project docs; archive metrics + cost rollups.
4. Write a one-page **conclusion**: recommended default for manual configs vs high-quality configs; link from `GROUNDED_INSIGHTS_GUIDE` or summarization docs only if we promote this out of WIP.
