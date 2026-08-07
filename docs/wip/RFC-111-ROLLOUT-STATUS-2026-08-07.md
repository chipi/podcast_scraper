# RFC-111 transcript-prefix caching — holistic rollout status (2026-08-07)

Branch `feat/naming-arc-and-corpus-prep`. **3 new commits** (`118536f5` Phase B core, `10bea0e4`
Phase C, `fdb4197e` Phase D) on top of the handover base. **Not pushed.** Gemini (Phase E) is the
only provider not done — it needs a design decision (below).

## What shipped

The cleaned transcript now leads the prompt as a **byte-stable cache block** so providers prefix-cache
it across an episode's LLM stages (summary → GI insights → KG). One shared, provider-family-agnostic
core drives it: `src/podcast_scraper/providers/common/transcript_cache.py`
(`relocate_transcript` + `openai_style_messages` + `anthropic_style_system`). Flag:
`cache_transcript_prefix` (default **on**; off = exact legacy layout, zero behaviour change).

**Byte-stability fix (important):** the transcript is normalised (`strip`) before it becomes the
block, because summary embeds the raw cleaned text (trailing `\n`) while GI/KG pre-strip it — without
normalisation their blocks differ and the cross-stage cache silently splits (real transcripts end in
`\n`, so this bit production). See `test_relocation_normalises_surrounding_whitespace`.

### Per-provider status + LIVE cache evidence

| Provider | Wired | Mechanism | Live cache result |
|---|---|---|---|
| openai | ✅ | layout (base) | RFC probe 99% (not re-run this session) |
| deepseek | ✅ | layout (base) | **isolated 97%; cross-stage summary→GI→KG 0→39→87%** (warms) |
| qwen | ✅ | layout (base) | RFC probe 69% (not re-run) |
| litellm (gateway) | ✅ | layout (base) | RFC probe 96% (deepseek route) |
| vllm | ✅ | layout (base) | backend-dependent (not probed; daemon) |
| grok | ✅ | layout (standalone) | **~3% — xAI barely auto-caches; safe no-op** |
| mistral | ✅ | layout (standalone) | unverified (probe SDK quirk; provider import works) |
| ollama | ✅ | layout (standalone) | backend-dependent (daemon down) |
| anthropic | ✅ | **cache_control block** | **summary writes 10450, GI reads 10450 = 100%** |
| **gemini** | ❌ | needs `cached_content` | implicit cache DEAD even @10k tok — see below |

Scope per provider = **summary + generate_insights + extract_kg_graph** (the three stages that send
the full cleaned transcript once/episode). `score_entailment` is not transcript-bearing (excluded).

### NOT covered (deliberately) — the honest gaps

1. **Bundled + quote stages** (`summarize_*_bundled`, `extract_quotes`, `extract_quotes_bundled`).
   These render the transcript via **shared builders** (`prompting/megabundle.py`,
   `providers/common/evidence_prompts.py`, `bundled_prompts.py`) that **strip/truncate the transcript
   internally**, so provider-level relocation can't match it verbatim (my guard safely falls back to
   legacy). The clean fix is to relocate **inside those shared builders** (they hold the exact
   embedded string) and return a transcript-first `(system, user)` — that covers ALL providers at
   once for those stages. Deferred; not yet done. NOTE the prod profile uses
   `gil_evidence_quote_mode: bundled`, so the quote stage currently does NOT cache.
2. **cleaning** stage — excluded (consumes the raw pre-clean transcript, a different string; runs once).
3. **Gemini** — see Phase E.
4. **Cache-token telemetry for anthropic/gemini** — `token_accounting.extract_token_usage` already
   parses their cache fields, but the anthropic/gemini providers don't forward the response to
   `emit_llm_cost_event` (only the OpenAI-compat base + summary path does, via `record_provider_call_cost(response=)`).
   So anthropic cache savings are real but not yet visible in `llm_cost` telemetry.

## Phase E — gemini (needs a DECISION)

Gemini's OpenAI-compat endpoint caches 0% (RFC) **and** native `generate_content` implicit caching is
dead even at ~10k tokens (probed both `gemini-2.5-flash` and `-flash-lite`: `cached_content_token_count=0`).
So a plain reorder gives gemini **nothing**. The only path is the **explicit `cached_content` API**:

- Create a `CachedContent` handle per episode (POST the transcript + a TTL), then pass
  `config.cached_content=<handle>` on each stage's `generate_content`, and clean the handle up.
- This is **stateful** (unlike every other provider) and has a **cost tradeoff**: creating/storing a
  per-episode handle costs money; it only pays off if the transcript is reused enough across stages
  (it is — 3+ stages) and the handle TTL covers the episode's processing. This is **RFC-111 §8's open
  question**.

**Decision needed:** implement explicit `cached_content` (with a per-episode create→reference→delete
lifecycle), or leave gemini on legacy? Gemini is currently safe on legacy (flag-on doesn't harm it).

## Findings that change the RFC's framing

- **The reorder changes output TEXT.** RFC §1's "caching never changes the answer" is true for the
  *cache*, but RFC-111 also *reorders* the prompt, which **does** change the generated text (temp=0
  isolation: transcript-first summaries are far more different from legacy than two legacy resamples
  are from each other — 3/4 episodes). This is the "heart surgery" risk RFC §4 flagged.
- **Quality is preserved.** Conclusive gate (temp=0, n=10, blind Opus, with a legacy-vs-legacy
  control): mean 7.90 (transcript-first) vs 8.00 (legacy); the win-lean toward legacy is explained by
  the judge's position bias (control forced a 6-2 winner on EQUAL inputs). Both layouts produce good,
  faithful summaries that surface slightly different facts.
- **Phase-1-summary-only gives ~0 single-run cache benefit** (the transcript-first summary prefix
  isn't reused within one run). The win needs the cross-stage set (summary+GI+KG), which is now wired
  → proven 0→39→87% live. Reprocessing (warm cache) approaches the isolated 97%.

## Open decision for the operator

The flag defaults **on** (per RFC). Given it rewrites output text on reprocess and the quote/bundled
stages + gemini aren't covered yet, consider whether to default it **off** until the shared-builder
pass + a broader quality eval land. Quality is preserved in the n=10 gate, but that rewrites every
summary/insight/KG output on the next reprocess.

## Next steps (ordered)

1. **Decide gemini** (`cached_content` vs leave legacy) and the **flag default** (on vs off pending).
2. **Shared-builder relocation** for quote/bundled stages → covers the prod quote path + all providers.
3. **Forward the response** in anthropic/gemini cost paths so cache savings show in `llm_cost`.
4. Re-run the cost proof (`gateway_spend.py` / cached-token telemetry) on a reprocess to quantify the
   real $/episode drop now that summary+GI+KG share one prefix.

## Update — smaller loose ends closed (2026-08-07, later)

1. **Flag default: ON** (decided by operator). `cache_transcript_prefix` stays `default=True` globally;
   the quality gate passed (neutral, deepseek+anthropic × summary+insights). Implication accepted: the
   next reprocess rewrites all outputs (same quality, different wording). Gemini remains its own
   `gemini_context_cache_enabled=False`.
2. **Cache-token telemetry (anthropic + gemini)**: their summarization cost path now forwards the raw
   `response` to `emit_llm_cost_event`, so `token_accounting.extract_token_usage` surfaces the
   cache-read tokens (anthropic `cache_read_input_tokens`, gemini `cached_content_token_count`) in the
   `llm_cost` events. Verified live (anthropic 9000, gemini 90). Guard tests added.
3. **Mega-bundle variants** (`summarize_mega_bundled` / `summarize_extraction_bundled`): the shared
   builders (`prompting/megabundle.py`) now relocate the (internally-truncated) transcript to the
   leading system block behind a `cache_transcript_prefix` param. Wired for the AUTO-CACHE providers
   only (openai/deepseek/qwen/litellm/vllm + grok + mistral) — a single mega-bundle call is a free
   reprocessing win there. Deliberately NOT wired for anthropic (cache-write premium) or gemini
   (storage rent): caching a lone, never-re-read call is a net LOSS on those. Ollama has no
   mega-bundle methods.

Remaining from the original plan: the reprocessing / finale re-derivation run (#2 + #3) — scope TBD.
