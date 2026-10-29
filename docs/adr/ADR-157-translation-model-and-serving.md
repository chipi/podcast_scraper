# ADR-157: TranslateGemma-12B, served co-resident, called through the completions route

- **Status**: Accepted — but the MODEL CHOICE is PROVISIONAL, see the amendment below
- **Date**: 2026-09-29 (amended 2026-10-02, 2026-10-09)
- **Authors**: Marko Dragoljevic
- **Issues**: [#2169](https://github.com/chipi/podcast_scraper/issues/2169) (epic),
  [#2186](https://github.com/chipi/podcast_scraper/issues/2186) (V.6a)
- **See Also**: [ADR-122](ADR-122-self-hosted-model-resilience-policy.md),
  [ADR-155](ADR-155-pin-every-model-checkpoint.md),
  [MULTILINGUAL_ARC](../architecture/MULTILINGUAL_ARC.md) §6.1

> **RENUMBERED 156 → 157 (2026-09-30).** This was authored as ADR-156 while `main` independently
> took that number for `ADR-156-topics-come-only-from-the-extractor.md`. Because the two
> filenames differ, git merges them with **zero conflict markers** and the repo ends up with two
> ADR-156s, each cited by different code. Commit messages elsewhere still say "ADR-156" and cannot be rewritten — they mean this document.
> `scripts/tools/check_doc_structure.py` now compares numbered docs against `origin/main`, which
> is the only place the collision is visible: every same-branch check passed on both sides.

## Context & Problem Statement

The multilingual arc needs a translation model: source-language transcript in, English out, so
every existing intelligence layer keeps reading one language. Gate V (V.3) is the decision point —
which model, served how, and called through which API.

This ADR records what was **deployed and measured**, not what was shortlisted. The shortlist work
is §6.1 of the arc notes; three of its assumptions turned out to be wrong in ways only a running
model revealed.

> **AMENDMENT 2026-10-02 (operator): the model choice here is a PRELIMINARY BEST BET, not a settled
> decision.** One model was picked so the arc could be built at all, and it works — but **no
> alternative has ever been run.** "Alternatives considered" below reasons from published scores,
> licences and serving shape; none of those candidates was benchmarked against this one on real
> episodes. The throughput figure that would have informed the comparison (4.3 tok/s) was taken on a
> heavily loaded GPU host, so it measures contention rather than capacity and cannot settle anything either.
>
> So read §1 as "this is what is deployed and it is good enough to build on", not "this is the
> model". The cross-model quality-and-throughput benchmark on an idle GPU host is **V.3b** in the arc,
> and it may replace this pick. Everything else in this ADR — the co-resident serving decision, the
> `/v1/completions` contract, the memory sizing, the Gemma licence position — stands on its own
> measurements and is unaffected.

## Decision

### 1. `google/translategemma-12b-it`, revision `d1b225e1caa1…`

The 12B over the 27B on measured WMT24++ numbers: MetricX **3.60** / Comet22 **83.5** against the
27B's 3.09 / 84.4 — and it beats *base* Gemma-3-27B (4.04), so the fine-tune is worth more than the
size step. The step up to the 27B buys 0.51 / 0.9 where the 4B→12B step bought 1.72 / 3.4: steep
diminishing returns.

> **CORRECTED 2026-09-30.** This paragraph read "Decisively, the 12B **co-resides with the 30B
> summary model** at bf16 and the 27B would not." That is a claim D-31 had already **retracted as
> wrong** two days earlier: ample memory was measured available on the GPU host with everything loaded, and
> the served LLM is FP4 at ~15–18 GB rather than bf16, so a 27B would co-reside. Writing this ADR
> from the shortlist reintroduced a claim the arc notes had withdrawn — and "decisively" did the
> damage, promoting a memory constraint that does not exist to the deciding factor. The real
> reason is quality-per-throughput plus leaving ample headroom for D-31's shared-service
> ambition; the 27B upgrade path stays open, and an FP8 27B sits at roughly a bf16 12B's
> footprint.

Revision pinned per [ADR-155](ADR-155-pin-every-model-checkpoint.md).

**The model id in the shortlist was wrong.** `google/translate-gemma-2-12b-it` 404s; the real id is
`google/translategemma-12b-it` — one word. It is `gated: manual`, and before the terms are accepted
the HF API returns **200 on metadata and 403 on files**, which presents as a boot hang rather than
an auth failure.

### 2. Served by its own endpoint, alongside the summary model

It is served by its own endpoint alongside the summary model; how and where is the deployment's
concern. Translation runs immediately before summary inside `generate_episode_metadata`, so one
episode needs both models at once and swapping per episode (two model loads per episode) is not a
usable shape.

### 3. Called through `/v1/completions`, NOT the chat route

`/v1/chat/completions` is unusable. It rejects even the exact structured content the model's own
`chat_template.jinja` documents (`content=[{type, source_lang_code, target_lang_code, text}]`):
vLLM transforms the content list before the template sees it, and the template's
`content | length != 1` guard then fires. The client renders the prompt itself:

```text
<start_of_turn>user
You are a professional {source_lang} ({src_code}) to {target_lang} ({tgt_code}) translator. Your
goal is to accurately convey the meaning and nuance of the original text.

{text}<end_of_turn>
<start_of_turn>model
```

The language-name map it needs (`es` → `Spanish`, plus hundreds of regional subtags) lives in
`chat_template.jinja` inside the model snapshot.

### 4. Sized to fit next to the summary model, `max-model-len` 4096

The memory fraction the translator claims must cover **weights plus KV cache** and fit within GPU
memory next to the summary model. A budget that left almost nothing for cache after the weights was
refused by vLLM (`No available memory for the cache blocks`), so sizing is weights plus cache, never
weights alone. The measured workload is far below any cache that fits: 141 production units,
serial, total tokens p95 141, max 433, so `max-model-len` 4096 is ample.

The reasoning error worth preserving: a co-resident stack is *allowed* a large fraction but may hold
far less. The fraction is a ceiling a stack may claim, not what it occupies — which is what makes
co-residency possible, and is knowable only by measuring.

## Evidence

es→en on the V.6a fixture:

> **in** — `Maya: Bienvenidos de nuevo a Sesiones de Sendero. … Maya: Este episodio es patrocinado
> por Strava. Comienza en strava.com/podcast.`
>
> **out** — `Maya: Welcome back to Trail Sessions. … Maya: This episode is sponsored by Strava.
> Visit strava.com/podcast to get started.`

Three consequences, each now evidence rather than assumption:

1. **Speaker labels survive verbatim** — S2.6's design (carry the label onto the English line,
   never through the translator) is compatible with how the model behaves.
2. **The English render is ad-detectable**: that output contains `sponsored by` **and**
   `visit strava.com` — two `_AD_PATTERNS` hits, against **zero** on the Spanish source. The
   ad-excision-after-translation ordering is demonstrated end to end.
3. **It translates the show title** (`Sesiones de Sendero` → `Trail Sessions`), which is why S2.4
   now carries an explicit title decision — drifting into it renames every show.

**Throughput is not yet known.** 4.3 tok/s (74 tokens in 17.1 s), measured while the GPU host was
under heavy production load. That is contention, not capacity. S2.10 needs a quiet host.

### 5. The model's input context is 2K, and the guard is client-side

The model card documents a **2K total input context**. The served container runs
`--max-model-len=4096` (8192 until the 2026-09-30 re-size in §4): `max-model-len` covers prompt
**plus** completion, so 2048 there would reject legitimate work — a 2K input needs room for its
~2K translation — and 4096 is exactly that, 2,048 in + 2,048 out.

The server does not protect against an oversized unit. Measured 2026-09-29: a unit at **4,800
prompt tokens was accepted** and returned a translation of its **first sentence only** —
32 completion tokens, `finish_reason: stop`. Non-empty, "successful", ~99% of the content gone,
and nothing downstream could tell.

So the budget is enforced in the provider, which refuses any unit over `MODEL_INPUT_TOKEN_LIMIT`
(2048) **before sending**, turning the worst failure shape available into an ordinary failed
unit. The count comes from vLLM's own `/tokenize`; when that is unreachable the fallback is
**2.2 chars/token — the FEWEST measured over 141 real Spanish units** (p05 2.88, median 4.06),
so the estimate over-counts and the guard errs toward refusing.

**Measured 2026-10-08 (#2187): the served endpoint reports `max_model_len` 4096, not 8192.** With
4096 a unit near the 2K input limit cannot also be given a ~2K completion: on the V.6b Spanish
feed two units asked for 1,144 + 2,953 = 4,097 tokens and the server refused both, failing the
episode's translation. The provider now reads the served value from `/v1/models` once and caps
each request's output budget at what the context leaves (`CONTEXT_MARGIN_TOKENS` = 32), logging
every cap; a translation that genuinely does not fit is then cut off and reported as truncated
rather than refused. 4096 is the intended value (§4: memory, and at 8192 the 4,800-token prompt
above came back 200 with its content gone, where at 4096 the server refuses it); the operator
kept it on 2026-10-09. Each ledger unit now records the request's `prompt_tokens`,
`completion_tokens` and `max_tokens_capped_from`, so how close real units come to the cap is
measurable from `translation.json`; before that it was not. A unit that does not fit is a failed
unit, which skips the episode's summary, GI and KG rather than running them on partial text.

## Licence

Gemma terms, accepted 2026-09-29. **§4.3: "Google claims no rights in Outputs you generate."**
**§1.5: "Outputs are not deemed Model Derivatives."** So §3.1's distribution obligations — the
`Notice` file, passing the agreement to recipients, propagating §3.2's use restrictions — bind
redistribution of **weights**, which we never do. They do not touch publication of translated text.

One clause to keep in view: §1.2 counts "making Gemma or its functionality available as a hosted
service via API" as Distribution. That matters only if the translator itself were exposed to users;
calling it internally and publishing its output is covered by §4.3.

## Consequences

- Translation and summarization co-reside, so a translated episode costs one pipeline pass rather
  than two model swaps.
- A sustained GPU-host outage stops ingest rather than degrading, because the DGX profiles now use
  ADR-122's `hold` — see the S0.7 note in the arc. A halted run is recoverable; a corpus of
  confidently wrong transcripts is not.
- `translate_api_base` / `translate_model` are **registry-governed**, so a profile cannot silently
  route translation somewhere unsanctioned, and the acceptance harness redirects the endpoint so a
  fixture run cannot dial the real GPU host.

  > **THIS WAS FALSE WHEN WRITTEN, AND IS TRUE AS OF 2026-09-30.** Both fields were declared on
  > `ProfilePreset` and listed in `REGISTRY_GOVERNED_FIELDS`, which is what the claim was based
  > on — but `resolve_profile_to_settings` never emitted them, and `governed_settings` narrows
  > with `if k in resolved`. So the drift check skipped them entirely. Measured: setting
  > `translate_model: totally/unsanctioned-model` in `prod_dgx_full.yaml` and running
  > `profiles-check` reported *"All 18 registry-governed profiles match the registry."* Being on
  > the governed list is not the same as being governed, and the difference was invisible because
  > the check's success message names a count of profiles rather than of fields. The resolver now
  > emits them, and `check_every_governed_field_is_emitted` fails any governed field the resolver
  > produces for no preset — these three were the only ones.
- The 27B stays available if quality proves insufficient. It would **not** force a swap-based
  serving model — see the correction in §1; the memory claim behind that was retracted in D-31.

## Alternatives considered

- **TranslateGemma-27B** — better on paper (MetricX 3.09). It WOULD co-reside (§1's correction;
  the contrary claim was retracted in D-31), so the reason it is not the pick is
  quality-per-throughput and D-31's shared-service headroom, not memory.
- **MiLMMT-46-12B / LMT-60-8B** — eligible licences; kept as fallbacks. LMT-60 is apache-2.0,
  which would remove the Gemma-terms question entirely if it ever became load-bearing.
- **Qwen3-30B (already served)** — no second model to deploy, but it is a general instruct model
  rather than a translation fine-tune, and using the summary model to translate its own input
