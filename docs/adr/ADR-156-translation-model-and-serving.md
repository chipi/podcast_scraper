# ADR-156: TranslateGemma-12B, served co-resident, called through the completions route

- **Status**: Accepted
- **Date**: 2026-09-29
- **Authors**: Marko Dragoljevic
- **Issues**: [#2169](https://github.com/chipi/podcast_scraper/issues/2169) (epic),
  [#2186](https://github.com/chipi/podcast_scraper/issues/2186) (V.6a)
- **See Also**: [ADR-096](ADR-096-dgx-spark-prod-primary-with-fallback.md),
  [ADR-122](ADR-122-self-hosted-model-resilience-policy.md),
  [ADR-155](ADR-155-pin-every-model-checkpoint.md),
  [MULTILINGUAL_ARC](../architecture/MULTILINGUAL_ARC.md) §6.1

## Context & Problem Statement

The multilingual arc needs a translation model: source-language transcript in, English out, so
every existing intelligence layer keeps reading one language. Gate V (V.3) is the decision point —
which model, served how, and called through which API.

This ADR records what was **deployed and measured**, not what was shortlisted. The shortlist work
is §6.1 of the arc notes; three of its assumptions turned out to be wrong in ways only a running
model revealed.

## Decision

### 1. `google/translategemma-12b-it`, revision `d1b225e1caa1…`

The 12B over the 27B on measured WMT24++ numbers: MetricX **3.60** / Comet22 **83.5** against the
27B's 3.09 / 84.4 — and it beats *base* Gemma-3-27B (4.04), so the fine-tune is worth more than the
size step. Decisively, the 12B **co-resides with the 30B summary model** at bf16 and the 27B would
not.

Revision pinned per [ADR-155](ADR-155-pin-every-model-checkpoint.md).

**The model id in the shortlist was wrong.** `google/translate-gemma-2-12b-it` 404s; the real id is
`google/translategemma-12b-it` — one word. It is `gated: manual`, and before the terms are accepted
the HF API returns **200 on metadata and 403 on files**, which presents as a boot hang rather than
an auth failure.

### 2. Served CO-RESIDENT on `:8005`, not swapped against `:8003`

Every other vLLM stack in the homelab shares `:8003` under a single-owner rule enforced by
`gpu-mode-swap.sh`. This one is different **by design**: translation runs immediately before
summary inside `generate_episode_metadata`, so one episode needs both models. Swapping per episode
would mean two model loads per episode, which is not a usable shape.

Therefore:

- `gpu-mode-swap.sh prod` brings up the serving vLLM **and** the translator.
- `stop_all_composes` deliberately excludes the translator, so swapping *into* prod cannot tear
  down the translator it is about to need; `free` uses a variant that does include it.
- `current_mode()` excludes it from the single-owner count — counting it would report
  `BROKEN-BOTH` for the correct state — so `status` reports it separately.
- A translator that fails to start **warns without failing the swap**: the summary model is
  already serving by then, and reporting `prod` as failed would say summarization is down when it
  is not.

The tailnet ACL grants `:8005` to the same three sources that reach `:8003` — `tag:prod`,
admin/gha-deployer, and `tag:homelab-host` for metrics — but **not** `tag:dr-drill`, whose grant
stops at `:8002` and which does not translate.

> **NOT YET APPLIED (2026-09-29).** That grant is authored in
> `podcast_scraper-infra/tailscale/policy.hujson` and **committed but unpushed**, so the LIVE
> policy does not carry `:8005`. Measured from the laptop: `:8003` completes a TCP connect in
> 17 ms while `:8005` times out, with the translator listening on `0.0.0.0:8005` and healthy on
> the box the whole time. The ADR previously stated the grant as fact because the Gate V access
> check was run ON the DGX against `127.0.0.1` — which proves the service works and proves
> nothing about the tailnet. Until the infra commit is pushed and the deployer applies it, the
> translator is reachable only from the box itself, which is where the S2.3 measurement rig was
> run for exactly this reason.

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

### 4. `--gpu-memory-utilization=0.32`

The fraction is of **total** unified memory (121.7 GiB) and must cover **weights plus KV cache**.

| | GiB |
| --- | --- |
| total usable (GB10) | 121.7 |
| prod-vllm's *actual* footprint | 29.1 |
| other compute apps | 5.1 |
| TranslateGemma-12B weights | **23.3** |

`0.20` (= 24.3 GiB) was tried and vLLM refused — `No available memory for the cache blocks` — with
~1 GiB left for cache. `0.32` gives a 38.9 GiB budget → **72,086 tokens of KV cache, 8.80×
concurrency** at `max-model-len` 8192.

The reasoning error worth preserving: prod-vllm is *allowed* 0.75 but **holds 29 GiB**. The
fraction is a ceiling a stack may claim, not what it occupies — which is what makes co-residency
possible, and is knowable only by measuring.

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

**Throughput is not yet known.** 4.3 tok/s (74 tokens in 17.1 s), measured while the box was at
~96% GPU under a production load. That is contention, not capacity. S2.10 needs a quiet box.

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
  than two GPU mode swaps.
- A sustained DGX outage stops ingest rather than degrading, because the DGX profiles now use
  ADR-122's `hold` — see the S0.7 note in the arc. A halted run is recoverable; a corpus of
  confidently wrong transcripts is not.
- `translate_api_base` / `translate_model` are **registry-governed**, so a profile cannot silently
  route translation somewhere unsanctioned, and the acceptance harness redirects the endpoint so a
  fixture run cannot dial the real DGX.
- The 27B stays available if quality proves insufficient, but it would force a swap-based serving
  model rather than co-residency.

## Alternatives considered

- **TranslateGemma-27B** — better on paper (MetricX 3.09) but does not co-reside with the 30B at
  bf16, which would cost a mode swap per episode.
- **MiLMMT-46-12B / LMT-60-8B** — eligible licences; kept as fallbacks. LMT-60 is apache-2.0,
  which would remove the Gemma-terms question entirely if it ever became load-bearing.
- **Qwen3-30B (already served)** — no second model to deploy, but it is a general instruct model
  rather than a translation fine-tune, and using the summary model to translate its own input
  would make the summary's provenance circular.
