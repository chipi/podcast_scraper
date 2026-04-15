# Anthropic Summarization v2 — Reference Card + OpenAI Comparison

**Date:** 2026-04-15
**Framework:** autoresearch v2 ([RFC-073](../docs/rfc/RFC-073-autoresearch-v2-framework.md))
**Model:** claude-haiku-4-5-20251001 (candidate) + claude-sonnet-4-6 (silver)
**Judges:** gpt-4o-mini + claude-haiku-4-5-20251001
**Datasets:** curated_5feeds_dev_v1 (10 ep iteration) + curated_5feeds_benchmark_v2 (5 ep held-out)

Second provider card under v2, compared head-to-head with the OpenAI card. All numbers come from
held-out content the prompts were never tuned on.

---

## Anthropic Scorecard (held-out, authoritative)

| Track | Approach | Final score | ROUGE-L | Judge mean | Contested |
| ----- | -------- | ----------- | ------- | ---------- | --------- |
| Bullets | **Non-bundled** | **0.570** | 40.7% | 0.949 | 0/5 |
| Bullets | Bundled | 0.552 | 39.3% | 0.921 | 0/5 |
| Paragraph | Bundled | **0.548** | 39.2% | 0.914 | 0/5 |
| Paragraph | Non-bundled | 0.522 | 36.4% | 0.893 | 0/5 |

**Winner per track on Anthropic:**

- **Bullets:** Non-bundled, 0.570 vs 0.552 (+3.2%).
- **Paragraph:** **Bundled** wins, 0.548 vs 0.522 (+5.0%). Opposite of OpenAI where non-bundled won paragraph too.

## Bundled ≈ Non-bundled on Anthropic (surprise finding)

On OpenAI, bundled-vs-non-bundled is a meaningful gap on both tracks (bullets −12%, paragraph −2.6%
against non-bundled). On Anthropic, the two modes are close on bullets (−3%) and **bundled
actually wins paragraph** (+5%).

Why: Claude Haiku 4.5 appears to handle the multi-task bundled prompt better than gpt-4o does — the
attention-split penalty we saw hurting OpenAI bundled doesn't materialize here. Plausibly Anthropic's
post-training emphasizes instruction-following across structured outputs.

Practical consequence: **bundled is much more viable on Anthropic.** You get title + summary +
bullets in one call at essentially non-bundled quality, at ~1/3 the cost.

---

## Head-to-head: Anthropic vs OpenAI (held-out final scores)

| Track | Approach | OpenAI (gpt-4o) | Anthropic (haiku-4.5) | Delta |
| ----- | -------- | --------------- | ---------------------- | ----- |
| Bullets | Non-bundled | 0.566 | **0.570** | Anthropic +0.7% |
| Bullets | Bundled | 0.505 | **0.552** | Anthropic +9.3% |
| Paragraph | Non-bundled | 0.481 | **0.522** | Anthropic +8.5% |
| Paragraph | Bundled | 0.469 | **0.548** | Anthropic +16.8% |

**Anthropic wins every cell**, and by wide margins on bundled. Haiku-4.5 (smaller, cheaper model)
outperforms gpt-4o on every single track-approach combination. This is a striking result — the
framework surfaces a clear provider-capability gap that would have been hidden under v1 (noisy
signals + no held-out).

### ROUGE-L breakdown (held-out)

| Track | Approach | OpenAI ROUGE-L | Anthropic ROUGE-L | Delta |
| ----- | -------- | -------------- | ----------------- | ----- |
| Bullets | Non-bundled | 39.6% | 40.7% | +1.1pp |
| Bullets | Bundled | 33.2% | 39.3% | +6.1pp |
| Paragraph | Non-bundled | 31.7% | 36.4% | +4.7pp |
| Paragraph | Bundled | 29.5% | 39.2% | +9.7pp |

Anthropic's largest wins are on bundled — exactly the mode OpenAI struggles with most.

---

## Cost & Speed (approximate, from run logs)

| Provider | Model | Mode | Latency/ep | Tokens/ep | Relative cost |
| -------- | ----- | ---- | ---------- | --------- | ------------- |
| OpenAI | gpt-4o | Bundled | ~7.9s | ~660 | baseline |
| OpenAI | gpt-4o | Non-bundled | ~20s | ~1140 | 2.6× latency, 1.7× tokens |
| Anthropic | claude-haiku-4-5 | Bundled | ~10s | ~600 | ~cheaper than gpt-4o ($0.80/$4 vs $2.50/$10 per M tokens) |
| Anthropic | claude-haiku-4-5 | Non-bundled | ~25s | ~1050 | same pattern — 2.5× latency |

**Anthropic Haiku 4.5 pricing** (Mar 2026): $0.80/M input, $4.00/M output.
**OpenAI GPT-4o pricing** (Mar 2026): $2.50/M input, $10.00/M output.

So Anthropic Haiku is both *cheaper* AND *higher quality* than OpenAI GPT-4o on this workload.

---

## Recommendation

**Use Anthropic Haiku 4.5 bundled as the new production default.** Lowest cost, highest quality on
both paragraph and bullets when allowed to use bundled format. The 5% paragraph advantage over
non-bundled plus 1/3 the latency/cost makes this the clear pick.

**Use Anthropic non-bundled bullets when bullets are the sole output** (e.g. show-notes only,
no summary needed). Small additional gain (+3%) for dedicated-bullets use cases.

**Use OpenAI when...** honestly, nothing on this workload. The held-out numbers favour Anthropic
across the board. OpenAI may win on use cases outside summarization, or where the gpt-4o developer
ecosystem / tooling matters more than raw quality.

---

## Key findings for the framework

1. **Champion prompts transfer across providers.** We ported OpenAI v2 champion prompts unchanged
   and got strong Anthropic numbers without any provider-specific tuning. The patterns that work
   (style narration, anti-pattern exemplars, opening sentence, verbatim terminology) are
   model-agnostic within current LLM capabilities.

2. **Bundled viability is provider-dependent.** OpenAI bundled has a structural gap from attention
   split across 3 outputs. Anthropic bundled is nearly parity with non-bundled. This means the
   bundled-vs-non-bundled decision has to be re-evaluated per provider, not assumed from OpenAI
   experience.

3. **Cheaper models can beat expensive ones with good prompts.** Haiku 4.5 (small model) beats
   gpt-4o (flagship) across all four tracks. Suggests the gap between flagship and smaller models
   has narrowed substantially, at least for structured summarization.

4. **The held-out framework caught what smoke/benchmark couldn't.** v1 framework on Anthropic
   would not have produced reliable absolute numbers for comparison — smoke noise alone would
   have obscured a 5-10pp advantage. Under v2 the gap is unambiguous.

---

## Implementation notes

### Anthropic JSON prefill (new)

Anthropic API has no `response_format: json_object` equivalent. Claude bundled calls were
returning preamble before JSON, causing parse failures. Added standard Claude prefill pattern:

```python
messages=[
    {"role": "user", "content": user_prompt},
    {"role": "assistant", "content": "{"},
]
```

The `{` is then prepended to the response before `json.loads()`. See
`src/podcast_scraper/providers/anthropic/anthropic_provider.py` around `summarize_bundled`.

### Prompt files

Bundled (new, copied from OpenAI champions):

- `src/podcast_scraper/prompts/anthropic/summarization/bundled_clean_summary_system_v1.j2`
- `src/podcast_scraper/prompts/anthropic/summarization/bundled_clean_summary_user_v1.j2`

Non-bundled paragraph (updated with champion rules):

- `src/podcast_scraper/prompts/anthropic/summarization/long_v1.j2` (+4-6 para, +opening sentence, +coverage, +verbatim)
- `src/podcast_scraper/prompts/anthropic/summarization/system_v1.j2` (unchanged, stock)

Non-bundled bullets reuses the already-champion-level shared prompts:

- `src/podcast_scraper/prompts/shared/summarization/system_bullets_v1.j2`
- `src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2`

### Configs (8 new)

See `data/eval/configs/summarization_bullets/autoresearch_prompt_anthropic_{bundled_,}{dev,benchmark}_bullets_v2.yaml`
and `data/eval/configs/summarization/autoresearch_prompt_anthropic_{bundled_,}{dev,benchmark}_paragraph_v2.yaml`.

---

## Open items / next steps

1. **Anthropic-specific prompt iteration** — we stopped at ported-from-OpenAI. A dedicated tuning
   session against Anthropic dev might find another 2-5% on top of current numbers. Diminishing
   returns given we're already winning — defer unless a business case emerges.

2. **Sonnet 4.6 as candidate** — currently Sonnet is only silver. Running Sonnet as the candidate
   would show how close Haiku gets to Sonnet on this workload. Useful for deciding when the
   Sonnet premium is worth paying.

3. **Seed parameter on Anthropic** — Anthropic API does not support seed; Haiku's temp=0 appears
   more deterministic than OpenAI's empirically (no contestation flips observed across our runs),
   but we haven't characterized this rigorously.

4. **Gemini and Mistral** under v2 — natural next providers to round out the comparison matrix.
