# OpenAI Summarization v2: Bundled vs Non-Bundled — Final Comparison

**Date:** 2026-04-14
**Framework:** autoresearch v2 (dev/held-out split, seed=42, fixed-rubric judges)
**Model (all):** gpt-4o
**Judges:** gpt-4o-mini + claude-haiku-4-5-20251001
**Datasets:** curated_5feeds_dev_v1 (10 ep, iteration) + curated_5feeds_benchmark_v2 (5 ep e03 ~32min, held-out)

This is the authoritative OpenAI reference card. It replaces the v1 comparison (smoke/benchmark) —
which was based on a contaminated test set (smoke ⊂ benchmark). All numbers below come from a
clean train/test split and are safe to compare across providers.

---

## Headline Scorecard

| Track | Approach | Dev (10 ep) | Held-out (5 ep) | Generalization |
| ----- | -------- | ----------- | --------------- | -------------- |
| **Bullets** | **Non-bundled** | **0.564** | **0.566** | +0.4% (holds) |
| Bullets | Bundled | 0.476 | 0.505 | +6% (holds) |
| **Paragraph** | Bundled | 0.467 | 0.469 | +0.4% (holds) |
| **Paragraph** | **Non-bundled** | **0.482** | 0.481 | −0.2% (holds) |

All four champions generalize cleanly — no overfitting detected.

## Winner per track (held-out, the authoritative number)

- **Bullets:** Non-bundled wins, 0.566 vs 0.505 (**+12.1%**). Big gap.
- **Paragraph:** Non-bundled wins, 0.481 vs 0.469 (**+2.6%**). Small but consistent.

This is a meaningful reversal from the v1 story. Under v2 (cleaner signal, held-out validation,
and prompt-tuning on non-bundled using the v2 judges), **non-bundled wins on both dimensions**.

## ROUGE-L breakdown (held-out)

| Track | Non-bundled | Bundled | Gap |
| ----- | ----------- | ------- | --- |
| Bullets | 39.6% | 33.2% | +6.4pp (non-bundled) |
| Paragraph | 31.7% | 29.5% | +2.2pp (non-bundled) |

## Cost / Speed

| Approach | Latency | Output tokens | Cost |
| -------- | ------- | ------------- | ---- |
| Bundled (1 call: title+summary+bullets) | ~7.9s | ~659 | 1x baseline |
| Non-bundled (2 calls: paragraph + bullets) | ~20s | ~1140 | ~2.6x latency, ~1.7x tokens |

---

## Recommendation

**Use non-bundled as the default for quality-critical outputs.** Both bullets and paragraphs
measurably cleaner against the held-out silver. Pay the 2.6× cost when quality matters.

**Use bundled for cost-sensitive / high-throughput paths.** The paragraph gap is small (2.6%) and
the cost savings real. If you're producing low-stakes summaries (e.g. transcribed podcast archives
not consumed directly by humans), bundled is the pragmatic default.

**Don't mix: pick one per product surface.** The prompt templates differ — running bundled where
non-bundled is specced (or vice versa) will regress quality silently.

---

## Key findings from v2 tuning

### Non-bundled bullets (accepted change, +7.2%)

Ported the bundled bullet champion to non-bundled: few-shot examples + style narration
("dense, noun-anchored, X rather than Y") + anti-pattern exemplars. The exact same prompt
material that worked in bundled also worked here. Lesson: **bullet style lessons transfer
across pipeline modes for the same model.**

### Non-bundled paragraph (accepted change, +4.7%)

Stacked four rules together (all previously validated or logical extensions):

1. 4–6 paragraph default (was 2–3)
2. Opening sentence pattern (domain + central argument)
3. Cover all major segments (coverage rule)
4. Verbatim terminology (preserve specific vocabulary)

### Non-bundled bullets rejected (+0.07%, below threshold)

Adding a coverage + verbatim rule *to the user prompt* gave almost nothing — the same rules
in the system prompt already cover it. Lesson: **rules work once; adding them a second time in
another prompt adds noise.**

### Non-bundled paragraph rejected (+0.9%, below threshold)

Opening sentence pattern alone without the 4-6 paragraph + coverage + verbatim stack only gave
+0.9%. The stack delivered because of the paragraph-count change, not the opening sentence.
Lesson: **structural changes (paragraph count) matter more than style changes for paragraph prompts.**

---

## What changed from v1 to produce these numbers

1. **v2 framework**: dev/held-out split eliminates overfitting noise. The v1 comparison
   (bundled smoke 33.4% → benchmark 28.0% for bullets) was an overfitting signature — we were
   tuning on smoke while calling benchmark the "validation." v2 fixes this properly.

2. **Seed plumbing**: OpenAI seed=42 stabilizes system_fingerprint across runs. Doesn't fully
   solve API non-determinism, but removes contestation-flip noise for borderline configs.

3. **Fixed-rubric judges**: fraction-based contestation (≥40% needed, not binary OR) + Efficiency
   dimension (replacing "Conciseness — reasonable length") — these from the earlier session in
   this branch. Under v2 judges, previously rejected non-bundled experiments now stabilize
   cleanly, unblocking the +7.2% and +4.7% improvements we couldn't measure before.

4. **gpt-4o throughout**: non-bundled was previously gpt-4o-mini. Using gpt-4o for both approaches
   makes the comparison apples-to-apples.

---

## Replication template for other providers

To benchmark another provider (Anthropic, Gemini, Mistral, etc.) against these numbers:

1. **Create 4 ratchet configs** (mirror structure):
   - `autoresearch_prompt_<provider>_bundled_dev_bullets_v2.yaml`
   - `autoresearch_prompt_<provider>_bundled_dev_paragraph_v2.yaml`
   - `autoresearch_prompt_<provider>_dev_bullets_v2.yaml` (non-bundled)
   - `autoresearch_prompt_<provider>_dev_paragraph_v2.yaml` (non-bundled)
   - …plus 4 matching `_benchmark_v2` configs for held-out validation
   - All targeting `curated_5feeds_dev_v1` and `curated_5feeds_benchmark_v2` respectively
   - Use provider-equivalent prompts (port OpenAI champions as starting point)

2. **Re-use the Sonnet 4.6 silvers** — they're the reference quality bar across all providers.

3. **Establish baseline → iterate on dev → validate on held-out.** Don't iterate against held-out.

4. **Compare head-to-head using the held-out numbers.**

---

## Current champion prompt files (OpenAI v2)

### Bundled (both bullets and paragraph, one prompt pair)

- `src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_system_v1.j2`
- `src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_user_v1.j2`

Includes: few-shot examples, style narration, anti-patterns, 4-6 paragraph default, opening
sentence pattern.

### Non-bundled bullets

- `src/podcast_scraper/prompts/shared/summarization/system_bullets_v1.j2`
- `src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2`

Includes: few-shot examples, style narration, anti-patterns, verbatim terminology, stock shape rules.

### Non-bundled paragraph

- `src/podcast_scraper/prompts/openai/summarization/system_v1.j2`
- `src/podcast_scraper/prompts/openai/summarization/long_v1.j2`

Includes: 4-6 paragraph default, opening sentence pattern, coverage rule, verbatim terminology.

---

## Open items (deferred to next session)

1. **Non-bundled paragraph still contests 2/5 on held-out** (below 40% threshold — not
   ROUGE-only, but borderline). Long episodes (~32min) are harder to judge consistently.
   More held-out episodes would shrink this.

2. **Multi-run averaging** not implemented. API non-determinism still contributes ~5% score noise
   per run. With seed+v2 we've reduced the variance, but a principled N=3 average at dev scale
   remains the most rigorous fix.

3. **RFC-057/015 v2 documentation** — process changes in this session have outgrown the
   original RFCs. Next break is for writing an RFC-057-v2 (or closure ADR) that captures the
   new framework, dataset structure, judging system, and held-out validation.

4. **Other providers** — Anthropic, Gemini, Mistral haven't been run through v2 yet. OpenAI is
   the reference now.
