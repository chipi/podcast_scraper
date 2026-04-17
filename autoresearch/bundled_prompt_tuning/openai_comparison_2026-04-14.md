# OpenAI Summarization: Bundled vs Non-Bundled Comparison

**Date:** 2026-04-14
**Model (all runs):** gpt-4o (bundled), gpt-4o-mini (non-bundled — existing champion)
**Judges:** gpt-4o-mini + claude-haiku-4-5-20251001 (fixed: fraction-based contestation, Efficiency rubric)
**Datasets:** `curated_5feeds_smoke_v1` (5 episodes) + `curated_5feeds_benchmark_v1` (10 episodes)
**Score formula:** `final = 0.70 * ROUGE-L + 0.30 * judge_mean` (ROUGE-only if contested >40% of episodes)

This is the reference template for evaluating other providers (Anthropic, Gemini, Mistral, etc.).

---

## Summary Scorecard

### Smoke scale (5 episodes) — higher variance, champion tuning signal

| Track | Approach | ROUGE-L | Judge mean | Contested | Final score |
| ----- | -------- | ------- | ---------- | --------- | ----------- |
| Paragraph | Bundled (p-r1-3) | **30.3%** | 0.912 | 0/5 | **0.486** |
| Paragraph | Non-bundled | 27.3% | 0.876 | ≥3/5 ⚠️ | 0.273 (ROUGE-only) |
| Bullets | Bundled (r7-1 + p-r1-3 regression) | 33.4% | 0.917 | 0/5 | 0.509 |
| Bullets | Non-bundled | **33.2%** | **0.953** | 0/5 | **0.519** |

### Benchmark scale (10 episodes) — authoritative signal

| Track | Approach | ROUGE-L | Judge mean | Contested | Final score |
| ----- | -------- | ------- | ---------- | --------- | ----------- |
| Paragraph | Bundled | **27.6%** | **0.928** | 0/10 | **0.471** |
| Paragraph | Non-bundled | 25.8% | 0.938 | 0/10 | 0.462 |
| Bullets | Bundled | 28.0% | 0.917 | 0/10 | 0.471 |
| Bullets | Non-bundled | **31.0%** | **0.950** | 0/10 | **0.502** |

**Winner per track at benchmark scale:**

- **Paragraph:** Bundled wins (+1.8pp ROUGE-L, +2.0% final).
- **Bullets:** Non-bundled wins (+3.0pp ROUGE-L, +6.6% final).

---

## Cost / Speed

| Approach | Latency (avg) | Output tokens (avg) | Cost signal |
| -------- | ------------- | ------------------- | ----------- |
| Bundled (1 call: title+summary+bullets) | ~7.9s | 659 | 1x baseline |
| Non-bundled (2 separate calls) | ~20s | ~1140 | ~2.6x latency, ~1.7x tokens |

Bundled produces both outputs in a single LLM call, eliminating the separate cleaning step
(bundled does mental cleaning in the same pass) and combining title+summary+bullets.

---

## Recommendation

**Use bundled mode when you need summary paragraphs** (the dominant use case for podcast archives
and RAG pipelines). Paragraph quality is meaningfully better (+1.8pp benchmark ROUGE-L), costs
~2.6x less, and latency is ~2.6x faster.

**Use non-bundled mode when bullets are the primary output** (e.g. show-notes generation, UI
chips, email digests). Non-bundled bullets are ~3pp better ROUGE-L and 0.05 higher judge mean —
the gap matters if bullets are consumed directly without a summary.

**Hybrid option:** bundled call for paragraphs, separate bullets-focused call when needed. Costs
more than pure bundled but avoids the paragraph-vs-bullets tradeoff identified in the bundled
p-r1-3 tuning.

---

## Key Findings from Tuning (for applying to other providers)

### What helped (bundled)

1. **Few-shot style examples (r3-1, +2.0%):** 3 silver-quality bullets as target style. Domain-neutral
   examples (motorsport, architecture, underwater) worked; technical examples hurt.
2. **Style narration (r3-3, +4.2%):** prose description of target writing style — dense, noun-anchored,
   "X rather than Y" contrasts, em-dash precision.
3. **Anti-pattern examples (r7-1, +2.2%):** 3 explicit negative→positive rewrites ("avoid: 'The
   episode discusses X' → 'X works by [mechanism]'").
4. **Opening sentence pattern (p-r1-3, +6.6% paragraph):** single clause in user prompt requiring
   first sentence to name episode's domain + central argument.
5. **Coverage + verbatim terminology (r1 experiments):** "cover ALL major discussion segments",
   "preserve terminology verbatim" — the largest single jumps in round 1.

### What hurt (bundled)

1. **Verbose examples (r7-2, −5.6%; p-r1-4, −6.3%):** full silver-quality summary examples bleed
   domain vocabulary into outputs. Keep examples short and domain-neutral.
2. **Structure narration in system prompt (p-r1-1, both tracks regress):** paragraph-level structural
   rules ("P1=thesis, P2=topics…") change whole output structure and tank bullets.
3. **Grammatical/subject mandates (r3-2, r7-3):** prescriptive "subject must be named entity" rules
   over-constrain the model. Prefer style narration over grammatical rules.
4. **Hard count caps (r7-4):** "do not exceed 8 bullets" reduces coverage. Soft aim-for ranges work
   better.
5. **Stacking rules (p-r1-5):** adding a second structural rule on top of an accepted first rule
   usually regresses. One rule per location.

### Rubric / judging lessons

- **Fraction-based contestation is necessary.** Binary OR (old rule) flipped entire runs to
  ROUGE-only on a single contested episode (~47% score swing). Fraction-based (≥40% must contest)
  reduced false rejection dramatically.
- **Efficiency dimension** (content-density, length not penalised per se) unblocked longer-form
  outputs that were being wrongly penalised as "not concise".
- **Prose extraction before judging** (parse JSON, show title/summary/bullets as clean text) was
  critical — judges were inconsistently interpreting raw JSON string length.

---

## Replication Template for Other Providers

To benchmark another provider (e.g. Anthropic, Gemini, Mistral):

1. **Prompt templates** — create provider-specific bundled variants under
   `src/podcast_scraper/prompts/<provider>/summarization/bundled_clean_summary_*_v1.j2`,
   porting the OpenAI champion prompts (paragraph + bullet structure + style narration + anti-patterns).

2. **Ratchet configs** — mirror the 4 OpenAI configs:
   - `data/eval/configs/summarization_bullets/autoresearch_prompt_<provider>_bundled_smoke_bullets_v1.yaml`
   - `data/eval/configs/summarization_bullets/autoresearch_prompt_<provider>_bundled_benchmark_bullets_v1.yaml`
   - `data/eval/configs/summarization/autoresearch_prompt_<provider>_bundled_smoke_paragraph_v1.yaml`
   - `data/eval/configs/summarization/autoresearch_prompt_<provider>_bundled_benchmark_paragraph_v1.yaml`

3. **Silver** — the existing `silver_sonnet46_*` silvers are provider-neutral reference quality.
   They can be reused for all OpenAI-compatible providers.

4. **Run benchmark-scale baselines first** — do not tune at smoke scale before confirming
   benchmark behaviour; smoke is too noisy (non-bundled paragraph contested 3/5 at smoke, 0/10
   at benchmark).

5. **Apply the champion prompt structure** — the learnings from this OpenAI tuning generalise:
   few-shot bullets, style narration, anti-patterns, opening sentence pattern. Start there before
   provider-specific experiments.

6. **Fill in this same scorecard** for each provider, then compare across the matrix to pick the
   best (provider × mode) combination per task.

---

## OpenAI Champion State (for reference)

- **Templates:** `src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_{system,user}_v1.j2`
- **Accepted rounds:** r1 (exp-1, exp-2, exp-4), r3 (r3-1, r3-3), redesign (r6-1 champion), r7 (r7-1), paragraph r1 (p-r1-3)
- **Model:** gpt-4o for bundled, gpt-4o-mini for non-bundled (legacy champion)
- **Bundled smoke scores:** paragraph 0.486 (ROUGE-L 30.3%), bullets 0.509 (ROUGE-L 33.4%)
- **Bundled benchmark scores:** paragraph 0.471 (ROUGE-L 27.6%), bullets 0.471 (ROUGE-L 28.0%)
- **Non-bundled benchmark scores:** paragraph 0.462 (ROUGE-L 25.8%), bullets 0.502 (ROUGE-L 31.0%)

---

## Open Questions / Next Work

1. **Why does bundled bullets decline more at scale** (−5.4pp smoke→benchmark) than non-bundled
   (−2.2pp)? Possibly smoke-scale overfitting from prompt tuning. Re-validate after any further
   bundled tuning on benchmark only.

2. **Non-bundled paragraph smoke contestation** (≥3/5 at smoke, 0/10 at benchmark) suggests
   sample-size instability rather than a judging bug. The paragraph rubric and judges are working
   correctly; 5 episodes is too small for stable paragraph signal.

3. **Hybrid mode** not yet implemented — bundled for summary + separate non-bundled for bullets
   could combine the best of both. Cost between pure bundled and pure non-bundled.

4. **Per-provider tuning** using this template should surface whether other providers have the
   same bundled-paragraph-wins / non-bundled-bullets-wins split, or whether they prefer one mode
   outright.
