# Mega-Bundle Experiment: Single-Call Summary + GI + KG

Test whether a single LLM call can produce all extraction outputs (summary +
GI insights + KG topics/entities) without unacceptable quality degradation.

**Status:** planned, after first-pass autoresearch completes for all streams.

**Depends on:** GI (#579), KG (#584) silver refs + standalone baselines established.

---

## Motivation

Current pipeline makes 3 LLM calls per episode, all reading the same transcript:

| Call | Input | Output | Cost share |
|------|-------|--------|-----------|
| 1. Summarization | full transcript | title + summary + bullets | ~33% |
| 2. GI extraction | full transcript | insights (if source=provider) | ~33% |
| 3. KG extraction | full transcript | topics + entities (if source=provider) | ~33% |

Bundling into 1 call = **2/3 cost reduction** + faster (one roundtrip vs three).

---

## Experiment design

### Prompt: mega-bundled JSON

Ask the LLM to produce all fields in one structured JSON response:

```json
{
  "title": "Episode title",
  "summary": "4-6 paragraph prose summary...",
  "bullets": ["Bullet 1", "Bullet 2", ...],
  "insights": [
    {"text": "Key insight", "type": "claim"}
  ],
  "topics": ["2-8 word noun phrase", ...],
  "entities": [
    {"name": "Full Name", "kind": "person", "role": "host"}
  ]
}
```

### Provider: OpenAI only (first pass)

Use gpt-4o-mini — stable JSON mode, well-understood baseline from v2 bundled
summarization experiments. Extend to other providers only if OpenAI results are
promising.

### Scoring: per-field quality vs standalone

Score each field independently against its silver reference:
- **Summary:** blended 0.70·ROUGE-L + 0.30·judge_mean (v2 methodology)
- **GI insights:** embedding coverage against GI silver (threshold=0.65)
- **KG topics:** embedding coverage against KG silver (threshold=0.65)
- **Entities:** entity_set F1 against KG silver entities

Compare each score to the standalone extraction baseline.

### Success criteria

| Field | Standalone baseline | Acceptable bundled | Unacceptable |
|-------|--------------------|--------------------|--------------|
| Summary (paragraph) | v2 gpt-4o-mini: 0.469 | ≥0.440 (<6% drop) | <0.420 (>10% drop) |
| GI insights (n=12) | 82% coverage | ≥75% (<7pp drop) | <70% (>12pp drop) |
| KG topics (n=10) | 71% coverage | ≥65% (<6pp drop) | <60% (>11pp drop) |
| Entities | ~1.000 F1 | ≥0.900 | <0.850 |

If ALL fields stay in the "acceptable" range: **bundle by default** for
cost-sensitive deployments, keep standalone for quality-sensitive.

---

## v2 bundled summarization lessons to apply

From the v2 eval (PR #568):

1. **Anthropic had zero bundled penalty** — may be true for mega-bundle too
2. **OpenAI/Gemini lost 5-12% on bundled bullets** — expect similar or worse
   with more fields
3. **JSON schema stabilises output** — response_format=json_object helps
4. **Prose extraction before judging** — extract prose before scoring summary
5. **Gemini JSON quirks** — strict=False, code-fence stripping needed

---

## What this does NOT test

- Whether the mega-bundle prompt needs per-provider tuning (likely yes)
- Whether different field counts (e.g., 8 insights + 10 topics vs 12 + 15)
  affect the attention split differently
- Whether the penalty varies by transcript length (longer = more attention
  competition)

These are follow-ups if the first-pass looks promising.

---

## Estimated effort

- Prompt design: ~30 min
- Run 5 held-out episodes on gpt-4o-mini: ~5 min
- Score all fields: ~15 min (reuse existing scorers)
- Compare to baselines: ~15 min
- **Total: ~1 hour, under $1 API cost**
