# Grounded Insights (GI) Autoresearch Plan

Evaluate and optimize Grounded Insights quality by establishing silver references
and measuring our pipeline's GI output against them. Same v2 methodology applied
to GI: generate authoritative silver, score our output against it.

**Related:** #570 (tier-2 cross-dataset), #575 (v2 provider matrix)

---

## The question

How good are our grounded insights? We have quality metrics (PRD-017: extraction
coverage, grounded rate, quote validity) but no **reference-based scoring** — no
"what should good GI look like for this episode?" to score against.

---

## Approach: silver GI references (same as v2 summarization)

### v2 summarization pattern (proven)

```
Sonnet 4.6 → silver summary → score champions against silver
```

### GI equivalent

```
Sonnet 4.6 → silver GI artifact → score our pipeline's GI output against silver
```

For each episode, the silver GI reference contains:
- 5-8 ideal insights (claims, recommendations, observations) from the transcript
- For each insight: 1-2 verbatim supporting quotes with char offsets
- `grounded=true` on all (Sonnet 4.6 verifies its own quotes against transcript)
- Insight types classified (claim / recommendation / observation / question)

This **decouples GI measurement from summary quality**. Instead of asking "do better
bullets make better GI?" (indirect), we ask "how close is our GI output to what a
frontier model would produce?" (direct).

---

## Current state

### GI pipeline (already built)

- **Pipeline:** `src/podcast_scraper/gi/pipeline.py` — `build_artifact()`
- **Flow:** summary bullets → insights → QA grounding → NLI entailment scoring
- **Evidence stack:** extractive QA (find quotes) + NLI (entailment ≥ 0.5)
- **Config:** `gi_insight_source="summary_bullets"`, `gi_require_grounding=True`

### Quality metrics (PRD-017, already built)

- Extraction coverage: ≥80% episodes have ≥1 insight + quote
- Grounded insight rate: ≥90% insights have `grounded=true`
- Quote validity rate: ≥95% quotes have valid spans + timestamps
- Tool: `scripts/tools/gil_quality_metrics.py`

### What's missing

- **No silver GI references** — can't answer "are these the RIGHT insights?"
- **No insight-level scoring** — PRD-017 measures structure (are insights grounded?)
  not content (are they the IMPORTANT insights?)
- **No cross-provider GI comparison** — haven't compared GI output from different
  summary sources (bart-led bullets vs gemini bullets vs qwen bullets → GI)

---

## Phase 1: Generate silver GI references (~$2-5)

### Step 1 — Design the Sonnet 4.6 prompt

Prompt Sonnet 4.6 with full transcript, ask it to produce:
```json
{
  "insights": [
    {
      "text": "The claim or observation",
      "insight_type": "claim",
      "supporting_quotes": [
        {
          "text": "exact verbatim quote from transcript",
          "char_start": 1234,
          "char_end": 1298
        }
      ]
    }
  ]
}
```

### Step 2 — Generate for current dataset

- 5 held-out episodes (curated_5feeds_benchmark_v2) + 10 dev episodes
- Verify char offsets programmatically (reject + retry if quote not found verbatim)
- Store as `data/eval/references/silver/silver_sonnet46_gi_benchmark_v2/`

### Step 3 — Build GI scorer

Extend or create a scorer that compares pipeline GI output against silver:
- **Insight coverage:** what fraction of silver insights are "covered" by pipeline insights?
  (semantic similarity match, not exact text match)
- **Quote overlap:** do pipeline quotes and silver quotes overlap for the same insights?
- **Precision:** are pipeline insights that DON'T appear in silver still reasonable?
  (may need judge for this)
- **Grounding accuracy:** when both pipeline and silver ground the same insight, do
  they point to the same transcript regions?

---

## Phase 1 results (2026-04-17)

Silver GI refs generated: 40 insights across 5 held-out episodes, 37/41
quotes verified verbatim (90%). Stored in
`data/eval/references/silver/silver_sonnet46_gi_benchmark_v2/`.

Insight coverage scored across 3 summary quality tiers:

| Tier | Summary source | v2 score | **GI coverage** | Avg similarity |
|------|---------------|----------|----------------|----------------|
| Low | bart-led | 0.206 | **8%** (3/40) | 0.393 |
| Mid | qwen3.5:9b bundled | 0.509 | **70%** (28/40) | 0.748 |
| High | gemini-2.5-flash-lite | 0.479 | **72%** (29/40) | 0.762 |

**Chain hypothesis confirmed:** summary quality dominates GI insight coverage
up to a threshold (~0.45 blended score), then plateaus at ~70%.

**Key implications:**
1. bart-led should never feed GI (8% = functionally useless)
2. qwen and gemini are interchangeable for GI quality (~70% coverage both)
3. The remaining 30% gap is the GI pipeline's ceiling — further summarization
   improvement won't close it

**Next levers to investigate (Phase 3):**
- QA/NLI threshold tuning (gi_qa_score_min, gi_nli_entailment_min)
- `gi_insight_source="provider"` — skip bullets, generate insights directly from
  transcript via LLM. May capture insights that summaries miss.
- QA/NLI model swap (current: roberta-squad2, nli-deberta-base)

## Phase 1b results: direct extraction + count scaling (2026-04-17)

Tested direct insight extraction from transcript (bypassing summary bullets)
at varying insight counts + across providers:

| Source | N | Coverage | Avg sim |
|--------|---|----------|---------|
| Summary bullets (bart-led) | ~8 | 8% | 0.393 |
| Summary bullets (qwen bundled) | ~9 | 70% | 0.748 |
| Summary bullets (gemini) | ~7 | 72% | 0.762 |
| **Direct (gemini) n=5** | 5 | 55% | 0.667 |
| **Direct (gemini) n=8** | 8 | 68% | 0.729 |
| **Direct (gemini) n=12** | 12 | **82%** | 0.800 |
| Direct (gemini) n=15 | 15 | 85% | 0.800 |
| Direct (gpt-4o-mini) n=12 | 12 | 82% | 0.812 |
| Direct (qwen3.5:9b local) n=12 | 12 | 80% | — |

**Finding 1:** Count matters more than provider. 5→12 insights = +27pp coverage.
All providers converge at 80-82% at n=12. Diminishing returns past 12 (n=15 = +3pp).

**Finding 2:** Direct extraction at n=12 beats summary-derived by ~10pp (82% vs 72%).
The summary is lossy; direct extraction from full transcript captures more.

**Finding 3:** ~18% ceiling is model-independent. The 7-8 uncovered insights are
genuinely hard (too abstract or too specific for any single-pass extraction).

**Practical recommendation:**
- Set `gi_max_insights=12` (done)
- Switch `gi_insight_source="provider"` for quality-sensitive deployments (+10pp)
- Provider choice doesn't matter for GI — pick on cost/latency
- The 18% residual gap would need multi-pass extraction to close

---

## Phase 2: Baseline measurement

### Step 1 — Run GI pipeline on 3 summary quality tiers

| Tier | Summary source | v2 bullets score | Expected GI quality |
|------|---------------|-----------------|-------------------|
| Low | bart-led (pure ML) | ~0.07 | Weak insights, poor grounding |
| Mid | qwen3.5:9b bundled | 0.529 | Reasonable insights |
| High | gemini-2.5-flash-lite | 0.564 | Best insights |

### Step 2 — Score each tier's GI against silver

This tells us:
- Does better summary quality → better GI? (the chain hypothesis)
- Or is GI quality dominated by the QA/NLI evidence stack? (the threshold hypothesis)
- Which specific insights does each tier miss?

### Step 3 — Compare PRD-017 metrics vs silver-based scoring

PRD-017 says "are insights structurally valid?" Silver scoring says "are insights
semantically correct?" These may diverge — an insight can be grounded but wrong.

---

## Phase 3: Optimize (based on Phase 2 findings)

Depending on what Phase 2 reveals:

**If summary quality dominates GI quality:**
- Focus on summarization optimization (already done in v2)
- GI quality improves "for free" as summaries improve

**If QA/NLI thresholds dominate:**
- Grid search on `gi_qa_score_min` (0.1–0.5) × `gi_nli_entailment_min` (0.3–0.7)
- Find the combination maximizing silver-scored GI quality
- Possibly swap QA/NLI models (current: roberta-squad2 / nli-deberta-base)

**If the insight source matters more than grounding:**
- Test `gi_insight_source="provider"` (LLM generates insights directly) vs
  `gi_insight_source="summary_bullets"` (current default)
- The provider path skips bullets entirely — may produce better insights

---

## Phase 4: Cross-dataset GI (after tier-2)

- Generate GI silvers for QMSum meetings
- Test whether GI pipeline works on meeting transcripts (different speaker structure)

---

## Estimated budget

- Silver generation: 15 episodes × ~$0.30/episode = ~$5
- GI pipeline runs: 3 tiers × 15 episodes = 45 runs (local, $0)
- GI scoring: mostly automated (no judge needed for structural comparison)
- **Total: under $10**
