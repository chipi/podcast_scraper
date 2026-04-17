# Next Steps: Eval Quality & Benchmark Improvements

> **Status:** Draft — proposed work items after RFC-057 closure (2026-04-06)

---

## Where we are

RFC-057 is closed. We have a complete eval system:

- 4 Sonnet-4.6 silver references (smoke/benchmark × paragraph/bullets)
- 72 configs covering 18 providers × 2 formats × 2 scales
- First 10-episode benchmark sweep documented in `EVAL_BENCHMARK_V1_2026_04.md`
- Four-tier summarization strategy with production defaults set

The system is **correct but narrow**. Rankings are stable but measured against a
limited dataset and a single model family's writing style.

---

## Proposed next steps (priority order)

### 1. Expand benchmark dataset — `curated_v2` (10+ feeds, 20-25 eps)

**Problem:** 10 episodes from 5 feeds is enough for stable rankings but not enough
to detect small quality differences (±1-2% ROUGE-L) or expose topic-coverage bias.
The current feeds likely skew toward tech/business — models that write in that
register score better against the Sonnet 4.6 silver.

**Proposal:**

- Expand to 20-25 episodes from 8-10 feeds covering more topic diversity
  (science, culture, narrative/storytelling, interview-heavy shows)
- Create `curated_10feeds_benchmark_v2` dataset and materialize it
- Re-run all providers against new benchmark; re-promote silver references for
  the new dataset
- The smoke dataset can stay at 5 eps — smoke is for fast iteration, not
  production numbers

**Effort:** Low-medium (curation + materialization, no new code).
**Value:** More confident rankings; exposes topic-coverage bias in current silver.

---

### 2. Human preference validation on top 3-4 models

**Problem:** ROUGE/embed measures similarity to the silver reference, not whether
the summary is actually good. Two models could score identically on ROUGE but one
could be clearly better to a human reader.

**Proposal:**

- Pick 5 episodes from the benchmark dataset
- Generate summaries from the top 3-4 models (Anthropic, DeepSeek, qwen3.5:35b,
  llama3.2:3b)
- Do a blind side-by-side preference rating (1 hour of manual work)
- Record whether ROUGE/embed rankings match human preference
- If they diverge, adjust metric weights in the composite scorer

**Effort:** Medium (mostly manual review time, ~1-2 hours).
**Value:** Validates whether the eval system measures what we care about. High
signal for deciding when to replace the silver reference.

---

### 3. Gold references for a small canonical set (5 episodes)

**Problem:** The silver reference is the ceiling — if Sonnet 4.6 writes a mediocre
summary for an episode, every model gets penalised for diverging from that.
Gold references (human-written) eliminate that bias.

**Proposal:**

- Pick 5 episodes that represent the hardest cases (long, topic-dense, many speakers)
- Write or carefully edit 5 gold summaries (paragraph + bullets)
- Freeze as `gold_human_v1` reference
- Run all providers against gold; compare gold vs silver rankings

**Effort:** High (2-4 hours of careful writing/editing).
**Value:** Ground truth that exposes silver reference bias. Necessary before
trusting silver-based rankings for a high-stakes deployment decision.

---

### 4. Factual accuracy metric (named entity + number retention)

**Problem:** A model can score 35% ROUGE-L while getting facts wrong — wrong speaker
name, wrong date, hallucinated statistic. ROUGE and embedding similarity are blind
to factual errors.

**Proposal:**

- Extend the scorer to compute named entity retention: what fraction of PERSON/ORG
  entities from the transcript appear in the summary
- The NER eval infrastructure (`ner_entities` task, spaCy pipeline) already exists —
  this is wiring it into the summarization scorer
- Add a `numbers_retained` metric (already partially implemented) to track numeric
  fidelity

**Effort:** Low-medium (scorer extension, NER pipeline already available).
**Value:** Catches hallucination cases that surface-overlap metrics miss. Especially
important for interview-heavy shows where speaker names matter.

---

## What not to do next

- **Do not replace the silver reference yet** — no evidence that a better frontier
  model is available that would materially change rankings
- **Do not expand Ollama model coverage** — the matrix already covers 12 models;
  adding more before improving dataset quality adds noise, not signal
- **Do not run a new prompt tuning round** — all tracks are at ceiling; a new round
  requires either a new silver reference or a fundamentally different prompt structure

---

## Suggested sequence

```text
1. curated_v2 dataset (feeds selection + materialization)
2. Re-run silver reference generation on curated_v2
3. Re-run all providers → new benchmark report
4. Human preference validation (can be done in parallel with 2-3)
5. Gold references for 5 canonical episodes
6. Factual accuracy metric extension
```
