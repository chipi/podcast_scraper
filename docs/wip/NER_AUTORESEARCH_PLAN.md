# NER Autoresearch Plan

Evaluate and optimize Named Entity Recognition (speaker name extraction) across
spaCy and LLM providers. Scoped to the current podcast dataset before cross-dataset
expansion.

**Related:** #570 (tier-2 cross-dataset), #575 (v2 provider matrix)

---

## The question

Are we doing a good job extracting host and guest names with spaCy and LLMs?
Can we do better? Which provider is best for NER specifically?

---

## Current state

### Infrastructure (already built)

- **Scorer:** `src/podcast_scraper/evaluation/ner_scorer.py` (782 lines)
  — 3 scoring modes: `mention_exact`, `mention_overlap`, `entity_set`
- **Gold references:** `data/eval/references/gold/ner_entities/ner_entities_smoke_gold_v1/`
  — 5 episodes (p01-p05, e01 only), PERSON + ORG labels annotated
  — fingerprints verified against dev_v1 materialized text (byte-identical)
- **Eval configs:** 5 configs in `data/eval/configs/ner/` (spaCy sm, spaCy trf, prod baseline)
- **Unified protocol:** all 8 providers (spaCy + 7 LLMs) implement `SpeakerDetector`

### Gold refs audit (2026-04-17)

| Aspect | Status | Action needed |
|--------|--------|---------------|
| Fingerprint alignment | ✅ matches dev_v1 e01 | None |
| PERSON labels | ✅ annotated (54-62 per episode) | None |
| ORG labels | ✅ annotated (3 per episode, show titles) | None |
| GPE/PRODUCT/EVENT | ⚠️ declared in scope but zero annotated | OK for now — our question is about names |
| Episode coverage | ⚠️ only e01 (5 of 10 dev episodes) | Expand to e02 if first pass shows value |
| Held-out coverage | ❌ no e03 gold | Create if NER becomes a serious research track |
| Mentions vs entities | ⚠️ same name annotated 20+ times | Use `entity_set` mode (deduplicates) |

**Verdict:** gold is usable for first-pass provider comparison using `entity_set`
scoring mode. No regeneration needed to start.

---

## Phase 1: Baseline sweep (half-day)

### Step 1 — Run existing + new NER configs on smoke 5 episodes

| Provider | Config status | Expected latency |
|----------|--------------|------------------|
| spaCy `en_core_web_sm` | ✅ config exists | ~2s total |
| spaCy `en_core_web_trf` | ✅ config exists | ~10s total |
| OpenAI `gpt-4o-mini` | needs new config | ~30s (API) |
| Gemini `gemini-2.5-flash-lite` | needs new config | ~15s (API) |
| Ollama `qwen3.5:9b` | needs new config | ~2 min (local) |

### Step 2 — Score all against gold (`entity_set` mode for PERSON)

Table we want to produce:

| Provider | PERSON P | PERSON R | PERSON F1 | ORG P | ORG R | ORG F1 |
|----------|----------|----------|-----------|-------|-------|--------|
| spaCy sm | ? | ? | ? | ? | ? | ? |
| spaCy trf | ? | ? | ? | ? | ? | ? |
| gpt-4o-mini | ? | ? | ? | ? | ? | ? |
| gemini-2.5-flash-lite | ? | ? | ? | ? | ? | ? |
| qwen3.5:9b | ? | ? | ? | ? | ? | ? |

### Step 3 — Analyze gaps

- Which provider misses which names?
- False positives: who hallucinates speakers that don't exist?
- Host vs guest: who correctly distinguishes roles?
- Cost/latency: is the LLM call worth it over free spaCy?

### Step 4 — Decision

- If LLM NER is clearly better: tune prompt (autoresearch loop on NER prompt)
- If spaCy trf is competitive: keep as default, save the API call
- If both miss the same entities: gold ref issue or structural limitation

---

## Phase 2: Prompt tuning (if Phase 1 shows LLM gap worth closing)

- Use autoresearch ratchet on LLM NER prompt templates
- Same pattern as v2 summarization: iterate on dev, validate on held-out
- Requires NER gold for e02 or e03 episodes (create at that point)

---

## Phase 3: Cross-dataset NER (after tier-2)

- Run NER on QMSum meetings — do our providers extract meeting participants correctly?
- Requires QMSum NER gold (create or source from QMSum metadata if available)

---

## Not in scope

- Multi-language NER (English only for now)
- Entity disambiguation (is "Sam" the same person across episodes?)
- Entity linking to external KBs (Wikidata, etc.)
