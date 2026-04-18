# Research-Powered Registry & Profile Presets

Capstone for the autoresearch programme: package all validated findings into
a registry of stage-specific defaults and named profile presets. One config
line replaces 20+ manual settings.

**Depends on:** #591 (pipeline validation), #592 (transcription research),
#590 (KG label quality), RFC review pass.

---

## The problem

Autoresearch produced clear, data-backed recommendations for every pipeline
stage. But the findings live in WIP notes, eval reports, and issue comments.
The actual pipeline still runs on old defaults unless someone reads all the
docs and manually flips the right flags.

**Current user experience:**
```yaml
# User has to discover and set 20+ options:
transcription_provider: openai
openai_transcription_model: whisper-1
summary_provider: gemini
gemini_summary_model: gemini-2.5-flash-lite
llm_pipeline_mode: bundled  # or not?
gi_insight_source: provider  # was summary_bullets
gi_max_insights: 12  # was 5
gi_require_grounding: true
kg_extraction_source: provider  # was summary_bullets
kg_max_topics: 10  # was 5
# ...more
```

**Target user experience:**
```yaml
profile: cloud_balanced
```

---

## Profile presets (derived from data)

Each profile is a frozen snapshot of "what autoresearch proved works best
for this use case." Every default links to the measurement that justified it.

### cloud_balanced (production default)

Best compound score (quality × cost × latency) from v2 eval + autoresearch.

| Stage | Setting | Value | Evidence |
|-------|---------|-------|----------|
| Transcription | provider | openai (whisper-1) | Baseline; #592 may change |
| Summary | provider + model | gemini / gemini-2.5-flash-lite | EVAL_HELDOUT_V2: 0.564 bullets, 1.5s, $0.00047/ep |
| Summary | mode | non-bundled | v2: bundled -5-12% on Gemini |
| GI | insight_source | provider | GI autoresearch: +10pp vs bullets |
| GI | max_insights | 12 | Coverage plateau at 12 (82%) |
| GI | require_grounding | true | Evidence providers auto-align to summary provider |
| KG | extraction_source | provider | KG autoresearch: +37pp vs bullets |
| KG | max_topics | 10 | Coverage plateau at 10 (75%) |
| KG | max_entities | 15 | Default, sufficient |
| Clustering | threshold | 0.70 | Production sweep: +47% clusters vs 0.75 |
| NER | provider | spacy (en_core_web_trf) | F1=1.000 with show-title fix, $0 |

### cloud_quality

Maximum quality, cost secondary. Uses best provider per stage.

| Stage | Setting | Value | Evidence |
|-------|---------|-------|----------|
| Transcription | provider | openai (whisper-1) | Best known |
| Summary | provider + model | deepseek / deepseek-chat | EVAL_HELDOUT_V2: 0.586 bullets (#1) |
| GI | insight_source | provider | Same |
| GI | max_insights | 12 | Same |
| GI | provider override | grok / grok-3-mini | GI matrix: 88% coverage (#1 for GI) |
| KG | extraction_source | provider | Same |
| KG | provider override | deepseek / deepseek-chat | KG matrix: 81% coverage (#1 for KG) |
| Clustering | threshold | 0.70 | Same |
| NER | provider | spacy (en_core_web_trf) | Same |

### local

Fully local, no API calls, $0 cost. Privacy / offline.

| Stage | Setting | Value | Evidence |
|-------|---------|-------|----------|
| Transcription | provider | whisper (local) | #592 will determine model size |
| Summary | provider + model | ollama / qwen3.5:9b bundled | EVAL_HELDOUT_V2: 0.529 bullets, 0.509 para |
| GI | insight_source | provider | Same methodology; Ollama as provider |
| GI | max_insights | 12 | Same |
| KG | extraction_source | provider | Same |
| KG | max_topics | 10 | Same |
| Clustering | threshold | 0.70 | Same |
| NER | provider | spacy (en_core_web_trf) | Same |

### dev

Fastest, cheapest, CI-friendly. Quality secondary.

| Stage | Setting | Value | Evidence |
|-------|---------|-------|----------|
| Transcription | provider | whisper (local, tiny/base) | Fast, low quality OK for CI |
| Summary | provider | transformers (bart-led) | Tier 2 floor, no deps |
| GI | insight_source | stub | Skip for speed |
| KG | extraction_source | stub | Skip for speed |
| Clustering | skip | — | No topics to cluster |
| NER | provider | spacy (en_core_web_sm) | Fast, F1=0.966 |

### airgapped

No network, no Ollama daemon, minimal dependencies.

| Stage | Setting | Value | Evidence |
|-------|---------|-------|----------|
| Transcription | provider | whisper (local, medium) | Best local quality |
| Summary | provider | transformers (bart-led) | Or SummLlama if wired (#571) |
| GI | insight_source | summary_bullets | No LLM for provider mode |
| GI | max_insights | 8 | Fewer bullets available |
| KG | extraction_source | summary_bullets | Same |
| KG | max_topics | 5 | From bullets, fewer available |
| Clustering | threshold | 0.70 | Same |
| NER | provider | spacy (en_core_web_sm) | Smallest model |

---

## Implementation

### What exists today

- **Model registry** (`src/podcast_scraper/providers/ml/model_registry.py`):
  ML model configs (BART, LED, hybrid). Stage-specific (summarization only).
- **Config profiles** (`config/profiles/`): YAML files with frozen settings.
  Currently used for cleaning profiles, not full pipeline presets.
- **config_constants.py**: scattered defaults per stage. Not unified.

### What to build

1. **Unified stage registry** — expand model_registry concept to cover ALL
   stages (transcription, summary, GI, KG, NER, clustering). Each entry
   has: provider, model, key params, and a `research_ref` linking to the
   eval data that justified the choice.

2. **Profile preset loader** — reads a profile name, resolves to a complete
   Config. Override any field: `profile: cloud_balanced` + `gi_max_insights: 15`
   keeps all defaults but bumps insight count.

3. **Research metadata** — each default carries provenance:
   ```python
   {
       "value": "gemini-2.5-flash-lite",
       "research_ref": "EVAL_HELDOUT_V2_2026_04.md#headline-matrix",
       "score": 0.564,
       "measured_at": "2026-04-16",
       "issue": "#575",
   }
   ```
   So when someone asks "why is this the default?" the answer is a link
   to measured data, not "someone thought it was good."

4. **CLI integration** — `--profile cloud_balanced` flag on CLI. Overrides
   individual settings. Shows which profile is active in run metadata.

5. **Validation** — `make pipeline-validate` (#591) runs each profile and
   verifies all stages work. Profile changes require validation pass.

---

## Migration path

1. **Phase 1: document profiles** — write the 5 profile YAML files with all
   settings. No code changes; users can `--config config/profiles/cloud_balanced.yaml`.
2. **Phase 2: profile loader** — add `profile:` field to Config that resolves
   a named profile. Override semantics (profile as base, explicit fields win).
3. **Phase 3: research metadata** — attach provenance to each default so
   `config_constants.py` becomes self-documenting.
4. **Phase 4: CLI flag** — `--profile` shorthand.

Phase 1 is zero-code: just create the YAML files. Already useful.

---

## When to do this

After:
- #591 (pipeline validation — confirms all providers work)
- #592 (transcription research — fills in the transcription tier)
- #590 (KG labels — finalizes KG settings)
- RFC review (updates RFCs with real data)

This is the **packaging step** — takes validated findings and makes them
the default experience. Do it last because every research stream feeds into
the profile definitions.
