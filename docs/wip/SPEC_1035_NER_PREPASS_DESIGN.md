# #1035 NER pre-pass — design spec (phase 1)

**Trigger**: #1033 step 2 cohort rerun showed 0% KG entity coverage across
all 7 candidates under the corrected `provider` pipeline + tightened
prompt. Confirmed Pattern B (universal LLM gap). #1035 adds a
deterministic NER pre-pass so the LLM operates on pre-identified entity
spans instead of discovering from scratch.

## Decisions

### Insertion point: pipeline-level, not per-provider

NER pre-pass runs in `kg/pipeline.py::_try_provider_extraction()` BEFORE
the call to `provider.extract_kg_graph(...)`. The resulting PERSON + ORG
spans are passed as a new `ner_entity_hints` parameter through the
provider's extract_kg_graph signature → into `build_kg_user_prompt` →
rendered as a candidate-list block in the new prompt v5 template.

**Why pipeline-level (not 7 provider edits)**:
- Single edit point, every provider gets it for free
- Re-uses the cached `_spacy_nlp` model on `summary_provider`
  (Issue #387 caching pattern already in place)
- Works on the SAME transcript text the LLM sees → no alignment skew
- Provider methods stay thin — only need to forward `ner_entity_hints`
  to `build_kg_user_prompt`

### Config flag: opt-in initially, flip to default-on after phase 3

```python
kg_extraction_use_ner_prepass: bool = Field(
    default=False,
    alias="kg_extraction_use_ner_prepass",
    description=(
        "When True, run spaCy NER on the cleaned transcript before LLM "
        "KG extraction and seed the prompt with PERSON+ORG candidate "
        "spans. The LLM still owns the final entities[] decision (it "
        "may reject misclassifications, fix spellings, and add missed "
        "entities). Closes the 0% entity-coverage gap surfaced by #1033 "
        "(see docs/wip/EVAL_1033_COHORT_RERUN_2026-06-19.md). "
        "Requires cfg.ner_model + cfg.speaker_detector_provider=spacy "
        "(both default-on in prod profiles). When False, prompt v4 is "
        "used and entity recall stays at the floor."
    ),
)
```

Phase 4 will flip the default to `True` for prod profiles after
validation sweep confirms a material entity-coverage win.

### spaCy model: reuse existing `cfg.ner_model`

- `prod_dgx_*`: `en_core_web_trf` (1.000 F1 on smoke set per #906)
- `airgapped`: `en_core_web_trf`
- `airgapped_thin` / `dev`: `en_core_web_sm` (0.966 F1, faster boot)
- No new dependency. spaCy + the model are already installed wherever
  `speaker_detector_provider: spacy` is set.

### Span source: PERSON + ORG only

The silver methodology (#1033 `silver_opus47_kg_dev_v1`) scores entity
coverage on PERSON + ORG. spaCy emits other labels (GPE, DATE, MONEY,
NORP, EVENT, …) which would dilute the candidate list and confuse the
LLM. Filter to PERSON + ORG at the NER stage; the LLM categorises into
`person`/`organization` (the existing entity_kind enum).

### Dedup + cap pre-LLM

Before injecting into the prompt:
1. Lowercase-dedup candidate text (`{"Maya", "maya"}` → `{"Maya"}`)
2. Strip leading/trailing whitespace + punctuation
3. Drop singleton initials, single-letter spans, all-digit spans
4. Cap at `min(max_entities * 3, 40)` candidates — gives the LLM
   slack to reject without bloating the prompt

The cap target `max_entities * 3` is conservative: empirically a 30-min
podcast yields 8–20 named entities; the silver expects 2–8 per episode.
Capping at 3× gives the LLM enough headroom to add missing entities
while keeping the prompt bounded.

### Prompt v5 template (new file)

`src/podcast_scraper/prompts/shared/kg_graph_extraction/v5.j2`

Builds on v4. New section between `Entity rules:` and `Return JSON:`:

```jinja
**Pre-extracted entity candidates (from NER pass):**

The following PERSON and ORG candidates were extracted from the transcript
by a named-entity recognizer. They are HINTS, not a confirmed final list.

{% for cand in ner_entity_hints %}- {{ cand.text }} ({{ cand.label }})
{% endfor %}

**How to use the candidate list:**
- For each candidate you confirm IS a real named entity, emit it in
  the `entities` array with the correct spelling + person/organization
  classification + a 1-2 sentence description of their role/relevance.
- You MAY skip candidates that are NOT real entities (generic words,
  transcription errors, "host"/"guest" fillers, possessive forms).
- You MAY add named entities the candidate list missed — extract any
  proper-noun person or organization actually mentioned in the
  transcript.
- One canonical spelling per entity (same rule as v4).
- Hard cap: at most {{ max_entities }} entries in `entities`.
```

When `ner_entity_hints` is empty (NER returned nothing), the template
renders with an empty bullet block + the LLM falls back to its own
discovery (equivalent to v4 with extra noise instructions).

When the Config flag is OFF, `kg/pipeline.py` passes prompt_version="v4"
and never invokes the NER stage — fully back-compatible.

### Per-provider changes — minimal

Each of the 7 LLM providers' `extract_kg_graph(transcript, episode_title,
max_topics, max_entities, params, pipeline_metrics)` gains an optional
`ner_entity_hints: list[dict] | None = None` kwarg + `prompt_version: str
= "v4"` kwarg. Each just forwards both to `build_kg_user_prompt`.

`kg/llm_extract.py::build_kg_user_prompt` already accepts
`prompt_version="v4"` — extend signature to accept
`ner_entity_hints=None` and forward it to the Jinja template.

### Failure modes + fallbacks

| Condition | Behavior |
|---|---|
| Flag off | Use v4. No NER call. No new code path active. |
| Flag on, `cfg.ner_model` empty | Log warning, fall back to v4 silently. |
| Flag on, spaCy import fails | Log warning, fall back to v4. |
| Flag on, NER returns 0 candidates | Use v5 with empty candidate block (LLM falls back to its own discovery). |
| Flag on, transcript > 100k chars | spaCy handles long docs natively; no extra truncation logic needed. |
| Flag on, provider doesn't forward kwargs | Soft-fail — prompt v5 still renders, but with empty hints. Caller log explains. |

### What we do NOT change

- Provider extract_kg_graph signature parameters that already exist
  stay the same; only ADD `ner_entity_hints` + `prompt_version` kwargs
  with safe defaults
- `gi_insight_source` path — out of scope (this is KG-only). GI gets
  the same treatment in a follow-up if entity coverage matters there
- Prefilled / mega_bundled / extraction_bundled paths — those
  short-circuit `_try_provider_extraction` before NER runs (correct;
  the bundle already contains entities)
- The `summary_bullets` path — N/A, it's gone (#1034)
- Silver methodology — same `silver_opus47_kg_dev_v1` entity slice
  scoring (no scoring code change)

## Phase 2 — implementation checklist

1. `kg/ner_prepass.py` (new) — `extract_kg_ner_hints(transcript, nlp,
   max_candidates)` → `list[dict]` returning the deduped + capped
   candidate list
2. `kg/pipeline.py::_try_provider_extraction` — when `cfg.kg_extraction_use_ner_prepass`
   is True, fetch nlp from `kg_extraction_provider._spacy_nlp` (cached)
   or `get_ner_model(cfg)`, call `extract_kg_ner_hints`, pass to
   `extract_fn(...)` as `ner_entity_hints=hints, prompt_version="v5"`
3. `kg/llm_extract.py::build_kg_user_prompt` — accept new kwargs;
   forward to Jinja
4. `prompts/shared/kg_graph_extraction/v5.j2` — new template (extends v4)
5. Provider methods (7 files) — accept + forward `ner_entity_hints` +
   `prompt_version` kwargs to `build_kg_user_prompt`
6. `config.py` — `kg_extraction_use_ner_prepass: bool = False` field
7. Tests:
   - `tests/unit/podcast_scraper/kg/test_ner_prepass.py` — unit test
     for `extract_kg_ner_hints` (filters, dedup, cap)
   - `tests/unit/podcast_scraper/kg/test_kg_llm_extract.py` — extend
     for v5 template + hint-passthrough
   - `tests/unit/podcast_scraper/kg/test_kg_pipeline.py` — extend for
     end-to-end NER hint flow with a mock provider
   - `tests/unit/podcast_scraper/prompts/prompt_contract_registry.py`
     — register v5 in the transcript-allowed list

## Phase 3 — validation sweep

After phase 2 implementation lands:
1. Set `kg_extraction_use_ner_prepass: true` in
   `data/eval/configs/kg_autoresearch_*.yaml` for the cohort
2. Re-run the 7-candidate cohort via
   `autoresearch/1033_cohort_rerun/run_sweep.sh` (re-uses the
   harness from #1033 step 2)
3. Score each candidate's KG output vs `silver_opus47_kg_dev_v1`
4. Compare entity coverage delta vs the 0% baseline
5. Hold-out validation on `curated_5feeds_benchmark_v2` + Sonnet 4.6
   silver (cross-dataset, cross-vendor)

**Material-win threshold**: ≥ 20pp entity coverage average across
the cohort, OR ≥ 30pp on the top-3 candidates (Qwen3.5-35B-A3B,
Cell F NVFP4, Gemma-4). Less than that → don't ship default-on,
revisit prompt design.

## Phase 4 — ship

If material:
- Flip `kg_extraction_use_ner_prepass` default to `True` in `Config`
- Update prod profiles to make the flag explicit (`true`) for clarity
- Land WIP report `docs/wip/EVAL_1035_NER_PREPASS_VERDICT.md` with
  per-candidate entity-coverage deltas
- Close #1035

If not material:
- Investigate failure mode (NER recall too low? LLM ignoring spans?)
- File follow-up ticket; leave #1035 open with diagnosis
