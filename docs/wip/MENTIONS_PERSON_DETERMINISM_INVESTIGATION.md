# MENTIONS_PERSON deterministic emission — investigation (#1076 chunk 4)

Background: `tests/stack-test/stack-person-profile.spec.ts` strictly
asserts the Person Profile **shell** but only conditionally exercises
the click-to-PositionTracker rich-data path because airgapped_thin
emits `MENTIONS_PERSON` best-effort. The reason lives in one function
in this repo, and the fix has two viable paths. This doc lays them out
so we can pick before committing.

## Today's emission logic

`src/podcast_scraper/gi/relational_edges.py:add_insight_entity_edges`:

```python
patterns = [
    (eid, kind, name, re.compile(r"\b" + re.escape(name) + r"\b"))
    for eid, (name, kind) in entity_index.items()
    if name and len(name) >= 2
]
...
for insight_id, text in insights:
    for entity_id, kind, surface_name, pattern in patterns:
        if pattern.search(text):
            edges.append({"type": "MENTIONS_PERSON", ...})
```

**Whole-word substring match** against the KG `Person.properties.name`.
Cheap, deterministic-given-the-text, no model load. But it only fires
when the **literal name** appears as a whole word in the Insight's
text. Under airgapped_thin the BART map-reduce summary paraphrases
many host/guest mentions to "the host", "the guest", "she", "they" —
the name doesn't appear in the bullet text and no edge fires.

## Path A — Swap whole-word match for spaCy NER

`src/podcast_scraper/speaker_detectors/entities.py` already exposes
`extract_person_entities(text, nlp)` which returns
`List[Tuple[str, float]]` (name + confidence). airgapped_thin loads
`en_core_web_sm` for speaker detection so the model is already in
process — no extra load.

Reshape `add_insight_entity_edges`:

```python
nlp = get_ner_model(cfg)  # already cached
for insight_id, text in insights:
    detected = extract_person_entities(text, nlp)  # spaCy PERSON spans
    for name, confidence in detected:
        # Match against the KG entity_index by canonical name; emit
        # MENTIONS_PERSON to the resolved Person id. Confidence rides
        # on the edge so the viewer can dim low-confidence rows.
        ...
```

**Pros**:

- Catches paraphrased mentions like "Bob and Maya disagreed" where
  the current path catches it, AND "Bob argued strongly" where the
  current path also catches it, AND any case where spaCy detects a
  PERSON span that resolves against the KG index.
- Determinism is now bounded by the spaCy model's output, which is
  reproducible across runs given the same input text.
- No new CI dependency — spaCy is already an airgapped_thin
  dependency.

**Cons / risks**:

- **False positives** from spaCy. en_core_web_sm has documented
  precision issues on short text and uncommon names. "Cascadia
  Alliance" gets tagged as PERSON sometimes. The fix doesn't add
  edges to entities that aren't in the KG index, so a false-positive
  only adds noise (an edge to a wrong Person) IF that wrong Person
  also exists in the KG index — rare but possible.
- **Performance**: spaCy doc parsing on every Insight is ~30 ms on
  M4 Pro. With ~30 insights per episode × 5 episodes in stack-test
  that's 4.5 seconds added to the pipeline. Worth it for the
  determinism, but it IS a cost.
- **Pipeline-stage placement**: today `add_insight_entity_edges`
  runs in the workflow's metadata-generation step after the GI
  artifact is emitted. The cfg isn't necessarily available at that
  call site. Would need either threading the cfg through or
  module-level model caching (the cache layer can handle it).

**Effort**: ~half a day. Code change is ~30 lines + test rewrite.

## Path B — Opt-in cloud-profile Tier-3 spec

cloud_thin emits MENTIONS_PERSON deterministically because the LLM
extracts named entities at insight-emit time, not via post-pass
substring match. Adding a cloud-profile stack-test would close the
determinism gap **without** changing the airgapped_thin emit logic.

Per `[[feedback_no_llm_in_ci]]` it can't run in default CI. Per
`[[feedback_cross_repo_apply]]` the cross-repo apply pattern is
established. Workable shape:

1. Add `tests/stack-test/stack-person-profile-cloud.spec.ts` gated on
   `STACK_TEST_PROFILE=cloud_thin` env var (already supported by the
   stack-test docker-compose).
2. The spec exercises the same walk as
   `stack-person-profile.spec.ts` but asserts the rich-data path
   strictly (not conditionally).
3. Operator runs it locally with API keys when validating prod-shape
   changes; never runs in CI.

**Pros**:

- Zero changes to production emit logic — no false-positive risk to
  cloud_thin / cloud_balanced production.
- Validates the same path real users see (the LLM emits, the viewer
  consumes).
- Cheap: one new spec file, one env-gated test.

**Cons / risks**:

- Doesn't help airgapped_thin users — their MENTIONS_PERSON
  determinism is unchanged. They still get the conditional rich-data
  experience.
- Adds API-key dependency for the operator's local Tier-3 runs.
  Already established by `stack-error-recovery.spec.ts` ("cloud_thin
  profile end-to-end smoke (real OpenAI Whisper + Gemini)") which
  proves the gate is workable.
- Cost: ~$0.05 per local run per the existing cloud_thin smoke's
  documented pricing.

**Effort**: ~2 hours. Mostly copy-paste from the existing spec.

## Recommendation

**Both paths are worth doing, in this order:**

1. **Path B first** (small, no production risk). Closes the
   stack-person-profile.spec.ts conditional path under cloud_thin
   immediately. Operator can run it before merging to prove the rich
   path stays alive.
2. **Path A second** (lift airgapped_thin's determinism floor).
   Wins for everyone deploying airgapped, including future
   Docker-stack-test users without API keys.

Path A's false-positive risk is the one thing that could derail it.
Mitigation: ship Path A behind a `cfg.gi_typed_mentions_use_ner` flag
(default `False`), validate against the chunk-2 v3 fixture corpus
(checked-in 2026-06-23) before flipping the default. The 23 fixtures
already pass our 4-field/edge contract; spaCy NER on those texts
gives a quick sanity read before turning on production.

## What this doc does NOT do

- No code change. The path A / B decision is operator's call.
- Doesn't benchmark the airgapped_thin BART → MENTIONS_PERSON miss
  rate on real episodes. Anecdotal "paraphrases to 'the host'" is the
  observation that motivates this; a real benchmark would parse N
  real episodes and count {hits, misses, false positives}.

## Files touched if the operator picks A

- `src/podcast_scraper/gi/relational_edges.py` — swap pattern match
  for spaCy NER call
- `src/podcast_scraper/gi/relational_edges.py` (new test file) —
  determinism + false-positive bound assertions
- `config/profiles/airgapped_thin.yaml` — flip
  `gi_typed_mentions_use_ner: true` after validation
- `docs/architecture/gi/ontology.md` — document the new emit
  surface

## Files touched if the operator picks B

- `tests/stack-test/stack-person-profile-cloud.spec.ts` (new) —
  cloud_thin variant of the existing spec
- `tests/stack-test/README.md` — note the env-var gate
- (Nothing in production code)
