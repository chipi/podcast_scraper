# RFC-125: Translation Confidence and Claim Verification

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Pipeline (GI, KG), Positions (read-time CIL arc), player (translated-content treatment), operator review
- **Related PRDs**:
  - `docs/prd/PRD-047-multilingual-ingest.md` — FR5 (trust and confidence), FR6.3 (review worklist)
  - `docs/prd/PRD-028-position-tracker.md` — the surface this RFC gates
- **Related RFCs**:
  - `docs/rfc/RFC-124-multilingual-transcription-and-translation.md` — produces the units this RFC scores; owns `resolve_units_for_span`
  - `docs/rfc/RFC-123-speaker-turns-artifact.md` — turn/sentence IDs used in citation
  - `docs/rfc/RFC-109-per-episode-observability-manifest.md`
- **Related ADRs**:
  - `docs/adr/ADR-155-pin-every-model-checkpoint.md` — the QE model is pinned; calibration is tied to the translation model revision
  - `docs/adr/ADR-108-nli-disagreement-enrichers-gated-dark.md` — the 2026-07-08 update that made stance-over-time a read-time query rather than an enricher; this RFC gates that query

## Abstract

A translated episode is analyzed on English text that no one has checked. This RFC makes the
pipeline **know when it is unsure**. There are three mechanisms:

1. A reference-free **quality-estimation (QE)** model scores every translation unit, and each score
   is mapped to a calibrated green / amber / red band per language.
2. Every claim derived from translated text (GI quote, insight, and KG edge with evidence)
   **inherits the weakest band** among the units it cites.
3. High-stakes or low-confidence claims are **verified against the source language**, surgically,
   and the read-time Positions query refuses to build an arc from a claim that failed.

Translation stays complete (RFC-124). What is surgical is the verification.

## Problem Statement

Translation errors are not uniformly distributed, and they are not uniformly harmful. A clumsy
phrase in a summary is cosmetic. A dropped "not", a hedge flattened into a certainty, or sarcasm
read literally will invert a position. The product's primary signal is a person's position changing
over time, so an inverted position manufactures a **false position change**. That is the most
damaging output this corpus can produce.

Without per-unit confidence, the pipeline has two bad options: trust all translated output, or
distrust all of it. A native-speaker review of every claim does not scale past a bake-off. The
system needs a cheap signal everywhere and an expensive check only where it matters.

**What "a stance" actually is in this codebase, and why that changes the design.** Earlier drafts of
this RFC assumed a stance-extraction stage with a "stance writer" to hook. There is none, and there
deliberately is none:

- Stances are **GI insights**. `gi/pipeline.py` attributes each insight to the speaker of the turn
  its first grounded quote sits in (`_speaker_for_insight`), and an unattributed one is explicitly
  not a stance — `_apply_route_and_tag` sets `surfaceable: False` and routes it to `connect` rather
  than `surface`, because "an unattributed stance is not a stance, it is a floating opinion that
  nobody holds".
- **Positions are a read-time query.** `position_arc` (`server/cil_queries.py:636`) and
  `topic_conversation_arc` (`:842`) build the per-(person, topic) arc at request time from GI
  insights, their supporting quotes and KG episode metadata. `enrichment/profile_sets.py:144-146`
  records the decision in the code itself: "Per-person / per-topic stance-over-time is now a
  read-time CIL query (conversation-arc / position-arc), not a gated enricher." ADR-108's 2026-07-08
  update removed the `stance_timeline` enricher for exactly this reason.

So there is one writer (the GI artifact builder) and the gate belongs in the **read path**. This is
better than the alternative, not a compromise: a gate in the read path applies to the 678 episodes
already in production the moment a verification record exists, with no re-extraction, and a
re-verification changes what the arc returns without rewriting a single artifact.

**Use cases:**

1. **Weak-link marking.** An insight grounded in a red-band unit is shown with a low-confidence
   marker.
2. **Position gate.** A translated insight enters a position arc only after its cited source text is
   shown to support it.
3. **Operator triage.** Contradicted claims appear in a review worklist with the source and the
   translation side by side.

## Goals

1. **Per-unit QE** on 100% of translated units, with a model- and language-calibrated band.
2. **Deterministic propagation**: every claim on a translated episode carries a
   `translation_confidence` derived from its citations.
3. **Position verification**: in v1, 100% of position-bearing translated insights get a verification
   outcome before they can appear in an arc.
4. **Cheap by default**: the verification cost scales with the number of claims, not with episode
   length.
5. **No new suppression path.** Confidence composes with the existing `surfaceable` / `routing_tag` /
   `salience` machinery instead of adding a second, parallel notion of "do not show this".

## Constraints & Assumptions

**Constraints:**

- Runs on the DGX. No cloud calls, for the same pinning and provenance reasons as RFC-124.
- A calibration is valid only for a (translation model revision, QE model revision, language)
  triple. If any of them changes, the band assignment is recomputed.
- It does not block the episode. QE and verification failures degrade confidence and gate the arc.
  They never unpublish the source transcript or the subtitles.
- **No new top-level keys in `gi.json`.** `GiArtifact` is `extra="forbid"`
  (`gi/contracts.py`), so every field this RFC adds lives inside node `properties` dicts. A change
  to the top-level shape would require a `schema_version` bump and a migration.

**Assumptions:**

- QE scores are **relative**. They rank units within a language well, but absolute thresholds do
  not transfer across languages. Thresholds are therefore calibrated per language against RFC-124
  §7 reviewer labels.
- Claims on translated episodes cite English char spans in the analysis transcript
  (`…en.adfree.txt`), and those spans resolve to units via RFC-124's
  `resolve_units_for_span` — the adfree→full-English→unit→source chain. This RFC does not
  reimplement that lookup.

## Design & Implementation

### 1. QE stage

It runs immediately after the translation stage and before the English ad-free base is built, so
every downstream artifact is produced from already-scored units.

- **Input**: `(src_text, en_text)` per unit from `translation.json`. The pair is exactly the
  translation unit (RFC-124 §5.1). This keeps scores 1:1 with units, cues and citations, and unit
  lengths (a sentence group of about 120 words or fewer) are inside the range QE models are trained
  on.
- **Model**: MetricX-24 in QE (reference-free) mode is the default candidate; the Unbabel family
  (COMETKiwi, and xCOMET for error spans) is the alternative. Several checkpoints in this space carry
  non-commercial licenses, so **license verification is part of adoption** alongside ADR-155
  pinning, and is unverified in this pass. The choice is made in the RFC-124 bake-off by correlation
  with reviewer labels, not by reputation.
- **Output**: written into `translation.json`, per unit:

```json
"qe": {
  "model": "<qe-model>@<sha>",
  "raw": 3.8,
  "band": "amber",
  "calibration": "el@tx:<sha>/qe:<sha>@2026-10-01"
}
```

- **Bands** come from `config/translation_qe_calibration.yaml`, one entry per language × model
  pair:

```yaml
el:
  translation_model: <winner>@<sha>
  qe_model: <qe-model>@<sha>
  direction: lower_is_better
  green_max: 2.5     # thresholds from bake-off reviewer labels
  amber_max: 5.0     # above amber_max = red
  fitted_on: 180 units, 2026-10-01
```

**Fitting.** Green is chosen so that at least 95% of reviewer-labeled green units were
meaning-preserving. Red is chosen so that at least 50% of reviewer-labeled red units had a meaning
error. Amber is everything in between. The fitting script and the labeled CSV are committed. Any
unit scored without a matching calibration entry gets band `uncalibrated`, which is treated as
amber.

### 2. Confidence propagation

**The rule: a claim's `translation_confidence` is the minimum band over all units its citations
intersect.** The weakest link governs. Averages would let one inverted sentence hide inside a
well-translated paragraph.

- **Span → units**: `resolve_units_for_span` (RFC-124 §4). A span that crosses a unit boundary
  intersects both units. A span adjacent to an excised ad range is still exact, because the ad-map
  shift is applied before the lookup.
- **Computed at write time, in one place.** The GI artifact builder is the only writer of quote and
  insight nodes, so `translation_confidence(spans, translation_map)` is called there, once per node,
  and the result is stored in the node's `properties`:

```json
"translation": {
  "translated": true,
  "source_language": "el",
  "confidence": "amber",
  "unit_ids": ["t0031.u02"],
  "verification": null
}
```

  The KG evidence writer does the same for edges that carry evidence spans. There is no third call
  site, because there is no stance stage.
- **Insights** take the minimum over their supporting quotes.
- **Composition with the existing gate.** `surfaceable`, `routing_tag` and `salience` are unchanged.
  Translation confidence is an additional, orthogonal property; a red band does not rewrite
  `routing_tag`. The consumers that already read `routing_tag` keep working, and the ones that care
  about translation read the new block. Only the Positions read path (§4) turns a band plus a
  verification outcome into an exclusion.
- **Aggregates** (summaries, topic clusters) do not carry per-claim confidence. They carry the
  episode's `translated: true` and a flagged-unit ratio in their manifest.
- **Invalidation** is provenance-keyed. If `translation.json` is regenerated (new model revision),
  its hash changes and every claim computed from it is recomputed (re-scored, not re-extracted)
  where the spans still resolve, or queued for re-extraction where they do not. Note RFC-124 §8: a
  regeneration also invalidates the episode's cached prompt prefix.

### 3. Verification

**3.1 Triggers (v1):**

| Claim type | Trigger |
|---|---|
| Insight that is position-bearing (attributed, `surfaceable`, linked to a topic via `ABOUT`) | **always** (any band) |
| Other GI quote / insight | band = red |
| KG edge with evidence | band = red |

Position-bearing insights are always verified in v1 because they are few per episode, they carry the
highest stakes, and green-band inversions are rare but not impossible. Revisit once there is
production data on how often green-band claims fail.

The trigger is computable from properties the pipeline already writes — `speaker_id` non-null,
`surfaceable: true`, and an `ABOUT` edge to a topic — which are exactly the conditions
`position_arc` itself uses to select rows.

**3.2 Methods**, in order:

1. **Source-grounded entailment (primary).** Give the multilingual LLM already served on the DGX
   (Qwen3-30B-A3B) the **source-language** text of the cited units, plus one turn of source context
   on each side, and the English claim. The model answers `supports | contradicts | insufficient`
   with a short rationale, constrained to JSON.
2. **Cross-model re-translation (on `insufficient`).** Re-translate the cited units with the
   runner-up model from the bake-off, then re-run the same entailment check on that English. If the
   claim holds, it is `supports`. If it inverts, it is `contradicts`. If it is still unclear, it is
   `insufficient`.

**Why these two methods.** Method 1 bypasses translation entirely, so it catches exactly the failure
we fear most. It uses a generalist model on source text, which is fine for a yes/no entailment
judgment even where that model is not the best translator. Method 2 catches failures where method 1
cannot decide, and it tests robustness to translation variance: a claim that survives two
independent translations is unlikely to be a translation artifact.

**GPU ordering note.** Method 1 needs the Qwen service and method 2 needs the translation service.
The DGX convention is one vLLM at a time (`gpu-mode`), so verification is a batched pass per episode
with method-2 work queued for the translation slot, not an interleaved per-claim call.

**3.3 Outcomes:**

| Outcome | Condition | Effect |
|---|---|---|
| `verified` | method 1 `supports`, or method 2 holds | eligible for position arcs; normal display with translated marker |
| `unverified` | both methods `insufficient` | visible on the episode with marker; **excluded from position arcs** |
| `contradicted` | either method `contradicts` | excluded from arcs and from episode stance display; added to review worklist |

A human review in the worklist can set `verified` or `rejected` manually, and that overrides the
automatic outcome, with provenance recorded as `verification.by: operator`.

**3.4 Verification record**, inside the node's `properties.translation.verification`:

```json
"verification": {
  "outcome": "verified",
  "method": "source_entailment",
  "model": "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4@<sha>",
  "rationale": "…",
  "at": "2026-10-03T12:00:00Z"
}
```

### 4. The Positions gate lives in the read path

`position_arc` selects insights for a (person, topic) pair and orders them by `Episode.publish_date`
then `position_hint`. It gains one filter:

> An insight whose `properties.translation.translated` is true is included only when
> `properties.translation.verification.outcome == "verified"`.

Consequences worth stating plainly:

- **Retroactive.** The filter is evaluated per request, so the day a verification pass finishes, the
  arcs change. No artifact rewrite, no reindex.
- **Fail-closed by default.** An insight from a translated episode with `verification: null` — the
  state during Phase 1 and 2, before verification ships — is *not* in an arc. That is the desired
  default and it means PRD-047's "translated positions are withheld until trust ships" is enforced
  by the absence of a record rather than by a feature flag someone must remember to set.
- **English episodes are untouched.** `translated` is absent on every existing node, so the filter is
  a no-op for the current corpus. This is testable as a byte-identical arc response on the English
  fixture corpus.
- `topic_conversation_arc` aggregates the same insights into weekly volume × sentiment buckets and
  applies the same filter, so a contradicted claim cannot influence the aggregate shape either.

### 5. Surfaces

- **Player** (details in a UXS): translated content carries a "Translated from Greek" chip.
  Amber and red show a confidence marker. **Green shows no extra marker**, to avoid noise on
  well-translated content. A reveal shows the source sentence and plays the source audio. Arc rows
  show the translated chip. Unverified claims never appear in arcs.
- **API**: the `translation` block (§2) is serialized on quotes, insights and arc rows.
  `SupportingQuote` gains an optional `translation` field (additive; it is a plain `BaseModel`, so
  this is safe).
- **Operator review worklist**: a JSONL export plus a minimal operator view listing contradicted
  and unverified claims. Each row has the source text, the English text, the claim, both method
  results, and audio timestamps. Actions are verify, reject, or re-translate (with a model
  override, recorded).
- **Manifest** (RFC-109):
  `translation.qe: {green, amber, red, uncalibrated}` counts and
  `translation.verification: {verified, unverified, contradicted}` counts.

### 6. Monitoring the machinery itself

- **Calibration drift**: a weekly sample of 20 units per enabled language goes to a native reviewer.
  If green-band precision falls below 90%, alert and refit.
- **Verification health**: the rate of `contradicted` per language and model. A rising rate means
  translation regressed or the calibration drifted.
- **Cost**: verification calls per episode, and GPU seconds for QE and verification per episode.
  The target is ≤ 15% of translation GPU time.
- **Gate efficacy**: the count of insights excluded from arcs per language, split by reason
  (`null` / `unverified` / `contradicted`). A large `null` bucket means a verification backlog, not a
  quality problem, and the two must not look alike on a dashboard.

## Key Decisions

1. **Minimum, not mean, for propagation.**
   - **Rationale**: position inversions live in single sentences, and averaging hides them.
2. **The Positions gate is a read-time filter in `position_arc`, not a write-time suppression.**
   - **Rationale**: Positions are already a read-time query, so this is where the decision belongs.
     It also makes the gate retroactive and fail-closed for free.
3. **Always verify position-bearing translated insights in v1.**
   - **Rationale**: the time axis makes false flips the costliest error, and the volume is small.
4. **Verify against source, not against another translation, first.**
   - **Rationale**: it is the only method that is independent of translation error.
5. **Per-language, per-model calibration.**
   - **Rationale**: QE scores are relative, and bands without calibration would imply a precision we
     do not have.
6. **Confidence degrades display. It never deletes the record.**
   - **Rationale**: the source transcript and subtitles are always available, and exclusion applies
     only to arcs and to contradicted claims.
7. **Everything lives in node `properties`, not in new top-level artifact keys.**
   - **Rationale**: `GiArtifact` forbids extra top-level fields, so the alternative is a
     `schema_version` bump and a corpus migration for what is additive per-node data.

## Alternatives Considered

1. **No QE; verify everything.**
   - **Cons**: verification cost scales with every claim, and there is no signal for quotes and
     insights shown outside arcs.
   - **Why rejected**: cost, and blindness everywhere except positions.
2. **Back-translation (en → source) similarity as confidence.**
   - **Cons**: known to mask errors, because a model often round-trips its own mistakes cleanly.
   - **Why rejected**: a weak signal.
3. **LLM self-rated confidence** (the translator scores itself).
   - **Cons**: poorly calibrated, and correlated with the errors it should catch.
   - **Why rejected**: not independent.
4. **Native-speaker review of all positions.**
   - **Why rejected**: does not scale. It is kept as the bake-off label source and the drift
     monitor.
5. **Block the whole episode's intelligence below a mean QE threshold.**
   - **Why rejected**: too blunt. One bad segment would hide a good episode, and one bad sentence
     inside a good episode would pass.
6. **Write the exclusion into `routing_tag` / `surfaceable` at extraction time.**
   - **Pros**: one gate instead of two, and existing consumers honour it with no change.
   - **Cons**: it conflates "nobody holds this claim" with "we do not trust the translation", which
     are different facts with different remedies; it is not retroactive; and a later verification
     would have to rewrite artifacts to undo it.
   - **Why rejected**: the two signals must stay separable. The read path composes them.

## Testing Strategy

- **Unit**: span → unit intersection via `resolve_units_for_span` (boundary-crossing spans; spans
  entirely inside one unit; spans next to an excised ad range); min-band propagation; the
  `uncalibrated` → amber fallback; outcome state transitions, including operator-override
  precedence; the `position_arc` filter's truth table over
  {absent, null, verified, unverified, contradicted}.
- **Integration**: a fixture translated episode with injected faults. A negation is deleted in one
  unit's English, and an insight is extracted from that unit. The test asserts the insight is
  `contradicted`, is absent from `position_arc` and from `topic_conversation_arc`, is still present
  on the episode view with a marker, and appears in the worklist.
- **Isolation**: `position_arc` and `topic_conversation_arc` responses are byte-identical on the
  English fixture corpus with this RFC's filter present and absent.
- **Calibration**: the fitting script is tested on a synthetic labeled set with known thresholds.
- **Eval**: on the bake-off set, measure verification recall on reviewer-labeled meaning errors in
  position-bearing units. The target is at least 90% of inversions caught as `contradicted` or
  `unverified`.

## Rollout & Monitoring

- **Phase 1**: the QE stage and bands (observability only; nothing gated). Collect distributions on
  gated feeds. Translated insights are already absent from arcs at this point, because the §4 filter
  is fail-closed on a missing verification record.
- **Phase 2**: propagation blocks written on all claims of translated episodes.
- **Phase 3**: verification pass plus the `position_arc` filter, enforced.
- **Phase 4**: player surfaces and the operator worklist.

**Success criteria:**

1. 100% of translated units have a band. 100% of position-bearing translated insights have a
   verification outcome before appearing in any arc.
2. At least 90% of reviewer-labeled position inversions are caught (contradicted or unverified) on
   the eval set.
3. Green-band precision stays at 90% or higher in weekly drift samples.
4. Zero change to English-corpus arc responses (isolation test).

## Open Questions

1. Should red-band quotes be excluded from shareable quote cards entirely (PRD-047 OQ4)? Proposed:
   yes.
2. Should verified translated claims be weighted equally with native-English ones in arc
   aggregation, or carry a visible provenance distinction only?
3. Should Method 1 use a stronger model than Qwen3-30B for Balkan languages, if the bake-off shows
   weak source comprehension there?
4. Is xCOMET-style error-span highlighting worth adopting later for the reveal UI, if a
   commercially licensed checkpoint exists?
5. `position_hint` is a 4-step waterfall arithmetic over insight properties (RFC-097). Does a
   translated, verified insight need a `position_hint` adjustment, or is the verification outcome
   sufficient? Unknown until the first translated corpus exists.

## References

- `src/podcast_scraper/gi/contracts.py` — `EvidenceSpan`, `SupportingQuote`, `GiArtifact` (`extra="forbid"`)
- `src/podcast_scraper/gi/grounding.py` — verbatim span grounding (QA + NLI)
- `src/podcast_scraper/gi/pipeline.py` — `_speaker_for_insight`, `_apply_route_and_tag` (`surfaceable`, `routing_tag`, `salience`)
- `src/podcast_scraper/server/cil_queries.py:636,842` — `position_arc`, `topic_conversation_arc`
- `src/podcast_scraper/enrichment/profile_sets.py:144-146` — stance-over-time is a read-time query
- `docs/adr/ADR-108-nli-disagreement-enrichers-gated-dark.md` — 2026-07-08 update retiring the `stance_timeline` enricher
- `docs/rfc/RFC-124-multilingual-transcription-and-translation.md` — `translation.json` schema, `resolve_units_for_span`, bake-off (§7)
