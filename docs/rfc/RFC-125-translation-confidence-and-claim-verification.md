# RFC-125: Translation Confidence and Claim Verification

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Pipeline (GI, KG), Positions (read-time CIL arc), player (translated-content treatment), operator review
- **Related PRDs**:
  - `docs/prd/PRD-047-multilingual-ingest.md` — FR5 (trust and confidence), FR6.3 (review worklist)
  - `docs/prd/PRD-028-position-tracker.md` — the surface this RFC gates
- **Related RFCs**:
  - `docs/rfc/RFC-124-multilingual-transcription-and-translation.md` — produces the units this RFC reasons over; owns `resolve_units_for_span`
  - `docs/rfc/RFC-123-speaker-turns-artifact.md` — turn/sentence IDs used in citation
  - `docs/rfc/RFC-109-per-episode-observability-manifest.md`
- **Related ADRs**:
  - `docs/adr/ADR-155-pin-every-model-checkpoint.md` — the verification model is pinned
  - `docs/adr/ADR-108-nli-disagreement-enrichers-gated-dark.md` — the 2026-07-08 update that made stance-over-time a read-time query rather than an enricher; this RFC gates that query
- **Arc notes**: `docs/architecture/MULTILINGUAL_ARC.md` (§4 slice plan, D-5, D-6)

## Abstract

A translated episode is analyzed on English text that no one has checked. This RFC is about closing
that gap — and about being precise that **v1 does not close it.**

**v1 records provenance and nothing else.** Every claim derived from translated text carries which
translation units it rests on, in which source language, resolved through RFC-124's unit map. That block
is written on every quote and insight, and it is **invisible to the user**. A translated episode has the
same standing as a native-English one on every surface: no gate, no filter, no marker.

That is a deliberate product decision (arc D-36, D-37), and the reasoning is worth stating because the
opposite was specified first. A gate is only meaningful if something can release what it holds. With no
verifier in v1, a gate's only possible behaviour is "hide every translated claim, permanently" — which
means paying for translation and extraction to produce data nobody can see. The gate belongs with the
verifier, so both are v2, along with the user-visible label.

**Everything else in this document is v2**, and it is written now because the v1 provenance block is
shaped to accept it without reprocessing:

1. **Source verification** — give a model the source-language text of the cited units and the English
   claim, and ask whether the source supports it (§2).
2. **A read-time gate** over the nine position-bearing surfaces, keyed on that outcome (§3).
3. **A user-visible label** wherever translated content renders (§4).
4. **Quality estimation** with calibrated bands (§7).

The v1 bet is explicit: we believe the translation, we choose a model on measured evidence at Gate V, and
quality is what v2 is *about* rather than something v1 hedges around by withholding output.

## Problem Statement

Translation errors are not uniformly distributed, and they are not uniformly harmful. A clumsy
phrase in a summary is cosmetic. A dropped "not", a hedge flattened into a certainty, or sarcasm
read literally will invert a position. The product's primary signal is a person's position changing
over time, so an inverted position manufactures a **false position change**. That is the most
damaging output this corpus can produce.

Without any trust machinery the pipeline has two bad options: trust all translated output, or
distrust all of it. A native-speaker review of every claim does not scale past a bake-off. The system
needs an expensive check exactly where the stakes are, and honest provenance everywhere else.

**What "a stance" actually is in this codebase, and why that changes the design.** Earlier drafts of
this RFC assumed a stance-extraction stage with a "stance writer" to hook. There is none, and there
deliberately is none:

- Stances are **GI insights**. `gi/pipeline.py` attributes each insight to the speaker of the turn
  its first grounded quote sits in (`_speaker_for_insight`), and an unattributed one is explicitly
  not a stance — the codebase's own words are "an unattributed STANCE is not a stance, it is a
  floating opinion that nobody holds". Mechanically, `surfaceable` is set by `_apply_voice_flags`
  (`gi/pipeline.py:1100-1124`); `_apply_route_and_tag` (`:2119-2143`) only *reads* it to derive
  `routing_tag` and `salience`.
- **Positions are a read-time query.** `position_arc` (`server/cil_queries.py:636`) and
  `topic_conversation_arc` (`:842`) build the per-(person, topic) arc at request time from GI
  insights, their supporting quotes and KG episode metadata. `enrichment/profile_sets.py:144-146`
  records the decision in the code itself: "Per-person / per-topic stance-over-time is now a
  read-time CIL query (conversation-arc / position-arc), not a gated enricher." ADR-108's 2026-07-08
  update removed the `stance_timeline` enricher for exactly this reason.

So there is one writer (the GI artifact builder) and any gate belongs in the **read path**. That is what
makes deferring it safe: a read-path gate applies to every episode already in the corpus the moment
verification records exist, with no re-extraction, and re-verifying changes what a surface returns without
rewriting a single artifact. Nothing about shipping the gate later costs more than shipping it now.

**Use cases:**

1. **Honest provenance (v1).** Any translated claim can say which source sentences it rests on, which
   makes every later mechanism possible.
2. **Position gate (v2).** A translated insight enters a position surface only once its cited source text
   is shown to support it.
3. **Labelled content (v2).** A reader can tell translated content from native, which matters most for
   quotes since those get passed on as somebody's words.
4. **Operator triage (v2).** Contradicted claims appear in a review worklist with source and translation
   side by side.

## Goals

**v1 — one goal:**

1. **Deterministic provenance**: every claim on a translated episode records the units it cites and the
   source language, computed from RFC-124's map rather than guessed — and shaped so that verification,
   gating, labelling and QE can all be added later **without reprocessing the corpus**.

**v2 — the rest:**

2. **Position verification**: every position-bearing translated claim gets a verification outcome.
3. **A gate keyed on that outcome**, across every position-bearing surface (§3).
4. **A user-visible label** wherever translated content renders (§4).
5. **Cheap by default**: verification cost scales with the number of position-bearing claims, not with
   episode length.
6. **No new suppression path.** Provenance and verification compose with the existing `surfaceable` /
   `routing_tag` / `salience` machinery rather than adding a second, parallel notion of "do not show
   this".
7. **Quality estimation** (§7) attaches without reshaping anything v1 wrote.

## Constraints & Assumptions

**Constraints:**

- Runs on the DGX. No cloud calls, for the same pinning and provenance reasons as RFC-124.
- It does not block the episode. Verification failures degrade display and gate the arc. They never
  unpublish the source transcript or the subtitles.
- **No new top-level keys in `gi.json`.** `GiArtifact` is `extra="forbid"` (`gi/contracts.py`), so
  every field this RFC adds lives inside node `properties` dicts. A change to the top-level shape
  would require a `schema_version` bump and a migration.
- The verification model is pinned (ADR-155). Its revision is recorded on every verification record,
  because a model change changes what "verified" meant.

**Assumptions:**

- Claims on translated episodes cite English char spans in the analysis transcript
  (`…en.adfree.txt`), and those spans resolve to units via RFC-124's `resolve_units_for_span` — which
  binary-searches `.en.adfree.segments.json` (offsets there are exact by construction), reads the
  `unit_id` carried on the matched segment, and looks that unit up in `translation.json`. It does
  **not** go through the ad-map: that cannot invert an ad-free transform which re-renders survivors
  (RFC-124 §4.1, measured). This RFC does not reimplement the lookup.
- A generalist multilingual LLM is good enough at a yes/no entailment judgment over source-language
  text even where it is not the best translator of that language. §5 monitors whether that holds;
  Open Question 3 is the escape hatch if it does not.

## Design & Implementation

### 1. Translation provenance on every claim — **the whole of v1**

**Computed at write time, wherever `gi.json` is written.** The GI artifact builder is the main writer
of quote and insight nodes, but it is **not the only one**: `add_spoken_by_edges(replace=True)` and
`gi/repair.py` both rewrite `gi.json` in place, so each must preserve or recompute the block rather
than drop it. Every writer calls `resolve_units_for_span(spans, segments, translation_map)` — which
resolves through `.en.adfree.segments.json` and the `unit_id` carried on each segment, **not** through
the ad-map (RFC-124 §4.1; the ad-map cannot invert the ad-free transform) — and stores the result in
the node's `properties`:

```json
"translation": {
  "translated": true,
  "source_language": "el",
  "unit_ids": ["t0031.u02"],
  "verification": null
}
```

There is no stance-stage call site, because there is no stance stage. There is also **no KG evidence
writer** to hook: a search of `kg/schema.py`, `kg/pipeline.py` and `kg/llm_extract.py` found no KG
edges carrying evidence spans at all. An earlier version of this RFC asserted one; it referred to
nothing. If KG evidence spans are added later, they join this list.

- **Insights** union the unit ids of their supporting quotes.
- **Spans that cross a unit boundary** intersect both units and record both.
- **English episodes** get no `translation` block at all. Its absence is the signal, so nothing about
  the existing corpus changes.
- **Aggregates** (summaries, topic clusters) do not carry per-claim provenance. They carry the
  episode's `translated: true` and its unit count in the manifest.
- **Composition with the existing gate.** `surfaceable`, `routing_tag` and `salience` are unchanged.
  Translation provenance is an additional, orthogonal property. Only the Positions read path (§3)
  turns it into an exclusion.
- **Invalidation** is provenance-keyed. If `translation.json` is regenerated (a new model revision),
  its hash changes; every claim computed from it has its `unit_ids` recomputed where the spans still
  resolve, and is queued for re-extraction where they do not. **Every verification record is
  invalidated**, because it was a judgment about text that no longer exists. Note RFC-124 §8: a
  regeneration also invalidates the episode's cached prompt prefix.

### 2. Verification

**2.1 What gets verified in v1.** Only **position-bearing** claims, and all of them:

| Claim type | v1 trigger |
| --- | --- |
| Insight that is **position-bearing**: it has a `SPOKEN_BY`-supported quote, an `ABOUT` edge to a topic, and `insight_type == "claim"` | **always** |
| Other GI quote / insight | not verified in v1 — carries provenance and a translated marker |

**The trigger is the read path's own predicate, copied deliberately.** `position_arc`
(`server/cil_queries.py:650-682`) selects rows on exactly those three conditions — it does **not**
read `surfaceable` or `speaker_id`, which an earlier version of this RFC claimed. Since the trigger
and the gate must not diverge, both are expressed with **one shared predicate helper** rather than two
hand-kept lists; that is what keeps them aligned, not a coincidence of properties.

Without QE there is no band to trigger on for the other rows, which is the honest consequence of
D-6: in v1 a non-position translated quote is **labelled but unchecked**. It is shown with a
"Translated from <Language>" chip and its source sentence one tap away, and it never enters a
timeline. v2's bands are what let us spend verification on the worst of those rows (§7).

**2.2 Methods**, in order:

1. **Source-grounded entailment (primary).** Give the multilingual LLM already served on the DGX
   (Qwen3-30B-A3B) the **source-language** text of the cited units, plus one turn of source context
   on each side, and the English claim. The model answers `supports | contradicts | insufficient`
   with a short rationale, constrained to JSON.
2. **Cross-model re-translation (on `insufficient`).** Re-translate the cited units with the
   runner-up model from the bake-off, then re-run the same entailment check on that English. If the
   claim holds, it is `supports`. If it inverts, it is `contradicts`. If it is still unclear, it is
   `insufficient`.

**Why these two methods.** Method 1 bypasses translation entirely, so it catches exactly the failure
we fear most, and it needs no calibration — which is what makes v1 possible without a reviewer in the
loop. Method 2 catches failures where method 1 cannot decide, and it tests robustness to translation
variance: a claim that survives two independent translations is unlikely to be a translation
artifact.

**Batching.** Method 1 needs the Qwen service and method 2 needs the translation service, so
verification runs as a batched pass per episode — all method-1 claims together, then any method-2
fallbacks together — rather than an interleaved per-claim call.

**2.3 Outcomes:**

| Outcome | Condition | Effect |
| --- | --- | --- |
| `verified` | method 1 `supports`, or method 2 holds | eligible for position arcs; normal display with translated marker |
| `unverified` | both methods `insufficient` | visible on the episode with marker; **excluded from position arcs** |
| `contradicted` | either method `contradicts` | excluded from arcs and from episode stance display; added to review worklist |

A human review in the worklist can set `verified` or `rejected` manually, and that overrides the
automatic outcome, with provenance recorded as `verification.by: operator`.

**2.4 Verification record**, inside the node's `properties.translation.verification`:

```json
"verification": {
  "outcome": "verified",
  "method": "source_entailment",
  "model": "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4@<sha>",
  "rationale": "…",
  "at": "2026-10-03T12:00:00Z"
}
```

### 3. The Positions gate lives in the read path — **v2**

**Not in v1.** v1 applies no filter anywhere; this section specifies the v2 gate, and it lives here
because the v1 provenance block is what it keys on.

`position_arc` selects insights for a (person, topic) pair and orders them by `Episode.publish_date`
then `position_hint`. It would gain one filter:

> An insight whose `properties.translation.translated` is true is included only when
> `properties.translation.verification.outcome == "verified"`.

**The filter applies to every position-bearing surface.** This list has been wrong twice — first at two
entries, then at five — so it is enumerated with its evidence, and the enumeration is itself a
deliverable of the slice rather than a claim in this document:

| Surface | Location | Why it counts |
| --- | --- | --- |
| `position_arc` | `cil_queries.py:636` | the per-(person, topic) arc |
| `topic_conversation_arc` | `:842` | reuses `topic_timeline` (def `:772`), so the filter belongs in `topic_timeline` or is duplicated |
| `topic_timeline_merged` | def `:891` | the same insights across multiple topics |
| `person_profile` insights-by-topic | defs `:709`, `:752` | a position surface in all but name |
| `topic_perspective_leaders` | def `:984` | ranks people by their claims on a topic; counts over **edge ids** and never materialises insight nodes, so a node-property predicate needs an id→node map first |
| `topic_perspectives` | `:1041-1147` | "who holds which take", served to the **consumer app** (`relational_view.build_topic_perspectives` ← `routes/app_relational.py:178`) **and to OG share images** (`server/og/build.py:217`, where a claim becomes the headline quote with a person's byline) |
| `gi_view` episode stance display | `server/gi_view.py:116` | §2.3 says contradicted claims are excluded from episode stance display; this is that surface |
| `search/relational_queries.positions_of` and neighbours | `:170`, plus `who_said`, `cross_show_synthesis`, `related_insights` | served at `routes/relational.py` and via MCP; these read `CorpusGraph`, not `gi.json` nodes, so whether the `translation` block even reaches them depends on the graph builder — verify before assuming the filter can be applied |
| `enrichment/enrichers/topic_consensus.py` | — | **write time.** No read-time filter reaches it. Needs provenance at extraction, or it corroborates unverified translated claims |

**Two predicates, deliberately, not one helper.** The verification *trigger* (§2.1) and the display
*gate* are different shapes and conflating them fails in a specific way:

- `is_position_bearing(gi, insight_id)` — an **edge** predicate: SPOKEN_BY-supported quote ∩ `ABOUT` ∩
  `insight_type == "claim"`. This selects what gets verified.
- `translated_claim_admissible(node)` — a **property** predicate on the node's `translation` block. This
  selects what renders.

If the gate excluded *every* unverified translated insight while only claim-type insights were ever
verified, then translated observations and recommendations on `topic_timeline` and `person_profile`
(neither of which applies a claim filter by default) would be excluded **permanently**, not until
verification ships. A test must assert the trigger population is a superset of the gated population on
every surface. Note also that `position_arc`'s claim filter is a *default*: `routes/cil.py:86-95` lets a
caller pass `insight_types=all`, so the two predicates can diverge for that caller unless the gate is
applied at selection regardless of type.

One further inconsistency to resolve in the slice: `person_profile` returns `quotes` alongside insights
(`:753-757`). Gating the insights while serving the quotes they were built from shows the reader the
translated evidence and hides the conclusion.

Consequences worth stating plainly:

- **Retroactive.** The filter is evaluated per request, so the day a verification pass finishes, the
  arcs change. No artifact rewrite, no reindex.
- **Fail-closed.** A translated insight with `verification: null` — the state for every translated
  episode until the verification pass runs — is *not* in an arc. So PRD-047's "translated positions
  are withheld until trust ships" is enforced by the absence of a record rather than by a flag
  someone must remember to set. This is why the filter can and should ship **before** the
  verification machinery (arc slice S3.1 before S3.2).
- **English episodes are untouched.** `translated` is absent on every existing node, so the filter is
  a no-op for the current corpus. This is testable as a byte-identical arc response on the English
  fixture corpus.
- `topic_conversation_arc` aggregates the same insights into weekly volume × sentiment buckets and
  applies the same filter, so a contradicted claim cannot influence the aggregate shape either.

### 4. Surfaces

- **Player** (details in a UXS): translated content carries a "Translated from Greek" chip. A reveal
  shows the source sentence and plays the source audio. Arc rows show the translated chip. Unverified
  claims never appear in arcs. **v1 has no per-claim confidence marker**, because without QE there is
  nothing calibrated to show; that arrives with v2.
- **API**: the `translation` block (§1) is serialized on quotes, insights and arc rows.
  `SupportingQuote` gains an optional `translation` field (additive; it is a plain `BaseModel`, so
  this is safe).
- **Operator review worklist**: a JSONL export plus a minimal operator view listing contradicted
  and unverified claims. Each row has the source text, the English text, the claim, both method
  results, and audio timestamps. Actions are verify, reject, or re-translate (with a model
  override, recorded).
- **Manifest** (RFC-109): `translation.verification: {verified, unverified, contradicted, pending}`
  counts.

### 5. Monitoring the machinery itself

- **Verification health**: the rate of `contradicted` per language and per translation model. A
  rising rate means translation regressed.
- **Entailment-model competence**: the rate of `insufficient` from method 1 per language. A high rate
  means the verifier does not understand the source language well enough, which is a different
  problem from bad translation and must not look like one on a dashboard.
- **Cost**: verification calls per episode and GPU seconds per episode. The target is ≤ 15% of
  translation GPU time.
- **Gate efficacy**: insights excluded from arcs per language, split by reason
  (`null` / `unverified` / `contradicted`). A large `null` bucket is a verification backlog, not a
  quality problem, and the two must be distinguishable at a glance.
- **Spot-check drift**: a monthly sample of 20 `verified` position-bearing claims per enabled
  language goes to a native reviewer. This is the only human loop in v1, it is small, and it is a
  check on the verifier rather than a dependency of the pipeline.

## Key Decisions

1. **QE is deferred to v2; v1 is verification-only.**
   - **Rationale**: QE bands need per-language threshold fitting against native-reviewer labels,
     which is a human bottleneck on the critical path, and they need a translated corpus that does
     not exist until RFC-124 runs. Source entailment needs neither and catches the inversions that
     actually cause harm. Arc note D-6.
2. **The Positions gate is a read-time filter across every position-bearing surface, not a write-time
   suppression.**
   - **Rationale**: Positions are already read-time queries, so this is where the decision belongs, and
     it makes the gate retroactive and fail-closed for free.
   - **Caveat, not a footnote**: one surface cannot be reached this way. `enrichment/enrichers/topic_consensus.py`
     corroborates insights across people at **write** time, so no read-time filter touches it — it needs
     the translation provenance at extraction time or it will corroborate unverified translated claims.
     §3 lists it separately for that reason.
3. **Verify all position-bearing translated claims in v1, not a sample.**
   - **Rationale**: the time axis makes false flips the costliest error, and the volume per episode is
     small enough that sampling buys little and costs trust.
4. **Verify against source, not against another translation, first.**
   - **Rationale**: it is the only method that is independent of translation error.
5. **Provenance is written even where verification is not run.**
   - **Rationale**: "labelled but unchecked" is an honest state; "unlabelled" is not. A reader can
     always reach the source sentence.
6. **Confidence degrades display. It never deletes the record.**
   - **Rationale**: the source transcript and subtitles are always available, and exclusion applies
     only to arcs and to contradicted claims.
7. **Everything lives in node `properties`, not in new top-level artifact keys.**
   - **Rationale**: `GiArtifact` forbids extra top-level fields, so the alternative is a
     `schema_version` bump and a corpus migration for what is additive per-node data.
8. **A re-translation invalidates every verification record it touches.**
   - **Rationale**: the record was a judgment about a specific English string. Keeping it against
     different text would be a stale claim wearing a fresh timestamp.

## Alternatives Considered

1. **Ship QE in v1 alongside verification** (the original draft).
   - **Pros**: a signal on every unit, including the claims v1 leaves unchecked; lets verification be
     spent selectively.
   - **Cons**: the calibration step needs a native reviewer per language before any band means
     anything, which puts a human on the critical path of the whole arc; and thresholds cannot be
     fitted until a translated corpus exists, so it cannot precede Phase 2 anyway.
   - **Why rejected for v1**: sequencing. It is additive later (§7) and nothing in v1 has to change
     to accept it.
2. **No verification either; label everything and trust the reader.**
   - **Cons**: a false position flip is the worst output this corpus can produce, and a label does not
     prevent it entering a timeline.
   - **Why rejected**: the timeline is the product.
3. **Back-translation (en → source) similarity as confidence.**
   - **Cons**: known to mask errors, because a model often round-trips its own mistakes cleanly.
   - **Why rejected**: a weak signal.
4. **LLM self-rated confidence** (the translator scores itself).
   - **Cons**: poorly calibrated, and correlated with the errors it should catch.
   - **Why rejected**: not independent.
5. **Native-speaker review of all positions.**
   - **Why rejected**: does not scale. It is kept as the bake-off label source and the §5 drift
     spot-check.
6. **Write the exclusion into `routing_tag` / `surfaceable` at extraction time.**
   - **Pros**: one gate instead of two, and existing consumers honour it with no change.
   - **Cons**: it conflates "nobody holds this claim" with "we have not checked the translation",
     which are different facts with different remedies; it is not retroactive; and a later
     verification would have to rewrite artifacts to undo it.
   - **Why rejected**: the two signals must stay separable. The read path composes them.

## Testing Strategy

- **Unit**: span → unit resolution via `resolve_units_for_span` (boundary-crossing spans; spans
  entirely inside one unit; spans next to an excised ad range); insight unit-id union over supporting
  quotes; the position-bearing trigger's truth table over (SPOKEN_BY-supported quote, `ABOUT` edge,
  `insight_type`) — the three conditions `position_arc` actually selects on, not `surfaceable`, which it
  never reads; a test that the trigger population is a superset of the gated population on every
  surface;
  outcome state transitions including operator-override precedence; the `position_arc` filter over
  {absent, null, verified, unverified, contradicted}; re-translation invalidating verification
  records.
- **Integration**: a fixture translated episode with injected faults. A negation is deleted in one
  unit's English, and an insight is extracted from that unit. The test asserts the insight is
  `contradicted`, is absent from `position_arc` and `topic_conversation_arc`, is still present on the
  episode view with a marker, and appears in the worklist.
- **Isolation**: `position_arc` and `topic_conversation_arc` responses are byte-identical on the
  English fixture corpus with this RFC's filter present and absent. This is the gate protecting the
  existing production corpus.
- **Fail-closed**: a translated insight with `verification: null` is absent from every arc, asserted
  without any verification pass having run.
- **Eval**: on the bake-off set, measure verification recall on reviewer-labeled meaning errors in
  position-bearing units. The target is at least 90% of inversions caught as `contradicted` or
  `unverified`.

## Rollout & Monitoring

Slice ids refer to `docs/architecture/MULTILINGUAL_ARC.md` §4 for v1 and
`docs/architecture/MULTILINGUAL_ARC_V2.md` §3 for v2.

- **v1 (S2.11)**: the provenance block on every claim of every translated episode. Nothing reads it yet,
  nothing is filtered, nothing is labelled. It exists so that none of the v2 work below needs the corpus
  reprocessed.
- **v2 (V2-A.1)**: the verification pass, so claims acquire an outcome.
- **v2 (V2-A.3)**: the read-time gate across the nine surfaces in §3, keyed on that outcome. Whether it
  is fail-closed everywhere or split between surfaces that *assert* a position and those that merely
  *describe* is a product judgement to make then.
- **v2 (V2-A.4)**: the user-visible label, across every serialization boundary in §4.
- **v2 (V2-B.1)**: quality estimation and calibrated bands (§7).

**Success criteria:**

For **v1**, exactly one: every claim derived from translated text carries a complete, resolvable
provenance block — so that verification, gating and labelling can all be added later without
reprocessing the corpus.

For **v2**: every position-bearing translated claim has a verification outcome before it appears on a
gated surface; at least 90% of reviewer-labeled position inversions are caught on the eval set; no
translated content renders without a label; and English-corpus responses are unchanged by any of it.

## Open Questions

1. Should unchecked (non-position) translated quotes be excluded from shareable quote cards
   entirely, given v1 has no band to distinguish them by? Proposed: yes, until v2 bands exist.
2. Should verified translated claims be weighted equally with native-English ones in arc
   aggregation, or carry a visible provenance distinction only?
3. Should Method 1 use a stronger model than Qwen3-30B for Balkan languages, if the bake-off shows
   weak source comprehension there? The §5 `insufficient` rate is the trigger for revisiting.
4. `position_hint` is a 4-step waterfall arithmetic over insight properties (RFC-097). Does a
   translated, verified insight need a `position_hint` adjustment, or is the verification outcome
   sufficient? Unknown until the first translated corpus exists.

## 7. Deferred to v2: quality estimation and calibrated bands

Recorded here so the seam is explicit and v1 does not have to be reshaped to accept it.

**What it adds.** A reference-free QE model scores every `(src_text, en_text)` pair from
`translation.json` and each score maps to a calibrated green / amber / red band, stored per unit
alongside the pair. Claims then inherit the **minimum** band over the units they cite — the weakest
link governs, because averaging lets one inverted sentence hide inside a well-translated paragraph.

**What it unlocks.**

- A verification trigger for the rows v1 leaves unchecked: band = red on a non-position quote.
- A per-claim confidence marker in the UI for amber and red, with green carrying no extra noise.
- Calibration-drift monitoring as an early warning that translation quality has regressed.

**What it costs, and why that is a v2 shape.** A band is meaningless without a calibration, and a
calibration is a fitted mapping per (language, translation model revision, QE model revision) triple
— green chosen so that ≥95% of reviewer-labeled green units were meaning-preserving, red so that
≥50% of reviewer-labeled red units had a meaning error. That needs a native reviewer per language and
a translated corpus to sample. Both exist only *after* Phase 2 runs. Candidate models (MetricX-24 in
QE mode; the Unbabel COMETKiwi / xCOMET family) also need licence verification, and several
checkpoints in that space are non-commercial.

**The seam.** v1 writes `properties.translation` with `translated`, `source_language`, `unit_ids` and
`verification`. v2 adds a `qe` block per unit in `translation.json` and a `confidence` key in that
same node block. No v1 field changes meaning, and the read-time filter in §3 keys on `verification`
regardless of whether a band is present. Arc note D-6 and slice V2.1.

## References

- `src/podcast_scraper/gi/contracts.py` — `EvidenceSpan`, `SupportingQuote`, `GiArtifact` (`extra="forbid"`)
- `src/podcast_scraper/gi/grounding.py` — verbatim span grounding (QA + NLI)
- `src/podcast_scraper/gi/pipeline.py` — `_speaker_for_insight`, `_apply_route_and_tag` (`surfaceable`, `routing_tag`, `salience`)
- `src/podcast_scraper/server/cil_queries.py:636,842` — `position_arc`, `topic_conversation_arc`
- `src/podcast_scraper/enrichment/profile_sets.py:144-146` — stance-over-time is a read-time query
- `docs/adr/ADR-108-nli-disagreement-enrichers-gated-dark.md` — 2026-07-08 update retiring the `stance_timeline` enricher
- `docs/rfc/RFC-124-multilingual-transcription-and-translation.md` — `translation.json` schema, `resolve_units_for_span`, bake-off (§7)
- `docs/architecture/MULTILINGUAL_ARC.md` — arc notes, slice plan, decisions D-5 / D-6
