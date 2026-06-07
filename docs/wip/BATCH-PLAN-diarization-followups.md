# Batch plan — diarization follow-ups (one PR on `feat/diarization-followups`)

Now that diarization has landed (#895/#901/#908), this batch closes the
speaker-attribution loop end to end and then validates the whole graph. All
items ship in **one PR** on `feat/diarization-followups`.

## Items

| # | Item | Issue | Type | Status |
| - | ---- | ----- | ---- | ------ |
| 1 | Full speaker-ID — **approach A**: unify `gi/speakers.py` on the diarizer's already-named `speaker_label` (reuse N-capable `map_speakers_to_names`); fix panels/multi-guest | #875 | implement | ready |
| 2 | Reprocess the 90 `whisper_transcription` episodes → corpus-wide `SPOKEN_BY` | #876 | implement | depends on #1 |
| 3 | Cross-episode person identity — *land + surface* the payoff of #1/#2 (validate corpus-wide `person_profile` / `who_said` / `positions_of`) | #909 | implement + validate | depends on #1/#2 |
| 4 | Coverage debt (B-tier tests) | — | tests | ready (see `COVERAGE-DEBT-deepgram-diarization-pr908.md`) |
| 5 | **Graph validation + all-tests focus** | — | validation | final phase (protocol below) |

## Approach decisions (locked)

- **#875 = approach A** (early-naming via diarized `speaker_label`, not the
  cluster-as-signal/late-naming re-architecture).

## Final phase — graph validation protocol (blind notes)

After items 1–4 land:

1. Operator does a full graph + test review and **does not share their gaps yet**.
2. Agent runs the **first few rounds of graph/test validation independently**.
3. Operator then shares their observed gaps (incl. issues noticed in 2.6.1).
4. **Compare notes** — the union drives the fix list.

This keeps the agent's testing unbiased by the operator's prior observations.

### Round-1 outcome (2026-06-07) — methodology correction

Round-1 was run against the GI **stub** path (`model_version="test"`) — which we
do NOT use for validation. Its findings (F1 "empty person profile", F2
"mentioned vs speaking divergence") were **stub artifacts, discarded**. The real
finding: **no current validation data carries the speaker layer.**

- `viewer-validation-corpus/v2` = synthetic **artifacts only** (no transcripts);
  its generator emits `Topic/Insight/Quote/Episode` + `MENTIONS/HAS_INSIGHT/HAS_QUOTE`
  — **zero `SPOKEN_BY` / `Person`**. So it cannot validate #875/#876/#909 or the
  viewer Person surfaces.
- The v2 **audio/transcript** family (`tests/fixtures/{audio,transcripts}/v2`, 32 eps)
  DOES carry named speakers (#111) + commercials (#109, 21/32) — but emitting
  `SPOKEN_BY` through it is the ML/diarization pipeline (heavy, gated).

### Next step after rebase — `viewer-validation-corpus/v3`

Build a **v3** synthetic generator that adds the speaker layer (deterministic, no
ML), so the person/speaker graph is validatable in ci-fast:

- `Person` nodes + `SPOKEN_BY` edges from named speaker turns.
- A **recurring guest** across ≥2 episodes (one `person:{slug}`) → #909.
- A **panel** episode (≥3 named speakers) → #875.
- A **mentioned-only** person (KG entity, no `SPOKEN_BY`) → models the F2
  distinction deliberately so we can decide on it with real-shaped data.
- Keep v2 content (commercials etc.). Bump to v3 (don't mutate v2 — keeps the
  N-1 corpus-compat tests stable).

Then: re-run graph/CIL/viewer validation against v3 → finally the
v2-audio→diarization e2e as the realistic proof (option B).

### v3 spec — production-faithful coverage (all shipped features)

Goal: v3 mirrors a real corpus as closely as a deterministic, no-ML synthetic can,
so every viewer surface + graph/CIL query has representative data. **Fidelity rule:**
derive the per-episode artifact shapes from what the **real pipeline emits** (run a
few v2-audio episodes through the full pipeline with all features on, snapshot the
node/edge shapes, reproduce them deterministically in the generator) — don't guess.

Must carry:

- **Speaker layer (#875/#876/#909):** `Person` nodes + `SPOKEN_BY` (Quote→Person) +
  derived `Person→Insight`; named host+guest per episode; a **recurring guest** across
  ≥2 episodes (one `person:{slug}`); a **panel** episode (≥3 named speakers); a
  **mentioned-only** person (KG entity, no `SPOKEN_BY`) for the F2 case.
- **GI:** Episode/Insight/Quote/Topic/Person nodes; `HAS_INSIGHT`/`HAS_QUOTE`/
  `SUPPORTED_BY`/`ABOUT` edges; quotes with **timestamps** (diarized segment timing) +
  `speaker_id`.
- **Commercials (#109 + commercial phases):** episodes with in-segment sponsor reads +
  the ad-region/cleaning metadata the pipeline produces.
- **KG + CIL (#851/#852/#854):** Entity (person/org/topic) with canonical ids; **aliases**
  (a person/org by variant names → one canonical id); cross-episode canonical recurrence.
- **Bridge (GI↔KG):** `bridge.json` identities joining the layers per episode.
- **Relational/search (PRD-033 / RFC-094):** `HAS_EPISODE` (Podcast→Episode), `MENTIONS`
  (Insight→Entity); cross-cutting umbrella topics (already in v2) → topic clusters;
  corpus-graph union reachability Person→Insight.
- **Index-buildable (#897/#899):** `make build-validation-index` works (FAISS +
  `topic_clusters.json`); `vector_embedding_provider` default.
- **Viewer API surfaces (`corpus/*.json`):** `feeds`/`episodes`/`persons-top`/`digest`/
  `coverage`/`stats` regenerated — `persons-top` now reflects **speaking** persons, not
  only mentioned.
- **Content patterns (#900):** recurring guests, cross-episode topics, **position arcs**
  (a person's stance evolving across episodes), edge cases.

Bump to **v3** (keep v2 for N-1 compat). Regenerate via the extended
`scripts/build_synthetic_validation_corpus.py`; keep it deterministic + idempotent.

### Graph viewer — diarization support (review 2026-06-07)

The viewer graph **works** with diarization data: Person nodes render (styled
`Entity_person`), SPOKEN_BY edges render, persons unify into **one cross-episode
node** (viewer `mergeGiKg` DEDUP + backend `corpus_graph` `_upsert_node`), and
node/edge filter chips auto-generate (Person + SPOKEN_BY visible by default). Gaps
are **polish, not blockers** — fix during/after v3 validation:

1. **No dedicated `SPOKEN_BY` (and `HAS_QUOTE`) edge style** — falls through to the
   generic muted-gray edge; the `(unknown)` fallback doesn't catch it.
   `cyGraphStylesheet.ts` `edgeStrokes` (~483-557) — add a `SPOKEN_BY` selector.
2. **No 1-hop Person→Insight graph edge** — backend `corpus_graph` derives a
   `STATES` shortcut but the Cytoscape graph only has the 2-hop
   Person←SPOKEN_BY←Quote→SUPPORTED_BY→Insight path (Episode→Person aggregate is
   off by default). Decide whether to surface a derived person→insight edge.
3. **Person filter chip count/swatch bug** — chip keys raw `"Person"` but the
   histogram/colors key the visual group `"Entity_person"` → shows `(0)` + gray
   swatch (toggle still works). `GraphTypesChip.vue` vs `GraphCanvas.vue`/`colors.ts`.

## Decided AGAINST (from the early identity-vision audit)

The five `docs/wip/rfc_*` / `product_ux_identity_vision` drafts (~1.5-month-old
"super-early thinking") were audited against the shipped system: ~70–80% already
delivered (CIL resolver, `person_profile`, all viewer surfaces via PRD-033
issues #882–#890, diarization). The surviving high-value thread is **#909**. Explicitly
**not** pursued (over-engineered for a host+guest corpus):

- Speaker-clusters-as-signal → late-naming / resolve-in-CIL re-architecture.
- Conversation graph / `RESPONDS_TO` edges (trivial in host↔guest).
- CIL confidence/provenance "single source of truth" consolidation.
- Voice-based linking for *unnamed* speakers (expensive ~5%).
- Conversation View UI — deferred unless concrete user pull.

→ The five draft files in `docs/wip/` are superseded by this audit + #909 and can
be deleted once the operator confirms.

## Out of this batch / inputs pending

- Operator's concrete **2.6.1 graph issues** — shared in the final-phase protocol,
  not before.
