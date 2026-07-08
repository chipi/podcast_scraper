# Fixture capability gap analysis — what the fake feeds must prove (v3+ / enrichment era)

Analysis for **#1148**. The fake-podcast fixtures were designed to test transcription →
extraction (pipeline quality). Since then we added **enrichers, topic/person assessment
surfaces, and digest/recommendation/discovery** — none of which the fixtures were built to
exercise. This maps every capability to its fixture need and defines what v3+ must add.

## Where we are

- **Source:** screenplay transcripts `tests/fixtures/transcripts/<version>/*.txt` → built into a
  corpus by `scripts/build_app_validation_corpus.py` (no pipeline; deterministic GI/KG).
- **Active = v2**; **v3 already exists** (9 shows p01–p09, ~24 eps) but per
  `docs/guides/eval-reports/EVAL_FIXTURES_V3.md` it targets the **pipeline-quality** era:
  ASR garble, entity-canon aliases, sponsor patterns, NER, chunk boundaries, position arcs,
  recurring guests, person density, and **multi-language voice tags** (fr-CA/de-DE/pt-BR/ar-EG).
- **The build script only wires 3 shows** (p05 investing, p02 software, p03 scuba) + cross-show
  umbrella topics. v3's richer set + the new capabilities are not exercised end-to-end.

**The gap in one line:** v3 proves *"we extract a clean graph from messy audio."* It does **not**
prove *"the enrichers, topic/person surfaces, and discovery produce the right thing."*

## Capability → fixture need → current coverage

| Capability | What the fixture must contain | v2/v3 today | Gap |
|---|---|---|---|
| **grounding_rate** | insights *with* and *without* supporting quotes | partial (`low_grounding` v3) | need a controlled mix + gold |
| **insight_density** | episodes of varying insight density | partial | need known-density eps + gold |
| **temporal_velocity** | episodes **dated across months** on recurring topics | **absent** — single-snapshot dates | **time spread** |
| **guest_coappearance** | **multi-guest** episodes (2+ named guests together) | partial (`high_person_density` v3) | need explicit 2-guest eps + gold pairs |
| **topic_cooccurrence_corpus** | topics that genuinely co-occur across eps | partial (umbrellas) | need designed co-occurrence + gold |
| **topic_theme_clusters / topic_similarity** | a topic vocabulary with real semantic clusters | partial (umbrellas force clusters) | need denser, meaningful neighbours + gold |
| **nli_contradiction / disagreement (#1144)** | two speakers taking **opposing stances on the same proposition** | **absent** (#1106: 0/150; #1144: 0/40) | **the hard one — engineered opposition** |
| **multi-perspective (#1146)** | **≥2 speakers, each with attributed insights, on the SAME topic** | **absent in committed corpus** (0 perspectives → e2e mocked) | **cross-speaker topic overlap** |
| **topic assessment** (cards, clusters, timeline) | topics discussed by multiple people over time | partial | time + overlap |
| **person assessment** (cards, positions, profile) | a person across episodes, with stances | partial (`position_arc`, `recurring_guest` v3) | wire + gold |
| **digest / trending** | velocity signal → "heating up" topics | absent (no time) | time spread |
| **personalized discovery / interests / ranking (#1139)** | **per-user** playback + captures → interest tokens | some per-user e2e state exists (`app/e2e/.app-state/users/`) | need seeded user profiles + gold rankings |
| **resurfacing / consolidation / scope=mine (#1149)** | per-user heard∪captured set intersecting real topics | partial | seeded users with heard eps on covered topics |

## What v3+ must add (the concrete design)

Six deliberate structures, each with **known gold labels** so enricher *accuracy* is
deterministically checkable in CI (not just "does it render"):

1. **Time spread** — stamp episodes across a multi-month window on recurring topics → unblocks
   `temporal_velocity`, trending, topic timeline, prediction-tracking.
2. **Cross-speaker topic overlap** — ensure ≥2 shows/speakers discuss the *same* topics, each
   with attributed insights → unblocks multi-perspective (#1146, de-mocks its e2e),
   `topic_cooccurrence`, `guest_coappearance`.
3. **Engineered opposition** — author a small set of pairs where two speakers assert *not-X* vs
   *X* on the same proposition (e.g. "index funds beat active" vs "active beats index"), and/or a
   panel/debate episode → the **only** reliable source of `nli_contradiction`/disagreement
   positives at fixture scale. Tag them as gold contradiction pairs.
4. **Multi-guest episodes** — 2+ named guests in one episode → real co-appearance edges.
5. **Grounding + density variation** — insights with/without quotes, high/low density eps, tagged.
6. **Seeded users** — per-user playback + capture fixtures whose heard set covers the topics above
   → exercise scope=mine (#1149), personalized ranking (#1139), resurfacing, with gold expected
   interests + rankings.

## Shaping the 5–6 fake feeds

Keep the domain structure (investing / software / scuba + umbrellas), but engineer:
- **Topic overlap is the master lever** ([[ONBOARDING-SHOWS-FOR-ENRICHER-VALUE]] logic applies to
  fixtures too): the same topics recurring across shows/speakers compounds perspectives +
  co-occurrence + co-appearance + disagreement at once. A niche topic in one episode proves nothing.
- One **recurring contested topic** across shows (e.g. "is X the best approach") with opposing
  stances → the disagreement positive.
- One **panel/2-guest** episode → co-appearance.
- Episodes **dated across a span** → velocity/timeline/trending.

## v3+ generator-evolution SPEC (all six — SPEC ONLY; authoring is a later joint step)

This section **specifies what to add and how it looks** — it is *not* the authored fixtures.
v3 is generated by `scripts/build_v3_fixtures.py` (`PodcastV3` / `EpisodeV3` dataclasses) with
per-episode gold in `tests/fixtures/v3/ground_truth/*.json`. v3 gold today covers *pipeline*
truth only (canonical ids, surface/garble forms, sponsor blocks, position-arc). **The core move:
extend the generator schema + emit ENRICHER gold**, so enricher *accuracy* is deterministic.

Legend: **schema** = new `EpisodeV3`/`PodcastV3`/corpus field · **gold** = new `ground_truth`
key · **unblocks** = capability made deterministically testable.

1. **Time spread** —
   *schema:* `EpisodeV3.publish_offset_days: int` (days from a corpus epoch) + a corpus epoch
   const; build stamps real dates instead of one run-tag.
   *gold:* `expected_velocity: {topic_id: ratio}` (which topics "heat up").
   *unblocks:* `temporal_velocity`, trending/digest, topic timeline, prediction-tracking.

2. **Cross-speaker topic overlap** — *(master lever)*
   *schema:* corpus-level `shared_topics: [topic_id]` (each authored into ≥2 shows/guests) +
   make `EpisodeV3.talking_points` topic-attributed — `{topic_id, claim, grounded: bool}` instead
   of bare strings, so each insight has a known ABOUT + speaker.
   *gold:* `expected_perspectives: {topic_id: {person_id: [insight_id]}}`,
   `expected_cooccurrence: [[topic_a, topic_b]]`.
   *unblocks:* multi-perspective (**de-mocks the #1146 e2e**), `topic_cooccurrence`, feeds
   co-appearance + disagreement.

3. **Engineered opposition** — *(the hard one)*
   *schema:* corpus-level `contradiction_pairs: [{topic_id, guest_a, claim_a, guest_b, claim_b}]`
   where `claim_b` negates `claim_a` on the same proposition; and/or a `panel_debate` episode kind
   with 2 guests arguing. New `failure_mode = "cross_person_contradiction"`.
   *gold:* `expected_contradictions: [{topic_id, insight_a_id, insight_b_id}]`.
   *unblocks:* `nli_contradiction` / disagreement *positives* (the #1106/#1144 dead-end) — the only
   reliable source at fixture scale. Authoring note: the opposed claims must be genuinely
   mutually-exclusive on one proposition, not just different takes.

4. **Multi-guest episodes** —
   *schema:* `EpisodeV3.additional_guests: [guest_key]` (first-class 2+ guests per episode, beyond
   the `high_person_density` tag) with attributed turns.
   *gold:* `expected_coappearance: [[person_a_id, person_b_id]]`.
   *unblocks:* `guest_coappearance`.

5. **Grounding + density variation** —
   *schema:* `EpisodeV3.insight_density: "high" | "low"` (reuse `low_grounding_filler_turns`);
   author some claims *without* a supporting quote turn.
   *gold:* `expected_grounding_rate: float`, `expected_insight_density` per episode.
   *unblocks:* `grounding_rate`, `insight_density`.

6. **Seeded users** —
   *schema:* corpus-level `seeded_users: [{user_id, heard: [ep_id], captured: [insight_id],
   playback_fraction}]` whose heard set covers the shared topics (#2). Build emits per-user files
   (mirrors `app/e2e/.app-state/users/`).
   *gold:* `expected_interests: [token]`, `expected_ranking: [ep_id]`,
   `expected_scope_mine_perspectives: {topic_id: [person_id]}`.
   *unblocks:* scope=mine (#1149), personalized ranking (#1139), resurfacing.

**Cross-cutting:** a `ground_truth` schema version bump + a generic "expected enricher output"
block so every enricher gets a deterministic gold row; a coverage test asserting each of the six
is present in ≥1 episode (mirrors the existing v3 failure-mode coverage test).

## Sequencing (per the operator's plan)

1. **This analysis** (#1148) — define the gaps + the six structures above.
2. **Author v3+ transcripts** — extend the existing v3 screenplay set + the build script
   (`build_synthetic_validation_corpus.py` / `build_app_validation_corpus.py`) to emit the six
   structures + gold labels; wire `FIXTURES_VERSION`.
3. **ElevenLabs voices (stretch)** — v3 already carries `voice=`/`host_voice=` locale tags; realize
   them as actual TTS audio (multi-accent, multi-language) for end-to-end audio + diarization value
   (#993 tracks an ElevenLabs TTS backend).

## References
- #1148 (this analysis) · [[CORPUS-EVOLUTION-FOR-COMPLEX-ENRICHERS]] (test-fixture track) ·
  [[ONBOARDING-SHOWS-FOR-ENRICHER-VALUE]] (eval-corpus track, distinct)
- `docs/guides/eval-reports/EVAL_FIXTURES_V3.md` (existing v3 design — pipeline era)
- `scripts/build_app_validation_corpus.py`, `scripts/build_synthetic_validation_corpus.py`
- Evidence: #1106 / #1144 / #1146 / #1105 evals (`scripts/eval/score/enrichment_*`,
  `disagreement_stance_*`)
