# RFC-131: Person identity — one home for cross-episode identity decisions

- **Status**: Draft
- **Authors**: Marko Dragoljevic
- **Stakeholders**: the operator (merge decisions are product decisions: a wrong name is worse than a hidden one)
- **Related ADRs**:
  - [ADR-166](../adr/ADR-166-person-identity-ledger-no-joins-without-it.md): the identity ledger; no ledger means no joins
  - [ADR-167](../adr/ADR-167-identity-guards-inside-the-join.md): guards inside the join; a cautious split is acceptable, a wrong merge is not
- **Related RFCs**:
  - [RFC-075](RFC-075-corpus-topic-clustering.md) (topic clustering, the model for entity clustering)
- **Related Documents**:
  - [Corpus change method](../guides/CORPUS_CHANGE_METHOD.md) (any change to served corpus files)
  - GitHub issue #2403 (Phase 5: the roster's own same-person rules)

## Abstract

- **What:** one module, `podcast_scraper.identity.person_match` (name final at implementation), owns every *cross-episode* person-identity decision:
  - whether two names are one person;
  - which spelling is shown;
  - what counts as a placeholder voice;
  - how a person id is minted.

  A deterministic step writes those decisions, with their evidence, to an **identity ledger**. The server reads the ledger and applies it at read time on every surface.
- **Why:** these decisions are made in many places with different rules today. The result, on the served corpus (3,069 episodes, 2026-10-30):
  - raw diarization labels are shown to listeners;
  - one person is split across spellings;
  - two co-presenters with near-identical first names are merged into one person.
- **How:** first a behaviour-neutral move into one home, proven against a saved baseline of today's identity map. Then behaviour changes, each measured before and after on the served corpus. No episode artifact is rewritten.

**Architecture alignment:** identity stays a read-time projection over corpus text, like the topic and entity clustering already applied through `CorpusGraph.build(identity_map=…)` (#852). This RFC moves that projection from an in-memory map with no record to a recorded ledger, and gives each decision one implementation.

## Problem Statement

A person reaches the listener through many paths, including:
- the insights panel;
- the transcript;
- episode people chips;
- person cards;
- top voices and perspectives;
- share cards;
- search results;
- MCP tools;
- emails, which are pushed and cannot be recalled.

Each path builds the name itself. A read-only report over the served corpus (`podcast_scraper.quality.name_report`) measured on 2026-10-30:

| Measure | Value |
|---|---|
| Raw labels shown as speakers in transcripts | 222,461 segments in 732 episodes |
| Raw labels shown as an insight's speaker | 66 insights in 21 episodes |
| Same-show spelling-variant pairs, names placed on a voice | 92 pairs in 58 groups, ≤ 1,110 served insights under a non-winning spelling |
| Identical names on two or more shows | 95 groups, 86 already joined by today's map, none checked for namesakes |
| Cross-show spelling variants | 55 pairs, 3 joined |

The causes are structural:

1. **One decision, many implementations.** "Are these two names one person?" is decided by about 25 functions across `kg/entity_clusters`, `search/corpus_graph`, `kg/filters`, `kg/speaker_coherence`, `identity/*`, `providers/ml/diarization/roster` and migrations, with different rules:
   - surname tolerance is decided eight ways;
   - whole-name similarity thresholds range from 0.6 to 0.91;
   - there are two nickname tables, five title lists and six copies of edit distance.

   On the same pair of names they disagree in both directions.
2. **Placeholder recognition has two homes and several patterns.**
   - `graph_id_utils.is_bare_speaker_label` misses the speaker detector's failure default (`unknown_guest_N`).
   - `enrichment/enrichers/_loaders.is_unresolved_speaker_placeholder` has its own regex.
   - `app_gi_view._speaker_name` applies neither, so a Person node named like a raw label is shown as a person, and its insights are published as attributed.
3. **Display names are built per surface.**
   - Fallbacks differ: `None`, the slug, the full `person:` id, or a lowercased slug in the web client.
   - The transcript passes a raw `SPEAKER_NN` through, by design (`roster.friendly_speaker_label`).
4. **The identity map has no record, no guard, and a chaining join.** `build_entity_canonical_map`:
   - joins a name to a cluster when it matches *any* member, so chains form;
   - reads every Person node, voices and mentions alike;
   - is built in memory and discarded. Nothing records why two ids were joined.

   It currently joins two co-presenters who appear as different voices in the same episode.
5. **Search maps no identity.** Index rows carry `speaker_id` raw (`search/indexer.py`), and no search read path applies the map.

**Use cases:**
1. **A listener reads a transcript.** Unnamed voices read as "Unidentified speaker 1", "Unidentified speaker 2", never as `SPEAKER_03`.
2. **A listener opens a person page.** It gathers every episode where that person spoke, across their spellings and across shows, and nobody else's.
3. **An operator asks why two people were joined or kept apart.** The ledger answers with the rule, the evidence and the build that decided it.

## Goals

1. **One home per decision.** Same-person rules, guards, the spelling rule, placeholder recognition, id minting, the display name and the similarity helpers each have one implementation. A guard test fails if a second appears.
2. **Recorded decisions.** Every cross-episode join, blocked join and chosen spelling is in the ledger with its evidence and the build that made it, plus an append-only change log.
3. **Safe failure.** A missing, unreadable or version-mismatched ledger means no cross-episode joins (ADR-166), never an unguarded map.
4. **No wrong merges.** Guards inside the join block a merge on evidence of two people (ADR-167).
5. **Measured outcomes.** Every behaviour change is measured on the served corpus before and after with `name_report`. A change ships only if it is better: splits down, no new wrong merges, no raw labels.

## Constraints & Assumptions

**Constraints:**
- Text-only and read-time first. No model calls, no re-run of diarization or extraction, no rewrite of episode artifacts. The only new corpus file is the derived ledger.
- Search-index rows are edited only if a read-time mapping cannot reach a surface, and then only under `CORPUS_CHANGE_METHOD`.
- No new third-party dependency.
- Migrations import live identity code (m0009, m0014, m0016, m0025, among others). A helper change must not silently change an applied migration's re-run or `verify`.

**Assumptions:**
- Evidence on disk is enough to decide most joins:
  - the roster in `content.speakers`, with `placed` and `voices` where present (older records place entries by segment label, the rule of m0025);
  - Person and Quote nodes with SPOKEN_BY edges;
  - publisher text;
  - the per-episode naming trace.
- A person split by caution is acceptable while it is listed; a wrong merge is not.

## Design & Implementation

### 1. Boundary

| In this RFC (cross-episode identity) | Not in this RFC |
|---|---|
| `kg/entity_clusters` (the identity map), `search/corpus_graph` host-variant collapse, display-name building on every server surface and in the web clients, placeholder recognition, person-id minting, the similarity helpers and constant sets | The roster's within-episode naming rules (`providers/ml/diarization/roster.py`, ~13 same-person questions). They share this module's helpers and constants, but their decisions are unchanged until Phase 5 (issue #2403). |

### 2. Module

`identity/person_match` holds:

- **Similarity helpers:** edit distance, Soundex, the name normaliser, nickname groups, title, credential and generational-suffix sets. These replace the copies listed in the problem statement.
- **Named rule modes,** one per current consumer:
  - `cluster` is the rule behind `_are_xep_variants`;
  - `host_variant` is the rule behind `_same_person_variant`.

  Folding modes together is a behaviour change, measured on its own (§7).
- **The guard interface** (ADR-167), evaluated inside the join test.
- **The spelling rule,** the display-spelling order:
  1. publisher text;
  2. the spelling matched to it at naming time;
  3. the most frequent spelling heard;
  4. on a tie, each episode keeps its own.
- **Placeholder recognition stays in its existing home,** `graph_id_utils`, which 27 modules already import. The copies elsewhere go there (S1b step 2): `_loaders.is_unresolved_speaker_placeholder` and the corpus graph's numbered-label regex. Numbered diarization labels and the detector default are recognised as names and as ids; role words are separate, because "Host" stays readable in its own episode.
- **The display-name function** with one fallback rule: no name rather than an id or a slug. This includes "Unidentified speaker N" for an unnamed substantive voice, numbered by first appearance in the episode at read time, never stored and never an id.
- **`place_roster_entry(entry, segment_labels)`,** the one live home of the placement rule m0025 applies to older records.

### 2a. Languages

Identity follows the language parameterisation the rest of naming uses: `speaker_detectors/naming_vocabulary.py` keeps every word list per language (en, de, es, fr, it, pt today), and callers pass `language`.

- **Rules take a language.** Every rule mode takes the episode's language (`languages.resolve_episode_language`).
  - Titles, credential and generational suffixes, particles and nickname groups come from `naming_vocabulary[language]`. Nothing English-only is built into the module.
  - The two nickname tables become one per-language table.
- **An unknown or unsupported language** gets only language-neutral rules: names equal after normalisation, plus the guards. A missing vocabulary can split a person, never merge two.
- **Phonetics and folding are per language.** Soundex is enabled per language. Fold rules (ü↔ue, ß↔ss, accents) and name order (family name first in zh, ja, ko, hu, vi) come from the vocabulary. <!-- codespell:ignore ue -->
- **Ids for any script.** One id-minting rule gives a stable id for a name in any script. Today the three minting functions give `person:topic`, `person:unknown`, or raise.
- **Labels are typed, not strings.** "Unidentified speaker N", "Brief speaker", "Advertisement" and "Host" reach clients as a kind and a number, so the client's i18n renders them in the listener's language. An English string remains as a fallback for MCP and email.
- **Publisher cues per language.** Cues like "On the show:" and introduction phrases come from the vocabulary.
- **Measured per language.** Every measure is broken down by episode language, and the truth set and its held-out split are stratified by language.
- **Across languages,** a person under two spellings (transliteration) joins only with independent evidence.
- **The guards (ADR-167) are language-neutral.** They read voices, publisher lists and graph nodes, not words.

### 3. Ledger (ADR-166)

- **Built by:** a deterministic step that runs last on every corpus-writing path: ingest, `upgrade apply`, relabel repair, enrich-edges.
- **Format:** extends the existing versioned clusters payload (`ENTITY_CLUSTERS_SCHEMA_VERSION`, `id_map_from_clusters_payload`). It adds:
  - a header: schema and rule version, code sha, corpus token (`corpus_mtime` over the existing four files, plus the newest-run kg count), time;
  - per cluster: canonical id, display spelling and its evidence, and members;
  - blocked joins: `(candidate, member, guard, episode)`;
  - an append-only change log comparing each build with the previous one.
- **Write:** atomic (temp file, fsync, `os.replace`), owned by the corpus owner. The ledger file joins the `perf_cache.corpus_mtime` tuple so the server picks it up without a restart, and it is excluded from its own token.
- **Read:** at warm-up, and whenever the file changes.
  - A version mismatch, or a missing or unreadable ledger, means no cross-episode joins, plus a warning and a metric.
  - A corpus newer than the ledger is served from the ledger as it is. New episodes stay unjoined until the next build, and the report shows the gap.
- **Not in the ledger:** per-episode detail stays in the naming trace (`*.speakers.diagnostics.json`).

### 4. Applying it

- Every surface that builds a person name calls the display function with the canonical id from the ledger. That covers the insight and quote speakers, transcript segments, episode chips, person cards, top voices, perspectives, share cards, MCP tools and push payloads.
- Search maps `speaker_id` through the ledger at read time.
- Web clients render the server's display name and never derive a name from an id.

### 5. Migrations

- Migrations stay as written.
- A migration that imports a helper moved by this RFC is re-checked before and after the move: its `verify`, or a dry run, must give identical output. If it would differ, the migration gets a pinned copy of the old helper, which is named in the migration and frozen.

### 6. Delivery order

1. **Behaviour-neutral move into the module, plus the ledger skeleton.** The replay compares the ledger's `variant → canonical` map, member order included, against a saved baseline of today's map. The baseline was dumped from unmodified code on the served corpus. The requirement is zero differences.
2. **Raw labels:** the placeholder predicate in its one home; unattributed treatment of raw-label insights; "Unidentified speaker N" in transcripts.
3. **Guards inside the join** (ADR-167), with an audit of today's joins.
4. **First deploy.**
5. **Variant joins:** hyphens and word breaks, middle initials, groups anchored on the winning spelling. Then one spelling on every surface, including search. Then roster vs voices, then cross-show joins with independent evidence.
6. **Second deploy.**
7. **The roster's own rules** (issue #2403).

### 7. Validation

- **Before and after on the served corpus.** Both runs use the same image in a read-only container; the after-run mounts the branch source, and a code marker proves which code ran. The "before" figures are frozen per stage.
- **A labelled truth set** of name pairs, read blind by two readers plus the operator.
  - Held out by show; cross-show pairs count only when both shows fall on the same side.
  - Correct spellings are recorded only where publisher text names the person.
- **Every new or removed join is listed.** Joins outside the truth set are sampled and read, and every join that newly reaches an insight surface is read.
- **The replay of each rule change** uses the old code from git, with one mutant per change.

## Alternatives considered

1. **Patch each surface where the bug shows** (filter `SPEAKER_NN` in the insight view; add a guard in `entity_clusters`). Rejected: this adds a seventh variant of each decision, and the next surface diverges again.
2. **Keep the in-memory map, add guards only.** Rejected: there would still be no record of why people were joined. And a server falling back to an unguarded build on any error would bring wrong merges back silently.
3. **Rewrite the corpus files with canonical names (a migration).** Rejected as the primary path:
   - it is irreversible without a corpus restore;
   - it needs the corpus-change method for every rule tweak;
   - read-time application reaches every served episode immediately, and rolls back with the image.
4. **Unify the roster's rules in the same step.** Deferred to Phase 5 (issue #2403):
   - it needs a voice-level truth set, not name pairs;
   - it reaches served episodes only through re-derivation;
   - the shadow namers built on it are being monitored.

## Testing

- **Unit:** each predicate shape; each rule mode; each guard and that it is checked against every member; the spelling rule; the display function; the ledger failure modes (missing, wrong version, old token). Properties: the rule is symmetric, there are no chains, and canonical is idempotent.
- **Integration:**
  - a synthetic corpus on two or three shows, through the ledger step, the server read and the routes, per surface;
  - guard tests: no second same-person rule or placeholder pattern, and every displayed person name goes through the display function.
- **End-to-end:** one Playwright spec per user-visible change, at phone size.
- **Outcome:** `name_report` on the served corpus, before and after.

## Open questions

- How a person page's URL id relates to the displayed spelling when the canonical id was minted from a non-winning spelling.
- Whether names already pushed or stored need a cleanup: the user's captures and favourites, LLM summaries, sent emails. These are counted, not changed, by this RFC.
- Non-ASCII names: today the three id-minting functions give three different ids (one of them `person:topic`). The module's minting rule settles this; the count on the served corpus is pending.
