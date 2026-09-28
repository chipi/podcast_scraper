# Multilingual ingest — arc notes (v1)

The one page that holds this arc together. The PRD says what the product needs, the three RFCs each
own a slice of the how, and this document is where the arc's **shape**, its **slice plan**, its
**verified code facts**, its **decisions** and its **running notes** live.

- **Arc**: multilingual ingest (source-language capture, English-normalized intelligence)
- **Scope**: **v1 only — a working pipeline and product, end to end.** Everything deferred lives in
  [MULTILINGUAL_ARC_V2](MULTILINGUAL_ARC_V2.md), with its slices and the reason it was deferred.
  Nothing was dropped.
- **Opened**: 2026-09-28
- **Status**: design — nothing implemented
- **Branch**: `feat/multilingual-ingest`
- **Documents**: [PRD-047](../prd/PRD-047-multilingual-ingest.md) · [RFC-123](../rfc/RFC-123-speaker-turns-artifact.md) · [RFC-124](../rfc/RFC-124-multilingual-transcription-and-translation.md) · [RFC-125](../rfc/RFC-125-translation-confidence-and-claim-verification.md)
- **Review state**: six adversarial reviews have run over these documents. §5.4 records every claim
  they found false — including errors introduced by an earlier *correction* pass — because the way
  those claims were wrong repeats and is worth recognising.

---

## 1. The arc in one page

The corpus is English-only by configuration, not by architecture. The bet is that we can ingest a
non-English show, keep **what was actually said** as the canonical record, and derive an English layer
that every existing intelligence stage reads without a per-language fork.

```text
audio (any enabled language)
  │
  ├─ transcribe + diarize IN SOURCE LANGUAGE ──────► ep1.txt            ← canonical, full timeline
  │   └─ speaker naming (on the source)              ep1.segments.json
  │                                                  ep1.turns.json     ← RFC-123
  │
  ├─ TRANSLATE every turn-bounded unit ────────────► ep1.translation.json  ← the traceability map
  │   (before summary — D-19)                        ep1.en.txt            ← derived, full timeline
  │                                                  ep1.en.segments.json  ← doubles as subtitles
  │
  ├─ ad-detect + excise ON THE ENGLISH ────────────► ep1.en.adfree.txt   ← THE ANALYSIS TRANSCRIPT
  │
  ├─ summary → GI → KG, all reading that file ─────► single-path, English
  │
  └─ every claim labelled; none on Position surfaces until v2 verifies them
```

Three properties make it work:

1. **The source is canonical, the English is derived.** A translation is an interpretation; the record
   is what was said. Every English span maps back to source text, source speaker and source audio time.
2. **Analysis reads one transcript.** Every transcript reader resolves through one function, so no
   stage decides for itself what "the transcript" means. That is **not** true today (§5.4 C-1) and
   making it true is part of the work, not a property the design can assume.
3. **The product never presents a translation as verbatim speech.** Labelled on every surface that
   renders it, traceable to the source, and kept off Position surfaces entirely until v2 can verify it.

## 2. Document map

| Document | Owns | Does **not** own |
| --- | --- | --- |
| [PRD-047](../prd/PRD-047-multilingual-ingest.md) | Why, for whom, what "done" means, phase gates, language policy, operator + listener surfaces | Any implementation shape |
| [RFC-123](../rfc/RFC-123-speaker-turns-artifact.md) | `turns.json` — turns and sentences as addressable units. v1 needs the artifact; its consumers are v2 | Anything language-specific |
| [RFC-124](../rfc/RFC-124-multilingual-transcription-and-translation.md) | Language resolution, source capture, the translation stage, the artifact set, the stage order, retrieval, model selection | Trust, gating |
| [RFC-125](../rfc/RFC-125-translation-confidence-and-claim-verification.md) | Translation provenance and the read-time Positions gate (v1); verification and QE (v2) | Producing the translation |
| This document | v1 arc shape, slice plan, verified code facts, decisions, running notes | Requirements or design detail |
| [MULTILINGUAL_ARC_V2](MULTILINGUAL_ARC_V2.md) | Everything deferred out of v1, with slices and reasoning | Anything v1 ships |

## 3. Phase ladder

Phase numbering means one thing across all four documents. The demand/model validation step is
**Gate V**, not a phase, because it produces evidence rather than software.

| Phase | What | Gate to start | Visible outcome |
| --- | --- | --- | --- |
| **0 — English as a declared language** | Language becomes a real, parsed, resolved, validated property of the corpus we already have | none; it is a correctness fix | An audit over real data showing the corpus's actual language distribution; language on the API; no code path that substitutes a language it was not given |
| **1 — Turns artifact (RFC-123)** | `turns.json` built, written for both variants, backfilled. Nothing reads it yet | none | The unit translation needs, available on today's corpus |
| **Gate V — Validate** | Demand check; the pilot-language investigation; model selection and its quality gate | Phase 0, because the bake-off cannot measure non-English transcription until it exists. **V.5 needs nothing and starts now** | A go/no-go with evidence, and a pilot language chosen on measurement |
| **2 — Translation (RFC-124)** | Reader routing, the stage-order change, translation, the English render, ad-free-on-English, labelling, same-language retrieval, the Positions gate | Gate V passed | One non-English feed processed end to end, findable in its own language, everything labelled |
| **3 — Surfaces** | The transcript/subtitle reading path and flag lifecycle. The badge and the language filter are v2 | Phase 2 | A listener can read the original or the English against the original audio |

**Why Phase 0 is a real phase.** Today a feed's declared language is never read, every episode is
stamped with the run configuration, and a non-English episode would be transcribed by a chain that can
fall back to `base`. Phase 0 makes language an *asserted, checked* fact — which is exactly the plumbing
translation needs. It is testable on the corpus that already exists, with no new models and no GPU, and
it is the recommended standalone ship.

## 4. Slice plan

Each slice is sized to be **one GitHub issue**: one goal, its own tests, its own acceptance criteria,
and shippable without leaving the tree half-built. Nothing is opened yet.

*Depends on* is a hard ordering constraint. *Size*: **S** = one sitting; **M** = a day-ish, multiple
files, real test surface; **L** = multi-day, new subsystem or a migration. *Ship alone?* asks whether
merging only this leaves `main` correct and coherent.

**A constraint that shapes several slices.** The operator plans to move **from a corpus to a database
over the coming weeks**. So: prefer additive, low-ceremony changes; do not build corpus-level
infrastructure the database move will obsolete; and where a change would force an index rebuild, ask
whether it can wait for the rebuild that migration brings anyway.

---

### Phase 0 — English as a declared language

**Ships as one release.** No new models, no GPU, no user-visible chrome — the badge moved to v2 (D-11).
At the end, "this episode is in English" is something the system parses and checks rather than assumes,
and the same machinery carries any other language.

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S0.1a** | Parse and persist the feed's declared language | Extract the channel `<language>` (a `channel.find("language")` in the pattern `rss/parser.py` already uses for title, author and description) and carry it on `RssFeed` and `FeedMetadata`; persist `feed.language_raw`, `feed.language`, `feed.language_source`, plus an **episode-level** `language` and `language_source`. Add the field to `RssFeed` as a defaulted field, never positional — it is constructed at ~53 test sites. Note there are two `FeedMetadata` types (a persisted pydantic one and a positional NamedTuple); the slice must name which. | — | M | Yes |
| **S0.1b** | Backfill language onto the existing corpus | **The real long pole, and its mechanism is an open decision** (§8.1): existing metadata holds no RSS language text to recover, the migration fixture has a `feed` block with no URL, and `skip_existing` is GUID-keyed so a normal run never rewrites processed episodes. Decide between a backfill CLI that refetches live, reading cached feed XML if prod keeps any, or accepting `language_source: unknown` for legacy episodes. Whether a formal corpus migration is needed at all is also open. | S0.1a | L | Yes |
| **S0.2** | Language-tag normalization and per-episode resolution | `normalize_language_tag` and `resolve_episode_language(feed_entry, feed_doc, cfg)`; `config/languages.yaml` registry with `en` the only enabled language. Must route the profile default through the same normalizer — `Config._normalize_language` only lowercases today, so `language: en-US` yields `en-us`, which fails `whisper_utils.py:50`'s `is_english` check: **a live bug this slice fixes**. The edge-case table and whether script is retained are open (§8.2). | S0.1a | M | Yes |
| **S0.3** | Per-feed language override | `rss/feeds_spec.py` already accepts a mapping per feed entry, but `RssFeedEntry` has explicit typed fields **and** `extra="forbid"` — so this needs both a model field and an `RSS_FEED_ENTRY_OVERRIDE_KEYS` entry. Consequence worth knowing: `merge_feed_entry_into_config` makes the per-feed `Config` the run's `cfg`, so the override reaches every existing `cfg.language` reader with no threading — but in a multi-feed batch the ML singleton is held across feeds, so feed 1's Whisper model persists and feed 2's language is silently ignored (see S0.7). | S0.2 | S | Yes |
| **S0.4** | Corpus language audit over real data | Read-only CLI in the existing `check_corpus` pattern: walk every feed and episode, resolve a language, report the distribution and every item not resolving to `en`, each with its resolution source. Commit the report — **S0.6 depends on it being clean.** Runs after S0.1b, or it can only re-read the run config and reports a guaranteed 100% `en`, which proves nothing. | S0.1b, S0.2 | S | Yes |
| **S0.5** | Expose language on the app API | Episode-level `language` (additive) on the episode list and detail responses; `AppPodcastItem.language` starts serving the normalized feed tag instead of the run config; **`CorpusFeedItem` gains the field too** — the operator viewer's shows library consumes that one and has no language data without it. Contract tests for present / absent / legacy values. | S0.1a, S0.2 | S | Yes |
| **S0.6** | One language reader: thread it, delete the substitutions | The transcription call site passes the episode's resolved language; the DGX provider stops **reporting** `"en"` for an auto-detected transcript; `ml_provider`'s `self.cfg.language or "en"` goes; `sniff_gate.py`'s four sites and `metadata_generation.py:957/:3832` are covered. **Hard dependency on S0.4's committed clean report**: after this slice a feed whose RSS says `de` is actually transcribed as German, so it must not land while the audit is unknown. Acceptance is a lint rule with an explicit whitelist — "exactly one reader" is false as stated, because the cloud providers and `ner.py` legitimately read `cfg.language` — and the lint must be wired into CI, which its cited precedent is not. | S0.2, S0.4 | M | Yes |
| **S0.7** | Select the transcription model from the resolved language | `MLProvider` resolves its Whisper model **once at init**, so per-episode selection is a provider change. **Decide first** whether the local tier serves non-English at all (§8.3): if not, this is S — the provider raises `deferred_quality_floor` for non-`en` — rather than L with a per-call model cache. Adds `min_model_for_non_english` and truncates the fallback chain. Byte-identical for English. | S0.6 | M/L | Yes |
| **S0.8** | Unsupported-language skip | `skipped_unsupported_language` for a language that is not `enabled`. Must land **after S0.3 and S0.4** or a mis-tagged English feed silently stops ingesting with no remedy. Needs the status-vocabulary decision (§8.4) — `EpisodeStatus.status` is `Literal["ok","failed","skipped"]` today. | S0.3, S0.4 | S | Yes |
| **S0.9** | Phase 0 observability | Manifest `language` block, plus log and metric surfacing for `skipped_unsupported_language` and `deferred_quality_floor`, and a runbook entry. Without this, the phase whose purpose is removing silent failures introduces new ones. | S0.7, S0.8 | S | Yes |
| **S0.10** | The English-artifact allow-list test | The instrument Phase 0's acceptance needs and which does not exist: serialize the metadata and manifest shapes before and after, assert `added_keys ⊆ ALLOWLIST`. Its own issue because "byte-identical" was a criterion with no command behind it, and because a new stage slot adds a `translation: skipped` ledger entry to **every** English episode (D-19) that has to be on that list. | S0.1a | M | Yes |

**Phase 0 acceptance** (a release checklist, not an issue): the audit reports the corpus's real language
distribution with every item's resolution source named; language is on the API; no stage can receive a
null language and substitute one; a non-English language selects a multilingual model at or above the
floor or fails loudly; the new failure modes are visible; and English artifacts are unchanged outside
the declared allow-list.

---

### Phase 1 — Turns artifact (RFC-123)

v1 needs the artifact to exist, because translation units are sentence groups inside a turn. Its three
consumers are **v2** ([V2-D](MULTILINGUAL_ARC_V2.md#6-turns-consumers)).

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S1.1** | `build_turns`: turns and sentences from a segments sidecar | The pure builder plus the invariants, property tests and committed goldens. Nothing reads it. | §8.5 decided | M | Yes |
| **S1.2** | Write `turns.json` in the pipeline for both variants | Emit per variant where the ad-free segments are produced, the manifest `turns` block, and the `turns: unavailable` legacy flag. | S1.1 | M | Yes |
| **S1.3** | Backfill `turns.json` across the existing corpus | No GPU, idempotent, coverage report. | S1.2 | S | Yes |

---

### Gate V — Validate (evidence, not software)

| # | Issue title | Goal | Depends on | Size |
| --- | --- | --- | --- | --- |
| **V.1** | ~~Verify translation model availability and licences~~ | **DONE 2026-09-28 — §6.1.** | — | closed |
| **V.5** | Choose the pilot language by measurement — **starts now** | No dependency on any phase. Three cheap actions: rescore FLEURS `sr` with `large-v3-turbo`, transliterating hypothesis *and* reference to Latin (if WER collapses toward Croatian's 13.4 the ASR objection is gone); probe Qwen3-30B for Serbian, since it is already served and Apache-2.0 and §6.1 never checked it; log in to the **gated** TranslateGemma card and settle its 55 languages. | — | M |
| **V.2** | Demand check with the beta cohort | Needs the instrument written first: question wording, cohort size, how "≥30%" is computed. | — | S |
| **V.3** | Model selection and the quality gate | The bake-off harness, the judge protocol (v2 doc §9 holds the rules; they apply here), and a run on 2–3 episodes of the pilot language plus a Spanish or Italian control. Reports RFC-124 §7's measurements including ad-detection survival. **Whether this stays a five-candidate comparison or trims to a single-model sanity check is open** (§8.6). | S0.10, V.5, V.6 | L |
| **V.4** | Gate V decision record | An **ADR** recording which language passed, with numbers, and the chosen model plus pinned revision. | V.2, V.3, V.5 | S |
| **V.6** | Source a test fixture in the pilot language | A short CC-licensed or TTS episode with an injected sponsor read. None exists, and the integration test is untestable without it. Do not hand-build audio. | V.5 | S |

---

### Phase 2 — Translation (RFC-124)

Behind `multilingual_ingest`. **The flag gates the pipeline; deciding when a translated episode becomes
visible is deciding when to add the feed to the production feed list** (D-16 withdrawn). So the
labelling and gating slices must be in before that feed is added.

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S2.1** | Route every transcript reader through one resolver | The work D-4 assumed away (§5.4 C-1). `load_processing_transcript` has two callers; the summary stage, **two** faithfulness reads, `search/indexer.py`, `gi/repair.py`, `gi/load.py`, `stages/processing.py`, `routes/corpus_text_file.py`, `segments_view.py` and `metadata_generation.py:1199` each resolve independently. Route them; add the `.en.adfree.txt` → `.en.txt` → `.adfree.txt` → `.txt` precedence. **Standalone bug fix — can ship any time, before Gate V.** | — | L | Yes |
| **S2.2** | Give translation a stage slot, in one seam | `CANONICAL_STAGE_ORDER` gains `translation`. **One insertion point, not two**: inside `generate_episode_metadata` immediately before summary, which covers ASR, transcript-cache hits, direct downloads, publisher-supplied transcripts, and every relabel/rediarize/retranscript cascade — all of which end there. Excludes translation wall time from the metadata deadline. ADR-151 means every English episode's ledger gains `translation: skipped` (S0.10's allow-list). | S2.1 | M | Yes |
| **S2.3** | Translation units and the vLLM client | Turn-bounded unit packing from the source variant; the `dgx_vllm_translate` client with per-episode batching, bounded concurrency and per-unit retry. | S1.2, V.4 | M | Yes — flag-off is a no-op |
| **S2.4** | `translation.json` and the English render | The unit map; the English screenplay and `.en.segments.json` rendered through the existing formatter, one pseudo-segment per unit **carrying `unit_id`**; `translation_pending` and failed-unit semantics; a stub-translator integration test. | S2.3 | M | Yes |
| **S2.5** | Ad-free base on English, and span→unit resolution | Build `.en.adfree.*` with the existing machinery. `resolve_units_for_span` resolves through `.en.adfree.segments.json` and `unit_id` — **not** the ad-map, which cannot invert the ad-free transform (§5.4 C-5, measured). Uses **overlap**, not containment, so a span touching a label prefix or inter-turn whitespace still resolves. Writes the English artifact set atomically and refuses on a provenance mismatch — including `excerpt != text[char_start:char_end]`, which catches a re-translation that a file hash alone would not. | S2.4 | L | Yes |
| **S2.6** | Speaker identity across scripts | Naming stays **before** translation, on the source (§5.4 C-6). Labels bypass the translator; CIL alias lookup then deterministic transliteration with a `_looks_like_person` guard and source-script fallback. **How much of the aliasing is v1 is open** (§8.7). | S2.4 | M | Yes |
| **S2.7** | Language-aware reprocess and invalidation | `_maybe_produce_adfree` has **five** call sites including the transcript-cache hit; on a non-English episode each would write the identity ad-free artifact this design says never exists, and strand `.en.*`. Make it language-aware; give every path that changes the source an explicit invalidation of `.en.*`, `translation.json` and the cached prompt prefix; add the per-episode reprocess command. Note `rederive_only` must **not** re-translate — that is the cheap repair path. | S2.5 | M | Yes |
| **S2.8** | Label translated content everywhere it renders | Not one chip. The marker must survive every serialization boundary that drops node properties: `AppInsight` / `AppQuote`, the MCP `InsightSummary` / `SupportingQuote` contracts, `hybrid_search._to_search_result` (Lance rows carry no `translated` column, so a translated quote would be served as verbatim speech in search, digest and trending), OG share images, and **snapshots** — favourites and captures are copies, so a translated insight saved to a library stays there unlabelled and no read-time gate can reach it. Plus `?lang=` on the segments contract and `translation_status` on episode detail. | S2.4 | L | Yes |
| **S2.9** | Same-language retrieval: a keyword-only table for non-English | Index both layers so a query in either language reaches the same episode. **Non-English chunks go in their own table with no vector column** (D-14, option B), consulted only for the keyword leg — a chunk with no vector cannot appear in a semantic result, which a row tag plus a filter could not guarantee. Leaves the existing `segments` table untouched, which should avoid the schema bump, the stale index and the full rebuild — **confirm that in the slice**, along with the read path tolerating the table's absence on older indexes. Also: chunk ids carry language, insight→segment linking filters on language, and a non-English query drops the dense leg via script detection or it returns English noise. Measure keyword recall through the English tokenizer before calling this done. | S2.5 | M | Yes |
| **S2.10** | Cost and capacity measurement | Translation GPU time and storage delta per episode, and the bake-off's own cost. RFC-124 OQ3's wall-time cap cannot be set without it. Note for model choice: a 27B translator and the served 30B model do not co-reside in the DGX's memory while a 12B does. | S2.3 | S | Yes |
| **S2.11** | Translation provenance on every claim | The `translation` block (`translated`, `source_language`, `unit_ids`, `en_sha256`) written into node `properties` via `resolve_units_for_span`, by **every** writer of `gi.json` — the artifact builder, `add_spoken_by_edges(replace=True)` and `gi/repair.py`. | S2.5 | M | Yes |
| **S2.12** | Keep translated claims off Position surfaces | The read-time filter, applied to **every** position-bearing surface — RFC-125 §3 enumerates nine, including `topic_perspectives`, which feeds the consumer app and OG share images, and one (`topic_consensus`) that is write-time and cannot be filtered at read time at all. Two predicates, not one helper: an edge predicate for what would be verified, a property predicate for what renders, with a test that the first is a superset of the second. Fail-closed: with no verification records in existence — v1's steady state — every translated claim is absent. | S2.11 | M | Yes |
| **S2.13** | Phase 2 gate | One non-English feed from audio to insights, findable in its own language, every translated claim labelled on every surface, none on a Position surface, and no English regression outside the allow-list. | S2.1–S2.12 | M | This is the ship |

---

### Phase 3 — Surfaces

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S3.1** | Transcript language toggle and original-text reveal | Read the source transcript or the English subtitles against the original audio; tapping a translated quote reveals the source sentence and plays the source span. UXS first. **Open: v1 or v2** (§8.8). | S2.8 | L | Yes |
| **S3.2** | Feature-flag lifecycle and rollback | What `multilingual_ingest` gates at each phase, its removal criterion, the per-phase rollback procedure, and what happens to a language that is **disabled** after episodes exist in it. | S2.13 | S | Yes |

---

### Critical path

```text
S0.1a → S0.1b → S0.4 ─┐
S0.1a → S0.2 → S0.3 ──┼→ S0.6 → S0.7 → S0.8 → S0.9 ═══ PHASE 0 SHIPS ═══╗
S0.1a → S0.5          │                                                  ║
S0.1a → S0.10 ────────┘                                                  ║
                                                                         ▼
S1.1 → S1.2 → S1.3 ═══ PHASE 1 ═══╗     V.5 (NOW) ──┬── V.6 → V.3 ──┬── V.4 ═══ GATE V ═══╗
S2.1 (standalone, any time) ══════╬═════ V.2 ────────┘               │                    ║
                                  ▼                                  ▼                    ▼
      S2.2 → S2.3 → S2.4 → S2.5 → {S2.6, S2.7, S2.8, S2.9, S2.10, S2.11 → S2.12} → S2.13 ══ PHASE 2
                                                                                        ▼
                                                                             {S3.1, S3.2}
```

Phase 0 is close to serial through `S0.1a → S0.2 → S0.6`, and S0.1b is its long pole. For one operator
the human-shaped item — V.2's demand instrument — has the longest lead time and should start early even
though nothing blocks on it.

## 5. Code facts this arc rests on

### 5.1 Findings that survived review

| Finding | Evidence | Consequence |
| --- | --- | --- |
| **There is no stance-extraction stage, and deliberately none.** Stances are GI insights; Positions are read-time queries. | `server/cil_queries.py:636` `position_arc`; `enrichment/profile_sets.py:144-146`; ADR-108's 2026-07-08 update retiring `stance_timeline`. A repo-wide word search for `stance` finds only comments. | The gate is a **read-path filter** — retroactive and fail-closed. No stance-extraction dependency exists. |
| **`vector_embedding_model` is genuinely wired to the search index.** | `search/indexer.py:577` → `build_two_tier_index`; the query side reads the model recorded in the index (`hybrid_search.py:238-244`). | A future encoder swap is coherent. v1 changes no embedding model (D-14). |
| **`GiArtifact` forbids extra top-level keys**; `EvidenceSpan` / `SupportingQuote` / `SegmentsResponse` are plain models. | `gi/contracts.py:130`, `:14`, `:25`; `server/schemas.py:24`. | Additive data lives in node `properties`; API fields are safe to add. |
| **The ad-free identity hazard is real.** | `gi/ad_regions.py:409-418`; `adfree_transcript.py:104-106`, `:129-137`; `load_processing_transcript:237-250` sets `is_adfree=True` on file existence alone. | Translation precedes ad detection (D-3). Caveat: with no segments `build_adfree_artifacts` returns `None`, so the identity artifact only appears for episodes with a segments sidecar. |
| **`adfree_transcript_relpath` composes `ep1.en.txt` → `ep1.en.adfree.txt`** unchanged. | `adfree_transcript.py:52-55`. | No new path helper needed. |
| **`formatting.py:81`'s passthrough tuple can carry `unit_id`** through to both sidecars. | `formatting.py:81`; `episode_processor.py:825`; `adfree_transcript.py:180` dump segment dicts unchanged. | The span→unit chain (C-5's fix) is implementable. |
| **The per-feed override reaches the whole run.** | `feeds_spec.py:250-291` `merge_feed_entry_into_config` does `cfg.model_copy(update=...)`; `service.py:143`. | An override needs no threading — but see C-4 for the multi-feed singleton hazard. |

### 5.2 The silent hazards for non-English audio

Five. Each fails quietly rather than erroring.

1. **Silent degradation to an unusable model.** The non-English fallback chain runs down to `tiny`, and
   prod's local default is `base.en`, so for Greek the chain is `["base", "tiny"]`. The DGX path never
   passes through the normalizer, so this lives entirely in the fallback tier. → **S0.7**
2. **No per-episode language path exists.** Providers read run-global config when the argument is
   absent; the call site passes exactly that global; `sniff_gate.py` threads it four more times.
   → **S0.6**
3. **The DGX provider misreports the language.** `whisper_provider.py:197` is the **returned result
   dict**; the request at `:429-430` *omits* `language` when it is `None`, so the server auto-detects
   and returns the detected value — which the client discards and overwrites with `"en"`. A provenance
   lie, not a forced English transcription. → **S0.6**
4. **Ad excision silently no-ops on non-English text.** `_AD_PATTERNS` all require an English token, so
   a Greek transcript yields no ranges and the ad-free base is an **identity** copy with
   `is_adfree: True` and the sponsor reads intact. Not absolute — a Greek host reading an English URL
   would match. → **S2.5**, measured by **V.3**
5. **The sniff gate keeps the cheap transcript on non-English audio.** It judges the small-model
   transcript by counting entities with spaCy `en_core_web_sm`; on Greek that count is ~0. Off in every
   profile today, one config line from live. → **S0.6**

### 5.3 Surfaces the arc touches

| Surface | Component / file | Used for |
| --- | --- | --- |
| Consumer episode list + toolbar | `views/CatalogView.vue:43-58` — single-select filter via `ListToolbar`, plus a show selector and sort | v2 filter |
| Episode + show items | `EpisodeRow/Tile/Card.vue`, `ShowRow/ShowTile.vue`, `PodcastView.vue` | v2 badge |
| Operator shows library | `library/{ShowsBrowse,ShowsView,ShowDetailView}.vue`, `LibraryFilterBar.vue` | S0.5 field, v2 badge/filter |
| Claim serialization | `AppInsight`/`AppQuote`, `gi/contracts.py` `InsightSummary`/`SupportingQuote`, `hybrid_search._to_search_result`, `server/og/build.py`, snapshot exports | **S2.8** — every one drops node properties today |
| Position surfaces | `cil_queries.py` (nine entry points, RFC-125 §3), `search/relational_queries.py:170 positions_of`, `enrichment/enrichers/topic_consensus.py` (write-time) | **S2.12** |

### 5.4 Claims that were WRONG, and the pattern behind them

Two rounds of review found false claims; the second found errors introduced by the *first* round's
corrections. **The recurring mistake is asserting completeness or behaviour from a partial search —
verifying a reader and assuming its writer, or enumerating what turned up and calling the list
complete.** Recorded so the next pass recognises it rather than repeating it.

- **C-1 — "`load_processing_transcript` is the single resolver all NLP consumers use" is a docstring,
  not a fact.** Two callers; nine or more independent resolvers (S2.1). **D-4's "consumers change
  nothing" was false.** A follow-on correction was itself wrong: `gi/load.py` reading the raw `.txt` is
  reached only from `gi inspect` / `gi show-insight`, so it is a CLI bug, not a live pipeline path.
- **C-2 — the RSS `<language>` tag is never parsed.** `feed.language` is the run config written back
  out, so `_feed_language` reads config and `AppPodcastItem.language` serves `"en"` by construction.
  The badge would have shown config, the audit would have been a tautology, and `en-US → en` had no
  input. One nuance: "no episode-level language field at all" was overstated — `TranscriptInfo.language`
  exists, also from config.
- **C-3 — the DGX hazard was misdescribed** (§5.2 hazard 3).
- **C-4 — model selection happens once at provider init**, and in a multi-feed batch the ML singleton is
  held across feeds, so feed 1's model persists and feed 2's language is silently ignored.
- **C-5 — the ad-map cannot invert the ad-free transform.** Measured twice. The mechanism is *not* "one
  label per excised range": the error appears even when a range removes a whole line cleanly, because
  newly adjacent same-speaker turns coalesce and their labels vanish. Conclusion stands; the fix
  resolves through segments and `unit_id`, using **overlap** rather than containment.
- **C-6 — naming cannot move after translation.** Citation corrected: `ml_provider.py:1061` is inside
  `detect_speakers`, not `transcribe_with_segments`; the load-bearing evidence is that labels are baked
  into the `.txt` at write time (`episode_processor.py:2869-2872`).
- **C-7 — D-15's containment was right by accident.** `gi_embedding_model` is read by nothing in GI; GI,
  the bridge, CIL identity, KG topic clustering, `hybrid_search` and `query_router` all hardcode MiniLM,
  and `insight_clusters.json` would not follow a config change. Moot for v1, load-bearing whenever a
  swap happens — v2 doc §7.
- **C-8 — RFC-125's code mechanics were wrong in three places.** `surfaceable` is set by
  `_apply_voice_flags`, not `_apply_route_and_tag`; `position_arc`'s predicate is SPOKEN_BY-supported
  quote ∩ `ABOUT` ∩ `insight_type == "claim"` and never reads `surfaceable` or `speaker_id` — and that
  type filter is a *default* a caller can drop; there are **no KG evidence spans** at all.
- **C-9 — the Positions surface list was wrong twice**, at two entries and then at five. It is nine, one
  of which is write-time and cannot be gated at read time. Enumerating it is a deliverable of S2.12, not
  a claim in a document.
- **C-10 — Appendix A was incomplete.** Every number matched the source, but French (8.3), Arabic,
  Azerbaijani and Maori were dropped and then "everything unlisted is above 40%" was asserted.
- **C-11 — two overstatements in opposite directions from one `?`.** "Greek is covered by every eligible
  MT model" and "Serbian by none".
- **C-12 — smaller ones.** The `ner.py` gate selects the *default* NER model and is not a gate on NER;
  the cloud providers never substitute `"en"`, so there was nothing to remove there; 15 bakeoff
  profiles, not 14; "the index build runs in Docker" was cited to this document, which never said it;
  "the player forbids hard-coded user-facing strings" is a rule I asserted and could not be found.

## 6. Model and ASR evidence (verified 2026-09-28)

### 6.1 Translation model shortlist

| Model | HF id | Licence | `el` | `sr` | Verdict |
| --- | --- | --- | --- | --- | --- |
| TranslateGemma 27B / 12B / 4B | `google/translategemma-{27b,12b,4b}-it` | `gemma` — commercial use, no territorial carve-out | ? | ? | **Eligible.** 55 languages claimed; the list is nowhere public and **the model card is gated**, so a login settles it (V.5) |
| MiLMMT-46-12B v1.0 | `xiaomi-research/MiLMMT-46-12B-v1.0` | `gemma` | ✅ | ❌ | Eligible; Serbian absent from its 46. Sizes 1B/4B/12B — no 27B |
| LMT-60-8B | `NiuTrans/LMT-60-8B` | **apache-2.0** | ✅ | ❌ | Eligible; Serbian absent from its 60. Self-described Chinese-English-centric |
| Qwen3-30B-A3B | already served | apache-2.0 | ? | **?** | Baseline; coverage **never checked** (V.5) |
| ~~Hunyuan-MT / HY-MT~~ | Tencent | Territory **excludes the EU**, UK, South Korea | — | — | **Excluded**, confirmed |
| ~~NLLB-200~~ | `facebook/nllb-200-3.3B` | **cc-by-nc-4.0** | ✅ | ✅ | **Excluded — non-commercial.** The only shortlisted model covering Serbian |

TranslateGemma's report confirms it was optimised with "an ensemble of reward models, including
MetricX-QE and AutoMQM" — an argument for QE later, and a caution that QE from the same lineage would
partly mark its own homework.

### 6.2 ASR evidence — Whisper FLEURS WER

**Source**: Whisper paper, Appendix D.2.4, **Table 13 "WER (%) on Fleurs"**, the **`large-v2`** row.
Every value below was verified against the table.

**Caveats:** these are not large-v3 or turbo numbers (`large-v3` appears nowhere in the paper) and our
DGX model is `large-v3-turbo`, so treat this as a **conservative prior**; FLEURS is read speech and real
podcasts are worse; and bigger is not monotonically better — Serbian regressed from `large` 29.2 to
`large-v2` 33.9. Clusters A and B are complete; C is complete to 40%.

**Under 5%** — Spanish 3.0 · Italian 4.0 · English 4.2 · Portuguese 4.3 · German 4.5

**5–10%** — Japanese 5.3 · Polish 5.4 · Russian 5.6 · Dutch 6.7 · Indonesian 7.1 · Catalan 7.3 ·
**French 8.3** · Turkish 8.4 · Swedish 8.5 · Ukrainian 8.6 · Malay 8.7 · Norwegian 9.5 · Finnish 9.7

**Over 10%** — Vietnamese 10.3 · Thai 11.5 · Slovak 11.7 · **Greek 12.5** · Czech 13.3 ·
**Croatian 13.4** · Danish 13.8 · Tagalog 13.8 · Korean 14.3 · Romanian 14.4 · Bulgarian 14.6 ·
Chinese 14.7 · Galician 15.4 · Bosnian 15.7 · Arabic 16.0 · Macedonian 16.5 · Hungarian 17.0 ·
Tamil 17.5 · Hindi 21.5 · Estonian 21.9 · Urdu 22.6 · Latvian 23.1 · Slovenian 23.1 ·
Azerbaijani 23.4 · Hebrew 27.1 · Lithuanian 28.1 · Persian 32.9 · Welsh 33.0 · **Serbian 33.9** ·
Afrikaans 36.7 · Kannada 37.0 · Kazakh 37.7 · Icelandic 38.2 · Marathi 38.3 · Maori 38.5 ·
Swahili 39.3 — then Armenian 44.6 and the remaining low-resource languages. Javanese is `nan`.

### 6.3 The pilot language is undecided

| | Greek | Serbian |
| --- | --- | --- |
| FLEURS WER (large-v2; Cyrillic for `sr`) | **12.5** | **33.9** |
| In MiLMMT-46 / LMT-60 | ✅ / ✅ | ❌ / ❌ |
| In an eligible MT model | TranslateGemma unconfirmed; two others yes | **unknown** |

Serbian looks like the weakest link on both axes, and an earlier version of this document **demoted it
outright**. That was premature: three cheap checks were skipped. Croatian sits at 13.4 and is mutually
intelligible with Serbian, and FLEURS scores `sr` in **Cyrillic** while `hr` is **Latin** — so the
penalty may be orthographic rather than linguistic, which is an hour's work to test. Qwen3 was never
probed. TranslateGemma's card is gated, not unknowable. **V.5 runs now and the pilot is chosen from its
result.** This also reclassifies the script question from display to **capability**.

## 7. Decisions taken

| # | Decision | Rationale | Date |
| --- | --- | --- | --- |
| D-1 | Translate once, to English; analyze English | Every downstream layer stays single-path | 2026-09-28 |
| D-2 | Source is canonical, English is derived | The record is what was said | 2026-09-28 |
| D-3 | **Translation precedes ad detection**; the ad-free base is built on English | `_AD_PATTERNS` is English, so the alternative is an identity ad-free base feeding sponsor reads to GI. Also collapses two translation passes into one | 2026-09-28 |
| D-4 | **Revised.** Analysis reads English through **one** resolver, and routing every reader to it is *part of the work* | The original claimed the resolver already existed as such; that was its docstring (C-1). Slices S2.1, S2.2 | revised 2026-09-28 |
| D-5 | **Revised.** The Positions gate is a read-time filter across **every** position-bearing surface | Nine surfaces, not two or five — and one is write-time and cannot be filtered at read time (C-9). Two predicates, not one | revised 2026-09-28 |
| D-6 | **QE is out of v1** | Calibration needs a translated corpus that does not exist until v1 runs. (The human-bottleneck half of the original rationale dissolved with D-20.) v2 doc §4 | 2026-09-28 |
| D-7 | Defer, don't substitute, on translation-model availability | A model swap invalidates the evidence the language was enabled on | 2026-09-28 |
| D-8 | Speaker labels bypass the translator, and **naming stays before translation** | Naming is baked into the `.txt`; moving it later means a relabel that merges turns and invalidates the unit map (C-6) | revised 2026-09-28 |
| D-9 | The language override is a feeds-spec key | Needs a model field **and** an allowlist entry, and it reaches the whole run via `model_copy` | revised 2026-09-28 |
| D-10 | **Phase 0 ships before Gate V**, on its own | A correctness fix on the existing corpus, and the bake-off depends on it | 2026-09-28 |
| D-11 | **Revised.** The badge and the filter are **v2**, shipping together | A language chip was deliberately deleted in #2115 because the corpus is monolingual; that reasoning expires exactly when a second language arrives. v1 delivers the data, not the chrome. Operator confirmed the reversal is intended | revised 2026-09-28 |
| D-12 | The language filter is its own control, not an option inside the played/downloaded filter | The dimensions are orthogonal | 2026-09-28 |
| D-13 | **Withdrawn.** Serbian is not demoted; the pilot is chosen by measurement (V.5) | Demoting on a Cyrillic-referenced large-v2 number plus two language lists skipped three checks costing about a day (§6.3) | withdrawn 2026-09-28 |
| D-14 | **Same-language retrieval only, via a separate keyword-only table.** Non-English chunks live in their own table with no vector column; no embedding model changes | A vector-less row **cannot** surface in a semantic result, which a row tag plus a filter cannot guarantee — and a zero vector would actively outrank most real results. It also leaves the existing table untouched, so it should avoid the stale-index outage a column addition forces. Cross-lingual semantics is v2, sequenced at the database migration when the rebuild is already being paid for | revised 2026-09-28 |
| D-15 | **Withdrawn.** No embedding-model change in v1, so its blast radius is moot | Superseded by D-14; findings preserved in v2 doc §7 | withdrawn 2026-09-28 |
| D-16 | **Withdrawn.** The flag gates the pipeline; visibility is controlled by **when the feed is added to the production feed list** | There is no per-episode serving gate: ~32 modules walk the corpus independently and the indexer walks metadata directly, so a catalog filter would not stop search, CIL, MCP or digest. A separate corpus root was considered and rejected — the corpus is becoming a database shortly and should not be complicated. Config, not code, and it survives that move | withdrawn 2026-09-28 |
| D-17 | **Labelling and the Positions gate ship in Phase 2**, before any translated episode is served | The gate keys on the marker, so a claim written without one slips through a gate that is only fail-closed when the marker exists | 2026-09-28 |
| D-18 | Decisions here graduate to **ADRs** as they are implemented | The engineering process puts decisions in ADRs; this many living only in an arc note is process drift. V.4 is the first | 2026-09-28 |
| D-19 | **Translation runs after transcription and diarization, before summary** — in **one** seam, inside `generate_episode_metadata` | Summary output feeds GI topic labels and KG topics, so translating later gives English insights on Greek topics and fragments cross-episode identity. One seam covers ASR, cache hits, direct downloads, publisher transcripts and every reprocess cascade; "after transcription" names a seam that does not exist for publisher-transcript episodes | refined 2026-09-28 |
| D-20 | **A native-speaker reviewer is replaced by an LLM judge**, validated by fault injection | The reviewer was never identified, had no protocol, and gated Phases 2–4. Operator decision. The protocol and its three trust rules are in v2 doc §9; they apply to V.3 | 2026-09-28 |

## 8. Open decisions

1. **S0.1b's backfill mechanism.** No RSS language text exists in current metadata to recover; the
   migration fixture has no feed URL; `skip_existing` means a normal run never rewrites processed
   episodes. Refetch live in a CLI, read cached feed XML if prod keeps any, or accept
   `language_source: unknown` for legacy episodes? And does this need a formal corpus migration at all,
   given the database move is weeks away and the repo's own rule exempts additive optional fields?
2. **The normalizer's contract.** Behaviour for empty, `und`, `zxx`, `mul`, three-letter codes, macro
   languages and case; and **whether script is retained** — `sr-Latn-RS` → `sr` throws away the one
   datum V.5 needs (§6.3).
3. **Does the local Whisper tier serve non-English at all?** "No" makes S0.7 an S; "yes" makes it an L
   with a per-call model cache.
4. **Status vocabulary.** `EpisodeStatus.status` is `Literal["ok","failed","skipped"]` and does not
   admit `skipped_unsupported_language`, `deferred_quality_floor` or `translation_pending`.
5. **Non-diarized episodes** — how many exist, and do they get turns? The offset derivation skips
   segments it cannot locate, which breaks the every-segment-in-one-turn invariant. Blocks S1.1.
6. **Does V.3 stay a five-candidate bake-off**, or trim to a single-model sanity check with the
   comparison moved to v2? (v2 doc §10.)
7. **How much speaker aliasing is v1?** (v2 doc §10.)
8. **Is the transcript toggle and source reveal v1 or v2?** (S3.1; v2 doc §10.)
9. **Legal.** Translations are derived works; the Gemma terms carry downstream obligations for served
   outputs; GDPR erasure must enumerate `.en.*`, `translation.json` and provenance records.
10. **Keyword recall in the pilot language** through the index's English tokenizer (stemming,
    stop-words, accent folding). S2.9 measures it; if poor, the options are a per-language FTS table or
    accepting it explicitly.

## 9. Running notes

**2026-09-28 — arc opened.** PRD-047 and RFC-123/124/125 landed, reworked against the code rather than
accepted as drafted. QE cut from v1. Phase 0 reframed as "English as a declared language". The
demand/bake-off step renamed **Gate V**. Slice plan added.

**2026-09-28 — evidence pass.** V.1 closed with primary sources (§6): the MT shortlist verified,
Hunyuan's EU carve-out and NLLB's non-commercial licence confirmed, the Whisper FLEURS table extracted
from the paper. Multilingual retrieval brought into scope on the operator's call. DGX capacity removed
as a deployment-time question the operator owns.

**2026-09-28 — first adversarial review; three load-bearing claims false.** C-1, C-2, C-5. New slice for
RSS parsing; D-4 revised; the span chain rebuilt on segments and `unit_id`. Serbian demotion withdrawn
(D-13). Retrieval split. Trust markers and the gate pulled into Phase 2 (D-17).

**2026-09-28 — second adversarial review; the corrections had their own errors.** Found: the Positions
surface list was still incomplete (nine, one of them write-time); `gi/load.py` is CLI-only, not a live
pipeline path; the reader inventory had a bogus entry and was short; `position_arc`'s type filter is a
caller-overridable default; the trigger and gate populations would have diverged, excluding translated
non-claim insights permanently; the marker does not survive four serialization boundaries or snapshots
at all; two chunk sets would break the transcript-lift path; and **D-14's narrowing did not remove the
reindex** — a column addition bumps the index schema version and takes search offline until a full
rebuild. Also: the badge reverses a deliberate deletion (#2115); the feeds-spec override needs a model
field and carries a multi-feed singleton hazard; `Config._normalize_language` only lowercases, so
`en-US` already fails the `is_english` check today.

**2026-09-28 — v1 trimmed, v2 document opened.** The operator's test — *"v1 is a working pipeline and
product end to end; v2 is translation quality and fine edges"* — moved these to
[MULTILINGUAL_ARC_V2](MULTILINGUAL_ARC_V2.md): QE, source verification and the operator worklist, the
badge and the language filter, all three turns consumers, and cross-lingual semantic retrieval. The
native-speaker reviewer was replaced by an **LLM judge validated by fault injection** (D-20), removing
the arc's only human dependency. **D-16 withdrawn** — no serving gate and no separate corpus root; the
corpus is becoming a database shortly and should not be complicated, so visibility is controlled by when
a feed is added to the production feed list. **D-14 settled on option B** — a separate keyword-only
table, which makes a non-English chunk structurally incapable of appearing in a semantic result and
should avoid the schema bump. Nothing implemented; no issues opened.

<!-- Append new entries above this line. Decisions go in §7 with a D-number; facts about the code go in
     §5; claims found false go in §5.4; anything deferred goes in MULTILINGUAL_ARC_V2.md, never deleted. -->
