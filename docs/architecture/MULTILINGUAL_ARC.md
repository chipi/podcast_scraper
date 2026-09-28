# Multilingual ingest — arc notes

The one page that holds this arc together. The PRD says what the product needs, the three RFCs each
own a slice of the how, and this document is where the arc's **shape**, its **slice plan**, its
**verified code facts**, its **decisions** and its **running notes** live — so none of that has to be
re-derived from four documents or rediscovered from the code.

- **Arc**: multilingual ingest (source-language capture, English-normalized intelligence)
- **Opened**: 2026-09-28
- **Status**: design — nothing implemented
- **Branch**: `feat/multilingual-ingest`
- **Documents**: [PRD-047](../prd/PRD-047-multilingual-ingest.md) · [RFC-123](../rfc/RFC-123-speaker-turns-artifact.md) · [RFC-124](../rfc/RFC-124-multilingual-transcription-and-translation.md) · [RFC-125](../rfc/RFC-125-translation-confidence-and-claim-verification.md)

---

## 1. The arc in one page

The corpus is English-only by configuration, not by architecture. The bet is that we can ingest a
Greek or Serbian show, keep **what was actually said** as the canonical record, and derive an English
layer that every existing intelligence stage reads without a single per-language fork.

```text
audio (any enabled language)
  │
  ├─ transcribe + diarize IN SOURCE LANGUAGE ──────► ep1.txt            ← canonical, full timeline
  │                                                  ep1.segments.json
  │                                                  ep1.turns.json     ← RFC-123
  │
  ├─ TRANSLATE every turn-bounded unit ────────────► ep1.translation.json  ← the traceability map
  │                                                  ep1.en.txt            ← derived, full timeline
  │                                                  ep1.en.segments.json  ← doubles as subtitles
  │
  ├─ ad-detect + excise ON THE ENGLISH ────────────► ep1.en.adfree.txt   ← THE ANALYSIS TRANSCRIPT
  │
  └─ summary · KG · GIL · CIL · search ────────────► unchanged, single-path
         │
         └─ Positions (read-time CIL arc) ─────────► gated on source verification
```

Three properties make it work:

1. **The source is canonical, the English is derived.** A translation is an interpretation; the
   record is what was said. Every English span maps back to source text, source speaker and source
   audio time.
2. **Analysis never learns about language.** The consumers already share one resolver
   (`adfree_transcript.load_processing_transcript`). It gains a branch; they change nothing.
3. **The product never presents a translation as verbatim speech.** Labelled everywhere, traceable
   on tap, and — for anything that feeds a Position timeline — verified against the source first.

## 2. Document map

Read in this order. Each owns a distinct question; none restates another.

| Document | Owns | Does **not** own |
| --- | --- | --- |
| [PRD-047](../prd/PRD-047-multilingual-ingest.md) | Why, for whom, what "done" means, phase gates, language policy, operator + listener surfaces | Any implementation shape |
| [RFC-123](../rfc/RFC-123-speaker-turns-artifact.md) | `turns.json` — turns and sentences as addressable units. **Ships alone, on English value.** | Anything language-specific |
| [RFC-124](../rfc/RFC-124-multilingual-transcription-and-translation.md) | Language resolution, source capture, the translation stage, the artifact set, the pipeline order, the model bake-off | Trust, confidence, gating |
| [RFC-125](../rfc/RFC-125-translation-confidence-and-claim-verification.md) | Source verification of claims, the read-time Positions gate, the review worklist. QE deferred to v2. | Producing the translation |
| This document | Arc shape, slice plan, verified code facts, decisions, running notes | Requirements or design detail |

## 3. Phase ladder

Phase numbering means one thing across all four documents. The demand/bake-off validation step is
**Gate V**, not a phase, because it produces evidence rather than software.

| Phase | What | Gate to start | Visible outcome |
| --- | --- | --- | --- |
| **0 — English as a declared language** | Language becomes an explicit, resolved, validated, *displayed* parameter — for the corpus we already have | none; it is a correctness fix | An `EN` badge on every show and episode; an audit proving the corpus is English; no code path that substitutes `"en"` silently |
| **1 — Turns (RFC-123)** | `turns.json` + backfill, then GI attribution, search chunking, sentence cues | none; English value stands alone | Better quote attribution and speaker-true search on today's corpus |
| **Gate V — Validate** | Demand interviews; model/licence verification; the per-language bake-off and its quality gate | Phase 0 shipped (the bake-off needs per-episode language to work) | A go/no-go with evidence, per language |
| **2 — Translation (RFC-124)** | Source capture, translation stage, English render, ad-free-on-English | Gate V passed | One or two gated non-English feeds fully processed |
| **3 — Trust (RFC-125)** | Source verification + the `position_arc` filter | Phase 2 producing artifacts | Translated claims can enter Position timelines — and cannot before this |
| **4 — Surfaces** | Language toggle, translated-quote treatment, confidence markers, **language filters** | Phase 2 | A listener can read the original or the translation, and filter the corpus by language |

**Why Phase 0 is a real phase and not a refactor.** Today a Greek feed would be transcribed *as
English*, by a chain that can silently fall back to `tiny`, and stamped `"language": "en"` in its own
metadata. Phase 0 makes English an *asserted, checked, visible* fact rather than an assumption —
which is also precisely the plumbing translation needs. It is testable end to end on the corpus that
already exists, with no new models and no GPU, and it is the recommended standalone ship.

## 4. Slice plan

Each slice below is sized to be **one GitHub issue**: one goal, its own tests, its own acceptance
criteria, and shippable on its own without leaving the tree in a half-state. None of them is opened
yet — this section is the proposal to review.

**How to read the tables.** *Depends on* is a hard ordering constraint, not a preference. *Size* is
relative effort, not time: **S** = one sitting, contained; **M** = a day-ish, multiple files, real
test surface; **L** = multi-day, new subsystem or a migration. *Ship alone?* asks whether merging
only this slice leaves `main` correct and coherent.

**On batching.** Slices within a phase can be collapsed into fewer PRs if you would rather move in
bigger steps — the boundaries are drawn so the *ordering* is safe, not to force six PRs. The one
boundary worth keeping hard is between phases, and especially around Phase 0, which is the
recommended standalone release.

---

### Phase 0 — English as a declared language

**Ships as one release.** Seven slices, no new models, no GPU, no user-visible behaviour change
except the badge and a normalized language tag. The point of the phase is that at the end of it,
"this episode is in English" is something the system *asserts, checks and displays* rather than
assumes — and the same machinery then carries any other language.

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S0.1** | Language tag normalization and per-episode language resolution | `normalize_language_tag` (`el-GR`→`el`, `sr-Latn-RS`→`sr`, `pt_BR`→`pt`, junk→`None`) + `resolve_episode_language(feed_entry, feed_doc, cfg)` + `config/languages.yaml` registry with `en` the only enabled language. `_feed_language` calls the normalizer so the catalog and pipeline cannot disagree. | — | M | Yes |
| **S0.2** | Corpus language audit: prove the existing corpus is English | Read-only CLI that walks every feed and every episode in a corpus, resolves a language for each, and reports the distribution plus an explicit list of anything that does not resolve to `en`. Run it on prod and commit the report. | S0.1 | S | Yes |
| **S0.3** | Expose episode-level language in the app API | Additive `language` field on the episode list and detail responses, read from episode metadata. Show-level already exists (`AppPodcastItem.language`). Contract tests for present / absent / non-normalized values. | S0.1 | S | Yes |
| **S0.4** | `EN` language badge on every show and episode | New `LanguageBadge.vue` (compact square, uppercase ISO code) rendered on `EpisodeRow`, `EpisodeTile`, `EpisodeCard`, `ShowRow`, `ShowTile`, `PodcastView`, and the operator viewer's show surfaces. Hidden when the language is unknown rather than guessing. | S0.3 | M | Yes |
| **S0.5** | Thread the resolved language as an explicit parameter; remove every silent `"en"` | The transcription call site passes the episode's resolved language instead of `cfg.language`; `whisper_provider.py:197`, `ml_provider.py:827/885` and the cloud providers stop substituting `"en"` for a missing language; `metadata_generation.py:2843` writes the resolved value; the manifest records the language and where it was resolved from. | S0.1 | M | Yes |
| **S0.6** | Select the Whisper model from the resolved language, with a non-English quality floor | `normalize_whisper_model_name` is driven by the resolved language: `.en` variants for English, multilingual otherwise, and a `min_model_for_non_english` floor that truncates the fallback chain and fails `deferred_quality_floor` rather than transcribing with `base`/`tiny`. Must be byte-identical for English. | S0.5 | M | Yes |
| **S0.7** | Phase 0 release gate: English end-to-end, proven | One full pipeline run with language explicitly declared and asserted at every stage; byte-identical artifacts on the English fixture corpus vs. pre-Phase-0; a regression test that an episode resolving to `el` can never ship `"language": "en"`; the S0.2 audit re-run clean. | S0.1–S0.6 | M | This **is** the ship |

**Phase 0 acceptance.** The audit reports 100% of existing shows and episodes as `en` with a named
resolution source; every episode and show renders an `EN` badge; no provider can receive a null
language and substitute English; a non-English language selects a multilingual model at or above the
floor or fails loudly; and the English corpus is byte-identical to before.

---

### Phase 1 — Turns (RFC-123)

Independent of everything multilingual. It pays for itself on the English corpus and it is the unit
translation later needs. Can run in parallel with Phase 0 — the two touch different files.

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S1.1** | `turns.json`: build turns and sentences from the segments sidecar | `build_turns` in `providers/ml/diarization/turns.py`, written per variant (raw + ad-free) with the §2.4 invariants asserted at build time. Property tests plus 5 committed goldens. Nothing reads it yet. | — | L | Yes |
| **S1.2** | Backfill `turns.json` across the existing corpus | `podcast-scraper turns backfill --corpus <dir>`, no GPU, idempotent, with a coverage report. | S1.1 | S | Yes |
| **S1.3** | Switch GI speaker attribution to turn lookup | Char-span binary search into turns, regex kept as the fallback when `turns.json` is absent. Gated on replaying the 7,101-quote #2062 set and reporting the full transition matrix — zero name→different-name transitions required to flip the default. | S1.2 | M | Yes |
| **S1.4** | Turn-bounded search chunking with per-chunk speakers | Chunk within turn boundaries; `SegmentDocument` and the LanceDB segment schema gain `speaker_ids` and `turn_ids`; reindex via the RFC-118 delta path; retrieval eval reporting recall@k and speaker-precision before/after. | S1.2 | L | Yes |
| **S1.5** | Player sentence-granularity cues | Optional `?granularity=sentence` on the segments contract, serving sentence cues from `turns.json`. Default stays raw segments. | S1.2 | S | Yes |

---

### Gate V — Validate (evidence, not software)

No production code. This is what the go/no-go decision rests on, and it needs Phase 0 finished
because the bake-off has to transcribe non-English audio properly to measure anything.

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **V.1** | ~~Verify translation model availability and licence terms~~ | **DONE 2026-09-28 — see §6.1.** Shortlist is TranslateGemma (4B/12B/27B, `gemma`), MiLMMT-46-12B (`gemma`), LMT-60-8B (`apache-2.0`), Qwen3-30B baseline. Hunyuan/HY-MT excluded (EU carve-out, confirmed); NLLB excluded (`cc-by-nc-4.0`, confirmed). QE candidates unverified — v2, off the critical path. | — | S | closed |
| **V.2** | Demand check with the beta cohort | What share of their listening is non-English, and which shows they would add. Target: ≥30% naming at least one. | — | S | n/a |
| **V.3** | Per-language bake-off harness and gate report | Bake-off profiles per candidate, the scoring script, native-reviewer CSV export, and the run on 2–3 **Greek** episodes plus one Spanish or Italian control. Reports all six RFC-124 §7 measurements including ad-detection survival. Greek, not Greek+Serbian — see §6.3 and V.5. | S0.7 | L | n/a |
| **V.5** | Settle whether Serbian is supportable at all | Closes §6.3. Three questions, in order: (a) is `sr` in TranslateGemma's 55 — read the full technical report or probe the model; (b) does Whisper's 33.9% FLEURS WER collapse toward Croatian's 13.4% on **Latin-script** Serbian, measured on 2 real episodes; (c) if Serbian has no eligible MT model, is treating it as Croatian defensible, and what does a native reviewer say about the output. Outcome is a yes/no on offering Serbian, not a model choice. | §6.1, §6.2 | M | n/a |
| **V.4** | Gate V decision record | An ADR or arc note recording which languages passed, with the numbers, and the chosen model + pinned revision per language. | V.2, V.3, V.5 | S | n/a |

---

### Phase 2 — Translation (RFC-124)

Behind `multilingual_ingest`, on one or two operator-chosen feeds. Everything here is testable with a
stub translator before any GPU is involved.

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S2.1** | Per-feed language override in feed config | Feed entries accept an optional mapping (`- url:` + `language:`) alongside the bare string form; the resolver reads it as the highest-precedence source; the shows library displays the resolved language and its source. | S0.1 | S | Yes |
| **S2.2** | Translation stage: units, `translation.json`, English render | Turn-bounded unit packing, the DGX `dgx_vllm_translate` client with per-episode batching and retry, `translation.json`, and the English screenplay + segments sidecar rendered through the existing formatter. Stub-translator integration test. | S1.1, V.4 | L | Yes — flag-off is a no-op |
| **S2.3** | Build the ad-free base on English; extend the transcript resolver | `load_processing_transcript` gains the `.en.txt` branch; the English ad-free artifacts are produced by the existing `build_adfree_artifacts`; `resolve_units_for_span` implements the adfree→full-English→unit→source chain. This is D-3 and D-4. | S2.2 | M | Yes |
| **S2.4** | Speaker identity across scripts | Labels bypass the translator; CIL alias lookup, then deterministic transliteration with a `_looks_like_person` guard and source-script fallback; the vLLM speaker detector runs on the English transcript with source-language metadata as context. | S2.2 | M | Yes |
| **S2.5** | Segments API `?lang=` and translation status on episode detail | Additive `language` / `machine_translated` / `translation_model` on `SegmentsResponse`; `translation_status` on episode detail; contract tests across English and translated episodes. | S2.2 | S | Yes |
| **S2.6** | Translation observability | Manifest `language`, `translation` and `adfree.built_on` / `ad_chars_removed` blocks; Grafana panels for wall time, units/sec, pending backlog, failures by language, and ad-chars-removed by language. | S2.2, S2.3 | S | Yes |
| **S2.8** | Multilingual retrieval: index both layers, search in either language | Tier-1 chunks are built from **both** the source-language transcript and the English analysis transcript, each tagged with its `language`, both pointing at the same episode — so a Greek query and an English query reach the same episode. Requires swapping the **search** embedding model to a multilingual one and a full corpus reindex. Gated on the existing retrieval eval: English recall@k must not regress. See RFC-124 §6.2 for why the blast radius is contained to search. | S2.3 | L | Yes — but the reindex is corpus-wide |
| **S2.9** | Language-aware search surfacing | Search results carry the `language` of the chunk that matched, so a hit can say "matched the Greek original" vs "matched the English translation"; the query workspace and MCP `search_corpus` pass a `language` filter through. | S2.8 | M | Yes |
| **S2.7** | Phase 2 gate: one non-English feed end-to-end | A gated feed processed from audio to insights **and findable by a query in either language**, with every GI quote resolving to source text and source audio, and the English corpus still byte-identical. | S2.1–S2.6, S2.8 | M | This is the ship |

---

### Phase 3 — Trust (RFC-125, verification-only)

QE is **not** here — see D-6. This phase is the source-verification pass and the read-time gate.

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S3.1** | Per-claim translation provenance on GI and KG nodes | The `translation` block (`translated`, `source_language`, `unit_ids`) written into node `properties` by the GI artifact builder and the KG evidence writer, via `resolve_units_for_span`. No new top-level artifact keys. | S2.3 | M | Yes |
| **S3.2** | Source-grounded entailment verification | The verification pass: source-language cited units plus one turn of context on each side, the English claim, a JSON `supports / contradicts / insufficient` verdict from the DGX Qwen service, and the verification record on the node. Includes the cross-model re-translation fallback on `insufficient`. | S3.1 | L | Yes |
| **S3.3** | Gate Positions on verification (read-time filter) | `position_arc` and `topic_conversation_arc` include a translated insight only when its outcome is `verified`. Truth-table tests over {absent, null, verified, unverified, contradicted}, plus a byte-identical-response isolation test on the English corpus. | S3.1 | M | Yes — fail-closed before S3.2 lands |
| **S3.4** | Operator review worklist | JSONL export plus a minimal operator view of contradicted and unverified claims, source and translation side by side, with verify / reject / re-translate actions and operator provenance. | S3.2 | M | Yes |

**Note on S3.3 ordering.** It can ship *before* S3.2 and is arguably better that way: with no
verification records in existence, the filter excludes every translated claim from timelines, which
is exactly the desired default. That makes the gate real before the machinery that satisfies it
exists.

---

### Phase 4 — Surfaces

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S4.1** | Filter shows and episodes by language | A language control in the consumer episode toolbar (`CatalogView`) and on show browse (`BrowseView`), plus the operator library filter bar. Built on the existing `TypeFilterBar`. Its own control, **not** an option inside the played/downloaded filter, so "Greek **and** unplayed" is expressible. Rendered only when the corpus holds more than one language. | S0.3, S2.7 | M | Yes |
| **S4.2** | Transcript language toggle and original-text reveal | Player reads the source transcript or the English subtitles against the original audio; tapping a translated quote reveals the source sentence and plays the source span. UXS first. | S2.5 | L | Yes |
| **S4.3** | Translated-content markers | "Translated from <Language>" chip wherever translated content renders; a confidence marker for amber/red once QE exists, nothing extra for green. | S3.1 | M | Yes |

---

### Deferred to v2

| # | Issue title | Why deferred |
| --- | --- | --- |
| **V2.1** | Per-unit quality estimation and per-language band calibration | D-6. Calibration needs a native reviewer per language to fit thresholds — a human bottleneck on the critical path — and it needs a translated corpus to calibrate against, which does not exist until Phase 2 has run. Verification alone catches the failure mode that matters. |
| **V2.2** | Word-level anchors via forced alignment | Per-language aligner checkpoints; `turns.json` is already forward-compatible (`timing: word_aligned`). |

---

### Critical path

```text
S0.1 → S0.5 → S0.6 → S0.7 ══════ PHASE 0 SHIPS ══════╗
  ├→ S0.2  (audit)                                    ║
  └→ S0.3 → S0.4  (badge)                             ║
                                                      ▼
S1.1 → S1.2 → {S1.3, S1.4, S1.5} ═ PHASE 1 SHIPS ═   {V.3, V.5} → V.4 ═ GATE V ═╗
       (parallel with Phase 0)                        (V.1 closed)               ▼
                            S2.2 → S2.3 → {S2.4, S2.5, S2.6, S2.8 → S2.9} → S2.7 ═ PHASE 2 ═╗
                                                                                        ▼
                                                    S3.3 (early) · S3.1 → S3.2 → S3.4 ═ PHASE 3
                                                                                        ▼
                                                                          {S4.1, S4.2, S4.3}
```

The only place the critical path is genuinely serial is `S0.1 → S0.5 → S0.6`, because each one
depends on the previous one's contract. Everything else forks.

## 5. Code facts this arc rests on

Verified against the source on 2026-09-28. These are the findings that changed the design, recorded
here so the next session does not re-derive them.

### 5.1 Load-bearing findings

| Finding | Evidence | Consequence |
| --- | --- | --- |
| **There is no stance-extraction stage, and deliberately none.** Stances are GI insights; Positions are a read-time query. | `server/cil_queries.py:636` `position_arc`; `enrichment/profile_sets.py:144-146`; ADR-108 update 2026-07-08 retiring `stance_timeline` | The trust gate is a **read-path filter**, so it is retroactive over the existing corpus and fail-closed on a missing record. The PRD has no stance-extraction dependency. |
| **`analysis_transcript_ref` already exists in essence.** `load_processing_transcript` is "the single resolver all NLP consumers use"; `ProcessingTranscript.transcript_ref` is already what quotes point at. | `workflow/adfree_transcript.py` | One added branch in an existing precedence (`.en.txt` → `.adfree.txt` → `.txt`), not a new field threaded through every consumer. `adfree_transcript_relpath` already composes `ep1.en.txt` → `ep1.en.adfree.txt`. |
| **Show-level language is already an API field.** | `server/schemas.py:1719` `AppPodcastItem.language` | The show badge needs no new field — but it carries the **raw** RSS tag, which is why normalization is S0.1. Episode-level language is **not** exposed, hence S0.3. |

### 5.2 The four silent hazards for non-English audio

Each fails quietly rather than loudly, which is what makes them worth writing down. Phase 0 closes
all four.

1. **Silent degradation to an unusable model.** `normalize_whisper_model_name` correctly drops `.en`
   for non-English, then builds a chain down to `base` and `tiny` from
   `FALLBACK_WHISPER_MODELS_MULTILINGUAL`. A DGX outage on a Greek episode yields unusable text
   rather than a failure. → **S0.6**
2. **No per-episode language path exists.** Providers read run-global config when the argument is
   absent — `ml_provider.py:885`, `gemini_provider.py:472`, `mistral_provider.py:356`,
   `deepgram_provider.py:281` — and the single call site passes exactly that global
   (`workflow/episode_processor.py:2296`). The DGX provider hard-defaults:
   `whisper_provider.py:197` sends `language or "en"`. → **S0.5**
3. **Episode metadata already asserts a language, and it will be wrong.**
   `workflow/metadata_generation.py:2843` writes `"language": cfg.language`. A Greek episode ships
   stamped `en`. A field that lies, not a field that is missing. → **S0.5**, proven by **S0.7**
4. **Ad excision silently no-ops on non-English text.** `gi/filters.py` `_AD_PATTERNS` are English
   regexes (`brought to you by`, `sponsored by`, `\w+ dot com slash`). On a Greek transcript nothing
   matches, `excise_ad_regions` returns no ranges, and `build_adfree_artifacts` emits an **identity**
   ad-free base — with `is_adfree: True` and the sponsor reads intact in the text GI, KG and search
   read. → **S2.3**, measured by **V.3**

Hazard 4 is why translation precedes ad detection. See D-3.

### 5.3 Surfaces the arc touches

| Surface | Component / file | Used for |
| --- | --- | --- |
| Consumer episode list + toolbar | `web/learning-player/src/views/CatalogView.vue` (`filterOptions`: all / unplayed / played / insights / downloaded, plus `show` and `sort`) | S4.1 — the language filter joins this toolbar as its **own** control |
| Consumer episode items | `EpisodeRow.vue`, `EpisodeTile.vue`, `EpisodeCard.vue` | S0.4 — episode language badge |
| Consumer show items | `ShowRow.vue`, `ShowTile.vue`, `PodcastView.vue`, `BrowseView.vue` | S0.4 badge, S4.1 show filter |
| Reusable chip control | `web/learning-player/src/components/TypeFilterBar.vue` — generic multi-select kind chips, built explicitly for reuse | S4.1 — a new instance, not new chip code |
| Operator shows library | `web/gi-kg-viewer/src/components/library/{ShowsBrowse,ShowsView,ShowDetailView}.vue`, `LibraryFilterBar.vue`, `chips/LibraryFeedChip.vue` | S0.4, S2.1, S4.1 |

No generic badge primitive exists in the player, so `LanguageBadge.vue` is new (small) component work.

## 6. Model and ASR evidence (verified 2026-09-28)

This section closes slice **V.1**. Everything here was checked against a primary source in this pass;
where something was **not** verified it says so. Read the caveats before quoting a number.

### 6.1 Translation model shortlist

| Model | HF id | Params | Licence | `el` | `sr` | `hr` | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TranslateGemma 27B / 12B / 4B | `google/translategemma-{27b,12b,4b}-it` | 27B / 12B / 4B | `gemma` — commercial use permitted, no territorial carve-out | ? | ? | ? | **Eligible.** 55 languages claimed, but the list is not enumerated on the model card, the launch blog, or the abstract, so `sr` coverage is **unconfirmed** |
| MiLMMT-46-12B v1.0 | `xiaomi-research/MiLMMT-46-12B-v1.0` | 12.19B | `gemma` | ✅ | ❌ | ✅ | **Eligible, but no Serbian.** Serbian is absent from its 46 |
| LMT-60-8B | `NiuTrans/LMT-60-8B` | 8.19B | **apache-2.0** | ✅ | ❌ | ✅ | **Eligible, but no Serbian.** Absent from its 60 (language tags are on the model card) |
| Qwen3-30B-A3B | already served | 30B MoE | apache-2.0 | — | — | — | Generalist baseline; also the RFC-125 verification model |
| ~~Hunyuan-MT / HY-MT~~ | Tencent | — | Community Licence: "Territory" **excludes the EU**, UK and South Korea | — | — | — | **Excluded.** Confirmed, not assumed |
| ~~NLLB-200~~ | `facebook/nllb-200-3.3B` | 3.3B | **cc-by-nc-4.0** | ✅ | ✅ | ✅ | **Excluded — non-commercial.** The only shortlisted model that covers Serbian |

Details worth carrying forward:

- **TranslateGemma** released 2026-01-15 in 4B / 12B / 27B (27B targets a single H100), gemma3
  architecture, technical report arXiv:2601.09012. That report confirms RFC-125's provenance claim in
  its own words: "we optimize translation quality using an ensemble of reward models, including
  MetricX-QE and AutoMQM". So the QE family we would adopt in v2 is the same one this model was
  trained against — which is an argument for QE later, and a caution that QE would be partly
  marking its own homework.
- **MiLMMT-46** is Gemma3-based, trained through continual pretraining on 143B tokens across 46
  languages, and claims to outperform TranslateGemma and HY-MT-1.5 while being competitive with
  Google Translate and Gemini 3 Pro (arXiv:2608.10812). Sizes are 1B / 4B / 12B — **there is no 27B**,
  so the earlier "MiLMMT-46-12B" naming was right and any 27B comparison is not available.
- **LMT-60** is Qwen3-based and self-describes as **Chinese-English-centric** across 234 directions.
  `el→en` is in scope, but its centre of gravity is not European.
- **Not verified in this pass**: the QE candidates (MetricX-24, COMETKiwi, xCOMET) — availability and
  licence. They are v2 (RFC-125 §7), so they are not on the v1 critical path.

### 6.2 ASR evidence — Whisper FLEURS WER

**Source**: Whisper paper (Radford et al.), Appendix D.2.4, **Table 13 "WER (%) on Fleurs"**. Numbers
below are the **`large-v2`** row, which is the strongest model in that table.

**Three caveats, all load-bearing:**

1. **These are not large-v3 or turbo numbers.** The string `large-v3` appears nowhere in the paper —
   it postdates it. large-v3 is generally better per language, and the per-language figures exist in
   the `openai/whisper` repo's `language-breakdown.svg`, but that is a figure and was **not** extracted
   here. Our DGX model is `faster-whisper-large-v3-turbo-ct2`, so treat these as a **conservative
   prior**, not as our expected error rate.
2. **FLEURS is read speech.** Podcast audio — crosstalk, music, informal register, two hours of it —
   is worse. Appendix A of PRD-047 says this and it remains true.
3. A bigger model is not monotonically better per language. Serbian got **worse** from `large`
   (29.2) to `large-v2` (33.9).

**Cluster A — under 5%**
Spanish 3.0 · Italian 4.0 · English 4.2 · Portuguese 4.3 · German 4.5

**Cluster B — 5% to 10%**
Japanese 5.3 · Polish 5.4 · Russian 5.6 · Dutch 6.7 · Indonesian 7.1 · Catalan 7.3 · Turkish 8.4 ·
Swedish 8.5 · Ukrainian 8.6 · Malay 8.7 · Norwegian 9.5 · Finnish 9.7

**Cluster C — over 10%**
Vietnamese 10.3 · Thai 11.5 · Slovak 11.7 · **Greek 12.5** · Czech 13.3 · **Croatian 13.4** ·
Danish 13.8 · Tagalog 13.8 · Korean 14.3 · Romanian 14.4 · Bulgarian 14.6 · Chinese 14.7 ·
Galician 15.4 · Bosnian 15.7 · Macedonian 16.5 · Hungarian 17.0 · Tamil 17.5 · Hindi 21.5 ·
Estonian 21.9 · Urdu 22.6 · Latvian 23.1 · Slovenian 23.1 · Hebrew 27.1 · Lithuanian 28.1 ·
Persian 32.9 · Welsh 33.0 · **Serbian 33.9** · Afrikaans 36.7 · Kazakh 37.7 · Icelandic 38.2 ·
Marathi 38.3 · Swahili 39.3 · Armenian 44.6 · Amharic 140.3 · Assamese 106.2 · Gujarati 102.7 ·
Kannada 37.0 · Malayalam 100.7 · and the remaining low-resource languages above 40%

### 6.3 The Serbian problem

The two axes agree, and they disagree with the plan as written:

| | Greek | Serbian |
| --- | --- | --- |
| Whisper FLEURS WER (large-v2) | **12.5** | **33.9** |
| In MiLMMT-46's languages | ✅ | ❌ |
| In LMT-60's languages | ✅ | ❌ |
| In an eligible MT model at all | yes | **only via TranslateGemma's unconfirmed 55** |
| Covered by NLLB | yes | yes — but NLLB is non-commercial |

Serbian is the weakest link on **both** axes simultaneously. PRD-047's Appendix A placed it in
"usable with review (10–15%)"; the source says 33.9%, which is more than double that, and in the tier
the same appendix calls "not ready".

**The likely cause is script, and that makes it testable.** Croatian sits at 13.4% while Serbian sits
at 33.9%, and the two are mutually intelligible. FLEURS Serbian is **Cyrillic**; FLEURS Croatian is
**Latin**. So the hypothesis is that Whisper's Serbian penalty is largely an orthography penalty, not
a language-comprehension one — and that Latin-script Serbian would behave much more like Croatian.

This reframes an open question. "Serbian script: normalize to Latin or keep Cyrillic?" was filed as a
**display** decision. On this evidence it is a **capability** decision that affects transcription
accuracy and whether any eligible MT model covers the language at all.

**Consequence for the plan**: **Greek is the pilot language.** Serbian gets its own investigation
slice (V.5) before it is promised to anyone, and it is removed from the Gate V "hardest likely-demanded
pair" framing, because the pair is not symmetric — one member is fine and the other may not be
supportable at all under an eligible licence.

## 7. Decisions taken

| # | Decision | Rationale | Date |
| --- | --- | --- | --- |
| D-1 | Translate once, to English; analyze English | Every downstream layer stays single-path; the prompts, evals and thresholds already exist for English | 2026-09-28 |
| D-2 | Source is canonical, English is derived | Grounding and provenance mean the record is what was said | 2026-09-28 |
| D-3 | **Translation precedes ad detection**; the ad-free base is built on English | `_AD_PATTERNS` is English, so the alternative is an identity ad-free base feeding sponsor reads to GI. Also collapses two translation passes into one, since subtitles need the full timeline anyway | 2026-09-28 |
| D-4 | `analysis_transcript_ref` is a branch in `load_processing_transcript`, not a new field | One function changes instead of every consumer; one answer to "which transcript" instead of two | 2026-09-28 |
| D-5 | The Positions gate is a **read-time** filter in `position_arc` | Positions are already a read-time query; this makes the gate retroactive and fail-closed | 2026-09-28 |
| D-6 | **QE is cut from v1.** Verification-only. | QE's per-language calibration needs a native reviewer to fit thresholds — a human bottleneck on the critical path — and a translated corpus that does not exist yet. Source-grounded entailment needs no calibration and catches the failure we actually fear (inverted negation/hedge) | 2026-09-28 |
| D-7 | Defer, don't substitute, on translation-model availability | A model swap invalidates the bake-off evidence and any later calibration | 2026-09-28 |
| D-8 | Speaker labels bypass the translator and must still satisfy `_looks_like_person` | Identity is CIL's job; a label that fails that check un-attributes a whole turn | 2026-09-28 |
| D-9 | The language override lives in feed config | RFC-104's shows library has no backend, and every other per-feed decision in this repo is config | 2026-09-28 |
| D-10 | **Phase 0 ships before Gate V**, on its own | It is a correctness fix on the existing corpus, not a feature bet, and the bake-off depends on it | 2026-09-28 |
| D-11 | The language **badge** ships in Phase 0; the language **filter** ships once a second language exists | A filter over a monolingual corpus is a dead control; a badge over one is a verified fact | 2026-09-28 |
| D-12 | The language filter is its own control, not an option inside the played/downloaded filter | The two dimensions are orthogonal; merging them makes "Greek and unplayed" unexpressible | 2026-09-28 |
| D-13 | **Greek is the pilot language. Serbian is an open investigation (V.5), not a launch target.** | §6.3: Serbian is 33.9% FLEURS WER against Greek's 12.5%, and it is absent from both MiLMMT-46 and LMT-60. It is the weakest link on ASR *and* MT at once, and the one model that covers it (NLLB) is non-commercial | 2026-09-28 |
| D-14 | **Multilingual retrieval is in v1, not deferred.** Both the source-language and the English layer are indexed, and a query in either language reaches the episode | If the content is in the corpus in that language, it has to be findable in that language from day one. Deferring it would ship a corpus you can only search in translation | 2026-09-28 |
| D-15 | The search embedding model is swapped; `gi_embedding_model` is **not** | `vector_embedding_model` is already a separate config key from `gi_embedding_model`, so search can go multilingual while GI ABOUT edges, chunked extraction and the CIL bridge stay on the pinned MiniLM. Containing the blast radius is the whole reason this is affordable | 2026-09-28 |

## 8. Open decisions

1. **Does TranslateGemma cover Serbian?** Its 55 languages are not enumerated in any source checked
   (§6.1). This is the difference between "Serbian is hard" and "Serbian has no eligible MT model".
   Closed by reading the full technical report or probing the model. Slice **V.5**.
2. **Serbian script is a capability decision, not a display one** (§6.3). Transcribe Cyrillic as-is,
   force Latin-script output, or romanize post-hoc? This affects WER, MT coverage and CIL identity at
   once.
3. **Source-language ad-free variant** — keep one, derived by mapping English ad ranges back through
   `translation.json`? It has a reader/player use, no analysis use.
4. **Code-switching** (English passages inside a Serbian episode): pass the unit through
   untranslated, or accept the damage?
5. **Ad-survival threshold.** There is no prior for "what fraction of sponsor reads survive
   translation into pattern-matchable English". The first language measured sets it.
6. **Badge scope.** Does the `EN` badge render on every surface that shows a show or episode, or only
   where other metadata chips already appear? S0.4 assumes the latter to avoid visual noise.
7. **Which multilingual embedding model** for the search index (RFC-124 §6.2 / slice S2.8), and
   whether the English retrieval eval holds after the swap. The eval gate decides; the model choice
   does not.

## 9. Running notes

**2026-09-28 — arc opened.** PRD-047 and RFC-123/124/125 landed on `feat/multilingual-ingest`,
reworked against the code rather than accepted as drafted: the stance-stage dependency does not exist
(§5.1), ad detection is English-keyed and reorders the pipeline (§5.2 hazard 4), and
`analysis_transcript_ref` collapses into an existing resolver (§5.1). QE cut from v1 (D-6). Phase 0
reframed from a silent refactor into "English as a declared language" with a badge and a corpus audit
as its visible outcome (D-10, D-11), and the demand/bake-off step renamed **Gate V** so phase numbers
mean one thing. Slice plan added (§4): 7 slices for Phase 0, 5 for Phase 1, 4 for Gate V, 7 for
Phase 2, 4 for Phase 3, 3 for Phase 4, 3 deferred to v2. No issues opened; no code written.

**2026-09-28 — evidence pass; Serbian demoted, retrieval promoted.** Slice V.1 closed with primary
sources (§6): the MT shortlist is verified, Hunyuan's EU carve-out and NLLB's non-commercial licence
both confirmed rather than assumed, and the Whisper FLEURS table (paper Table 13) pulled for every
language. Two plan changes fell out. **Serbian is not a launch target** (D-13): 33.9% WER vs Greek's
12.5%, absent from both MiLMMT-46 and LMT-60, and covered only by the one model we cannot use
commercially — so Greek is the pilot and Serbian becomes investigation slice V.5, with the script
question reclassified from display to capability (§6.3). **Multilingual retrieval moved into v1**
(D-14, slices S2.8/S2.9) on the operator's call that content in the corpus must be searchable in its
own language from day one; the blast radius stays in search because `vector_embedding_model` is
already separate from `gi_embedding_model` (D-15). PRD-047 Appendix A replaced with the sourced
three-cluster table. DGX capacity removed from the documents — it is a deployment-time question the
operator owns, not a design constraint.

<!-- Append new entries above this line, newest last. Keep each to a few lines: what changed, what it
     cost, what it invalidated. Decisions go in §6 with a D-number; facts about the code go in §5. -->
