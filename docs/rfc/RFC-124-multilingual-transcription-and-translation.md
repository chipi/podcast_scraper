# RFC-124: Multilingual Transcription and English Translation Stage

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Pipeline (transcription, diarization, ad-free base, GI/KG/summary), DGX serving, player (segments contract), operator feed curation
- **Related PRDs**:
  - `docs/prd/PRD-047-multilingual-ingest.md` — product requirements this RFC implements (FR1–FR4, FR6)
  - `docs/prd/PRD-044-operator-shows-library.md` — per-show language override surface
- **Related RFCs**:
  - `docs/rfc/RFC-123-speaker-turns-artifact.md` — prerequisite: turns/sentences define translation units
  - `docs/rfc/RFC-125-translation-confidence-and-claim-verification.md` — provenance and source verification over this RFC's outputs (QE deferred to v2)
  - `docs/rfc/RFC-005-whisper-integration.md`, `docs/rfc/RFC-058-audio-speaker-diarization.md`
  - `docs/rfc/RFC-106-tiered-dgx-service-fallback.md` — tiered fallback semantics
  - `docs/rfc/RFC-109-per-episode-observability-manifest.md` — manifest fields
  - `docs/rfc/RFC-115-transcript-prefix-caching-llm-stages.md` — LLM stages cache the analysis transcript as a prompt prefix
- **Related ADRs**:
  - `docs/adr/ADR-155-pin-every-model-checkpoint.md` — the translation checkpoint is pinned
- **Arc notes**: `docs/architecture/MULTILINGUAL_ARC.md` (§4 slice plan, decisions D-1 … D-4)

## Abstract

This RFC makes language a per-show property instead of a global setting. Episodes are transcribed
and diarized in their source language, and that transcript stays canonical. A new **translation
stage** then produces an English derived transcript (`<stem>.en.txt`, `<stem>.en.segments.json`)
plus a unit map (`<stem>.translation.json`) linking every English span back to source text, audio
time and speaker. The existing ad-free machinery then runs on the **English** text — not the source
— producing `<stem>.en.adfree.txt`, which becomes the transcript every analysis stage reads. No
consumer changes, because the resolver those consumers already share
(`adfree_transcript.load_processing_transcript`) simply gains one more branch. The translation model
is chosen per language by a bake-off with an explicit quality gate.

## Problem Statement

The pipeline is English by configuration. `language: en` is global
(`config/profiles/prod_dgx_full.yaml:70`). The DGX Whisper provider sends `language or "en"`
(`providers/tailnet_dgx/whisper_provider.py:197`). `speaker_detectors/ner.py:133` gates NER on
`cfg.language == "en"`. The DGX Whisper (`faster-whisper-large-v3-turbo-ct2`) and pyannote
community-1 are multilingual already, so transcribing Greek is close to a parameter change.

Analysis is English-shaped by design, and it should stay that way. GIL grounding requires quotes
that are verbatim substrings with char offsets (`gi/grounding.py`, `EvidenceSpan`). Extraction
prompts, QA/NLI entailment, MiniLM embeddings and spaCy NER are all English. Making each layer
multilingual would fork every prompt, eval and threshold per language. Translating once, to
English, keeps every downstream layer single-path.

Three things are missing: a place in the pipeline to translate, an artifact model that keeps
translated text **traceable** to what was actually said, and a way for a per-episode language to
reach the transcriber at all. Without the second, a translated quote is indistinguishable from
verbatim speech, which is incompatible with grounded objectivization.

**Four hazards the current code creates for non-English audio.** Each is verified against the
source, and each is a silent failure rather than an error:

1. **Silent degradation to an unusable model.** `whisper_utils.normalize_whisper_model_name`
   correctly drops `.en` for non-English, but it then builds a chain from
   `FALLBACK_WHISPER_MODELS_MULTILINGUAL` down to `base` and `tiny`. For a Greek episode, a DGX
   outage would silently produce text that is unusable rather than failing.
2. **No per-episode language path.** Every provider reads the run-global config when the argument is
   absent — `ml_provider.py:885` (`self.cfg.language or "en"`), `gemini_provider.py:472`,
   `mistral_provider.py:356`, `deepgram_provider.py:281` — and the single call site passes exactly
   that global (`workflow/episode_processor.py:2296`, `language=cfg.language`). A run that
   processes feeds in two languages cannot express that today.
3. **The episode metadata already asserts a language, and it will be wrong.**
   `workflow/metadata_generation.py:2843` writes `"language": cfg.language` onto every episode. Until
   hazard 2 is fixed, a Greek episode ships stamped `en`. This is not a missing field; it is a field
   that will lie.
4. **Ad excision silently no-ops on non-English text.** `gi/ad_regions.py` matches `gi/filters.py`'s
   `_AD_PATTERNS`, which are English regexes (`brought to you by`, `sponsored by`,
   `\w+ dot com slash`). On a Greek transcript none of them match, so `excise_ad_regions` returns no
   ranges and `build_adfree_artifacts` produces an **identity** ad-free base. The artifact exists,
   `is_adfree` is `True`, and the sponsor reads sit in the analysis text feeding GI, KG and search.
   Nobody would notice.

Hazard 4 is the one that reorders this design. It is why translation runs **before** ad detection,
not after.

**Use cases:**

1. **Native show in the library.** A Greek show is processed end to end, and its insights and
   positions sit in the same corpus as English ones.
2. **Subtitles.** The player plays the Greek audio with English cues aligned to it.
3. **Traceable quote.** Tapping a translated quote plays the original audio span and shows the
   original sentence.

## Goals

1. **Per-show language routing** with an operator override and explicit skip for unsupported
   languages.
2. **The source language stays canonical.** It is never overwritten, and it is always servable.
3. **Complete, turn-aligned English translation**, with a lossless map from English spans back to
   source spans.
4. **Zero change to analysis layers.** They keep calling the resolver they already call.
5. **Ad handling that actually works on non-English episodes**, rather than an identity pass that
   looks like success.
6. **Pinned, measured models.** The translation model is chosen per language by bake-off, and
   every checkpoint is pinned.

## Constraints & Assumptions

**Constraints:**

- English episodes: byte-identical artifacts and no added latency when the feature flag is on.
- No silent model substitution for translation. A different translation model invalidates the
  bake-off evidence the language was enabled on — and, once QE lands in v2, its calibration too
  (RFC-125 §7). An unavailable model therefore means **defer**, not **fall back**.
- Only models whose license permits this deployment (EU operator, commercial product) are eligible.
- Runs on the DGX Spark as its own served process. English ingest keeps priority in the work queue.
- **One translation pass per episode.** The full-timeline text is translated exactly once. Every
  other English artifact is derived from it by existing deterministic machinery.

**Assumptions:**

- RFC-123 turns exist for the episode. Translation does not run without them.
- Whisper punctuation in enabled languages is good enough for sentence splitting. The per-language
  gate (§7) checks this.
- Conversational podcasts translate acceptably at the turn or sentence-group level, without
  cross-turn context. §7 tests this.
- Sponsor reads in a non-English episode survive translation as recognizable English sponsor
  language. §7 measures this; it is the assumption the §3 ordering rests on.

## Design & Implementation

### 1. Language resolution

This produces `episode.language` (ISO 639-1) in `resolve_episode_language(feed_entry, feed_doc, cfg)`:

1. **Operator override** on the feed entry (§1.1).
2. **RSS `<language>`**, normalized: `el-GR` → `el`, `sr-Latn-RS` → `sr`, `pt_BR` → `pt`.
3. **Profile default** `language` (currently `en`).

**The RSS tag is read but not normalized today.** `server/corpus_catalog.py:159` `_feed_language`
returns the raw tag (`feed.get("language")`, stripped) and nothing more. Normalization is new code,
not a lift: a shared `normalize_language_tag(raw) -> str | None` that lowercases, splits on `-`/`_`,
takes the primary subtag, validates it against the registry, and returns `None` for anything
unparsable. `_feed_language` then calls it, so the catalog and the pipeline cannot disagree.

**1.1 Where the override lives.** PRD-047 FR1.1 says "operator override (shows library)", but there
is no store for it. RFC-104's own header states **"Backend: none (reuses existing endpoints)"** — the
shows library is a browse mode over `GET /api/corpus/feeds` — and the feed list
(`config/corpus-expansion.feeds.yaml`) is a flat list of bare `- url:` entries with no per-feed keys
anywhere in `config/*.yaml`. So this RFC must choose a home rather than assume one:

- **Proposed**: extend the feed entry to an optional mapping, keeping the bare string form valid.

  ```yaml
  feeds:
    - url: https://example.com/feed.xml          # unchanged, resolves to profile default
    - url: https://example.gr/feed.xml
      language: el                                # operator override
  ```

  The loader accepts either form. This is config-as-source-of-truth, matches how every other
  per-feed decision in this repo is made, and needs no new persistence layer or API.
- **Deferred**: a writable override in the operator UI. That needs a feed-record store, which is a
  larger change than this feature justifies. FR6.1's "override control" therefore reduces to
  *display* the resolved language and its source in the shows library, with editing done in config.

This is the one place where the PRD asks for a surface that does not exist. It is called out rather
than absorbed.

**Supported-language registry.** `config/languages.yaml`:

```yaml
languages:
  en: { tier: excellent, enabled: true,  translation: none }
  es: { tier: excellent, enabled: false, translation: { model_ref: tx-default } }
  el: { tier: review,    enabled: false, translation: { model_ref: tx-default } }
  sr: { tier: review,    enabled: false, translation: { model_ref: tx-default },
        script: latin }   # display normalization of source text (PRD-047 OQ2)
models:
  tx-default: { id: <winner>, revision: <sha>, serve: dgx_vllm_translate }
```

An episode whose language is not `enabled` gets status `skipped_unsupported_language` and is never
transcribed. The registry is the single switch per language, and enabling one is a reviewed config
change backed by bake-off evidence (§7).

**Sanity check, not routing.** After transcription, Whisper language ID runs on a 30 s window taken
from the middle third of the episode, which avoids intros, music and ads. If the result is not the
declared language with probability ≥ 0.8, the manifest records `language_mismatch: {declared,
detected, p}` and the episode is flagged for the operator. It is **never** rerouted automatically,
because intros and ad reads regularly fool language ID.

### 2. Source-language transcription and diarization

- **Thread the resolved language to the call site.** `episode_processor._transcribe_one` passes
  `language=cfg.language` today. It passes the episode's resolved language instead. The providers
  need no signature change — `language` is already on the `TranscriptionProvider` protocol
  (`transcription/base.py:48`) and every implementation accepts it.
- **Remove the English default on the DGX path.** `whisper_provider.py:197` sends
  `"language": language or "en"`. With per-episode language threaded, a missing language must be an
  error or an explicit auto-detect, never a silent `"en"`. Same for
  `ml_provider.py:827/885`'s `self.cfg.language or "en"`.
- **Quality floor for non-English.** `normalize_whisper_model_name` gains a
  `min_model_for_non_english` (default `large-v3`, or turbo on DGX). For non-`en` languages the
  fallback chain is truncated at that floor. If no tier can meet it, the attempt fails with
  `deferred_quality_floor` rather than transcribing with `base` or `tiny`.
- Diarization (pyannote community-1) is language-agnostic, so nothing changes.
- **Speaker naming.** `ner.py` skips detection for non-`en` today, and that stays off. For
  non-English episodes, speaker naming runs on the **English** transcript after translation (§5.5),
  using the existing vLLM speaker detector.
- Source artifacts stay at their existing paths (`<stem>.txt`, `<stem>.segments.json`) and gain a
  top-level `language` field. RFC-123 `turns.json` is built from them as usual.
- **No source-language ad-free variant is written.** See §3.

### 3. Pipeline order, and why translation precedes ad detection

For an English episode, today's order is: transcribe → segments → ad-free base → analysis. The
ad-free base is what analysis reads.

For a non-English episode the naive extension — ad-free base, then translate it — fails on hazard 4:
`_AD_PATTERNS` are English, so the source-language ad-free base is an identity copy and the ads
travel into analysis. It also costs more, because subtitles need the **full** timeline while
analysis needs the **ad-free** text, which would mean translating two overlapping texts.

The order is therefore inverted for non-English episodes:

```text
audio
 └─ transcribe + diarize (source language)      → ep1.txt, ep1.segments.json
     └─ turns (RFC-123)                          → ep1.turns.json
         └─ TRANSLATE every unit, full timeline  → ep1.translation.json
             └─ render English screenplay        → ep1.en.txt, ep1.en.segments.json
                 ├─ turns over English           → ep1.en.turns.json
                 └─ ad-free base ON ENGLISH      → ep1.en.adfree.{txt,segments.json,admap.json}
                     └─ ANALYSIS reads this      (GI, KG, summary, search)
```

Three things fall out of this, all of them using machinery that already exists:

1. **Ad detection works**, because it runs on English text with the English patterns it was written
   for. No per-language ad lexicon is needed, now or later.
2. **One translation pass.** The full-timeline units are translated once. They are simultaneously the
   subtitle source and the input to the ad-free derivation. `build_adfree_artifacts` already drops
   the segments inside detected ad ranges and **re-renders the survivors** through the same
   formatter, so the English ad-free text and its offsets come out exact, with no new code.
3. **The existing path-naming helper composes unchanged.** `adfree_transcript_relpath` is
   `splitext(path)` + `.adfree`, so `ep1.en.txt` yields `ep1.en.adfree.txt` with no modification.
   `load_processing_transcript("…", "ep1.en.txt")` then finds that ad-free sibling by the rule it
   already implements.

**Bonus, not a goal:** ad ranges discovered in English map back through `translation.json` to source
char ranges and source times, so the source-language reader and the source audio player can hide or
skip ads too — something the English-only pattern set could not otherwise give a Greek episode.

### 4. Artifact model

For a non-English episode with stem `ep1`:

```text
transcripts/
  ep1.txt                        # source language (canonical), screenplay, FULL timeline
  ep1.segments.json              # source segments, language: "el"
  ep1.turns.json                 # RFC-123, source variant
  ep1.translation.json           # unit map (this RFC)
  ep1.en.txt                     # English screenplay, derived, FULL timeline → subtitles
  ep1.en.segments.json           # English cues: one per translation unit, derived
  ep1.en.turns.json              # RFC-123, English variant
  ep1.en.adfree.txt              # English ad-free — THE ANALYSIS TRANSCRIPT
  ep1.en.adfree.segments.json
  ep1.en.adfree.admap.json       # excised ranges, in ep1.en.txt coordinate space
  ep1.en.adfree.turns.json
```

English episodes are untouched: `ep1.txt`, `ep1.segments.json`, `ep1.adfree.*`, `ep1.turns.json`.

**`translation.json`**:

```json
{
  "version": "1.0",
  "derived": true,
  "source_language": "el",
  "target_language": "en",
  "source": { "turns_ref": "…turns.json", "turns_sha256": "…" },
  "target": { "transcript_ref": "…en.txt", "en_sha256": "…" },
  "model": { "id": "<winner>", "revision": "<sha>", "serve": "dgx_vllm_translate" },
  "units": [
    {
      "unit_id": "t0007.u01",
      "turn_id": "t0007",
      "sent_ids": ["t0007.s01", "t0007.s02", "t0007.s03"],
      "speaker_label": "Κυριάκος Μητσοτάκης",
      "start_ms": 2471200, "end_ms": 2489100,
      "src_char_start": 18356, "src_char_end": 18702,
      "src_text": "…",
      "en_char_start": 20114, "en_char_end": 20488,
      "en_text": "…",
      "status": "ok"
    }
  ]
}
```

`en_char_*` are offsets into `ep1.en.txt` — the full-timeline English text — captured from
`format_diarized_screenplay_with_offsets`.

**Resolving a claim back to a unit.** A GI citation's char span lives in the analysis transcript,
which is `ep1.en.adfree.txt`. The chain is:

```text
span in ep1.en.adfree.txt
  → shift by ep1.en.adfree.admap.json            (ad-free → full English space)
  → range-lookup over en_char_start/en_char_end  (→ unit_id, turn_id, speaker)
  → src_char_*, start_ms/end_ms                  (→ source text + source audio)
```

The first hop is the same reconciliation the ad-map was built for; the docstring in
`adfree_transcript` states that purpose explicitly. A single helper,
`resolve_units_for_span(span, admap, translation_map)`, owns the whole chain so no consumer
reimplements it. RFC-125 calls exactly this helper.

**English transcript construction.** `ep1.en.txt` is rendered with the **same screenplay formatter**,
feeding it one pseudo-segment per unit (`text = en_text`, times from the unit, speaker label resolved
per §5.4). That makes `ep1.en.segments.json` a valid segments sidecar, so everything that reads
segments — timing lookup for quotes, player cues, RFC-123 turns, the ad-free builder — works on the
English layer unchanged.

**Derived marking.** `.en.*` artifacts and `translation.json` carry `derived: true` and
`translated_from`. The source artifacts do not.

### 5. Translation stage

**5.1 Units.** Sentence groups inside a single RFC-123 turn of the **source** variant:

- Greedily pack consecutive sentences up to about 120 source words (configurable). Never cross a
  turn boundary.
- A sentence longer than the cap is its own unit. It is never split mid-sentence.
- Backchannel turns (`backchannel: true`) are translated as one unit each, without context.

Units are a deterministic function of `ep1.turns.json`. They are not stored separately beyond
`translation.json`.

**5.2 Context.** v1 translates each unit **on its own text**. Prepending the previous turn and
stripping it from the output is fragile, because boundaries drift. The bake-off (§7) measures
whether unit-only translation meets the gate. If a candidate model supports a native context field,
`context_turns: N` is an allowed per-model setting.

**5.3 Serving.** A dedicated vLLM instance (`dgx_vllm_translate`, its own port). Requests are batched
per episode (all units, ordered) with a bounded concurrency budget. The stage runs on the same DGX
work queue as other GPU stages, and English episodes are scheduled ahead of translation batches.

- **Failure semantics.** Per RFC-106 tiering, the translation tier is DGX-only in v1. If it is
  unavailable, the episode moves to `translation_pending` and is retried with backoff. There is no
  cloud translation fallback, because substituting a model would make quality untraceable — the
  episode's text would no longer be the text the language's gate evidence was measured on.
- **Partial output.** A unit that errors or returns empty text is retried up to 2 times, then
  marked `status: "failed"`. Any failed unit blocks analysis for the episode (PRD-047 FR3.4) —
  concretely, the English ad-free base is not built, so the resolver finds no analysis transcript
  and the analysis stages do not run. The source transcript and the partial subtitles remain
  servable.

**5.4 Names and identity.** Speaker labels are in source script (e.g. `Κυριάκος Μητσοτάκης`), and the
translation model may transliterate names inconsistently. Two rules:

1. **Speaker labels are never sent through the translator.** They are resolved separately. The
   source-script label is looked up in CIL aliases. If a canonical person exists, the English
   screenplay uses the canonical Latin name. Otherwise a deterministic transliteration (ICU
   `Any-Latin; Latin-ASCII` plus per-language rules) becomes the display label, and the
   source-script form is written as a CIL alias when the person node is minted.
2. **In-text names** are left to the model. Downstream CIL resolution then matches English-text
   mentions against aliases that include transliteration variants. The bake-off measures name
   consistency explicitly (§7).

A label must round-trip: the English screenplay's `Label: ` prefix is what
`build_unverified_named_turns` and the turn builder read, and `gi/speakers.py:_looks_like_person`
requires ≥2 tokens and no publisher token. A transliteration that collapses to one token
(or to a publisher-like string) silently un-attributes every quote in that turn. The transliteration
step therefore asserts the label still satisfies `_looks_like_person`, and falls back to the
source-script label when it does not.

**5.5 Speaker naming for non-English episodes.** The existing vLLM speaker detector runs on the
English transcript and the English feed metadata, with the source-language feed and episode
description passed as extra context. That context matters: show notes often name the guest in
source script.

### 6. API and player

- **Segments contract** (`GET /api/app/episodes/{slug}/segments`,
  `server/routes/app_episodes.py:463`) gains `?lang=`:
  - default: source-language segments (unchanged for English);
  - `lang=en` on a translated episode: English unit-level cues from `.en.segments.json`.
  - `SegmentsResponse` gains additive fields: `language`, `machine_translated: bool`, and
    `translation_model` when translated. It is a plain `BaseModel` with no `extra="forbid"`, so the
    addition is safe.
- **Episode detail** exposes `language` and `translation_status` (`none | ok | pending | failed`).
- **Quotes** from translated episodes expose `translated: true` and `source_language`, plus
  `source_excerpt` (the source text of the covering unit(s)) and the source time range. Audio
  playback uses the source times, which are identical to the English cue times by construction.
- **Search** indexes **both** layers — see §6.2.

UI treatment (the translated marker, the original-text reveal, the language toggle) is deferred to
a UXS.

**6.1 Language visibility and filtering** (PRD-047 FR7). Two additive API changes and one new
component carry it:

- **Show language already exists**: `AppPodcastItem.language` (`server/schemas.py:1719`), described as
  "Feed language tag (e.g. 'en') if known". It currently serves the **raw** RSS tag, because
  `_feed_language` does not normalize — so `en-US` and `en` are two values for one language. Routing
  it through `normalize_language_tag` (§1) is what makes it usable as a badge and as a filter key.
- **Episode language is not exposed** and gains an additive `language` field on the episode list and
  detail responses, read from the episode metadata that `metadata_generation.py:2843` already writes.
- **`LanguageBadge.vue`** is new: a compact squared chip with the uppercase code. No generic badge
  primitive exists in `web/learning-player`, so this is the primitive. It renders on `EpisodeRow`,
  `EpisodeTile`, `EpisodeCard`, `ShowRow`, `ShowTile` and `PodcastView`, and alongside the operator
  viewer's existing `library/chips/` components. Absent language renders nothing rather than a guess.
- **The filter reuses `TypeFilterBar.vue`**, whose own docstring states the intent — "Type filtering
  is one concept, so it is one component: a new surface passes its options and a testid prefix". The
  language filter is a new instance of it on the `CatalogView` toolbar (which today carries
  all / unplayed / played / insights / downloaded plus a `show` selector and a sort), on show browse,
  and on the operator `LibraryFilterBar`. It is a **separate** control from the played/downloaded
  filter, because language is orthogonal to listening state — collapsing them would make "Greek and
  unplayed" unexpressible — and it renders only once the corpus holds more than one language.

**6.2 Multilingual retrieval** (PRD-047 FR4.5). A Greek episode must be findable by a Greek query and
by an English one, and both must land on the same episode. That is not what indexing the English layer
alone gives, so retrieval is part of v1 rather than a later addition (arc note D-14).

**Two things have to change, and exactly two.**

1. **Index both layers.** `build_segment_documents` produces Tier-1 chunks from the analysis
   transcript today. For a translated episode it produces **two** chunk sets — one from the
   source-language transcript, one from the English analysis transcript — each carrying its own
   `language`, both keyed to the same `episode_slug`. A lexical (BM25) match on a Greek query then hits
   the Greek chunks, an English query hits the English chunks, and the episode is reachable either way.
   `SegmentDocument` and the LanceDB segment schema gain `language` alongside the `speaker_ids` /
   `turn_ids` that RFC-123 adds, so it is one schema change, not two.
2. **A multilingual embedding model for the search index.** `DEFAULT_EMBEDDING_MODEL` is
   `sentence-transformers/all-MiniLM-L6-v2` (`config_constants.py:350`, revision-pinned at `:407`) and
   it is English. Lexical matching alone would make a Greek query find Greek chunks but never the
   English ones, so *cross*-lingual recall — the property that makes one corpus out of two languages —
   needs a multilingual encoder on the dense side.

**Why the blast radius is contained.** MiniLM is referenced in four places: the search vector index,
GI `ABOUT` edges (`gi/about_edges.py:28`, hardcoded), GI chunked extraction
(`gi/chunked_extraction.py:49`, hardcoded default), and the CIL bridge builder
(`builders/bridge_builder.py:121`, `"minilm-l6"`). Swapping all four would put topic linking, insight
extraction and cross-episode identity through an unmeasured model change on a 678-episode corpus.

It is already avoidable: **`vector_embedding_model` is a distinct config key from `gi_embedding_model`
and `embedding_model`** (`config.py:3615`, `:3124`, `:3083`). So this RFC changes
`vector_embedding_model` only. GI, KG and the bridge keep the pinned MiniLM and are untouched. That
containment is the reason this is affordable in v1 (arc note D-15).

**What it costs.** A corpus-wide reindex, through the existing delta path (RFC-118). The dense vectors
for every chunk change, so this is a full rebuild rather than an incremental one, and per the arc's
own operational notes the index build runs in Docker.

**The gate.** The swap changes English retrieval, not just Greek. So it is gated on the existing
retrieval eval over the current query set: **English recall@k must not regress**, reported before and
after alongside the speaker-precision metric RFC-123 adds. A multilingual encoder that helps Greek and
costs English recall is not an acceptable trade, and the eval is what makes that visible rather than
discovered in production. Candidate encoders are a Gate V question, decided by that measurement rather
than by reputation.

**Query side.** No query-language detection and no query translation. A query goes to both layers as
it is, and hybrid RRF (RFC-090) fuses the hits. Results carry the matched chunk's `language`, so the UI
can say whether a hit came from the Greek original or the English translation, and `search_corpus`
gains an optional `language` filter.

### 7. Model selection: bake-off and per-language gate

This reuses the existing bake-off pattern (`config/profiles/bakeoff_*.yaml`, 14 profiles today), with
one profile per candidate.

**Candidates** (operator's shortlist; **availability and license terms are unverified in this pass**
and are a Phase 0 task, not an established fact):

| Candidate | Why | License note (to verify) |
|---|---|---|
| TranslateGemma 27B | translation-specialized Gemma 3; strongest prior | Gemma terms |
| TranslateGemma 12B | same family, cheaper; check the quality gap | Gemma terms |
| MiLMMT-46-12B (Xiaomi) | reports gains over TranslateGemma; newer | confirm `el`/`sr` coverage |
| LMT-60 8B | permissive fallback option | Apache-2.0 |
| Qwen3-30B-A3B (current vLLM) | generalist baseline, already served | Apache-2.0 |

Excluded on licence grounds, subject to the same verification: Tencent Hy-MT (EU exclusion), NLLB
(non-commercial).

**Eval set per language:** 3 real episodes from shows likely to be requested, about 60 units each,
including every unit that the current GI pipeline cites in those episodes.

**Measurements:**

1. **ASR check** (pre-translation): a native reviewer rates 20 random source units as
   usable / minor errors / broken.
2. **Automatic**: latency and GPU memory per model. Optionally a reference-free QE model as a cheap
   comparative signal **if** a permissively licensed checkpoint is available — it would inform the
   model choice, but no thresholds are calibrated and nothing ships from it, because QE is a v2
   addition (RFC-125 §7). The gate below does not depend on it.
3. **Native reviewer, blind to model**: for each position-bearing unit, whether the translation is
   meaning-preserving (yes / minor / no), with error type (negation, hedge, name, omission, other).
4. **Position agreement**: run GI extraction on each model's English output and compare the
   attributed stance against the reviewer's reading of the original.
5. **Name consistency**: the fraction of person mentions whose English form resolves to the correct
   CIL identity.
6. **Ad-detection survival** (new, gates the §3 ordering): on episodes that contain sponsor reads,
   the fraction of those reads that `_AD_PATTERNS` catches in the translated English. A low number
   means §3 buys less than claimed and a per-language cue list is back on the table.

**Gate** (per language; all must pass to set `enabled: true`):

- ASR: ≥ 90% of sampled units usable or minor.
- ≥ 90% of position-bearing units meaning-preserving.
- ≥ 90% position agreement.
- ≥ 95% name consistency.
- Ad-detection survival reported, with a threshold set from the first language measured (there is no
  prior to set it from).

The winning model per language is recorded in `config/languages.yaml` with its pinned revision. The
default expectation is one model for all enabled languages. Per-language models are allowed but
cost a served instance each.

### 8. Interaction with transcript prefix caching (RFC-115)

`cache_transcript_prefix` embeds the analysis transcript as the leading, stage-invariant block of the
system prompt so providers prefix-cache it across stages
(`config.py:3071`, `prompting/megabundle.py:60`). Switching that transcript from source to English is
transparent to the mechanism. Two consequences are not:

- A re-translation (a model revision bump, or RFC-125's cross-model re-translation of selected
  units) changes the English text and therefore invalidates the whole episode's cached prefix, not
  just the changed span. Re-translation is an episode-level cost, not a per-unit one.
- Translated episodes reach the LLM stages with a different token distribution (English rendered
  from Greek), so cache-hit-rate dashboards should be read per language rather than in aggregate.

### 9. Observability

The per-episode manifest (RFC-109) gains:

```json
"language": { "declared": "el", "source": "rss", "mismatch": null },
"translation": {
  "status": "ok", "model": "…@<sha>", "units": 412, "failed_units": 0,
  "src_words": 11820, "en_words": 12954, "wall_s": 318
},
"adfree": { "built_on": "en", "ad_chars_removed": 4120 }
```

`adfree.built_on` is the field that would have made hazard 4 visible: an English episode reads
`source`, a translated one reads `en`, and an `ad_chars_removed: 0` on a language where the gate
measured survival is now a question rather than a silence.

Grafana: translation wall time per episode, units/sec, pending backlog, failures by language, and
ad-chars-removed by language.

## Key Decisions

1. **Translate once, to English, and analyze English.**
   - **Rationale**: every downstream layer stays single-path, and the evals, prompts and
     thresholds already exist for English.
2. **The source is canonical; English is derived.**
   - **Rationale**: grounding and provenance mean the record is what was said, and a translation is
     an interpretation of it.
3. **Translation precedes ad detection; the ad-free base is built on English.**
   - **Rationale**: `_AD_PATTERNS` is English, so the alternative is an identity ad-free base that
     silently feeds sponsor reads to GI. It also collapses two translation passes into one, and
     makes ad-skipping available to the source-language reader as a by-product.
4. **`analysis_transcript_ref` is a branch in an existing resolver, not a new metadata field.**
   - **Rationale**: `load_processing_transcript` is already "the single resolver all NLP consumers
     use", and `ProcessingTranscript.transcript_ref` is already the ref that quote and viewer
     references point at. Adding `.en.txt` to its precedence changes one function; inventing a
     parallel field would mean touching every consumer and having two answers to the same question.
5. **The English transcript is a real screenplay-format transcript with a segments sidecar.**
   - **Rationale**: timing lookup, player cues, turns and the ad-free builder all work on it with
     zero consumer changes.
6. **Defer, don't substitute.**
   - **Rationale**: a translation model swap silently invalidates the bake-off evidence the language
     was enabled on, and later its QE calibration too.
7. **Speaker labels bypass the translator, and must still look like people.**
   - **Rationale**: identity is CIL's job. MT transliteration is inconsistent, and a label that
     fails `_looks_like_person` un-attributes a whole turn.
8. **Declared language routes; detection only warns.**
   - **Rationale**: intros, music and ads regularly fool language ID. The operator override exists
     for wrong RSS tags.
9. **The language override lives in feed config, not in a new store.**
   - **Rationale**: the shows library has no backend to hold it, and every other per-feed decision
     in this repo is config.

## Alternatives Considered

1. **Whisper's built-in `translate` task.**
   - **Pros**: one pass; no new model.
   - **Cons**: large-v3-turbo was not trained on translation data. It loses the source transcript
     (no canonical record, no diarized source text) and cannot be QE-scored against a source.
   - **Why rejected**: quality, and it destroys provenance.
2. **Multilingual analysis: extract directly from source-language text.**
   - **Pros**: no translation error inside extraction.
   - **Cons**: forks every prompt and eval per language, breaks English-only QA/NLI/embedding
     assumptions, and GIL verbatim grounding would produce non-English quotes the product cannot
     display coherently.
   - **Why rejected as the primary path**: cost and fragmentation. It is **kept as a verification
     method** in RFC-125, where it is used surgically.
3. **Translate the whole transcript as one document.**
   - **Pros**: maximum context.
   - **Cons**: output cannot be aligned back to turns or times, and one hallucination can corrupt
     long spans.
   - **Why rejected**: breaks subtitles and traceability.
4. **Ad-free first, then translate the ad-free text** (the naive ordering).
   - **Pros**: matches the English pipeline's shape exactly.
   - **Why rejected**: `_AD_PATTERNS` is English, so the source ad-free base is an identity copy and
     ads enter analysis silently. It also needs a second translation pass for full-timeline
     subtitles.
5. **Per-language ad-cue lexicons, keeping ad-free before translation.**
   - **Pros**: no reordering.
   - **Cons**: a new hand-maintained lexicon per language, unmeasurable until each language has a
     corpus, and duplicated maintenance forever.
   - **Why rejected for v1**: §7's ad-survival measurement decides whether this is ever needed.
6. **Cloud MT API** (DeepL / Google Translate).
   - **Pros**: strong quality, no GPU.
   - **Cons**: content leaves the infrastructure, per-character cost at corpus scale, and the model
     can change underneath (unpinnable, against ADR-155).
   - **Why rejected**: pinning and provenance. It may be used as a **reference** in the bake-off
     only.
7. **Auto-detect language per episode.**
   - **Why rejected**: false detections on intros and ads. Used as a warning only.

## Testing Strategy

- **Unit**: language-tag normalization (`el-GR`, `sr-Latn-RS`, `pt_BR`, junk) and resolution
  precedence; the registry skip path; the quality-floor truncation in
  `normalize_whisper_model_name`; the removal of the `or "en"` defaults; unit packing (never crosses
  turns, never splits sentences); `translation.json` ↔ `.en.segments.json` offset consistency (every
  English char maps to exactly one unit); `resolve_units_for_span` across the
  adfree→full→unit→source chain, including spans that cross a unit boundary and spans adjacent to an
  excised ad range; the transliteration `_looks_like_person` guard.
- **Integration**: a fixture Greek episode (short, CC-licensed or synthetic TTS) through transcribe →
  diarize → turns → translate (stub translator returning deterministic text) → English render →
  ad-free-on-English → GI, asserting that `EvidenceSpan.transcript_ref` names
  `…en.adfree.txt`, that its offsets resolve into that text, and that the span maps back to source
  times. Include one episode with an injected English sponsor read in the translated output, to
  assert the ad-free base actually excises it.
- **Contract**: `SegmentsResponse` with and without `lang`, on English and translated episodes.
- **Isolation**: the English fixture corpus produces byte-identical artifacts with the feature flag
  on and off. This is the gate that protects the 678-episode production corpus.
- **Regression for hazard 3**: an episode whose resolved language is `el` must not have `"language":
  "en"` in its metadata.
- **Bake-off**: the §7 harness is committed as profiles plus a scoring script. Reviewer sheets are
  exported as CSV, and results are committed as an eval report.

## Rollout & Monitoring

Phase names match PRD-047 and `docs/architecture/MULTILINGUAL_ARC.md`; slice ids (S0.x, S2.x) refer to
that document's §4 slice plan.

- **Phase 0 — English as a declared language (S0.1–S0.7), ships first and alone.** Language
  resolution and tag normalization, the registry (every non-`en` language `enabled: false`), the
  per-episode language threading, removal of the `or "en"` defaults, model selection from the resolved
  language with the non-English quality floor, the corpus audit, and the language badge. It precedes
  **Gate V**, because the bake-off cannot measure non-English transcription until this exists. The
  only behavioral change for English is that a misconfigured language now fails loudly instead of
  transcribing as English.
- **Gate V — Validate.** Model/licence verification, demand interviews, the §7 bake-off and its gate
  report. Evidence, not code.
- **Phase 2 (S2.1–S2.7)**: translation stage, English render, ad-free-on-English and the artifact set,
  behind `multilingual_ingest: true`, on 1–2 operator-chosen feeds in one gated language.
- **Phase 3**: RFC-125 provenance, source verification and the read-time Positions gate. (QE is a v2
  addition — RFC-125 §7.)
- **Phase 4**: player `lang=` toggle, translated-quote treatment (UXS) and the language filters. Then
  enable further languages as each passes the gate.

**Success criteria:**

1. At least 95% of episodes in enabled languages reach `translation_status: ok` without manual
   intervention.
2. Zero English-episode regressions (isolation test, plus production manifest comparison).
3. Every GI quote on a translated episode resolves to source text and source audio.
4. `adfree.ad_chars_removed` on translated episodes is non-zero at a rate comparable to English
   episodes of similar shows, or the §7 survival number explains why not.

## Relationship to Other RFCs

- **RFC-123** supplies the units, and it ships first on its own merits.
- **RFC-125** consumes `translation.json` and adds per-claim translation provenance and source
  verification, calling this RFC's `resolve_units_for_span`. QE scores and calibrated bands are its
  v2 addition, not a v1 dependency.
- **Positions** are a read-time CIL query (`position_arc`), not a stage. Nothing in the position
  path changes here; RFC-125 gates what that query returns.

## Open Questions

1. Code-switching: when a unit's language ID disagrees strongly with the declared language (an
   English passage in a Serbian show), should that unit be passed through untranslated? This is
   cheap to detect per unit after the fact.
2. Serbian source display: normalize to Latin at render time, or store normalized? Proposed:
   render-time only, because stored text stays exactly as transcribed.
3. The GPU budget per translated episode is unknown until the bake-off. Given that `relabel_only`
   already costs ~17 min/episode on the DGX and runs near-serial, a translation wall-time cap
   (for example ≤ 50% of transcription wall time) as a gate criterion?
4. Should English episodes eventually route through the same resolver branch to pick a cleaned
   variant, unifying this with `save_cleaned_transcript`? The resolver now generalizes, so this is a
   config question rather than an architectural one.
5. Do we keep a source-language ad-free variant at all, derived by mapping the English ad ranges
   back through `translation.json`? It has no analysis consumer, only a reader/player one.

## References

- `src/podcast_scraper/providers/tailnet_dgx/whisper_provider.py:197` — `language or "en"` default
- `src/podcast_scraper/providers/ml/whisper_utils.py` — `normalize_whisper_model_name`
- `src/podcast_scraper/providers/ml/ml_provider.py:885` — `self.cfg.language or "en"`
- `src/podcast_scraper/workflow/episode_processor.py:2296` — the single `language=` call site
- `src/podcast_scraper/workflow/metadata_generation.py:2843` — `"language": cfg.language`
- `src/podcast_scraper/workflow/adfree_transcript.py` — `load_processing_transcript`, `adfree_transcript_relpath`, `build_adfree_artifacts`
- `src/podcast_scraper/gi/filters.py` — `_AD_PATTERNS` (English)
- `src/podcast_scraper/gi/ad_regions.py` — `excise_ad_regions`, `excise_ad_regions_with_offsets`
- `src/podcast_scraper/speaker_detectors/ner.py:133` — en-only NER gate
- `src/podcast_scraper/server/corpus_catalog.py:159` — `_feed_language` (no normalization)
- `src/podcast_scraper/providers/ml/diarization/formatting.py` — screenplay + offsets
- `src/podcast_scraper/gi/contracts.py` — `EvidenceSpan`, `SupportingQuote`
- `src/podcast_scraper/server/schemas.py:24` — `SegmentsResponse`
- `src/podcast_scraper/server/routes/app_episodes.py:463` — the segments route
- `src/podcast_scraper/prompting/megabundle.py` — transcript prefix caching
- `config/profiles/prod_dgx_full.yaml:70,111` — `language`, `transcription_fallback_providers`
- `config/corpus-expansion.feeds.yaml` — bare `- url:` feed entries
