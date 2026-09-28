# PRD-047: Multilingual Ingest (source-language capture, English-normalized intelligence)

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: post-beta (gated on Phase 0 — see §Phasing)
- **Related RFCs**:
  - [RFC-123](../rfc/RFC-123-speaker-turns-artifact.md) — speaker turns as a first-class artifact (prerequisite; ships independently)
  - [RFC-124](../rfc/RFC-124-multilingual-transcription-and-translation.md) — language routing, source-language transcription, translation stage
  - [RFC-125](../rfc/RFC-125-translation-confidence-and-claim-verification.md) — quality estimation, confidence propagation, claim verification
  - [RFC-005](../rfc/RFC-005-whisper-integration.md), [RFC-058](../rfc/RFC-058-audio-speaker-diarization.md), [RFC-106](../rfc/RFC-106-tiered-dgx-service-fallback.md), [RFC-109](../rfc/RFC-109-per-episode-observability-manifest.md)
- **Related ADRs**: [ADR-155](../adr/ADR-155-pin-every-model-checkpoint.md) (pin every model checkpoint), [ADR-108](../adr/ADR-108-nli-disagreement-enrichers-gated-dark.md) (stance-over-time is a read-time query)
- **Related PRDs**: [PRD-028](PRD-028-position-tracker.md) (Position Tracker — the surface trust gates), [PRD-036](PRD-036-foundation-identity.md) (player `segments.json` contract), [PRD-039](PRD-039-player.md) (player), [PRD-044](PRD-044-operator-shows-library.md) (operator shows library)
- **Arc notes**: [MULTILINGUAL_ARC](../architecture/MULTILINGUAL_ARC.md) — arc shape, the slice plan (§4), decisions and running notes
- **Scope**: pipeline (ingest → transcript → translation → ad-free base → existing intelligence layers), player transcript/quote surfaces, language visibility and filtering, operator feed config. No non-English UI.

---

## Summary

Ingest podcasts in languages other than English. Each episode is transcribed and diarized **in its
source language**, which stays the canonical record. A translation stage then produces an **English
derived transcript**, aligned turn by turn with the original. The existing ad-removal step then runs
on that English text, and its output is what every existing layer (summary, KG, GIL, CIL, search,
Positions) reads — without modification. The listener hears the original audio, can read the
original transcript, and can switch to English subtitles. Every claim that rests on translated text
is marked as such, records the source sentences it rests on, and — where it could enter a Position
timeline — is verified against the source first.

**The first deliverable is not translation.** It is making language an explicit, resolved, validated
and *visible* property of the corpus we already have: every show and episode carries a language
badge, an audit proves the existing corpus is English, and no code path can silently substitute
English for a language it was not given. That work is a correctness fix on today's corpus, it ships
on its own, and it is exactly the plumbing translation needs.

The product bet is two-sided. For listeners, the hypothesis is that internationally-minded people
who consume English podcasts also keep a native-language minority in their diet. For the corpus,
native-language shows add **source divergence across language communities**, meaning how a Greek or
Serbian show frames the same question as an English one. English-only tools structurally cannot see
this, and it compounds the cross-show-synthesis moat.

## Background & Context

- **Listener behavior (hypothesis, anecdotal).** Two informal data points: a Greek listener at
  roughly 70–80% English / 20–30% Greek, and Serbian listeners at roughly 80–90% English /
  10–20% Serbian. This is not evidence of demand yet. Phase 0 exists to test it with the beta cohort.
- **The pipeline is English-pinned by configuration, not by architecture.** `language: en` is a
  single global setting (`config/profiles/prod_dgx_full.yaml:70`). The DGX Whisper provider defaults
  to `"en"` per request. `speaker_detectors/ner.py:133` gates NER on `cfg.language == "en"`. The DGX
  transcriber (`deepdml/faster-whisper-large-v3-turbo-ct2`) and pyannote community-1 are
  multilingual already. The feed `<language>` tag is already read
  (`server/corpus_catalog.py:159`), though not normalized.
- **Everything downstream of the transcript is English-shaped and should stay that way.** GIL
  grounding requires verbatim substrings with char offsets (`gi/grounding.py`, `EvidenceSpan`).
  Prompts, QA/NLI checks, embeddings (MiniLM) and spaCy NER are all English. Translating once to
  English is far cheaper and safer than making every layer multilingual.
- **Ad removal is English-keyed, and that is not cosmetic.** The ad detector matches English regexes
  (`gi/filters.py` `_AD_PATTERNS`: "brought to you by", "sponsored by", "… dot com slash"). On a
  non-English transcript it matches nothing, and the ad-free base — the text all analysis reads — is
  silently produced as an identity copy with the sponsor reads still in it. RFC-124 therefore
  translates **before** removing ads, which fixes this and costs one translation pass instead of two.
- **Grounded objectivization raises the bar.** A mistranslated negation or hedge can invert a
  position, and the time axis (position change over 12–18 months) is the primary signal in this
  corpus. A false position flip caused by translation is the worst failure this feature can produce.
  Trust machinery (RFC-125) is therefore part of the feature, not a later polish item.
- **Positions are already a read-time query, not a pipeline stage.** `position_arc`
  (`server/cil_queries.py:636`) builds the per-(person, topic) arc at request time from GI insights;
  ADR-108's 2026-07-08 update retired the `stance_timeline` enricher in favour of exactly this. That
  is load-bearing for this PRD in two ways: the trust gate is a read-path filter (so it is
  retroactive and fail-closed), and **this feature has no dependency on shipping a stance-extraction
  stage**, because there is none to ship.
- **Why not now.** Beta onboarding and the corpus rebuild come first. This PRD is written now so the
  beta interviews can test the hypothesis and so RFC-123, which has standalone value, can be
  scheduled on its own merits.

## Goals

- **G1.** Ingest, transcribe and diarize episodes in any **enabled** language, with the
  source-language transcript as the canonical artifact.
- **G2.** Produce a complete, turn-aligned English transcript per episode that doubles as subtitles.
- **G3.** Run every existing intelligence layer on non-English episodes with no per-language forks
  in those layers.
- **G4.** Never present translated text as verbatim speech. Every translated quote, insight and
  position is labeled, records the source units it rests on, and traces back to the original audio
  and text.
- **G5.** Claim language support only where a measured quality gate passes, per language.
- **G6.** Remove ads from non-English episodes as effectively as from English ones, rather than
  appearing to.
- **G7.** Make the language of every show and episode **visible and filterable** in the product, so a
  listener can see what language something is in and find things by it.
- **G8.** Make English an explicitly declared, checked language before any second language exists, so
  the multilingual path is proven end to end on a corpus we can already verify.

## Non-Goals

- Translating the product UI, or translating English content into other languages.
- Dubbing, voice cloning, or TTS of translations.
- Real-time or streaming translation.
- Automatic language detection as the primary routing signal. Detection is a sanity check only
  (RFC-124 §1).
- Supporting languages below the quality gate, even when Whisper accepts them.
- Multilingual retrieval (querying in Greek). Search indexes the English layer in v1.
- Code-switching within an episode (e.g. Serbian with English passages) beyond "transcribe as the
  declared language". Tracked as an open question.
- A writable per-show override in the operator UI. The override is config (see FR1.1); a UI editor
  needs a feed-record store this feature does not justify building.
- Per-unit quality-estimation bands in v1. Calibrating them needs a native reviewer per language and
  a translated corpus that does not exist until the pipeline has run, so QE is a v2 addition
  (RFC-125 §7). v1's trust mechanism is source verification.

## Personas

- **Bilingual listener** (the hypothesis user): mostly English podcasts plus a few native-language
  shows.
  - Needs: native shows in the same library, with the same intelligence as the English ones.
  - Gets: one corpus. A Greek guest's position on EU policy sits on the same Position timeline as an
    English host's.
- **Curious non-speaker**: follows a topic, and a relevant show exists only in another language.
  - Needs: to understand what was said, and how far to trust the understanding.
  - Gets: English subtitles and insights, clearly marked as machine-translated, with confidence
    shown where it is low.
- **Operator** (Marko): curates the corpus and decides which languages are live.
  - Needs: per-language quality evidence, per-episode translation health, and a review queue for
    claims that failed verification.

## User Stories

- _As a bilingual listener, I can add a Greek podcast to my library and see its episodes processed
  like any other show._
- _As a listener, I can read the transcript in the original language or switch to English
  subtitles while the original audio plays._
- _As a listener, I can tap a translated quote and hear the original speaker saying it, and see the
  original sentence next to the translation._
- _As a listener, I can tell at a glance which quotes and positions come from a translation, and
  which are low-confidence._
- _As an operator, I can set a show's language when the RSS tag is wrong or missing._
- _As an operator, I can see which languages pass the quality gate and why, and enable only those._
- _As an operator, I can review claims that failed source verification before they reach a
  Position._
- _As a listener, I can see at a glance what language a show or an episode is in, without opening it._
- _As a listener, I can filter shows and episodes by language, the same way I filter by whether I have
  played or downloaded them._

## Functional Requirements

### FR1: Language declaration and routing

- **FR1.1**: Each show resolves a language in this order: operator override > normalized RSS
  `<language>` (`el-GR` → `el`) > profile default (`en`). The override lives on the **feed entry in
  config** (`config/corpus-expansion.feeds.yaml`, extended to accept an optional mapping per feed).
  The shows library *displays* the resolved language and its source; it does not edit it. Rationale:
  RFC-104 states the shows library has no backend, and no feed record store exists to hold an
  override.
- **FR1.2**: The resolved language is recorded on every episode's metadata and observability
  manifest. Note that episode metadata already writes `"language": cfg.language`
  (`workflow/metadata_generation.py:2843`) from the run-global config — so until FR1.1 and FR2.1 land
  together, that field actively asserts `en` for a non-English episode. Correcting it is part of this
  requirement, not a follow-up.
- **FR1.3**: A supported-language registry (`config/languages.yaml`) lists each language's tier and
  an `enabled` flag. Episodes in a language that is not enabled are skipped with explicit status
  `skipped_unsupported_language`. They are **never** transcribed as English.
- **FR1.4**: A language-ID sanity check samples audio away from the intro. On a mismatch with the
  declared language, the episode is flagged for the operator. The check never silently reroutes.
- **FR1.5**: The existing corpus is **audited** before any second language is enabled: every feed and
  every episode resolves to a language, and anything that does not resolve to `en` is reported
  explicitly. The expected result is that the whole current corpus is English; the audit is what turns
  that expectation into evidence. It is read-only and re-runnable.
- **FR1.6**: The resolved language is **carried as an explicit parameter** through transcription,
  diarization, metadata writing and the observability manifest. No stage may substitute a default
  language for one it was not given, and the transcription model is selected *from* the resolved
  language (see FR2.1).

### FR2: Source-language capture (canonical)

- **FR2.1**: Transcription and diarization run in the **episode's resolved** language, threaded to
  the provider call rather than read from run-global config. A tier that cannot honor the language —
  an English-only `.en` Whisper model, or any model below the non-English quality floor — fails the
  attempt rather than producing output. No provider may substitute `"en"` for a missing language.
- **FR2.2**: The source-language transcript and segments are persisted at the existing artifact
  paths, carry a `language` field, and are the canonical record for the episode. They keep the
  **full** timeline (ads included), as English episodes' raw transcripts do today.
- **FR2.3**: The player can play the original audio and display the source-language transcript for
  any non-English episode, even while translation is pending or has failed.

### FR3: English translation

- **FR3.1**: Every translation unit is translated. Units are sentence groups within a single speaker
  turn (RFC-123 turns; RFC-124 §5.1). Coverage is complete, not selective. The full timeline is
  translated exactly once per episode.
- **FR3.2**: The English transcript is aligned unit by unit to source timestamps and speakers, and
  is served as subtitles.
- **FR3.3**: English artifacts are marked `derived: true` and `translated_from: <lang>`, and record
  the translation model and pinned revision (ADR-155).
- **FR3.4**: If translation fails or any unit fails, the episode shows status `translation_pending`
  and the English ad-free base is not built, so analysis layers cannot run on a partial translation.
  The source transcript and partial subtitles stay servable.

### FR4: Ad removal and intelligence on non-English episodes

- **FR4.1**: Ad detection and excision run on the **English** transcript, producing the ad-free
  English base that analysis reads. A non-English episode must not receive an identity ad-free base
  that misrepresents itself as ad-free. `ad_chars_removed` and the variant the base was built on are
  recorded per episode.
- **FR4.2**: Summary, KG, GIL, CIL and search run on that base with no per-language code paths, via
  the resolver they already share.
- **FR4.3**: Every quote from a translated episode is labeled as a translation, shows the original
  sentence on demand, and plays the **original** audio span.
- **FR4.4**: People resolve to a single CIL identity across languages and scripts. A canonical Latin
  name is kept, with the source-script name as an alias. A transliterated speaker label that would
  break existing attribution heuristics falls back to the source-script label rather than
  silently un-attributing the turn.

### FR5: Trust and confidence

- **FR5.1**: Every claim derived from translated text records its **translation provenance**: that it
  is translated, the source language, and the specific source units it rests on — so any translated
  quote can show the original sentence and play the original audio.
- **FR5.2**: A translated claim that has not been verified is **labelled but not silently trusted**:
  it renders with a translation marker, its source is one tap away, and it never enters a Position
  timeline. (Per-unit quality bands that would let us rank *how* unsure we are on the unverified
  remainder are a v2 addition — RFC-125 §7.)
- **FR5.3**: A translated, position-bearing insight appears in a Position timeline only when it has
  a `verified` verification outcome. This is enforced as a filter in the read-time `position_arc` /
  `topic_conversation_arc` query, which means it is retroactive and **fail-closed**: a translated
  claim with no verification record is absent from timelines by default, with no feature flag to
  remember. A contradicted claim is excluded and queued for review. An unverifiable claim stays
  visible on the episode, marked unverified.
- **FR5.4**: The player shows a "Translated from <Language>" marker on translated content. v1 adds no
  further per-claim confidence decoration, because without calibrated bands there is nothing honest to
  grade; the amber/red marker arrives with v2.

### FR6: Operator surfaces

- **FR6.1**: The shows library displays each show's resolved language and the source of that
  resolution (override / RSS / default). Editing is via config (FR1.1).
- **FR6.2**: The per-episode manifest (RFC-109) records language, translation model, unit count, the
  QE distribution, flagged-unit count, verification outcomes, and which transcript variant the
  ad-free base was built on.
- **FR6.3**: A review worklist lists contradicted and unverified claims, each with source and
  translation side by side.

### FR7: Language visibility and filtering

- **FR7.1**: Every **show** and every **episode** displays a compact language badge — a small squared
  chip carrying the uppercase ISO code (`EN`, `EL`, `SR`) — wherever that item's metadata renders:
  the consumer episode rows, tiles and cards, the show rows, tiles and show detail page, and the
  operator shows library. The badge is omitted, not guessed, when the language is unknown.
- **FR7.2**: Episode language is exposed by the app API. Show language already is
  (`AppPodcastItem.language`), and it is served **normalized** (`en-US` → `en`) so the badge shows one
  token per language rather than one per feed's spelling of it.
- **FR7.3**: Shows and episodes can be **filtered by language** on the surfaces that already offer
  filters — the consumer episode toolbar (which today filters all / unplayed / played / insights /
  downloaded and selects a show), show browse, and the operator library filter bar.
- **FR7.4**: The language filter is its **own** control, not another option inside the
  played/downloaded filter, because the two dimensions are orthogonal: "Greek **and** unplayed" must
  be expressible.
- **FR7.5**: The language filter is shown only when the corpus holds more than one language. A filter
  with a single value is a dead control, so it appears with the second language rather than shipping
  inert. The badge (FR7.1) has no such condition — it is informative even when everything is English,
  and that is how a listener sees the audit's result.

## Supported-language policy

A language goes live only after it passes the RFC-124 §7 bake-off gate: acceptable transcription
quality on real episodes of that language, acceptable translation quality measured by a native
reviewer, position agreement between translation and source at or above the threshold, and a
measured ad-detection survival rate. The ASR tiers in Appendix A are a **prior** for deciding what to
test first. They are not a support claim.

External wording for beta conversations: "Western European languages plus Russian, Polish,
Japanese and Korean are the first candidates; Balkan and Nordic languages are in trial; South Asian
languages are not supported yet."

## Phasing

Phase numbering means one thing across this PRD and RFC-123/124/125. The demand and bake-off
validation step is **Gate V**, not a phase, because it produces evidence rather than software. The
per-slice breakdown — each slice sized as one issue, with dependencies and acceptance criteria — lives
in [MULTILINGUAL_ARC §4](../architecture/MULTILINGUAL_ARC.md#4-slice-plan).

- **Phase 0: English as a declared language.** **Ships on its own, before Gate V.** Tag
  normalization and per-episode resolution; the corpus audit (FR1.5); episode language on the API and
  the `EN` badge on every show and episode (FR7.1–FR7.2); the resolved language threaded as an
  explicit parameter with every silent `"en"` substitution removed (FR1.6); transcription-model
  selection driven by the resolved language, with a non-English quality floor (FR2.1). This is a
  correctness fix on the corpus that exists — no new models, no GPU — and it is what makes the
  multilingual path provable end to end before any translation happens.
- **Phase 1: Turns (RFC-123).** Independent of everything multilingual; improves quote attribution,
  search chunking and player cues for English episodes today. Can run in parallel with Phase 0.
- **Gate V: Validate (no build).** Ask the beta cohort what share of their listening is non-English
  and which shows. Confirm candidate translation-model availability and licence terms (unverified
  today). Run the RFC-124 bake-off on 2–3 episodes each of Greek and Serbian (hardest likely-demanded
  pair) and one of Spanish or Italian (easy control). Requires Phase 0, because the bake-off has to
  transcribe non-English audio correctly to measure anything. **Decision gate:** proceed only if the
  demand signal and the quality gate both pass.
- **Phase 2: Translation (RFC-124)** behind a feature flag, on one or two operator-chosen feeds.
- **Phase 3: Trust (RFC-125).** Source verification and the read-time Positions gate. Required before
  any translated position enters a timeline — though note FR5.3 is fail-closed, so the absence of this
  phase already withholds them.
- **Phase 4: Surfaces.** Language toggle, translated-quote treatment, and the language filters
  (FR7.3–FR7.5). UXS to follow for the transcript toggle and quote treatment.
- **v2:** per-unit quality estimation and calibrated bands (RFC-125 §7).

## Success Metrics

- **Phase 0:** the audit reports 100% of existing shows and episodes as `en` with a named resolution
  source; every show and episode renders a language badge; no provider can receive a null language and
  substitute English; and the English corpus is byte-identical to before the phase.
- **Demand (Gate V):** at least 30% of the beta cohort names one or more non-English shows they
  would add to the product.
- **Quality gate (per language):** at least 90% of position-bearing units judged meaning-preserving
  by a native reviewer, and at least 90% agreement between positions extracted from the translation
  and the reviewer's reading of the original.
- **Trust:** zero translated quotes rendered without a translation label, and zero translated
  positions in timelines without a verified outcome. At least 90% of reviewer-labeled position
  inversions caught on the bake-off eval set.
- **Findability (Phase 4):** a listener can reach every non-English show in the corpus using the
  language filter alone.
- **Coverage:** at least 95% of episodes in enabled languages reach English analysis without manual
  retry.
- **Ad removal:** `ad_chars_removed` on translated episodes is non-zero at a rate comparable to
  English episodes from similar shows.
- **Isolation:** no measurable change in English-episode pipeline time or outputs when the feature
  is enabled, including byte-identical `position_arc` responses on the English corpus.

## Dependencies

- RFC-123 turns artifact (unit boundaries for translation, and the citation unit for confidence).
- DGX capacity for one additional served model (translation), plus a QE model. Note the existing
  one-vLLM-at-a-time convention on the DGX: this is a capacity question to settle in Phase 0, not an
  assumption.
- A native-speaker reviewer per candidate language **for the bake-off** (Gate V) and for the monthly
  verification spot-check. Cutting QE from v1 removes the reviewer from the *pipeline's* critical path;
  it does not remove them from the decision to enable a language.
- Candidate translation models confirmed available under a licence that permits this deployment.
- Phase 0 shipped, before Gate V can measure anything.

**Not dependencies (corrected):**

- **A stance-extraction stage.** Positions are a read-time CIL query over GI insights (ADR-108,
  2026-07-08), so there is no extraction stage to ship first and Phase 3 is not blocked on one.
- **A QE model or its calibration.** Deferred to v2 (RFC-125 §7). v1's trust mechanism is source
  verification, which needs no per-language threshold fitting.

## Risks

- **Translation inverts meaning** (negation, hedging, sarcasm). Mitigated by RFC-125 verification;
  residual risk is accepted only for non-position content with an amber or red marker.
- **Sponsor reads enter the analysis text.** The cause is English-only ad patterns; mitigated by
  translating before ad removal (FR4.1) and measured by the bake-off's ad-survival metric. If
  survival is low, per-language ad cues come back on the table.
- **Episode metadata asserts the wrong language** until FR1.2/FR2.1 land together. Mitigated by
  shipping them in the same phase (1b) and by a regression test.
- **Identity fragmentation across scripts.** Mitigated by FR4.4 and CIL alias rules (RFC-124 §5.4).
- **Demand does not materialize.** Phase 0 gate; RFC-123 still pays for itself.
- **Model licensing.** Some leading MT models exclude EU use or forbid commercial use. The bake-off
  shortlist only includes models whose license permits this deployment, and that check is a Phase 0
  task rather than a settled fact.
- **GPU contention on the DGX.** Translation adds a served model. It is scheduled in batches, and
  English ingest keeps priority.

## Open Questions

1. How is code-switching handled (English passages inside a Serbian episode)? Transcribe as
   declared and accept the damage, or segment-level language ID?
2. Serbian script: normalize output to Latin, keep Cyrillic, or follow the feed? This affects the
   display, not the English layer.
3. Should verified translated positions be visually distinguishable from native English ones in
   Position timelines, or is verification enough to treat them as equals?
4. Do we ever show amber-band quotes in shareable quote cards?
5. Do we keep a source-language ad-free transcript at all, derived by mapping the English ad ranges
   back to source offsets? It has a reader/player use, not an analysis one.

## Appendix A: ASR prior by tier (Whisper large-v3, FLEURS; turbo tracks closely)

Real podcast error rates are higher than these benchmark tiers because of crosstalk, music and
informal speech.

- **Excellent (<~6% WER):** Spanish, Italian, Korean, Portuguese, English, Polish, Catalan,
  Japanese, German, Russian, Dutch, French.
- **Good (6–10%):** Indonesian, Ukrainian, Turkish, Malay, Swedish, Mandarin, Finnish*, Norwegian,
  Romanian, Vietnamese*, Slovak, Arabic (MSA; dialects are worse), Thai (turbo weaker).
- **Usable with review (10–15%):** Czech, Croatian, Greek, Serbian, Danish, Bulgarian, Hungarian,
  Filipino, Bosnian, Galician, Macedonian.
- **Not ready (>15%):** Hindi, Estonian, Slovenian, Tamil, Latvian, Azerbaijani, Urdu, Lithuanian,
  Hebrew, Welsh, Persian, Icelandic, Kazakh, Afrikaans, Kannada, Marathi, Swahili, Telugu, Maori,
  Nepali, Armenian, Belarusian, Gujarati, Punjabi, Bengali.

\* Finnish and Vietnamese score much worse on Common Voice (noisier audio). Treat them as
borderline.
