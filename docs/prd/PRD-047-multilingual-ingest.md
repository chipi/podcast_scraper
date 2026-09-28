# PRD-047: Multilingual Ingest (source-language capture, English-normalized intelligence)

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: post-beta (gated on Phase 0 — see §Phasing)
- **Related RFCs**:
  - [RFC-123](../rfc/RFC-123-speaker-turns-artifact.md) — speaker turns as a first-class artifact (prerequisite; ships independently)
  - [RFC-124](../rfc/RFC-124-multilingual-transcription-and-translation.md) — language routing, source-language transcription, translation stage
  - [RFC-125](../rfc/RFC-125-translation-confidence-and-claim-verification.md) — translation provenance (v1); source verification, labelling, the read-time gate and quality estimation (all v2)
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
Positions) reads. Those layers need no per-language logic — but getting them to read one agreed
transcript is real work, because six of them resolve the transcript independently today. The listener
hears the original audio and reads the transcript in English by default, with a control to switch to
the original. Every claim that rests on translated text **records which source sentences it rests
on** — invisible to the user in v1, and the foundation v2's labelling, verification and gating are
built on. v1 otherwise treats a translated episode exactly like an English one.

**The first deliverable is not translation.** It is making language an explicit, resolved and validated
property of the corpus we already have: the feed's declared language is actually parsed and persisted, an
audit reports what the corpus is really in, and no code path can silently substitute English for a
language it was not given. That work is a correctness fix on today's corpus, it ships on its own, and it
is exactly the plumbing translation needs. The visible chrome — a badge and a language filter — follows
in v2, when there is a second language to distinguish.

The product bet is two-sided. For listeners, the hypothesis is that internationally-minded people
who consume English podcasts also keep a native-language minority in their diet. For the corpus,
native-language shows add **source divergence across language communities** — how a Spanish or German
show frames the same question as an English one. English-only tools structurally cannot see this, and
it compounds the cross-show-synthesis moat.

## Background & Context

- **Listener behavior (hypothesis, anecdotal).** Informal data points suggest internationally-minded
  listeners run roughly 70–90% English with a native-language remainder. This is not evidence of
  demand yet. **Gate V** exists to test it with the beta cohort.
- **The pipeline is English-pinned by configuration, not by architecture.** `language: en` is a
  single global setting (`config/profiles/prod_dgx_full.yaml:70`). The DGX transcriber
  (`deepdml/faster-whisper-large-v3-turbo-ct2`) and pyannote community-1 are multilingual already, so
  transcribing another language is close to a parameter change.
- **But the feed's declared language is never read.** There is no `<language>` extraction anywhere in
  `rss/`, and `RssFeed` has no such field. What looks like a feed language is the run config written
  back out: `metadata_generation.py:957` sets `FeedMetadata(language=cfg.language)`, so
  `server/corpus_catalog.py:159` reads the profile's `en` and `AppPodcastItem.language` serves it.
  There is no episode-level language field at all. Parsing and persisting the real tag — with a
  per-show backfill migration — is therefore the **first** piece of work, not a lift (FR1.1).
- **Everything downstream of the transcript is English-shaped and should stay that way.** GIL
  grounding requires verbatim substrings with char offsets (`gi/grounding.py`, `EvidenceSpan`).
  Prompts, QA/NLI checks, embeddings (MiniLM) and spaCy NER are all English. Translating once to
  English is far cheaper and safer than making every layer multilingual.
- **Ad removal is English-keyed, and that is not cosmetic.** The ad detector matches English regexes
  (`gi/filters.py` `_AD_PATTERNS`: "brought to you by", "sponsored by", "… dot com slash"). On a
  non-English transcript it matches nothing, and the ad-free base — the text all analysis reads — is
  silently produced as an identity copy with the sponsor reads still in it. RFC-124 therefore
  translates **before** removing ads, which fixes this and costs one translation pass instead of two.
- **Grounded objectivization raises the bar, and v1 accepts a known risk.** A mistranslated negation
  or hedge can invert a position, and the time axis is the primary signal in this corpus, so a false
  position flip is the worst failure this feature can produce. v1's mitigation is **model choice**,
  measured at Gate V — not withholding output. The trust machinery that verifies, labels and gates
  individual claims is v2, and v1 records the provenance it will need.
- **Positions are already a read-time query, not a pipeline stage.** `position_arc`
  (`server/cil_queries.py:636`) builds the per-(person, topic) arc at request time from GI insights;
  ADR-108's 2026-07-08 update retired the `stance_timeline` enricher in favour of exactly this. That
  is load-bearing for this PRD in two ways: a future gate is a read-path filter, so **v2 can add it
  retroactively over the whole corpus** with no re-extraction — which is why deferring it costs
  nothing — and **this feature has no dependency on shipping a stance-extraction stage**, because
  there is none to ship.
- **Why not now.** Beta onboarding and the corpus rebuild come first. This PRD is written now so the
  beta interviews can test the hypothesis and so RFC-123, which has standalone value, can be
  scheduled on its own merits.

## Goals

- **G1.** Ingest, transcribe and diarize episodes in any **enabled** language, with the
  source-language transcript as the canonical artifact.
- **G2.** Produce a complete, turn-aligned English transcript per episode that doubles as subtitles.
- **G3.** Run every existing intelligence layer on non-English episodes with no per-language forks
  in those layers.
- **G4.** Record, for every claim derived from translated text, exactly which source units it rests
  on — so a translated quote can be traced to the original sentence and the original audio, and so
  labelling, verification and gating can all be added in v2 without reprocessing. **v1 records this
  and shows the user nothing**; distinguishing translated from native content in the UI is a v2 goal.
- **G5.** Claim language support only where a measured quality gate passes, per language.
- **G6.** Remove ads from non-English episodes as effectively as from English ones, rather than
  appearing to.
- **G7.** Make the language of every show and episode **knowable** — parsed, persisted and exposed on
  the API — so that seeing and filtering by it (v2) is a rendering question rather than a data one.
- **G8.** Make English an explicitly declared, checked language before any second language exists, so
  the multilingual path is proven end to end on a corpus we can already verify.

## Non-Goals

- Translating the product UI, or translating English content into other languages.
- Dubbing, voice cloning, or TTS of translations.
- Real-time or streaming translation.
- Automatic language detection as the primary routing signal. Detection is a sanity check only
  (RFC-124 §1).
- Supporting languages below the quality gate, even when Whisper accepts them.
- Code-switching within an episode (e.g. Serbian with English passages) beyond "transcribe as the
  declared language". Tracked as an open question.
- A writable per-show override in the operator UI. The override is config (see FR1.2); a UI editor
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

- **FR1.1**: The feed's declared `<language>` is **parsed from the RSS channel and persisted** — it is
  not read today at all. Each feed records `language_raw`, the normalized `language`, and
  `language_source`; each episode records a `language` and `language_source` of its own. The existing
  corpus is backfilled by a one-off migration that works **per show**: fetch each feed once, read its
  `<language>`, and write it onto that show and every episode under it. The language is a property of the
  feed, so one fetch covers all of its episodes. It runs as a versioned, re-runnable migration with a
  dry-run that prints the distribution first, and it reports per show so the output doubles as FR1.6's
  first data. A feed whose URL is missing from the metadata is reported and skipped, never guessed.
- **FR1.2**: Each show resolves a language in this order: **operator override > the feed's normalized
  declared language (`el-GR` → `el`) > profile default (`en`)**. The profile default is the only place
  the run-global setting may be read. The override is a key on the feed entry in the feeds spec, which
  already accepts a per-entry mapping — so this is an allowlist addition, not a new config surface.
  The shows library *displays* the resolved language and its source; editing stays in config, because
  RFC-104's shows library has no backend.
- **FR1.3**: The resolved language is recorded on every episode's metadata and observability manifest,
  and is **carried as an explicit parameter** through transcription, diarization, metadata writing and
  the manifest. No stage may substitute a default language for one it was not given. This is enforced
  as an invariant — the run-global setting has exactly one reader — rather than as a checklist of
  known call sites, because that list was already incomplete.
- **FR1.4**: A supported-language registry (`config/languages.yaml`) lists each language's tier and an
  `enabled` flag. Episodes in a language that is not enabled are skipped with explicit status
  `skipped_unsupported_language`, and are **never** transcribed as English. **The skip must not ship
  before the override in FR1.2**: declared feed tags are routinely wrong, and an English show tagged
  `de` would otherwise stop ingesting with no remedy available.
- **FR1.5**: A language-ID sanity check samples audio away from the intro. On a mismatch with the
  declared language, the episode is flagged for the operator. The check never silently reroutes.
- **FR1.6**: The existing corpus is **audited over real data** before any second language is enabled:
  every feed and episode resolves to a language, with its resolution source named, and anything not
  resolving to `en` is reported explicitly. The audit runs *after* FR1.1 has persisted real feed tags —
  run against today's metadata it would only re-read the run config and report a guaranteed 100% `en`,
  which proves nothing. Read-only and re-runnable.
- **FR1.7**: Every new failure mode this section introduces — `skipped_unsupported_language`,
  `deferred_quality_floor`, `language_mismatch` — is **visible** to the operator in logs, metrics and a
  runbook entry. A phase whose purpose is removing silent failures must not add three.

### FR2: Source-language capture (canonical)

- **FR2.1**: Transcription and diarization run in the **episode's resolved** language, threaded to
  the provider call rather than read from run-global config. A tier that cannot honor the language —
  an English-only `.en` Whisper model, or any model below the non-English quality floor — fails the
  attempt rather than producing output. No provider may substitute a language it was not given, and
  none may **report** a language it did not actually transcribe in: the DGX provider today omits the
  language from its request when it is absent and then reports `"en"` for whatever the server
  auto-detected, which is a provenance error rather than a transcription one.
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
- **FR4.2**: Summary, KG, GIL, CIL and search run on that base with no per-language code paths. They
  must all resolve the transcript through **one** shared resolver — which they do not today: two call
  the shared one and six resolve independently, one of them reading the raw transcript. Routing them is
  part of this requirement.
- **FR4.3**: A quote from a translated episode **plays the original audio span** — the source times
  and the English cue times are identical by construction, so this falls out of the artifact model
  rather than needing UI work. Showing the original sentence beside the translation, and marking the
  quote as translated at all, are v2 (FR5.2).
- **FR4.4**: Speaker labels are **never sent through the translation model**, which would rename the
  same person inconsistently between units. The label is carried onto the English transcript verbatim.
  **No transliteration and no alias minting in v1**: every tier-1 language is Latin script and a person's
  name is usually the identical string across those languages, so there is nothing to convert. Resolving
  one identity across *scripts* — canonical name, transliteration, source-script aliases — is deferred
  until the first non-Latin-script language is enabled.
- **FR4.4b** *(deferred with the above)*: a transliterated speaker label that would
  break existing attribution heuristics falls back to the source-script label rather than
  silently un-attributing the turn.
- **FR4.5**: **A non-English episode is findable by a query in its own language, and by an English
  query, and both reach the same episode.** Search indexes both the source-language transcript and the
  English analysis transcript, each chunk tagged with its language, so the corpus is searchable in the
  language it was spoken in rather than only in translation. A result says which layer matched.
  **No embedding model changes**: non-English chunks are keyword-matched only. Matching *by meaning*
  across languages — an English query finding Greek content it shares no words with — is explicitly out
  of scope; it needs a multilingual encoder, a corpus-wide re-embed and a migration, and it belongs with
  internationalizing the application, when a listener can search in whatever language they choose.

### FR5: Trust and confidence

**v1 believes the translation.** A translated episode has the same standing as a native-English one on
every surface: nothing is gated, nothing is filtered, and nothing in the UI marks it as translated. That
is deliberate — the listener sees one simple thing with no doubt attached, and translation *quality* is
what v2 is about rather than something v1 hedges around by withholding output. The bet is mitigated by
choosing the model on measured evidence at Gate V, not by hiding its results.

The honest consequence, stated rather than buried: **in v1 a listener cannot tell a translated quote from
a native-English one.** That matters most for quotes, because a quote can be passed on as somebody's
words — which is exactly why the label is the first thing v2 adds after verification.

- **FR5.1**: Every claim derived from translated text records its **translation provenance** — that it is
  translated, the source language, and the specific source units it rests on. This is written on every
  quote and insight and is **invisible to the user**. It exists so that verification, labelling and
  gating can all be added in v2 **without reprocessing the corpus**.
- **FR5.2** *(v2)*: A user-visible "Translated from <Language>" marker wherever translated content
  renders, with the original sentence reachable on demand.
- **FR5.3** *(v2)*: Source verification — give a model the source-language text a claim rests on and the
  English claim, and record whether the source supports it.
- **FR5.4** *(v2)*: A read-time gate keyed on that outcome, across every surface that attributes a claim
  to a person on a topic. RFC-125 §3 enumerates them. Whether it is fail-closed everywhere or split
  between surfaces that *assert* a position and those that merely *describe* is a judgement to make when
  the verifier exists — a gate with nothing able to release what it holds would simply hide translated
  output permanently, which is why it is not in v1.
- **FR5.5** *(v2)*: Per-unit quality estimation with calibrated bands, so the confidence of an individual
  claim can be ranked rather than treated as binary.

### FR6: Operator surfaces

- **FR6.1**: The shows library displays each show's resolved language and the source of that
  resolution (override / RSS / default). Editing is via config (FR1.2).
- **FR6.2**: The per-episode manifest (RFC-109) records language, translation model, unit count, and
  which transcript variant the ad-free base was built on. Verification outcomes join it in v2.
- **FR6.3** *(v2)*: A review worklist lists contradicted and unverified claims, each with source and
  translation side by side. It has no rows until verification exists.

### FR7: Language visibility and filtering

**The badge and the filter ship together, in v2**, once the corpus actually holds more than one
language. A language chip on every show existed in the player and was **deliberately removed**
(#2115, 2026-09-17) with the reason recorded in its own test: *"every show in the corpus is English,
so it was a constant that cost a wrap in a 144px column."* That reasoning is correct while the corpus
is monolingual and stops applying the moment it is not — so reinstating the badge belongs with the
second language, not before it. Shipping it in Phase 0 would re-add exactly what was deleted.

What v1 does deliver is the **data**: language parsed, resolved, persisted and exposed on the API
(FR1.1, FR1.3), plus the audit report (FR1.6). That is what makes the corpus's language knowable; the
chrome follows when it has something to distinguish.

- **FR7.1**: Episode language is exposed by the app API, and show language — which exists today as
  `AppPodcastItem.language` but currently serves the run config — starts serving the **normalized**
  feed tag (`en-US` → `en`), so a language is one token rather than one per feed's spelling. The
  operator viewer's feed response gains the same field. **v1.**
- **FR7.1b**: The **transcript panel defaults to English**, with a small control to switch to the
  original language. English is the default because summaries, insights and search are all English, so a
  source-language transcript would be the inconsistent choice. The control serves the **full-timeline**
  English transcript, not the ad-free analysis text, so it stays in sync with the audio where ads were
  cut. **v1** — and it is why the badge *component* exists in v1 even though badges as metadata
  decoration do not.
- **FR7.2**: Every show and episode displays a compact language badge — a small squared chip with the
  uppercase code — wherever that item's metadata already renders: consumer episode rows, tiles and
  cards, show rows, tiles and detail page, and the operator shows library. Omitted, not guessed, when
  the language is unknown. **v2.**
- **FR7.3**: Shows and episodes can be **filtered by language** on the surfaces that already offer
  filters — the consumer episode toolbar (today: all / unplayed / played / insights / downloaded, plus
  a show selector), show browse, and the operator library filter bar. **v2.**
- **FR7.4**: The language filter is its **own** control, not another option inside the
  played/downloaded filter, because the dimensions are orthogonal: "Greek **and** unplayed" must be
  expressible.
- **FR7.5**: Both the badge and the filter render only when the corpus holds more than one language.

## Supported-language policy

A language goes live only after it passes the RFC-124 §7 gate: acceptable transcription quality on real
episodes, acceptable translation quality judged per the LLM-judge protocol, position agreement between
translation and source at or above the threshold, and a measured ad-detection survival rate. The ASR
figures in Appendix A are a **prior** for deciding what to enable first. They are not a support claim.

**The roadmap is three tiers.** The registry lists all of them; only `en` is enabled.

**Tier 1 — the initial set and the focus of v1.** Dutch, German, Italian, Spanish, Catalan, French,
Portuguese, Swedish, Norwegian. Every one is under 10% FLEURS WER and five are under 5% — Spanish (3.0),
Italian (4.0), Portuguese (4.3) and German (4.5) transcribe *better* than English (4.2) on that
benchmark. All of tier 1 is **Latin script**, which is why no transliteration or identity aliasing is
built (FR4.4).

**Tier 2 — Eastern Europe.** Russian, Serbian, Bulgarian, Romanian. Introduces Cyrillic, which is the
trigger for the deferred identity work.

**Tier 3 — Asia and the Middle East.** Korean, Japanese, Chinese, Arabic. Two v1 assumptions must be
fixed first: translation units are packed by word count, which is meaningless without spaces, and the
speaker-name heuristic requires two or more tokens, which a single-token CJK name fails.

The tiers are ordered by intent rather than difficulty, and they diverge in two places worth knowing:
**Russian (5.6) and Japanese (5.3) are technically easier than Norwegian (9.5)**, and **Serbian (33.9) is
the hardest language in all three tiers** by more than double the next one.

**Pilot: Spanish or Italian.** Lowest error rate, Latin script, covered by every candidate model.

External wording for beta conversations: "Western European and Nordic languages are the first set;
Eastern European languages follow; Asian and Middle Eastern languages are later." Do not promise a
specific language before it has passed the gate.

## Phasing

Phase numbering means one thing across this PRD and RFC-123/124/125. The demand and bake-off
validation step is **Gate V**, not a phase, because it produces evidence rather than software. The
per-slice breakdown — each slice sized as one issue, with dependencies and acceptance criteria — lives
in [MULTILINGUAL_ARC §4](../architecture/MULTILINGUAL_ARC.md#4-slice-plan).

- **Phase 0: English as a declared language.** **Ships on its own, before Gate V.** Parsing and
  persisting the feed's declared language with its per-show backfill (FR1.1); normalization and
  per-episode resolution plus the feed override (FR1.2); the language threaded as an explicit parameter
  with the one-reader invariant (FR1.3); the registry and skip path, ordered after the override
  (FR1.4); the language-ID check (FR1.5); the corpus audit over real data (FR1.6); the new failure
  modes made visible (FR1.7); transcription-model selection from the resolved language with a
  non-English floor (FR2.1); episode language on the API and the badge (FR7.1–FR7.2). A correctness fix
  on the corpus that exists — no new models, no GPU — and what makes the multilingual path provable end
  to end before any translation happens.
- **Phase 1: Turns (RFC-123).** Independent of everything multilingual; improves quote attribution,
  search chunking and player cues for English episodes today. Can run in parallel with Phase 0.
- **Gate V: Validate (no build).** Ask the beta cohort what share of their listening is non-English
  and which shows. Confirm candidate translation-model availability and licence terms (unverified
  today — largely **done**, see Appendix B). Run the RFC-124 bake-off on 2–3 **Greek** episodes plus
  one Spanish or Italian control, and settle Serbian separately (arc slice V.5). Requires Phase 0,
  because the bake-off has to transcribe non-English audio correctly to measure anything.
  **Decision gate:** proceed only if the demand signal and the quality gate both pass.

  Greek currently looks like the stronger pilot — 12.5% WER on read speech, present in both MT models
  whose language lists we could read, while Serbian is 33.9% and absent from both (Appendix A,
  Appendix B). But neither claim is stronger than that: TranslateGemma's coverage is unconfirmed for both
  languages because its card is gated, and Qwen3's was never checked. **The pilot is decided by
  measurement, not by these lists** — the arc's V.5 runs now and settles it, including whether Serbian's
  gap is largely a Cyrillic-versus-Latin artifact.
- **Phase 2: Translation (RFC-124)** behind a feature flag that gates **serving as well as the
  pipeline**, on one operator-chosen feed. Includes the trust marker on every translated claim and the
  read-time Positions gate, both of which would otherwise arrive too late — see Phase 3.
- **Phase 3: Trust (RFC-125).** The source-verification pass and the operator worklist. Note the
  **marker and the gate ship in Phase 2, not here**: the gate keys on the marker, so a translated claim
  written without one passes straight through and "fail-closed" stops being true. Serving translated
  episodes before both exist would put unverified, unlabelled claims into Position timelines — the
  failure this PRD calls the worst it can produce.
- **Phase 4: Surfaces.** Language toggle, translated-quote treatment, and the language filters
  (FR7.3–FR7.5). UXS to follow for the transcript toggle and quote treatment.
- **v2:** per-unit quality estimation and calibrated bands (RFC-125 §7).

## Success Metrics

- **Phase 0:** the audit reports 100% of existing shows and episodes as `en` with a named resolution
  source; every show and episode renders a language badge; no provider can receive a null language and
  substitute English; and English artifacts are unchanged outside a **declared allow-list** of added
  metadata and manifest keys. Unqualified byte-identity is not the criterion and never was achievable:
  this phase deliberately adds language fields, and the new stage slot adds a `translation: skipped`
  entry to every English episode's stage ledger.
- **Demand (Gate V):** at least 30% of the beta cohort names one or more non-English shows they
  would add to the product.
- **Quality gate (per language):** at least 90% of position-bearing units judged meaning-preserving
  by a native reviewer, and at least 90% agreement between positions extracted from the translation
  and the reviewer's reading of the original.
- **Provenance:** every claim derived from translated text carries a complete, resolvable set of
  source-unit references — the one v1 trust requirement, and what makes v2's labelling, verification
  and gating addable without reprocessing.
- **Translation quality (Gate V, not a shipped metric):** at least 90% of position-bearing units
  judged meaning-preserving, and at least 90% agreement between positions extracted from the
  translation and the judge's reading of the source. This is how the model is *chosen*; v1 does not
  re-check it per episode.
- **Findability (Phase 4):** a listener can reach every non-English show in the corpus using the
  language filter alone.
- **Coverage:** at least 95% of episodes in enabled languages reach English analysis without manual
  retry.
- **Ad removal:** `ad_chars_removed` on translated episodes is non-zero at a rate comparable to
  English episodes from similar shows.
- **Isolation:** no measurable change in English-episode pipeline time, and no change to English
  artifacts outside the declared allow-list, when the feature is enabled — including byte-identical
  `position_arc` responses on the English corpus (English nodes carry no `translation` block, so the
  gate is a no-op there).

## Dependencies

- RFC-123 turns artifact (unit boundaries for translation, and the citation unit for confidence).
- One additional served model on the DGX for translation.
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
  residual risk is accepted only for non-position content, which is labelled as translated but not
  individually verified in v1.
- **Sponsor reads enter the analysis text.** The cause is English-only ad patterns; mitigated by
  translating before ad removal (FR4.1) and measured by the bake-off's ad-survival metric. If
  survival is low, per-language ad cues come back on the table.
- **There is no episode language to be right or wrong yet.** Nothing parses the feed's declared tag and
  no episode carries a language of its own, so the badge, the audit and the normalization step all have
  no input until FR1.1 lands. Mitigated by making FR1.1 the first slice, with its per-show backfill,
  rather than assuming the data was already there.
- **Identity fragmentation across scripts.** Mitigated by FR4.4 and CIL alias rules (RFC-124 §5.4).
- **Demand does not materialize.** Gate V; Phase 0 and RFC-123 still pay for themselves.
- **Serbian may not be supportable under an eligible licence.** Whisper is at 33.9% FLEURS WER on
  Serbian against Greek's 12.5%, and Serbian is absent from both MiLMMT-46 and LMT-60 while the one
  shortlisted model that covers it (NLLB) is non-commercial. Mitigated by deciding the pilot language by
  **measurement** (arc slice V.5, running now) rather than by these lists, and by promising no language
  until it does. The likely cause of the WER gap is script — Croatian, mutually intelligible, sits at
  13.4% and FLEURS scores it in Latin while Serbian is Cyrillic — and that is an hour's work to test.
- **Model licensing.** Verified rather than assumed: Hunyuan/HY-MT excludes the EU, NLLB is
  non-commercial; both are out. TranslateGemma, MiLMMT-46 and LMT-60 are eligible (arc §6.1).
- **Greek keyword search is weaker than English-on-English.** FR4.5 keeps the existing embedding model,
  so non-English chunks are keyword-matched only, and the index's full-text tokenizer is English
  (stemming, stop-words, accent folding) with one tokenizer per index. Mitigated by measuring Greek
  recall before calling the slice done; if it is poor the options are a per-language full-text table or
  accepting it explicitly. **This risk replaces an earlier, much larger one** — an embedding-model swap
  that would have re-embedded all 678 existing English episodes against an eval of 25 fixture anchors
  with no CI gate. That is now out of scope entirely.
- **The backfill depends on feeds still being reachable.** FR1.1's migration fetches each show's feed to
  read its declared language. A feed that has gone away, moved, or dropped the tag leaves its episodes
  with `language_source: unknown` — reported, not guessed. Known cost, existing machinery, bounded by the
  number of shows rather than episodes.

## Open Questions

1. How is code-switching handled (English passages inside a Serbian episode)? Transcribe as
   declared and accept the damage, or segment-level language ID?
2. Serbian script: normalize output to Latin, keep Cyrillic, or follow the feed? **This was filed as a
   display question and the evidence reclassified it as a capability question.** Whisper is at 33.9% WER
   on (Cyrillic) Serbian and 13.4% on (Latin) Croatian — two mutually intelligible languages — so the
   script plausibly drives most of the gap, and it also decides whether any eligible MT model covers the
   language. Arc slice V.5 tests it.
3. Should verified translated positions be visually distinguishable from native English ones in
   Position timelines, or is verification enough to treat them as equals?
4. Do we ever show amber-band quotes in shareable quote cards?
5. Do we keep a source-language ad-free transcript at all, derived by mapping the English ad ranges
   back to source offsets? It has a reader/player use, not an analysis one.

## Appendix A: ASR evidence by language (Whisper FLEURS WER)

**Source**: Whisper paper (Radford et al.), Appendix D.2.4, **Table 13 "WER (%) on Fleurs"**. Figures
are the **`large-v2`** row — the strongest model in that table. Verified against the paper on
2026-09-28; the fuller discussion, including the Serbian finding, is in
[MULTILINGUAL_ARC §6.2–§6.3](../architecture/MULTILINGUAL_ARC.md#62-asr-evidence-whisper-fleurs-wer).

**Read these caveats before using a number:**

1. **Not large-v3 or turbo.** `large-v3` postdates the paper and appears nowhere in it. Our DGX model
   is `faster-whisper-large-v3-turbo-ct2`, which is generally better per language, so treat this as a
   **conservative prior**. The per-language large-v3 figures exist in the `openai/whisper` repo's
   `language-breakdown.svg`, which is a figure and has not been transcribed here.
2. **FLEURS is read speech.** Real podcast rates are higher — crosstalk, music, informal register.
3. **Bigger is not monotonically better per language.** Serbian regressed from `large` (29.2) to
   `large-v2` (33.9).

**Cluster A — under 5% WER**
Spanish 3.0 · Italian 4.0 · English 4.2 · Portuguese 4.3 · German 4.5

**Cluster B — 5% to 10% WER**
Japanese 5.3 · Polish 5.4 · Russian 5.6 · Dutch 6.7 · Indonesian 7.1 · Catalan 7.3 · **French 8.3** ·
Turkish 8.4 · Swedish 8.5 · Ukrainian 8.6 · Malay 8.7 · Norwegian 9.5 · Finnish 9.7

**Cluster C — over 10% WER**
Vietnamese 10.3 · Thai 11.5 · Slovak 11.7 · **Greek 12.5** · Czech 13.3 · **Croatian 13.4** ·
Danish 13.8 · Tagalog 13.8 · Korean 14.3 · Romanian 14.4 · Bulgarian 14.6 · Chinese 14.7 ·
Galician 15.4 · Bosnian 15.7 · Arabic 16.0 · Macedonian 16.5 · Hungarian 17.0 · Tamil 17.5 ·
Hindi 21.5 · Estonian 21.9 · Urdu 22.6 · Latvian 23.1 · Slovenian 23.1 · Azerbaijani 23.4 ·
Hebrew 27.1 · Lithuanian 28.1 · Persian 32.9 · Welsh 33.0 · **Serbian 33.9** · Afrikaans 36.7 ·
Kannada 37.0 · Kazakh 37.7 · Icelandic 38.2 · Marathi 38.3 · Maori 38.5 · Swahili 39.3 — then
Armenian 44.6 and the remaining low-resource languages. Javanese is `nan` in the source.

Clusters A and B are complete. Cluster C is complete up to 40%; above that only a sample is shown.

**What changed from earlier drafts of this appendix.** The first version was recollection and placed
Serbian at "10–15%" (actual **33.9**), Macedonian in the same band (16.5) and Hungarian too (17.0).
The second version was sourced but **incomplete**: it silently dropped French at 8.3 — the one Western
European language named in the candidate wording above — along with Arabic, Azerbaijani and Maori, and
then asserted everything unlisted was above 40%, which was false. Each language here is placed by its
measured number.

## Appendix B: translation model shortlist

Verified 2026-09-28. Full detail, including what was **not** verified, in
[MULTILINGUAL_ARC §6.1](../architecture/MULTILINGUAL_ARC.md#61-translation-model-shortlist).

| Model | Licence | `el` | `sr` | Eligible? |
| --- | --- | --- | --- | --- |
| TranslateGemma 27B / 12B / 4B | `gemma` (commercial OK, no territory carve-out) | ? | ? | Yes — its 55 languages are not enumerated on the card, the blog or the abstract, and **the card is gated**, so a login settles it |
| MiLMMT-46-12B v1.0 | `gemma` | ✅ | ❌ | Yes, Serbian absent from its 46 |
| LMT-60-8B | `apache-2.0` | ✅ | ❌ | Yes, Serbian absent from its 60 |
| Qwen3-30B-A3B (already served) | `apache-2.0` | ? | **?** | Baseline / verification model — its coverage was **never checked**, and it is already served |
| Hunyuan-MT / HY-MT | Territory **excludes the EU** | — | — | **No** |
| NLLB-200 | `cc-by-nc-4.0` | ✅ | ✅ | **No** — non-commercial, and the only one covering Serbian |

Two `?` cells are load-bearing and neither should be read as a "no": TranslateGemma's coverage of either
language, and Qwen3's. Closing them is the first work in the arc's slice V.5.
