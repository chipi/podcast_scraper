# RFC-124: Multilingual Transcription and English Translation Stage

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Pipeline (transcription, diarization, translation, ad-free base, summary/GI/KG), DGX serving, player (segments contract), search, operator feed curation
- **Related PRDs**:
  - `docs/prd/PRD-047-multilingual-ingest.md` — product requirements this RFC implements (FR1–FR4, FR6, FR7)
- **Related RFCs**:
  - `docs/rfc/RFC-123-speaker-turns-artifact.md` — prerequisite: turns/sentences define translation units
  - `docs/rfc/RFC-125-translation-confidence-and-claim-verification.md` — provenance and source verification over this RFC's outputs (QE deferred to v2)
  - `docs/rfc/RFC-005-whisper-integration.md`, `docs/rfc/RFC-058-audio-speaker-diarization.md`
  - `docs/rfc/RFC-090-hybrid-retrieval.md` — the two-tier index and RRF this RFC's retrieval section extends
  - `docs/rfc/RFC-106-tiered-dgx-service-fallback.md` — tiered fallback semantics
  - `docs/rfc/RFC-109-per-episode-observability-manifest.md` — manifest fields
  - `docs/rfc/RFC-115-transcript-prefix-caching-llm-stages.md` — LLM stages cache the analysis transcript as a prompt prefix
- **Related ADRs**:
  - `docs/adr/ADR-155-pin-every-model-checkpoint.md` — the translation checkpoint is pinned
- **Arc notes**: `docs/architecture/MULTILINGUAL_ARC.md` — slice plan (§4), verified code facts and the
  claims the adversarial reviews found false (§5.4), decisions D-1 … D-20

## Abstract

This RFC makes language a per-show property instead of a global setting. Episodes are transcribed and
diarized in their source language, and that transcript stays canonical. A **translation stage** — which
runs **directly after transcription and diarization, before summary** (D-19) — produces an English
derived transcript (`<stem>.en.txt`, `<stem>.en.segments.json`) plus a unit map
(`<stem>.translation.json`) linking every English span back to source text, audio time and speaker.
The existing ad-removal machinery then runs on the **English** text, and the result becomes the
transcript every analysis stage reads. Reaching that state requires routing the transcript readers
through one resolver, which the codebase does not do today (§2.1). Search indexes both layers so an
episode is findable in its own language. The translation model is chosen per language by a bake-off
with an explicit quality gate.

## Problem Statement

The pipeline is English by configuration. `language: en` is global
(`config/profiles/prod_dgx_full.yaml:70`). The DGX Whisper (`faster-whisper-large-v3-turbo-ct2`) and
pyannote community-1 are multilingual already, so transcribing Greek is close to a parameter change.

Analysis is English-shaped by design, and it should stay that way. GIL grounding requires quotes that
are verbatim substrings with char offsets (`gi/grounding.py`, `EvidenceSpan`). Extraction prompts,
QA/NLI entailment, MiniLM embeddings and spaCy NER are all English. Making each layer multilingual
would fork every prompt, eval and threshold per language. Translating once, to English, keeps every
downstream layer single-path.

Four things are missing: a **place in the stage order** to translate, an artifact model that keeps
translated text **traceable** to what was actually said, a way for a per-episode language to reach the
transcriber at all, and — discovered in review — a **single resolver** that decides which transcript
analysis reads.

**The five silent hazards.** Each is verified against the source, and each fails quietly rather than
erroring. The arc notes (§5.2) carry the same list with slice ids.

1. **Silent degradation to an unusable model.** `whisper_utils.normalize_whisper_model_name` drops
   `.en` for non-English, then builds a chain from `FALLBACK_WHISPER_MODELS_MULTILINGUAL` down to
   `tiny`. Prod's local fallback default is `base.en` (`config_constants.py:332`), so for Greek the
   chain is `["base", "tiny"]`. The DGX path never passes through the normalizer — `dgx_whisper_model`
   is an HF id — so this hazard lives entirely in the fallback tier.
2. **No per-episode language path exists.** Providers read run-global config when the argument is
   absent (`ml_provider.py:827`, `:885`), the transcription call site passes exactly that global
   (`workflow/episode_processor.py:2296`), and `workflow/sniff_gate.py:116,130,144,177` threads it four
   more times.
3. **The DGX provider misreports the language.** `whisper_provider.py:197` sets `language or "en"` in
   the **returned result dict**; the request (`:429-430`) *omits* `language` when it is `None`. So the
   server auto-detects and the provider then reports `"en"`. The failure is a provenance lie, not a
   forced English transcription.
4. **Ad excision silently no-ops on non-English text.** `gi/filters.py:35-77` `_AD_PATTERNS` all
   require an English token (`brought to you by`, `sponsored by`, `\w+ dot com slash`), so a Greek
   transcript yields no ranges (`gi/ad_regions.py:409-418`) and `build_adfree_artifacts` produces an
   **identity** ad-free base with `is_adfree: True` and the sponsor reads intact in the analysis text.
   Not absolute: a Greek host reading an English URL would match. Caveat: with no segments,
   `build_adfree_artifacts` returns `None` and no artifact is written at all.
5. **The sniff gate keeps the cheap transcript on non-English audio.** `sniff_gate.py:63-70` judges
   the small-model transcript by counting entities with spaCy `en_core_web_sm`; on Greek that count is
   ~0, so the sniff transcript is kept. Off in every profile today, one config line from live.

Hazard 4 is why translation precedes ad detection (§3).

**Use cases:**

1. **Native show in the library.** A Greek show is processed end to end, and its insights and
   positions sit in the same corpus as English ones.
2. **Subtitles.** The player plays the Greek audio with English cues aligned to it.
3. **Traceable quote.** Tapping a translated quote plays the original audio span and shows the
   original sentence.
4. **Findable in its own language.** A Greek query reaches the Greek episode.

## Goals

1. **Per-show language routing** with an operator override and explicit skip for unsupported
   languages.
2. **The source language stays canonical.** Never overwritten, always servable.
3. **Complete, turn-aligned English translation**, with a lossless map from English spans back to
   source spans.
4. **One analysis transcript, resolved in one place.** Analysis layers keep no per-language code, and
   no stage decides for itself what "the transcript" means.
5. **Ad handling that actually works on non-English episodes**, rather than an identity pass that
   looks like success.
6. **An episode findable in the language it was spoken in.**
7. **Pinned, measured models**, chosen per language by bake-off.

## Constraints & Assumptions

**Constraints:**

- English episodes: artifacts unchanged outside a declared allow-list of added metadata keys, and no
  added latency when the feature flag is on. (Unqualified byte-identity is not achievable — this
  feature deliberately adds language fields to metadata and the manifest.)
- No silent model substitution for translation. A different model invalidates the bake-off evidence
  the language was enabled on, so an unavailable model means **defer**, not **fall back**.
- Only models whose license permits this deployment (EU operator, commercial product) are eligible.
- Runs on the DGX Spark as its own served process. English ingest keeps priority in the work queue.
- **One translation pass per episode.** The full-timeline text is translated exactly once; every other
  English artifact is derived from it by existing deterministic machinery.
- **No embedding-model change anywhere in this RFC** (D-14).

**Assumptions:**

- RFC-123 turns exist for the episode. Translation does not run without them.
- Whisper punctuation in enabled languages is good enough for sentence splitting. The per-language
  gate (§7) checks this.
- Conversational podcasts translate acceptably at the turn or sentence-group level, without cross-turn
  context. §7 tests this.
- Sponsor reads in a non-English episode survive translation as recognizable English sponsor language.
  §7 measures this; it is the assumption §3's ordering rests on.

## Design & Implementation

### 1. Language resolution

`resolve_episode_language(feed_entry, feed_doc, cfg)` produces `episode.language` (ISO 639-1):

1. **Operator override** on the feed entry (§1.2).
2. **The feed's declared `<language>`**, normalized: `el-GR` → `el`, `sr-Latn-RS` → `sr`, `pt_BR` → `pt`.
3. **Profile default** `language` (currently `en`) — the lowest precedence, and the only place
   `cfg.language` may be read.

**1.1 The feed's language is not read today. This is new plumbing, not a lift.** A sweep of `rss/`
finds nothing language-related, and `models/entities.py:16-44` `RssFeed` has no language field. What
exists is a *write* of the run config: `metadata_generation.py:957` sets
`FeedMetadata(language=cfg.language)`, so `server/corpus_catalog.py:159` `_feed_language` reads back
the profile's `en`, and `AppPodcastItem.language` (`server/schemas.py:1719`, served at
`app_episodes.py:156`) serves that. The only episode-level language today is `TranscriptInfo.language` (`:3832`), also written from config;
`metadata_generation.py:2843` writes one into `processing.config_snapshot`, which is an accurate
snapshot of configuration and not a claim about the episode.

So the first slice of this design is:

- extract the channel `<language>` in `rss/parser.py` and carry it on `RssFeed` and `FeedMetadata`;
- persist `feed.language_raw`, `feed.language` (normalized) and `feed.language_source`;
- add an **episode-level** `language` and `language_source`;
- `_feed_language` calls the normalizer, so the catalog and the pipeline cannot disagree;
- and because this changes `*.metadata.json`, it carries a migration under `upgrade/migrations/`, a
  `corpus_format_version` bump, a reader-support bump, a `CORPUS_UPGRADE.md` row and a backfill.

Without it the language badge would display the run config, the corpus audit would be a tautology, and
the `en-US → en` normalization would have no input.

**1.2 The operator override is an allowlist addition.** `rss/feeds_spec.py:26-56` already accepts a
mapping per feed entry, validated with `extra="forbid"` (`:77`) against
`RSS_FEED_ENTRY_OVERRIDE_KEYS` — whose keys must exist on `Config`, and `language` does. So:

```yaml
feeds:
  - url: https://example.com/feed.xml          # unchanged bare form stays valid
  - url: https://example.gr/feed.xml
    language: el                                # operator override
```

needs a `language` field on `RssFeedEntry` (it has explicit typed fields **and** `extra="forbid"`) plus the allowlist entry — small, but not zero. One consequence worth knowing: `merge_feed_entry_into_config` makes the per-feed `Config` the run's `cfg`, so the override reaches every existing `cfg.language` reader with no threading at all. The hazard is the other side of that: in a multi-feed batch the ML singleton is deliberately held across feeds, so feed 1's Whisper model persists and feed 2's language is silently ignored (§2, per-call resolution). The shows library
*displays* the resolved language and its source (RFC-104's shows library has no backend, so editing
stays in config).

**Supported-language registry.** `config/languages.yaml`:

Seeded with the full roadmap (arc §6.3) and only `en` enabled. `tier` is the rollout tier, not a quality
claim; `wer` records the measured prior so nobody has to look it up again.

```yaml
languages:
  en: { tier: 0, wer: 4.2, enabled: true,  translation: none }

  # Tier 1 — the initial set. All Latin script, all under 10% WER.
  es: { tier: 1, wer: 3.0, enabled: false, translation: { model_ref: tx-default } }
  it: { tier: 1, wer: 4.0, enabled: false, translation: { model_ref: tx-default } }
  pt: { tier: 1, wer: 4.3, enabled: false, translation: { model_ref: tx-default } }
  de: { tier: 1, wer: 4.5, enabled: false, translation: { model_ref: tx-default } }
  nl: { tier: 1, wer: 6.7, enabled: false, translation: { model_ref: tx-default } }
  ca: { tier: 1, wer: 7.3, enabled: false, translation: { model_ref: tx-default } }
  fr: { tier: 1, wer: 8.3, enabled: false, translation: { model_ref: tx-default } }
  sv: { tier: 1, wer: 8.5, enabled: false, translation: { model_ref: tx-default } }
  no: { tier: 1, wer: 9.5, enabled: false, translation: { model_ref: tx-default } }

  # Tier 2 — Eastern Europe. Introduces Cyrillic.
  ru: { tier: 2, wer: 5.6,  enabled: false, translation: { model_ref: tx-default } }
  ro: { tier: 2, wer: 14.4, enabled: false, translation: { model_ref: tx-default } }
  bg: { tier: 2, wer: 14.6, enabled: false, translation: { model_ref: tx-default } }
  sr: { tier: 2, wer: 33.9, enabled: false, translation: { model_ref: tx-default } }

  # Tier 3 — Asia and the Middle East. Needs the CJK prerequisites first (v2 doc §10).
  ja: { tier: 3, wer: 5.3,  enabled: false, translation: { model_ref: tx-default } }
  ko: { tier: 3, wer: 14.3, enabled: false, translation: { model_ref: tx-default } }
  zh: { tier: 3, wer: 14.7, enabled: false, translation: { model_ref: tx-default } }
  ar: { tier: 3, wer: 16.0, enabled: false, translation: { model_ref: tx-default } }

models:
  tx-default: { id: <chosen model>, revision: <sha>, serve: dgx_vllm_translate }
```

An episode whose language is not `enabled` gets status `skipped_unsupported_language` and is never
transcribed. **The skip must not ship before the override** (§1.2): feed tags are routinely wrong, and
an English show tagged `de` would otherwise stop ingesting with no remedy.

**Sanity check, not routing.** After transcription, Whisper language ID runs on a 30 s window from the
middle third of the episode. If the result is not the declared language with probability ≥ 0.8, the
manifest records `language_mismatch: {declared, detected, p}` and the episode is flagged. It is
**never** rerouted automatically, because intros, music and ad reads regularly fool language ID.

### 2. Source-language transcription and diarization

- **Thread the resolved language to the call site.** `episode_processor._transcribe_one` passes the
  episode's resolved language instead of `cfg.language`. Providers need no signature change —
  `language` is already on the protocol (`transcription/base.py:48`).
- **Delete the substitutions.** `ml_provider.py:827/:885`'s `self.cfg.language or "en"` goes, and the
  DGX provider stops reporting `"en"` for an auto-detected transcript (hazard 3). The cloud providers
  fall through to `None`/auto and never substitute `"en"`, so there is nothing to remove there. The
  acceptance criterion is a **lint rule** — `cfg.language` readable in exactly one place — rather than
  a hand-maintained list, because the list was already four sites short (`sniff_gate.py`).
- **Model selection is per call, not per provider.** `ml_provider.py:566-568` resolves the Whisper
  model **once at init** from run-global language, so per-episode selection needs per-call resolution
  in `MLProvider`. `normalize_whisper_model_name` gains a `min_model_for_non_english` (default
  `large-v3`); for non-`en` the chain truncates at that floor, and if no tier meets it the attempt
  fails `deferred_quality_floor` rather than transcribing with `base`. Simplest defensible policy:
  the local tier does not serve non-English at all.
- **NER.** `speaker_detectors/ner.py:131-136` selects the *default* NER model only when
  `cfg.ner_model` is unset — it is not a gate on NER itself, so a configured model would run on any
  language. Non-English episodes must skip transcript NER explicitly rather than relying on that.
- Diarization (pyannote community-1) is language-agnostic; nothing changes.
- **Speaker naming stays before translation.** See §5.4.
- Source artifacts keep their existing paths (`<stem>.txt`, `<stem>.segments.json`) and gain a
  top-level `language`. RFC-123 `turns.json` is built from them as usual.
- **No source-language ad-free variant is written** (§3).

**2.1 One resolver, and routing every reader to it.** `workflow/adfree_transcript.load_processing_transcript`
describes itself as "the single resolver all NLP consumers use". It is not. Two callers use it — GI
(`metadata_generation.py:4835`) and KG (`:5082`) — while these resolve independently:

| Reader | Location | Today's behaviour |
| --- | --- | --- |
| Summary | `metadata_generation.py:2926-2928` | opens `transcript_file_path` raw |
| Faithfulness / QA flags | `:2401-2405` **and** `:4265-4270` | two separate raw reads; the second is easily missed |
| Recurrent-host scan | `workflow/stages/processing.py:1089` | its own `.adfree` → `.cleaned` → raw precedence |
| Viewer transcript serving | `server/routes/corpus_text_file.py:65-104` | its own `.adfree` → raw fallback |
| Segments view | `server/segments_view.py:21-35` | its own precedence |
| Segments sidecar pick | `metadata_generation.py:1199` | `.adfree.segments.json` then `.segments.json` |
| Search indexer | `search/indexer.py:96-111` | its own `.adfree` preference |
| GI repair | `gi/repair.py:161-171` | its own resolution |
| GI evidence loading | `gi/load.py:19-37` | **raw `.txt`, no ad-free preference at all** |

That last row is a genuine coordinate-space bug that exists **today**, independent of this feature — GI
computes offsets in ad-free space while this loader reads the raw text. Its severity is bounded, though:
`load_artifact_and_transcript` is reached only from `gi inspect` and `gi show-insight` (`cli.py:2917`,
`:2991`), so it is a CLI inspection bug rather than a pipeline or server path. Worth fixing in the same
pass; not worth describing as a production defect.

The list above is what a search turned up, and this inventory has already been wrong twice. **Producing
the complete list is part of S2.1's work**, not a claim this document makes — the slice starts by
enumerating every reader of `transcript_file_path` and every independent `.adfree` precedence.

Routing them all through one resolver is a standalone bug fix that this design also needs, and it is
where the `.en.txt` branch belongs — precedence `.en.adfree.txt` → `.en.txt` → `.adfree.txt` → `.txt`.

The resolver also gains a **provenance check**: `.en.adfree.*` carries the `en_sha256` of the
`.en.txt` it was built from, and a mismatch refuses rather than silently anchoring spans into stale
text.

### 3. Stage order: translation directly after transcription, before summary

For an English episode today: transcribe → segments → ad-free base → summary → GI → KG.
`CANONICAL_STAGE_ORDER` is `("asr", "diarization", "naming", "summary", "gi", "kg")`
(`workflow/processing_manifest.py:54`) with no translation or ad-free slot, and summary runs in the
same episode job immediately after transcription (`workflow/stages/transcription.py:266`).

**Translation is inserted directly after transcription and diarization, before summary** (D-19).
Two reasons, and the second is the load-bearing one:

1. **Ad detection needs English** (hazard 4). Running ad excision on Greek produces an identity
   artifact that quietly feeds sponsor reads to every analysis layer.
2. **Summary output feeds topic identity.** The summary's prefilled topics become GI's topic labels
   (`metadata_generation.py:4890ff`) and KG's topics come from summary bullets (`:5100-5110`).
   Translating after summary would give English insights hanging off **Greek** topic labels — and
   topics are how episodes connect to one another across the corpus, so cross-episode topic identity
   would fragment by language. That is the opposite of "English-normalized intelligence".

The resulting order for a non-English episode:

```text
audio
 └─ transcribe + diarize (source language)      → ep1.txt, ep1.segments.json
     └─ speaker naming (source, §5.4)
         └─ turns (RFC-123)                      → ep1.turns.json
             └─ TRANSLATE every unit, full timeline
                 │                               → ep1.translation.json
                 │                               → ep1.en.txt, ep1.en.segments.json
                 ├─ turns over English           → ep1.en.turns.json
                 └─ ad-free base ON ENGLISH      → ep1.en.adfree.{txt,segments.json,admap.json}
                     └─ summary → GI → KG, all reading ep1.en.adfree.txt
```

**One seam, not "after transcription".** `call_generate_metadata` is invoked from **two** places
(`stages/transcription.py:266` in the sequential loop and `stages/processing.py:3138` on the processing
thread), and an episode whose publisher supplies its own transcript never enters
`transcribe_media_to_text` at all (`episode_processor.py:4440-4490`) — which is why `retranscript_only`
exists. So "directly after transcription" names a seam that does not exist for those episodes.

The insertion point is therefore **inside `generate_episode_metadata`, immediately before the summary
call** — the one site every path converges on: ASR, transcript-cache hits, direct downloads,
publisher-supplied transcripts, and every `relabel_only` / `rediarize_only` / `retranscript_only` /
`rederive_only` cascade. Two consequences to build in:

- **Exclude translation wall time from the metadata deadline** (`processing.py:3120-3130`), or a
  several-minute translation eats a budget the LLM stages already strain.
- **`CANONICAL_STAGE_ORDER` gains a `translation` slot**, and ADR-151 requires every stage to record an
  outcome — so **every English episode's stage ledger gains `translation: skipped`**. That changes every
  English `metadata.json`, which is exactly why S0.10's allow-list test exists.

Retries use a `translate_only` reprocess mode rather than a new per-episode pending queue: the pipeline
has no asynchronous per-episode state machine, and inventing one is larger than this feature needs. The
honest cost of that choice is that a translation-service outage needs an operator-run reprocess, so the
"95% reach English analysis without manual retry" metric holds only when the service is up — state it
that way rather than claiming otherwise.

Three things fall out, all using machinery that already exists:

1. **Ad detection works**, on English text with the English patterns it was written for. No
   per-language ad lexicon, now or later.
2. **One translation pass.** The full-timeline units are both the subtitle source and the input to the
   ad-free derivation. `build_adfree_artifacts` already drops segments inside detected ad ranges and
   **re-renders the survivors** through the same formatter, so offsets come out exact with no new code.
3. **The path helper composes unchanged.** `adfree_transcript_relpath` is `splitext` + `.adfree`, so
   `ep1.en.txt` yields `ep1.en.adfree.txt` (`adfree_transcript.py:52-55`).

**Reprocess paths must become language-aware.** `_maybe_produce_adfree` is called on the **source** text
from **five** sites: `relabel_only` (`episode_processor.py:2869-2882`), `rediarize_only` (`:3191-3204`),
the ASR path (`:3784-3805`), direct download (`:4460-4483`), and the **transcript-cache hit** (`:1310`),
which is the one most easily missed. On a non-English episode each would write the identity
`ep1.adfree.txt` this design says never exists, and leave `.en.*` stale with pre-relabel labels.

Every path that changes the source must invalidate `.en.*`, `translation.json`, any provenance records
and the episode's cached prompt prefix. Two consequences worth stating rather than discovering:

- A relabel that merges two voices into one name **shifts every turn id**, so the unit map is invalid and
  the episode needs a full re-translation — on top of the GI cost the relabel already carries.
- `rederive_only` must **not** re-translate; it reads the existing `.en.adfree` base. That is the cheap
  repair path and losing it would make every repair expensive.

### 4. Artifact model

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
  ep1.en.adfree.segments.json    # carries unit_id per segment (§4.1)
  ep1.en.adfree.admap.json
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
      "sent_ids": ["t0007.s01", "t0007.s02"],
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

`en_char_*` index into `ep1.en.txt`, the full-timeline English text.

**4.1 Resolving a claim back to its source — through segments, not the ad-map.** A GI citation's span
lives in `ep1.en.adfree.txt`. An earlier version of this RFC proposed shifting that span into
full-English space with `ep1.en.adfree.admap.json` and then range-looking it up over `en_char_*`.
**That is wrong, and it was measured.** On the diarized branch `build_adfree_artifacts` does not cut
the complement of the excised ranges; it **drops** overlapping segments and **re-renders** the
survivors, re-emitting the `Label:` prefix and its trailing space. So `_shift_for`
(`gi/ad_regions.py:439`) does not invert the transform. Measured twice, and the mechanism is **not**
simply "one label per excised range": the error appears even when a range removes a whole screenplay line
cleanly, because the two turns that become adjacent afterwards coalesce and their labels vanish too. The
conclusion is what matters — the ad-map cannot reconstruct pre-excision offsets — and nothing in the
codebase used it for reconciliation, so the error had never surfaced.

The correct chain uses the artifact that is exact by construction:

```text
span in ep1.en.adfree.txt
  → binary-search ep1.en.adfree.segments.json for the segment(s) OVERLAPPING [char_start, char_end)
    (char_start/char_end there are exact — the formatter guarantees
     screenplay_text[char_start:char_end] == text)
  → read unit_id off the matched segment(s)
  → look each unit up in translation.json → src_char_*, start_ms/end_ms, speaker
```

**Overlap, not containment.** A span can begin inside a `Label:` prefix, a joining space or the newline
between turns — none of which belongs to any segment — so a containment test would resolve to nothing.
The existing precedent does it the right way: `_char_range_to_ms` (`gi/pipeline.py:841-880`) takes the
first and last *overlapping* segments. (`:810-830` is `_segment_char_spans`, which builds the spans; the
lookup is the later function.)

This requires the English pseudo-segments to **carry `unit_id`**, which means adding it to the
formatter's passthrough tuple (`providers/ml/diarization/formatting.py:81`, currently
`("speaker", "speaker_role", "voice_type")`). The field is additive and absent on English-episode
segments, so nothing changes for the existing corpus. `resolve_units_for_span(span, segments,
translation_map)` owns the chain; RFC-125 calls it and no consumer reimplements it.

**English transcript construction.** `ep1.en.txt` is rendered with the **same screenplay formatter**,
one pseudo-segment per unit (`text = en_text`, times from the unit, `unit_id` carried, speaker label
resolved per §5.4). That makes `ep1.en.segments.json` a valid segments sidecar, so timing lookup,
player cues, RFC-123 turns and the ad-free builder all work on the English layer unchanged.

**Atomicity.** `ep1.en.txt`, `ep1.en.segments.json`, `translation.json` and the `.en.adfree.*` set are
written as one atomic group (temp + rename). `save_adfree_artifacts` currently writes three files
non-atomically and only warns on failure (`adfree_transcript.py:176-188`), which would leave an
episode with English text and no analysis base — a state §5.3 has to refuse rather than half-serve.

**Provenance, checked at the edge that matters.** `.en.adfree.*` carries the `en_sha256` of the
`.en.txt` it was built from, and the resolver refuses on a mismatch. That binds the two files — but it
does **not** bind a `gi.json` span to either, so a *consistent* re-translation would pass the hash check
while every stored span pointed into text that no longer exists. The cheap guard uses data that already
exists: `EvidenceSpan` carries `excerpt`, so `resolve_units_for_span` refuses when
`excerpt != en_adfree_text[char_start:char_end]`. Nodes also record `en_sha256` at write time (RFC-125
§1), which makes stale claims detectable without re-reading the text.

**Derived marking.** `.en.*` and `translation.json` carry `derived: true` and `translated_from`. The
source artifacts do not.

### 5. Translation stage

**5.1 Units are the translation context; sentences are the alignment atom.** These are two different
jobs and collapsing them into one ~120-word block breaks both of the others it was given.

- A **unit** is a sentence group inside a single RFC-123 turn of the source variant — greedily packed,
  never crossing a turn boundary, backchannel turns one unit each. It exists so the model sees enough
  context to translate well. Units are a deterministic function of `ep1.turns.json`.
- A **sentence** is what gets aligned, stored and rendered. The request sends the unit's sentences
  **numbered**, and requires numbered output of the same length; a length mismatch retries once, then
  falls back to translating the unit as one block and marking the unit `alignment: "unit"`.

Why this matters, concretely. The ad-free builder **drops any segment that overlaps an excised range**
(`adfree_transcript.py:129-137`). Today a segment is a 5–15 s Whisper fragment, so an ad boundary costs
at most one fragment. With one pseudo-segment per ~120-word unit, every ad boundary would drop up to
~45 seconds of real speech. And `.en.segments.json` is served as subtitle cues — a 120-word cue is a
paragraph, not a subtitle.

So `translation.json` units carry a `sentences` array:

```json
"sentences": [
  { "sent_id": "t0007.s01", "en_text": "…", "en_char_start": 20114, "en_char_end": 20188 }
]
```

and the English render emits **one pseudo-segment per sentence**, carrying both `unit_id` and `sent_id`.

**This is the one decision v1 cannot cheaply reverse.** Changing alignment granularity later re-translates
every episode and invalidates every provenance block. Word-count packing is also wrong for Japanese,
Korean and Chinese, which do not delimit words with spaces — a tier-3 prerequisite recorded in the v2
notes.

**5.1b Unit identity is content-keyed, and translations are remembered.** A unit's `unit_id`
(`t0007.u01`) is turn-ordinal, so anything that renumbers turns changes it. Two cheap additions make that
harmless:

- each unit also carries a **content key** — a hash of `(source_language, the unit's source sentence
  texts)` — which is stable across renumbering; and
- a **translation memory** keyed by `(source_language, model@revision, src_text)`.

With both, a relabel, a re-render or a merge becomes re-render plus re-key at **zero GPU cost**, and
provenance blocks whose units still resolve by content key survive untouched. Without them, every naming
repair on a translated show costs a full re-translation — and naming repair is the most common repair in
this corpus. Note also that a rename does **not** merge turns: coalescing is by equal *adjacent* labels
(`formatting.py:60-66`), so renaming one voice changes `Label:` prefix lengths and therefore offsets, but
never the unit text itself. That is why the memory hits on every unit.

**5.2 Context.** v1 translates each unit on its own text. Prepending the previous turn and stripping
it from the output is fragile because boundaries drift. §7 measures whether unit-only translation
meets the gate; a model with a native context field may use `context_turns: N`.

**5.3 Serving and failure semantics.** A dedicated vLLM instance (`dgx_vllm_translate`, its own port).
Requests are batched per episode (all units, ordered) with bounded concurrency, on the same DGX work
queue as other GPU stages, with English episodes scheduled ahead of translation batches.

- **Tier.** Per RFC-106, DGX-only in v1. Unavailable → the episode moves to `translation_pending` and
  is retried via `translate_only`. No cloud fallback: substituting a model would make quality
  untraceable, because the text would no longer be what the language's gate evidence was measured on.
- **Partial output.** A unit that errors or returns empty is retried twice, then marked
  `status: "failed"`.
- **Blocking analysis is an explicit gate, not an absence.** The resolver always returns *something*,
  so "analysis does not run" cannot be expressed by withholding a file. `generate_episode_metadata`
  checks the episode's resolved language and translation status: a non-`en` episode without a complete,
  provenance-matching `.en.adfree` set skips summary, GI and KG with an explicit status. Without that
  check, a pending translation would silently fall through the resolver's precedence to `.adfree.txt`
  or `.txt` and run English prompts over Greek text.
- **`transcript_ref` must follow the resolver.** GI updates its ref only `if loaded.is_adfree`
  (`metadata_generation.py:4831, 4839-4840`), so a present `.en.txt` with a missing `.en.adfree.*`
  would stamp `EvidenceSpan.transcript_ref` with the *source* path while the offsets are English. Fix
  the ref to follow whatever the resolver returned.

**5.4 Speaker naming — the same trick as ad detection, for the same reason.**

**The problem.** Naming is three layers and **two of them are English vocabulary**. The deterministic cue
matchers look for `i'm`, `i am`, `my name is`, `Hosted by` (`providers/ml/diarization/roster.py:1472-1494`,
`speaker_detectors/hosts.py:1506-1520`) — none of which fire on *soy*, *me llamo*, *bienvenidos a*,
*presentado por*. Guest candidates come from `en_core_web_trf` NER over the title and description
(`speaker_detectors/detection.py:60-68`). The only language-agnostic layer is LLM voice resolution
(`diarization/pipeline.py:389-455`) — and it is **closed-list**: *"The candidate list is closed. The model
picks a name from it or says null"* (`speaker_detectors/resolution.py:202-204`). So on a Spanish feed the
candidates are starved and voices stay `SPEAKER_01`.

**Why that is not a cosmetic problem.** `_looks_like_person` rejects `SPEAKER_01` (`gi/speakers.py:74-77`),
so no SPOKEN_BY edge is written, so `position_arc`'s predicate — SPOKEN_BY-supported ∩ `ABOUT` ∩ claim
(`server/cil_queries.py:650-665`) — matches nothing. **A translated episode with unnamed voices produces
zero position-bearing insights**, which means the read-time gate has nothing to gate and v2's verification
has nothing to verify. The product's primary signal would simply not exist for non-English shows.

**The fix is the ad-detection trick applied to naming.** We did not write Spanish ad patterns; we
translate first and run the existing English patterns on the English text. Do exactly the same here:

```text
transcribe + diarize            → anonymous labels (SPEAKER_01, SPEAKER_02)
  └─ turns on anonymous labels
      └─ TRANSLATE units        (unit text contains no labels — a turn's char_start is
      │                          AFTER the "Label: " prefix, so labels are never in the payload)
      └─ run the existing ENGLISH cue matchers on ep1.en.txt
      │   → "I'm Ana García", "our guest today is Pablo Ruiz", sign-offs
      └─ map each discovered name to its speaker id via the turn the cue appeared in
          └─ re-render BOTH transcripts with the real names
```

The final step is a relabel, and it is cheap **because of §5.1b**: renaming a voice changes `Label:`
prefix lengths and therefore offsets, never the unit text — so the translation memory hits every unit and
the re-render costs no GPU. The two changes are one design; neither works cleanly alone.

**Also translate the title and description.** They are a few hundred tokens, and once they are English the
existing NER candidate discovery works unmodified — which closes the candidate-starvation problem rather
than routing around it.

**What this changes about the earlier decision.** What must stay before translation is **diarization**, not
naming. Naming *resolution* moves after it. The one thing that does not move is rule 1 below.

**What it does not fix.** A speaker nobody names aloud and who is absent from the feed metadata stays
unnamed. That is equally true in English, so non-English stops being the weaker case — and §7 measures it
directly rather than assuming.

Two rules, and v1 only needs the first:

1. **Speaker labels never go through the translator**, and are carried onto the English line
   **verbatim**. An MT model renames the same person inconsistently between units — "Putin", "Vladimir
   Putin", something odd — which is exactly the fragmentation the identity layer cannot absorb.
2. **In-text names** are left to the model. §7 measures name consistency.

**No transliteration and no alias minting in v1.** Every tier-1 language is Latin script, and a person's
name is usually the identical string across Dutch, German, Italian, Spanish, Catalan, French, Portuguese,
Swedish and Norwegian — so there is nothing to convert and nothing to reconcile. The work that *does*
become necessary when a non-Latin-script language is enabled (canonical-name lookup, deterministic
transliteration, source-script forms written as aliases, and the guard that a transliterated label must
still satisfy `gi/speakers.py:_looks_like_person` or it silently un-attributes a whole turn) is specified
in `docs/architecture/MULTILINGUAL_ARC_V2.md` §10 and triggered by tier 2.

### 6. API, player and search

- **Segments contract** (`GET /api/app/episodes/{slug}/segments`, `server/routes/app_episodes.py:463`)
  gains `?lang=`: default source-language segments (unchanged for English); `lang=en` on a translated
  episode serves English unit-level cues. `SegmentsResponse` gains additive `language`,
  `machine_translated` and `translation_model` (it is a plain `BaseModel`, so this is safe).
- **Episode detail** exposes `language` and `translation_status` (`none | ok | pending | failed`).
- **Quotes** from translated episodes expose `translated: true`, `source_language`, `source_excerpt`
  (the covering unit's source text) and the source time range. Audio playback uses source times, which
  equal the English cue times by construction.
- **The "Translated from <Language>" chip ships with the pipeline, not later.** No translated content
  is served unlabelled at any point.

**6.1 Language on the API** (PRD-047 FR7.1). Episode language becomes a new additive field on the
episode list and detail responses; `AppPodcastItem.language` starts carrying the normalized feed tag
instead of the run config it carries today; and `CorpusFeedItem` gains the field, because that is what
the operator viewer's shows library consumes.

**The badge and the language filter are v2.** A language chip existed on the show page and was
deliberately removed in #2115 because every show in the corpus was English. That reasoning expires when a
second language arrives, so the chrome ships then — see
`docs/architecture/MULTILINGUAL_ARC_V2.md` §5. v1 delivers the data.

**6.2 Same-language retrieval** (PRD-047 FR4.5, D-14). A non-English episode must be findable by a query
in its own language, and by an English query through its translation. Both resolve to the same episode.

**Both layers are indexed.** Tier-1 chunks are built from the source-language transcript *and* the
English analysis transcript, both keyed to the same `episode_slug`. A Greek query matches the Greek
chunks lexically; an English query matches the English chunks.

**Non-English chunks go in their own table, with no vector column.** This is the mechanism, and the
alternative was considered and rejected:

- *Rejected:* add a `language` column to the existing `segments` table and filter the vector leg to
  English. Both legs share one `where` clause today, so this needs per-signal filters in the query
  contract — and it is a rule every future query path must remember. Worse, the natural implementation
  stores a placeholder vector, and a **zero vector sits at L2 distance exactly 1.0 from any unit-norm
  query while a genuinely related document sits further away** — so those rows would rank *above* most
  real results on every query. The failure is not subtle.
- *Chosen:* a separate table holding non-English chunks with no `embedding` column at all, consulted only
  for the keyword leg. A row with no vector **cannot** appear in a semantic result — structural rather
  than filtered. It also leaves the existing `segments` table untouched, which should mean no
  `LANCE_SCHEMA_VERSION` bump and therefore no stale index, no `no_index` outage and no full rebuild of
  the existing corpus. **Confirm both properties in the slice** — that the version constant stays put,
  and that the read path tolerates the new table being absent on older indexes.

**Four details that are easy to miss:**

- **Chunk ids must carry language.** `f"{episode_id}_chunk_{i}"` (`search/segments.py:47`) and
  `chunk:{scope_tag}:{i}` (`indexer.py:449`) carry none, and Lance merges on id — so suffix the
  non-English ids and leave the English ones byte-identical, which avoids invalidating stored
  `source_segment_id` references.
- **Insight→segment linking must be language-aware.** The primary linker matches by *text*
  (`two_tier_indexer.py:704-714`), which is language-safe by nature; the time-based fallback
  (`search/segments.py:59-80`) is not, and both layers share timestamps, so an English insight could
  link a Greek chunk.
- **A non-English query must drop the dense leg.** Otherwise MiniLM embeds it into a meaningless vector,
  the dense leg returns English neighbours, and RRF fuses that noise 1:1 with the real keyword hits.
  Script detection on the query string is enough — one regex, no model.
- **The transcript-lift path matches by char-range overlap** (`search/transcript_chunk_lift.py:268-320`)
  with no coordinate-space check, so a source-language chunk could be "lifted" as evidence under an
  English insight. Lift only where the chunk's language matches the analysis language.

**What does not change: the embedding model.** Cross-lingual *semantic* matching — an English query
finding Greek content by meaning — is explicitly out of scope for this arc. It would require a
multilingual encoder, which re-embeds every existing English episode, changes vector dimensionality
(migration plus reader-support bump), forces a corpus-wide reindex with cutover and rollback, and drags
along `search/insight_clusters.json`, `kg/topic_clustering`, `search/query_router` and
`search/quality_metrics.py`'s hardcoded `_ZERO_VECTOR_DIM = 384` — all gated on a prod-derived eval set
that does not exist. It belongs with internationalizing the application, when a listener can search in
whatever language they choose. See the arc notes, "Explicitly not in this arc".

**Consequences of keeping MiniLM**, stated plainly:

- **Non-English chunks are keyword-only.** They are excluded from the dense stage, because MiniLM
  vectors for Greek text are noise that would otherwise surface in English semantic results.
- **Greek keyword recall is weaker than English-on-English.** `create_fts_index("text", replace=True)`
  (`search/backends/lancedb_backend.py:266`) builds one English tokenizer per index — stemming,
  stop-words, accent folding — and Greek is heavily inflected. Quantify the cost before calling this
  done; if it is bad, the options are a per-language FTS table or accepting it explicitly.
- **Chunk ids must carry language.** `f"{episode_id}_chunk_{i}"` (`search/segments.py:47`) and
  `chunk:{scope_tag}:{i}` (`indexer.py:449`) carry none, and LanceDB merges on id — so the second
  chunk set would overwrite the first.
- **Insight→segment linking must filter on language.** `link_insights_to_segments`
  (`search/segments.py:59-80`) links by time, and both layers share timestamps, so an English insight
  would otherwise link a Greek chunk.

**Query side.** No query-language detection and no query translation. A query goes to the index as it
is, and hybrid RRF (RFC-090) fuses the hits. Results carry the matched chunk's `language` so a hit can
say whether it came from the original or the translation, and `search_corpus` gains an optional
`language` filter.

### 7. Model selection: bake-off and per-language gate

Reuses the existing bake-off pattern (`config/profiles/bakeoff_*.yaml`, 15 profiles today), one profile
per candidate. The shortlist and its licences are **verified** — see the arc notes §6.1; they are not
open questions any more. Availability of the QE candidates remains unverified and is not on the v1 path
(RFC-125 §7).

**Eval set per language:** 2–3 real episodes of the pilot language plus one Spanish or Italian control,
about 60 units each, including every unit the current GI pipeline cites in those episodes. Requires a
test fixture in the pilot language, which does not exist yet.

**Measurements:**

All judgements are made by the **LLM judge** under the protocol in
`docs/architecture/MULTILINGUAL_ARC_V2.md` §9 — a model from a different family than the translator,
blind to which candidate produced which output, and itself gated on a measured fault-injection detection
rate before its verdicts count.

1. **ASR sanity**: the judge rates 20 random source units usable / minor / broken. A text judge cannot
   hear the audio, so this is weaker than a native ear; cross-ASR agreement (two models, divergence as a
   proxy) is the automatable supplement, and the FLEURS prior in §6.2 of the arc notes is the floor.
2. **Cost and capacity**: latency, GPU memory, and translation wall time per episode — which is what sets
   the cap in OQ 3.
3. **Meaning preservation**: per position-bearing unit, meaning-preserving (yes / minor / no) with error
   type (negation, hedge, name, omission, other).
4. **Position agreement**: run GI extraction on the English output and compare the attributed stance
   against the judge's reading of the source.
5. **Attribution — the measurement that decides whether the product works at all.** Named-voice share of
   talk time on the pilot episodes, **against the English corpus baseline**. §5.4 explains why: with
   unnamed voices a translated episode yields zero position-bearing insights, so this number gates the
   whole value proposition, not a quality dimension. Report the naming path each name arrived by (feed
   metadata, `known_hosts`, English cue on the translated text, NER on the translated description) so it
   is clear which layer is carrying the load.
6. **Naming-cue survival**: the fraction of source self-introductions and guest hand-offs that emerge as
   English text the existing cue matchers recognise. The exact analogue of measurement 7, and the thing
   §5.4's design rests on.
7. **Ad-detection survival, measured relative to the English baseline.** Not "what fraction of ad
   characters get excised" in absolute terms — English detection has its own gaps (**issue #2168**: the
   scan windows cover only ~5 minutes at each end, and mid-roll cuts are bounded by the pattern hits with
   no expansion), so an absolute number would look acceptable while both paths were weak. Measure the same
   content English-vs-translated and report the delta, so the number means *translation did not make it
   worse*.

**Gate** (per language; all must pass to set `enabled: true`): ASR ≥ 90% usable-or-minor; ≥ 90% of
position-bearing units meaning-preserving; ≥ 90% position agreement; named-voice share within a stated
margin of the English baseline; naming-cue and ad-detection survival reported with thresholds set from the
first language measured.

The chosen model is recorded in `config/languages.yaml` with its pinned revision, and one model serves
every enabled language. **It runs as its own service** — a dedicated vLLM alongside the existing Whisper
and diarization services, intended to be reusable for translation beyond this project. That fixes the size
class: it must co-reside with the LLM already served, so the **12B tier** rather than a 27B.

### 8. Interaction with transcript prefix caching (RFC-115)

`cache_transcript_prefix` embeds the analysis transcript as the leading, stage-invariant block of the
system prompt (`config.py:3071`, `prompting/megabundle.py:59-72`). Switching that transcript from source
to English is transparent to the mechanism. Two consequences are not:

- A re-translation (model revision bump, or RFC-125's cross-model re-translation) changes the English
  text and invalidates the whole episode's cached prefix, not just the changed span. Re-translation is
  an episode-level cost.
- Translated episodes reach the LLM stages with a different token distribution, so cache-hit dashboards
  should be read per language rather than in aggregate.

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

`adfree.built_on` is the field that would have made hazard 4 visible. Phase 0's own failure modes —
`skipped_unsupported_language`, `deferred_quality_floor`, `language_mismatch` — need log and metric
surfacing plus a runbook entry, or the phase that exists to remove silent failures introduces three.

Grafana: translation wall time, units/sec, pending backlog, failures by language, and ad-chars-removed
by language.

## Key Decisions

1. **Translate once, to English, and analyze English.** Every downstream layer stays single-path, and
   the evals, prompts and thresholds already exist for English.
2. **The source is canonical; English is derived.** Grounding and provenance mean the record is what
   was said.
3. **Translation runs directly after transcription and diarization, before summary.** Ad detection
   needs English, and summary output feeds GI/KG topic identity — translating later would give English
   insights on Greek topics and fragment cross-episode identity by language.
4. **One resolver decides the analysis transcript, and routing every reader to it is part of the
   work.** An earlier version of this decision claimed `load_processing_transcript` was already that
   resolver and consumers would not change; that was its docstring, not the code (§2.1).
5. **Span→unit resolution goes through `.en.adfree.segments.json` and `unit_id`, not the ad-map.** The
   ad-map cannot invert an ad-free transform that re-renders survivors; measured (§4.1).
6. **Speaker naming stays before translation.** Naming is baked into the source `.txt`; moving it later
   means a relabel that merges turns and invalidates the unit map.
7. **Defer, don't substitute**, on translation-model availability.
8. **Speaker labels bypass the translator and must still look like people.**
9. **Declared language routes; detection only warns.**
10. **The language override is a feeds-spec allowlist key**, not a new config surface.
11. **Same-language retrieval only; no embedding-model change.** Indexing both layers satisfies the
    requirement; cross-lingual semantics belongs with application internationalization.
12. **Blocking analysis is an explicit gate on language + translation status**, because the resolver
    always returns something and absence cannot express refusal.

## Alternatives Considered

1. **Whisper's built-in `translate` task.** One pass, no new model — but large-v3-turbo was not trained
   on translation data, it loses the source transcript entirely, and it cannot be scored against a
   source. Rejected: quality and provenance.
2. **Multilingual analysis: extract from source-language text.** No translation error inside
   extraction — but it forks every prompt and eval per language, breaks English-only QA/NLI/embedding
   assumptions, and GIL verbatim grounding would produce non-English quotes the product cannot display
   coherently. Rejected as the primary path; **kept as a verification method** in RFC-125.
3. **Translate the whole transcript as one document.** Maximum context, but the output cannot be
   aligned back to turns or times and one hallucination can corrupt long spans. Rejected.
4. **Ad-free first, then translate the ad-free text.** Matches the English pipeline's shape — but
   `_AD_PATTERNS` is English, so the source ad-free base is an identity copy and ads enter analysis
   silently; it also needs a second translation pass for full-timeline subtitles. Rejected.
5. **Translate after summary, leaving the stage order alone.** Smaller diff — but topics would be
   Greek while insights are English, fragmenting cross-episode identity. Rejected (KD 3).
6. **Per-language ad-cue lexicons, keeping ad-free before translation.** No reordering, but a new
   hand-maintained lexicon per language, unmeasurable until each language has a corpus. Rejected for
   v1; §7's survival metric decides whether it is ever needed.
7. **A per-episode `translation_pending` queue.** Correct long-term shape, but the pipeline has no
   asynchronous per-episode state machine and building one is larger than this feature.
   `translate_only` reprocess covers retries.
8. **A multilingual embedding model for cross-lingual search.** Rejected for this arc (KD 11).
9. **Cloud MT API** (DeepL / Google Translate). Strong quality, no GPU — but content leaves the
   infrastructure, per-character cost at corpus scale, and the model can change underneath
   (unpinnable, against ADR-155). May be a **reference** in the bake-off only.
10. **Auto-detect language per episode.** Rejected: false detections on intros and ads. Warning only.

## Testing Strategy

- **Unit**: language-tag normalization (`el-GR`, `sr-Latn-RS`, `pt_BR`, junk) and resolution precedence
  including the feeds-spec override; the registry skip path; the quality floor and per-call model
  resolution; the lint rule that `cfg.language` has one reader; unit packing (never crosses turns, never
  splits sentences); `translation.json` ↔ `.en.segments.json` offset consistency; `resolve_units_for_span`
  across the segment→unit→source chain, including spans crossing a unit boundary and spans adjacent to
  an excised ad range; the `en_sha256` provenance refusal; the transliteration `_looks_like_person`
  guard.
- **Integration**: a fixture episode in the pilot language through transcribe → diarize → naming →
  turns → translate (stub translator) → English render → ad-free-on-English → summary → GI, asserting
  that `EvidenceSpan.transcript_ref` names `…en.adfree.txt`, that offsets resolve into that text, that
  spans map back to source times, and that topic labels are English. Include an injected English
  sponsor read in the translated output to assert the ad-free base excises it.
- **Refusal**: a non-`en` episode with `translation_pending`, and one with `.en.txt` present but
  `.en.adfree.*` missing, both assert that summary/GI/KG do not run and that no `transcript_ref` names
  a source-language file with English offsets.
- **Reprocess**: `relabel_only` on a translated episode invalidates `.en.*` and `translation.json`
  rather than stranding them, and writes no source-language identity ad-free artifact.
- **Contract**: `SegmentsResponse` with and without `lang`, on English and translated episodes.
- **Search**: both chunk sets present with distinct ids, no insight linked across languages,
  non-English chunks absent from dense results, and a measured Greek keyword-recall number.
- **Isolation**: the English fixture corpus produces artifacts identical outside the declared
  allow-list of added metadata keys, with the flag on and off.
- **Regression**: an episode resolving to `el` can never ship `"language": "en"`.
- **Bake-off**: the §7 harness committed as profiles plus a scoring script; reviewer sheets exported as
  CSV; results committed as an eval report.

## Rollout & Monitoring

Phase names match PRD-047 and the arc notes; slice ids (S0.x, S2.x) refer to the arc's §4.

- **Phase 0 — English as a declared language (S0.1–S0.11), ships first and alone.** Feed-language
  parsing and persistence with its migration, normalization and resolution, the feeds-spec override, the
  corpus audit, episode language on the API, the badge, the one-reader lint, model selection from the
  resolved language, the skip path and language-ID check, and observability. Precedes **Gate V**,
  because the bake-off cannot measure non-English transcription until it exists.
- **Gate V — Validate.** The Serbian investigation (V.5, running now), demand interviews, the §7
  bake-off and its gate report, and the pilot-language fixture.
- **Phase 2 (S2.1–S2.12)**: reader routing, the stage-order change, the translation stage, the English
  render, ad-free-on-English, naming, reprocess invalidation, the segments API and the translated chip,
  same-language retrieval, cost measurement, and the trust markers — behind `multilingual_ingest`,
  which gates **serving** as well as the pipeline, on one operator-chosen feed.
- **Phase 3**: source verification and the operator worklist. (The read-time Positions gate ships in
  Phase 2 with the markers.)
- **Phase 4**: language toggle, original-text reveal, language filters, flag lifecycle and rollback.

**Success criteria:**

1. At least 95% of episodes in enabled languages reach `translation_status: ok` without manual
   intervention.
2. Zero English-episode regressions outside the declared allow-list.
3. Every GI quote on a translated episode resolves to source text and source audio.
4. `adfree.ad_chars_removed` on translated episodes is non-zero at a rate comparable to English
   episodes of similar shows, or §7's survival number explains why not.
5. An episode is reachable by a query in its own language and by an English query.
6. No translated content is served without a translation label at any point.

## Relationship to Other RFCs

- **RFC-123** supplies the units and ships first on its own merits. Its schema change to the LanceDB
  segment table should land together with §6.2's `language` column.
- **RFC-125** consumes `translation.json` and adds per-claim provenance and source verification,
  calling this RFC's `resolve_units_for_span`. QE is its v2 addition, not a v1 dependency.
- **Positions** are a read-time CIL query; nothing in that path changes here. RFC-125 gates what it
  returns.

## Open Questions

1. Code-switching: when a unit's language ID disagrees strongly with the declared language, pass it
   through untranslated? Cheap to detect per unit after the fact.
2. Serbian script — **a capability question, not a display one** (arc §6.3): transcribe Cyrillic
   as-is, force Latin output, or romanize post-hoc? It affects WER, MT coverage and CIL identity at
   once. V.5 settles it.
3. The GPU budget per translated episode is unknown until the bake-off. A translation wall-time cap
   (for example ≤ 50% of transcription wall time) as a gate criterion, given `relabel_only` already
   costs ~17 min/episode and runs near-serial?
4. Do we keep a source-language ad-free variant at all, derived by mapping the English ad ranges back
   through `translation.json`? It has a reader/player use, no analysis one.
5. Should English episodes eventually route through the same resolver branch to pick a cleaned variant,
   unifying this with `save_cleaned_transcript`? The resolver generalizes, so this is a config question.
6. How much Greek keyword recall does the English FTS tokenizer cost (§6.2)?

## References

- `src/podcast_scraper/rss/parser.py`, `models/entities.py:16-44` — where `<language>` is **not** parsed
- `src/podcast_scraper/rss/feeds_spec.py:26-56,77` — per-entry mapping + `RSS_FEED_ENTRY_OVERRIDE_KEYS`
- `src/podcast_scraper/workflow/metadata_generation.py:957,2843,3832` — the `cfg.language` writes
- `src/podcast_scraper/server/corpus_catalog.py:159` — `_feed_language` (reads the config back)
- `src/podcast_scraper/server/schemas.py:1719` — `AppPodcastItem.language`
- `src/podcast_scraper/providers/tailnet_dgx/whisper_provider.py:197,429-430` — result dict vs request
- `src/podcast_scraper/providers/ml/ml_provider.py:566-568,827,885,1061` — init-time model choice, naming
- `src/podcast_scraper/providers/ml/whisper_utils.py`, `config_constants.py:332,610` — fallback chain
- `src/podcast_scraper/workflow/episode_processor.py:2296,2869,3191,3784,4460` — call site + reprocess paths
- `src/podcast_scraper/workflow/sniff_gate.py:63-70,116,130,144,177` — hazard 5 + four language sites
- `src/podcast_scraper/workflow/processing_manifest.py:54` — `CANONICAL_STAGE_ORDER`
- `src/podcast_scraper/workflow/stages/transcription.py:266` — summary in the same job
- `src/podcast_scraper/workflow/adfree_transcript.py:52-55,104-137,176-188,230-261` — paths, identity base, writes, resolver
- `src/podcast_scraper/gi/ad_regions.py:409-418,439` — no-match return, `_shift_for`
- `src/podcast_scraper/gi/filters.py:35-77` — `_AD_PATTERNS`
- `src/podcast_scraper/gi/load.py:19-37`, `gi/repair.py:161-171`, `search/indexer.py:96-111` — the independent resolvers
- `src/podcast_scraper/gi/pipeline.py:810-830` — span→segment resolution this design reuses
- `src/podcast_scraper/gi/speakers.py:74-77` — `_looks_like_person`
- `src/podcast_scraper/providers/ml/diarization/formatting.py:81` — the passthrough tuple `unit_id` joins
- `src/podcast_scraper/search/segments.py:47,59-80`, `search/indexer.py:449`, `search/backends/lancedb_backend.py:266` — chunk ids, linking, FTS
- `src/podcast_scraper/speaker_detectors/ner.py:131-136` — default-model selection, not a NER gate
- `config/profiles/prod_dgx_full.yaml:70,111,112` — language, fallback providers
