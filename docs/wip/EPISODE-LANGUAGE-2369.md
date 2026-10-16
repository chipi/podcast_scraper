# An episode in another language than its show (#2369)

Design note, 2026-10-16. Nothing is built. Direction from the operator: the show keeps its language
setting; an episode may be in another language; the check must be cheap, and episodes in the
show's language must pay almost nothing for it. Every episode is assumed to be in one language
(an episode split between two languages is out of scope). An episode we cannot handle is refused
and skipped, as today; no new publishing status is added.

## What was measured

Prod snapshot `snapshot-prod-20261015`, 2,684 served episodes (newest run each).

- The metadata cannot answer it: every feed declares `en`, and the episode block inherits the
  feed's language (`metadata_generation.py`, "S0.1a has no per-episode override"). RSS has no
  standard per-episode language tag, and the feed here has none; this publisher labels items with a
  `<category>Türkçe</category>`, which is its own convention.
- From the transcript text: **2 of 2,684 episodes are not English.** Both are Turkish, from
  Ottoman History Podcast ("Soykırımın Bürokrasisi", 2026-05-16; "Cümle Fener Burada", 2026-09-30),
  host Can Gümüş, with Turkish titles and descriptions. The live feed labels 5 items `Türkçe`; the
  other 3 were not checked against the corpus.
- The ASR wrote correct Turkish even though `en` was requested (the service echoes the requested
  language; see `languages_guard.py`). "Cümle Fener Burada" also names a voice after the feed's
  English-series host (Chris Gratien, ASR "Chris Grayton"), who is not in it (#2352).
- Both were ingested before the language guard existed (2026-09-12 and 2026-10-02). Today's guard
  flags both (0.0% and 0.7% English function words; a correctly declared episode measures at least
  5%), so on current code they would be refused, not served.
- English function-word share, our word list: English episodes lowest 1% 0.321, median 0.395; the
  2 Turkish 0.014 / 0.021; the 15 v3 fixtures (es, pt, fr, de, it) 0.007-0.074.
- Picking WHICH language by word lists fails between close languages: Portuguese fixtures read as
  Italian and Italian as Portuguese (es, fr, de, tr right).

## What exists

- **Detection, right after ASR:** `languages_guard.transcript_language_contradiction`
  (`episode_processor.py`, before diarization) on the whole transcript: at least 200 words; flags
  when the declared language's function words are under `DECLARED_LANGUAGE_MIN_SHARE` = 0.02, or
  the text confidently reads as another language.
- **Refuse and skip, on a flag:** the error is logged; a hard incident (`TranscriptLanguageMismatch`)
  and an unresolved transcript are recorded; `<corpus>/.language_refusals/<sha1(episode id)>.json`
  stores `{requested, detail, title}`; nothing else is written (no transcript, metadata or speakers),
  so the episode is absent from every serving surface. Later runs skip it before download while the
  declared language is unchanged (`_still_refused`); a changed declaration or a forced reprocess
  retries it.
- **Supported languages:** `config/languages.yaml` `enabled` = en, es, it, pt, de, fr. Punctuation
  prompts, ad patterns and naming cues cover the same six.
- **Language is per run, not per episode:** readers go through `transcription_language(cfg)`, and
  `apply_overrides.gate_languages` enforces one language per run. `cfg.model_copy(update=
  {"language_override": ...})` is the existing seam for an episode-specific language.
- The ASR's reported language is no help: the service echoes the requested one. It detects only
  when the request carries no language.

## Design

The only new step sits between the existing detection and the existing refusal.

1. **Detect (every episode, no extra cost).** The existing post-ASR check, unchanged.
2. **Identify (flagged episodes only).** Score each stretch of the transcript against the show's
   language. From stretches that clearly are NOT the show's language (so an ad, a quote or a cameo
   in the show's language is skipped), take three clips of about 30 s, spread early / middle / late,
   and send each to the DGX ASR with **no** language, so the server detects. Accept a language only
   when all three agree. Cost: three short calls on a flagged episode (2 in 2,684 today).
3. **Route.**
   - Identified and **enabled** -> the episode carries that language from here on (diarization,
     naming, metadata, translation), through a per-episode `cfg` copy; its metadata records
     `language_source: detected`. Translation runs as for any non-English show. The transcript
     already in hand is kept (the ASR wrote the spoken language).
   - Identified and **not enabled**, or the clips disagree, or detection fails -> **refuse and skip
     as today**, with the detected language (if any) added to the refusal record, so a later
     mechanism that admits such an episode knows what it is.

What a refusal costs: the transcript is not kept, so admitting the episode later means paying for
ASR again; only the rare refused episode pays it.

## How it is proven before it ships

- Detection: unchanged code; re-run over the snapshot (expect exactly the 2) and the v3 fixtures
  declared `en` (expect all 15 flagged).
- Identification: the three-clip call on the 15 fixtures and the 2 Turkish episodes (DGX, a handful
  of short calls, operator's go), every answer read.
- Routing: unit tests for each branch (enabled -> continues with the episode's language; not
  enabled / disagreement / failure -> the existing refusal, record carries the detected language).
- Replay over the snapshot: with the change, the only episodes whose outcome changes are the 2.

## Not covered

- An episode split between two languages (out of scope by decision).
- A language outside the seven word lists the guard knows: the guard still flags it (the show's
  share is low), identification then decides; untested on any real example.
- Publisher transcripts in another language (the guard runs on our ASR output; the publisher path
  was not checked).
- The 3 other `Türkçe` items in the live feed.
- An operator surface for refused episodes, and the mechanism that admits one later.
