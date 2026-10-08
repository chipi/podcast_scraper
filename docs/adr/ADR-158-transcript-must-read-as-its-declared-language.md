# ADR-158: A transcript that does not read as its declared language fails the episode

- **Status**: Accepted
- **Date**: 2026-10-08
- **Authors**: Marko Dragoljevic
- **Issues**: [#2187](https://github.com/chipi/podcast_scraper/issues/2187)
- **See Also**: [RFC-124](../rfc/RFC-124-multilingual-transcription-and-translation.md) §2.2,
  [ADR-159](ADR-159-recover-untranscribed-speech-through-a-clip-call.md)

## Context & Problem Statement

The pipeline sends each episode's declared language (the feed's `<language>`, or an operator
override) to ASR. Whisper honours it. Spanish audio requested as `en` came back as **correct
Spanish text** with `language: en` reported, because the service echoes the request (V.6b
fixture p10_e01, 2026-10-07). The `asr_language_mismatch` flag (requested vs reported) therefore
cannot fire for the case that matters: a feed whose tag is wrong would send source-language text
through every English stage — cleaning, summary, GI, KG, search — labelled English.

The text is the only witness left.

## Decision

Right after ASR, before diarization and before anything is written, the transcript is read with
the function-word counts `ad_signatures` already runs over the corpus
(`languages_guard.transcript_language_contradiction`). The episode **fails** when:

- the text confidently reads as another language (`detect_language`), or
- the declared language's own function words are under 2% of its words (this half catches
  Portuguese, whose thin word list leaves `detect_language` undecided on 2 of 3 fixtures).

It has no opinion under 200 words or for a language with no word list. The failure is an error
log, a hard incident and a ledger row (`TranscriptLanguageMismatch`), like every other refusal.
The reason names the remedy: an operator language override (#2283), then re-run.

**No auto-correction.** The guard does not re-transcribe in the language the text reads as.

## Consequences

- **Positive**: a mislabelled feed is loud on its first episode instead of polluting the corpus.
  On the six V.6b real feeds (es, it, fr, de, pt-PT, pt-BR; 2026-10-08) every episode passed:
  the declared share was 9.6-21.3%, against the 2% floor.
- **Negative**: an episode that is genuinely bilingual, or mostly music with a little speech in
  another language, can be refused. That is the intended trade: an operator decision instead of a
  silent wrong label.
- **Neutral**: languages without a function-word list pass unjudged; adding a list turns the
  guard on for them.

## Alternatives Considered

1. **Trust the ASR's reported language.** Rejected: measured to echo the request.
2. **Auto-correct**: re-transcribe in the detected language. Rejected by the operator
   (2026-10-07): a wrong tag is a feed fact someone should look at, and an automatic second
   transcription doubles ASR cost on exactly the episodes whose metadata is already wrong.
3. **Detect the language from audio first and never trust the tag.** Rejected for now: the feed's
   tag is right on every measured feed, and audio language-ID adds a call to every episode to
   catch the rare one.
