# ADR-160: Remove the lines Whisper invents, by whole segment only

- **Status**: Accepted
- **Date**: 2026-10-08
- **Authors**: Marko Dragoljevic
- **Issues**: [#2187](https://github.com/chipi/podcast_scraper/issues/2187)
- **See Also**: [RFC-124](../rfc/RFC-124-multilingual-transcription-and-translation.md) §2.2,
  [ADR-159](ADR-159-recover-untranscribed-speech-through-a-clip-call.md)

## Context & Problem Statement

Whisper learnt from subtitled video, so over music, silence or speech it cannot follow it can emit
the credit line that ended its training subtitles. The V.6b French episode (2026-10-08) opened with
nine segments of "Sous-titrage Société Radio-Canada" while the diarizer heard the guest — and
re-transcribing those stretches returned real French speech, the host's self-introduction among
it. The Spanish episode ended on "Gracias por ver el video." The prod snapshot of 2026-09-20 has
the same lines inside English episodes ("Untertitelung im Auftrag des ZDF,", "Sous-titrage Société
Radio-Canada .").

Neither existing check catches them: each is said once per segment (no loop), at avg_logprob -0.1
to -0.5 (confident).

## Decision

After every fresh ASR result (primary, chunked, failover), a segment whose **whole** text — case,
accents and punctuation folded — is one of a fixed list of credit lines and video sign-offs, or a
dated ZDF credit, is removed (`invented_lines.drop_invented_lines`). The removed lines are recorded
in `.asr.json` (`invented_lines_removed`) and flagged (`asr_invented_lines_removed`). Gap recovery
rejects them from clips too, judged on the whole clip segment.

Their time is then uncovered, so where the diarizer heard speech it becomes a gap and is
recovered like any other (ADR-159).

## Consequences

- **Positive**: on the prod snapshot (2,628 transcripts) the list matched 2 segments, both
  credits inside English episodes; on the six V.6b real feeds, 11, all invented. The French
  opening's real speech comes back through recovery.
- **Negative**: a real speaker who says exactly "Thank you for watching." as a whole Whisper
  segment loses those words. The one prod candidate was a real sign-off at the end of a longer
  Whisper segment, so it stays. **Accepted by the operator, 2026-10-09**: an invented line can
  reach a summary, an insight or a quote; a lost whole-segment sign-off reaches nothing.
- **Neutral**: the list grows by observation, not by guessing; it holds the observed lines, the
  Amara.org community credits, and the common sign-offs.

## Alternatives Considered

1. **Substring match.** Rejected: real speech contains these words — a German sentence about a
   copyright question on the V.6b feed, and sign-offs inside longer turns on the prod snapshot.
2. **Use `no_speech_prob`.** Rejected: the DGX server returns 0.0 for every segment.
3. **Lower the confidence floor.** Rejected: the invented lines are confident (-0.1 to -0.5), so
   any floor that catches them drops real speech first.
