# ADR-159: Recover untranscribed speech by re-transcribing the gap through a one-request clip call

- **Status**: Accepted
- **Date**: 2026-10-08
- **Authors**: Marko Dragoljevic
- **Issues**: [#2187](https://github.com/chipi/podcast_scraper/issues/2187)
- **See Also**: [RFC-124](../rfc/RFC-124-multilingual-transcription-and-translation.md) §2.2,
  [ADR-122](ADR-122-self-hosted-model-resilience-policy.md),
  [ADR-131](ADR-131-speech-normalized-coverage-gate.md),
  [ADR-160](ADR-160-remove-whole-segment-invented-lines.md)

## Context & Problem Statement

Whisper's long-form decoding can skip a stretch of speech outright. On 80k_03 (raw audio) its
segments jump from 23.4 s to 48.8 s while one speaker talks throughout, losing 77 words; the V.6b
Spanish fixture lost a 19 s ad read. The diarizer hears those stretches, and the same audio
transcribes when it is cut out and sent alone.

Two questions: how to find the loss, and how to get the words back without harming the rest of
the pipeline.

## Decision

1. **Detect (A1).** `untranscribed_speech`: every stretch of diarized speech of 3 s or more that
   no ASR segment covers, recorded in `.asr.json` (`untranscribed_speech`) and flagged
   (`asr_untranscribed_speech`).
2. **Recover (A2).** Each gap is cut with 1 s of context either side, transcribed in the episode's
   language, and the words whose midpoint lies inside the gap are spliced in as segments tagged
   `recovered: true`, before speaker alignment. A recovered segment must pass: avg_logprob ≥ -1.0,
   compression ratio ≤ 2.4, no repeated short phrase, and not an invented line (ADR-160).
3. **Through a clip call, never the episode path.** `transcribe_clip` makes ONE request under the
   DGX single-flight lock, with no response guardrail, no retry policy, no breaker and no
   punctuation retry. A provider without it gets no recovery. Off-switch:
   `transcription_recover_untranscribed_speech`.
4. **Stretched words are evidence, not gaps.** A word timed 3 s or longer over diarized speech is
   recorded (`stretched_words`), never counted, flagged or recovered.

## Consequences

- **Positive**: on the six V.6b real feeds (2026-10-08) 57 of 57 gaps were recovered (1,295
  words), no clip failed, and 0 of 55 seams repeat the neighbouring word. On the French feed the
  recovered speech includes the host's self-introduction, which invented credit lines had covered.
  On the 80k English set: 80k_03 raw WER 10.63% → 8.38% and its 77-word deletion gone; 80k_06 raw
  9.07% → 8.60%; 80k_06 preprocessed 8.81% → 8.55% (15 words remain under a stretched word).
  Episodes with no gap are untouched.
- **Negative**: one extra ASR request per gap. Recovered speech can be an inserted ad in another
  language (the V.6b French download carried a Dutch ad); ad handling stays downstream.
- **Neutral**: `no_speech_prob` is not used: the DGX server returns 0.0 for every segment, so it
  cannot tell speech from silence there. The diarizer's turn is the speech evidence.

## Alternatives Considered

1. **Recover through `transcribe_with_segments`.** Built first, and measured on the production
   profile (HOLD): a 5.5 s clip with 3 words — a real answer for a clip — failed the
   episode-scale length floor (6 words), was retried 3 times, opened the breaker, then held the
   process-wide DGX lock through 900 s of pause-and-probe and raised `ResilienceFuseOpenError`
   after 1,068 s with an "alerting operator" log. Every other episode in the worker waits on that
   lock. Rejected.
2. **Treat a stretched word as a gap and recover it.** Re-transcribing 11 such spans (80k_06, one
   V.6b Italian, nine prod) found lost speech under 3, only the word / a stutter / fillers under 6,
   and 2 unclear. Reporting them as gaps was wrong 6 times in 11, and a splice would duplicate
   the word, or put real lost speech in the wrong place (the word's true position in its span is
   unknown). Rejected; kept as evidence.
3. **Re-transcribe the whole episode on another model when coverage is low** (ADR-131's
   failover). Kept for its own case; too expensive to spend on a 4 s gap.
