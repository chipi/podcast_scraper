# ADR-161: Judge punctuation per 10-minute window, and repair a broken window by re-transcribing it

- **Status**: Accepted
- **Date**: 2026-10-08
- **Authors**: Marko Dragoljevic
- **Issues**: [#2187](https://github.com/chipi/podcast_scraper/issues/2187),
  [#2284](https://github.com/chipi/podcast_scraper/issues/2284)
- **See Also**: [RFC-124](../rfc/RFC-124-multilingual-transcription-and-translation.md) §2.2,
  [ADR-159](ADR-159-recover-untranscribed-speech-through-a-clip-call.md)

## Context & Problem Statement

The DGX Whisper server decodes long audio window by window, each 30-second window conditioned on
the previous one's text. #2284 found transcripts unpunctuated from the first word (121 of 2,378 on
prod), detected them on the whole text and fixed them with an English prompt on retry.

V.6b (2026-10-08) found the same failure starting **part-way**: punctuation falls from 35-67
sentence ends per 1,000 words to 0-3 after minute 10-40 and stays there. Five of the six real
feeds did it (es, it, fr, de, pt-PT), and so do **121 of 2,421 prod episodes of 20+ minutes**
(English; prod snapshot 2026-09-20), 90 of them from the break to the end. The whole-episode
figure still passes (10-30 per 1,000), so nothing noticed — and the retry prompt is English-only.
Everything that needs sentences downstream (cleaning, quotes, translation units, summaries)
degrades on the broken part.

## Decision

1. **Judge per window.** `punctuation.unpunctuated_windows`: each 10-minute window is judged with
   the existing `is_unpunctuated` (≥ 300 words, mostly Latin script, < 5 sentence ends per 1,000
   words); a window list is reported only when the whole text passes.
2. **Repair a broken window on its own.** Cut it at segment edges, transcribe it through the
   provider's one-request clip call (ADR-159) with a neutral punctuated prompt **in the episode's
   language** (`WINDOW_PROMPTS`: en, es, it, fr, de, pt), and replace the window's segments only
   when the new decode:
   - is punctuated and does not echo the prompt;
   - has no segment over Whisper's loop threshold (compression ratio 2.4);
   - keeps at least 90% of the old window's words (the same speech, nothing dropped);
   - carries at most 1.3× the old words **where the old decode had text**, and at most 4.5
     words per second **where it had none**.

   Otherwise the window is kept as it was. Invented lines (ADR-160) are removed again afterwards:
   a repaired window is a fresh decode.
3. **Record.** `.asr.json`: `punctuation_repair` (each attempt, before/after) and
   `unpunctuated_windows` (what is still broken). Manifest: `asr_punctuation_repaired`,
   `asr_partly_unpunctuated`. Off-switch: `transcription_repair_unpunctuated_windows`.

## Consequences

- **Positive**: measured through the production function and provider (2026-10-08):
  - five V.6b feeds (es, it, fr, de, pt-PT): **25 of 25** broken windows repaired, none left;
    e.g. pt-PT 4.2 / 1.6 / 2.7 → 74.5 / 83.0 / 99.5 sentence ends per 1,000 words; each repair
    kept 91.8-98.3% of the old window's words.
  - two English prod episodes: **9 of 10** repaired, each carrying 22-55% more words — filling
    the broken decode's empty seconds — while keeping 97.9-99.5% of the old words.
  - the 80k English set flags no window, so it is never called and is untouched.
  The repair also brings back speech the broken decode dropped (German minutes 40-50: +25%).
- **Negative**: one extra ASR request per broken window — 1-3 minutes each on the DGX as
  measured — only on episodes that break (~5% of prod by the snapshot). A repaired window is a
  fresh decode and Whisper's output there is not deterministic: the one refused English window
  carried a looping segment in one decode and none in the next, so it stays as it was.
- **Neutral**: a language without a window prompt is detected and recorded, not repaired.

## Alternatives Considered

1. **Prompt the whole file in the episode's language.** Measured on pt-PT: windows still broken
   (53.3, 0.6, 3.6, 5.5, 2.0 per 1,000 by window) — the prompt shapes only the first window.
2. **`vad_filter`.** Measured on pt-PT: the break moved from minute 10 to minute 30 (62.6, 60.3,
   66.4, 1.6, 1.9) but still came. Rejected as a fix.
3. **Turn off conditioning on previous text.** The DGX server's API does not expose it
   (`/openapi.json`: file, hotwords, language, model, prompt, response_format, stream, temperature,
   timestamp_granularities, vad_filter).
4. **Judge the whole episode only (#2284 as it was).** Misses every partial break above.
5. **Bound the repair by total word count (90-130%).** Built first. It held on all 25 non-English
   windows (2-25% more words) and refused 9 of 10 English windows: on an English prod episode the
   broken decode had also DROPPED speech — the old windows covered 377-417 of 600 s with 18-24 s
   holes, and the re-decode filled them at speech rate (35-88 words per hole) while keeping
   98-99.6% of the old words. A total-word cap cannot tell that from invention; the old decode's
   coverage can.
6. **Gap recovery's short-phrase loop rule** (`is_repetitive`). Refused two good Spanish windows
   over one ordinary filler run each ("bla, bla, bla, bla", compression ratio 1.6) out of 113
   segments. A loop over a window shows in the compression ratio and the coverage bounds.
