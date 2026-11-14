---
id: w1_whisper_repeat_failures
class: light
status: open
issue: 2475
touches:
  - src/podcast_scraper/providers/guardrails/transcription.py
  - src/podcast_scraper/providers/resilience/policy.py
  - src/podcast_scraper/providers/tailnet_dgx/whisper_provider.py
---

# w1_whisper_repeat_failures — the two episodes that fail transcription every night stop failing as outages

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md), light class.

## 0. Why light

Two episodes failed transcription every nightly (issues 2475, 2458, 2428, 2429); evidence from the
mini's VictoriaLogs, the DGX's faster-whisper log, both feeds' public RSS and a replay of each
episode's public audio against the DGX on 2026-11-14:

- **c18f57c6** (ChinaTalk, "Roasting China's AI Industry (中文版)", 3816 s; failed 11-04 to 11-14).
  The episode is in Chinese under a feed that declares `en`. Whisper transcribed it in Chinese
  (replay: 21,749 characters), and the length floor, which counts `text.split()`, read that as
  1,791 "words" against a floor of 4,769 and raised a hard error before the language guard ran.
  The guard reads the same text as not English (1.4 % English function words against 5 %).
  Change: the floor counts each Chinese or Japanese character as a word. Text without those
  characters is counted exactly as before.
- **26d7aade** (Colombia Calling 639, 2678 s; failed 11-11 to 11-14). faster-whisper answered 500
  (`cudaErrorInvalidDevice` in word alignment) on every attempt and every probe while it served
  130 to 188 other episodes a night; the reprocess policy probed with the same audio, held for
  900 s each night and alerted "endpoint did not recover". A replay of the public enclosure,
  whose bytes differ from prod's archived copy (inserted ads), answered 200 with and without word
  timestamps, so the cause of the 500 is not reproduced. Change: when all attempts end in an HTTP
  error status and the server's health check passes, the input fails
  (`ResilienceInputFailedError`, a hard transcription incident) with no hold and no fuse-open alert.
  A health check that fails, or a timeout, is held for as before.

What it writes: nothing a listener or reader is shown. c18f57c6 goes from a hard failure to the
language guard's refusal (`zh` is not enabled in `config/languages.yaml`), which writes the
`.language_refusals/` record and serves nothing, as for any episode in a disabled language;
26d7aade stays untranscribed and is parked after two counted nights (`episode_park`, already in
this release). No transcript, gi, kg, bridge, segments or metadata of any served episode changes,
and no scoreboard number can move: both episodes serve nothing before and after.

Measured with unit tests that fail before and pass after:
`TestWhisperGuardrail::test_a_full_chinese_transcript_is_not_a_short_one`,
`test_a_server_error_on_one_input_fails_that_input_without_holding` (policy) and
`test_a_500_from_a_server_that_is_up_fails_the_input_without_holding` (provider, mutation-checked:
without the provider's `answered` wiring it fails). Kept unchanged by tests:
`test_latin_script_counts_words_as_before`, `test_a_short_chinese_answer_still_fires`,
`test_a_server_error_while_the_endpoint_is_down_still_holds`,
`test_a_timeout_is_an_outage_even_when_the_health_check_answers`,
`test_a_500_while_the_server_is_down_still_holds`. Both failure kinds are parked:
`test_the_other_repeat_failures_are_parked_too`.

Not covered: why faster-whisper's alignment fails on prod's archived copy of 26d7aade. A replay of
that exact file needs prod's audio archive, which this workstation cannot reach.
