---
id: s4_segment_char_spans_public
class: light
status: landed
issue: none
touches:
  - src/podcast_scraper/gi/pipeline.py
---

# s4_segment_char_spans_public — the char-span-to-segment mapping moves out of the GI pipeline

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md), light class.

## 0. Why light

`_segment_char_spans` and `SEGMENT_TRANSCRIPT_ALIGNMENT_MAX_DELTA` move verbatim from
`gi/pipeline.py` to the public `transcript_spans.py` (E04 S4). The search indexer times every
transcript chunk with this mapping (`search.indexer.body_timestamps`), so public needs it without
the GI stage. `gi/pipeline.py` imports both names back, so `gi.pipeline._segment_char_spans` is
the same function object for every caller that reaches it there (m0031, m0038,
`quality/corpus_scoreboard`, `quality/gold_insights`, the GI tests).

What it writes: nothing. No corpus file and no index row changes; the function's body is
identical (checked by AST against `main`), and every call returns what it returned before.

Effect on landed outcomes: the code digest of every migration whose closure reaches
`gi/pipeline.py` moves, as any edit there does; those digests already differ from `main`'s
earlier records, and a landed record is no longer compared with today's code (method v0.7).

## Landed

On prod at the latest with the core release (public d12e9acb0, release d4033b3, 2026-11-17); the
news release (085bdc27b, release d8bb835, 2026-11-23) carries it too. Code only: no corpus
change, no prod step.
Status set to landed 2026-11-24 in the record cleanup after the news release (the change gate's
records are open or landed; this one shipped).
