# Corpus comparison — v2-cloud vs v3-dgx

Shared episodes (joined by GUID): **9**

Deterministic metrics only — no LLM judge. Summary/insight *text* quality needs a
cross-vendor judge panel (autoresearch/JUDGING.md) and is scored separately.

| metric | v2-cloud | n | v3-dgx | n | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `has_summary` | 1.0 | 9 | 1.0 | 9 | +0.00 |
| `insights` | 12.0 | 9 | 9.78 | 9 | -2.22 |
| `kg_edges` | 26.11 | 9 | 23.78 | 9 | -2.33 |
| `kg_nodes` | 27.11 | 9 | 22.11 | 9 | -5.00 |
| `kg_person_placeholders` | 0.0 | 9 | 0.0 | 9 | +0.00 |
| `kg_persons` | 0.0 | 9 | 8.44 | 9 | +8.44 |
| `n_segments` | 1067.56 | 9 | 1052.78 | 9 | -14.78 |
| `quote_attributed_pct` | 100.0 | 9 | 100.0 | 9 | +0.00 |
| `quote_ts_valid_pct` | 100.0 | 9 | 100.0 | 9 | +0.00 |
| `quotes` | 24.67 | 9 | 2.33 | 9 | -22.34 |
| `summary_chars` | 1488.89 | 9 | 1284.33 | 9 | -204.56 |
| `timeline_error_pct` | 1.92 | 9 | 0.19 | 9 | -1.73 |
| `transcript_span_s` | 4017.69 | 9 | 4100.49 | 9 | +82.80 |
| `voices` | 8.11 | 9 | 7.44 | 9 | -0.67 |
| `voices_named` | 2.67 | 9 | 6.11 | 9 | +3.44 |

## How to read the load-bearing rows

- **`timeline_error_pct`** — how far short of the audio the transcript ends. This is the
  #1173 signature: a corpus transcribed from silence-stripped audio ends early, and every
  timestamp derived from it is wrong. Near 0 is correct.
- **`quote_ts_valid_pct`** — a *range* check only: does the timestamp fall inside
  `[0, duration]`? A quote drifted by a minute still does, so this cannot see drift. It
  scored 99.6% on the corpus whose timestamps were broken. Never read it as accuracy.
- **`voices_named`** vs **`voices`** — voices resolved to a real person rather than left
  as `SPEAKER_NN`. Unresolved placeholders must never reach an entity surface (#1167).
- **`kg_person_placeholders`** — should be 0. Anything above it is a leak.
