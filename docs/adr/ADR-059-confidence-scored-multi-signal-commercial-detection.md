# ADR-066: Confidence-Scored Multi-Signal Commercial Detection

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-060](../rfc/RFC-060-diarization-aware-commercial-cleaning.md)

## Context & Problem Statement

The current sponsor detection (`remove_sponsor_blocks`) uses four hardcoded English
phrases and deletes text until the next blank line or 2000 characters. This fails when
hosts paraphrase ads, weave sponsor messages into conversation, or use any phrasing
outside the four patterns. Commercial content leaking into summaries, GIL quotes, and
KG topics undermines the pipeline's core value.

ADR-020 established a single place for preprocessing (provider-agnostic cleaning), but
the detection mechanism itself was never formalized as an architectural decision.

## Decision

We replace binary pattern matching with a **confidence-scored multi-signal detection
architecture** via a consolidated `CommercialDetector`.

1. **Confidence-scored candidates**: Each candidate sponsor segment receives a float
   confidence score from combined signals. A configurable threshold (default 0.65)
   determines removal. This replaces the binary "match or not" approach.
2. **Pattern matching is primary, diarization adjusts confidence**: Expanded text
   patterns (categorized as intro/body/CTA/outro with base confidence scores) and
   positional heuristics (ad-break windows at 0–15%, 35–65%, 80–100%) form the primary
   detection layer. When diarization data is available (RFC-058), it provides confidence
   boosts (host monologue in mid-episode) or penalties (guest speaking during
   candidate → disqualify).
3. **Precision over recall**: Default threshold is tuned for >= 90% precision even at
   the cost of some recall. Removing genuine content is worse than leaving an ad in —
   summarization prompts provide a second line of defense.
4. **Consolidated `CommercialDetector`**: A single implementation replaces duplicate
   `remove_sponsor_blocks` in `preprocessing/core.py` and `providers/ml/summarizer.py`.
   All sponsor detection flows through `cleaning/commercial/detector.py`.
5. **Precise boundary detection**: Searches backward from match for a `block_start`
   pattern and forward for a `block_end` pattern, replacing the crude "delete to next
   blank line or 2000 chars" heuristic.

## Rationale

- **Confidence scores** make the system tunable (adjust threshold), auditable (log
  scores for debugging), and extensible (add new signal layers without rewriting
  detection logic). Binary detection offers none of these properties.
- **Pattern-primary**: Patterns work without diarization, without audio, and across
  all transcription providers (local Whisper, OpenAI, Gemini, Mistral). Diarization is
  valuable but optional.
- **Precision-first**: Ad leakage into summaries is annoying; removing real content
  destroys trust. The asymmetry justifies a conservative threshold.
- **Consolidation**: Two implementations of the same logic in different files is a
  maintenance and correctness hazard. One `CommercialDetector` eliminates the
  duplication.

## Alternatives Considered

1. **LLM-only detection**: Rejected as default; too expensive (full transcript per
   episode), too slow. Kept as hybrid fallback for uncertain candidates.
2. **Audio-level jingle/music detection**: Rejected; many host-read ads have no audio
   cue. Requires audio analysis ML, adds heavy dependency, doesn't solve the main
   problem.
3. **Trained classifier on labeled ad segments**: Rejected; insufficient labeled data
   currently. The confidence-scoring architecture can evolve into a learned model later
   if training data accumulates.
4. **Keep binary four-phrase detection**: Rejected; fails on paraphrased ads, woven-in
   ads, and any phrasing outside the four patterns. Boundary detection is crude.

## Consequences

- **Positive**: Measurably better sponsor detection. Summaries and GIL/KG outputs
  cleaner. Works across all transcription providers. Confidence scores enable tuning
  and auditing.
- **Negative**: Expanded pattern library and positional heuristics add code complexity
  compared to four hardcoded phrases. Threshold tuning requires evaluation on real
  podcast episodes.
- **Neutral**: `cleaning/commercial/` module added. Brand name list is extensible via
  config. The `clean_for_summarization()` API gains an optional `diarization_result`
  parameter for Phase 2 (backward compatible — `None` gives pattern+position mode).

## Implementation Notes

- **Module**: `src/podcast_scraper/cleaning/commercial/` — `detector.py`,
  `patterns.py`, `positions.py`, `diarization_signals.py`
- **Pattern**: `CommercialDetector` with `detect()` (returns scored candidates) and
  `remove()` (detects and removes above threshold)
- **Integration**: `clean_for_summarization()` calls `CommercialDetector.remove()`
  instead of `remove_sponsor_blocks()`
- **Phase 1**: Expanded patterns + positional heuristics (all providers, no audio
  dependency)
- **Phase 2**: Diarization-aware signals (after RFC-058, local Whisper only)
- **Relationship to ADR-020**: ADR-020 established provider-agnostic preprocessing as
  the single cleaning entry point; this ADR specifies the detection architecture within
  that entry point

## References

- [RFC-060: Diarization-Aware Commercial Cleaning](../rfc/RFC-060-diarization-aware-commercial-cleaning.md)
- [ADR-012: Provider-Agnostic Preprocessing](ADR-012-provider-agnostic-preprocessing.md)
- [RFC-058: Audio Speaker Diarization](../rfc/RFC-058-audio-speaker-diarization.md)
