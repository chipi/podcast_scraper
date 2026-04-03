# ADR-062: Sentence-Boundary Transcript Chunking

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-061](../rfc/RFC-061-semantic-corpus-search.md)
- **Related PRDs**: [PRD-021](../prd/PRD-021-semantic-corpus-search.md)

## Context & Problem Statement

Semantic search (RFC-061) indexes transcript chunks as one of four document types. The
chunking strategy directly affects embedding quality, search relevance, and index size.
Chunks that split mid-sentence produce poor embeddings. Chunks that are too long dilute
semantic signal. The system needs a chunking approach that balances quality, simplicity,
and no external dependencies.

## Decision

We adopt **sentence-boundary chunking** with configurable overlap:

1. **Sentence splitting**: Simple regex (`(?<=[.!?])\s+` with `\n` fallback). No
   external tokenizer dependency.
2. **Target chunk size**: ~300 tokens (estimated via whitespace split). Sentences are
   grouped until the target is reached.
3. **Overlap**: ~50 tokens of trailing sentences carried from the previous chunk to
   preserve cross-chunk context.
4. **Character tracking**: Each chunk records `char_start` and `char_end` offsets into
   the original transcript.
5. **Timestamp interpolation**: When Whisper segment timestamps are available,
   `timestamp_start_ms` and `timestamp_end_ms` are interpolated from character position
   alignment.

## Rationale

- **Embedding quality**: Sentence boundaries preserve semantic coherence. Mid-sentence
  splits degrade embedding vectors measurably.
- **No external dependency**: Regex splitting avoids adding spaCy sentence segmentation
  or NLTK punkt as dependencies. The project already uses spaCy for NER but not for
  chunking — keeping the dependency isolated.
- **Predictable sizing**: Target + overlap parameters produce consistent chunk sizes
  across episodes, making index behavior predictable.
- **Configurable**: `vector_chunk_size_tokens` and `vector_chunk_overlap_tokens` in
  config allow tuning without code changes.

## Alternatives Considered

1. **Fixed-token chunking (no sentence awareness)**: Rejected; frequently splits
   mid-sentence, degrading embedding quality.
2. **Paragraph-based chunking**: Rejected; podcast transcripts often lack paragraph
   structure. Would produce wildly variable chunk sizes.
3. **Recursive character splitting (LangChain-style)**: Rejected; adds dependency,
   more complex than needed, and the recursive fallback logic is unnecessary when
   sentence splitting works well for transcripts.
4. **spaCy sentence segmentation**: Rejected; adds a heavy dependency for chunking
   when regex splitting is sufficient for English transcripts.

## Consequences

- **Positive**: Consistent, high-quality chunks. No new dependencies. Configurable
  parameters. Timestamp tracking enables "jump to audio" in viewer.
- **Negative**: Regex sentence splitting may fail on edge cases (abbreviations, URLs,
  decimal numbers). Acceptable for podcast transcripts which are conversational text.
- **Neutral**: Overlap increases index size by ~15-20% vs non-overlapping chunks.
  Trade-off for better cross-chunk retrieval.

## Implementation Notes

- **Module**: `src/podcast_scraper/search/chunker.py`
- **Function**: `chunk_transcript(text, target_tokens=300, overlap_tokens=50,
  timestamps=None) -> list[TranscriptChunk]`
- **Config**: `vector_chunk_size_tokens: int = 300`,
  `vector_chunk_overlap_tokens: int = 50`

## References

- [RFC-061: Semantic Corpus Search — Transcript Chunker](../rfc/RFC-061-semantic-corpus-search.md)
- [ADR-060: VectorStore Protocol](ADR-060-vectorstore-protocol-with-backend-abstraction.md)
