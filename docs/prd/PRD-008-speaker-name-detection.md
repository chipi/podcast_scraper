# PRD-008: Automatic Speaker Name Detection

- **Status**: ✅ Implemented (v2.1.0)
- **Related RFCs**: RFC-010

## Summary

Automatically detect and label speaker names (hosts and guests) in Whisper transcriptions using Named Entity Recognition (NER), eliminating the need for manual `--speaker-names` configuration.

## Background & Context

When Whisper generates transcriptions, speakers are labeled generically as "Host" and "Guest" unless the user manually provides names via `--speaker-names`. This manual step:

- Requires feed-specific knowledge that users may not have
- Breaks when episodes have unique guests
- Adds friction when running at scale or embedding the pipeline
- Reduces transcript quality and readability

Users want transcripts that automatically use real speaker names without requiring manual configuration for each podcast or episode.

## Goals

- Automatically extract host names from RSS feed metadata (author tags, titles, descriptions)
- Automatically detect guest names from episode metadata (titles, descriptions)
- Provide speaker names to Whisper screenplay formatting for higher-quality transcripts
- Maintain backward compatibility with manual `--speaker-names` as fallback
- Support multiple languages through configurable NER models

## Non-Goals

- Voice fingerprinting or audio-based speaker identification
- Multi-speaker diarization beyond host/guest distinction
- Real-time speaker detection during transcription

## User Stories

1. **As a podcast archivist**, I want transcripts to automatically show "Lenny Rachitsky" and "guest name" instead of "Host" and "Guest" so my archive is more searchable and readable.

2. **As a researcher**, I want to process hundreds of episodes without manually looking up speaker names for each one.

3. **As a developer integrating the pipeline**, I want automatic speaker detection so I don't need to maintain a database of podcast hosts.

## Functional Requirements

### Host Detection (Feed-Level)

1. Extract host names from RSS author tags (highest priority):
   - RSS 2.0 `<author>` tag
   - iTunes `<itunes:author>` tag
   - iTunes `<itunes:owner><itunes:name>` tag

2. Fall back to NER extraction from feed title/description if no author tags

3. Validate detected hosts by checking if they appear in episode content

### Guest Detection (Episode-Level)

1. Extract guest names from episode titles (e.g., "Interview with John Smith")
2. Extract from episode descriptions using NER
3. Filter out detected hosts to avoid duplication

### Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `auto_speakers` | `true` | Enable automatic speaker detection |
| `language` | `"en"` | Language for NER model selection |
| `ner_model` | derived from language | Override NER model (e.g., `en_core_web_sm`) |
| `speaker_names` | `[]` | Manual fallback names if detection fails |

### CLI Flags

```bash
# Enable/disable auto detection
--auto-speakers / --no-auto-speakers

# Set language (affects both NER and Whisper)
--language en

# Override NER model
--ner-model en_core_web_sm

# Manual fallback
--speaker-names "Lenny" "Guest"
```

## Success Criteria

- ✅ Host names correctly detected from RSS author metadata
- ✅ Guest names correctly extracted from episode titles/descriptions
- ✅ Detected names appear in Whisper screenplay output
- ✅ Manual `--speaker-names` works as fallback when detection fails
- ✅ No regression in transcript quality when names aren't detected
- ✅ Performance impact < 100ms per episode for NER processing

## Dependencies

- **spaCy**: NER library for named entity extraction
- **spaCy language models**: `en_core_web_sm` (English), others for additional languages

## Related Documents

- [RFC-010: Automatic Speaker Name Detection](../rfc/RFC-010-speaker-name-detection.md) - Technical implementation
- [PRD-002: Whisper Fallback Transcription](PRD-002-whisper-fallback.md) - Whisper integration
- [Architecture](../ARCHITECTURE.md) - System design

