# PRD-001: Transcript Acquisition Pipeline

## Summary

Create a resilient pipeline that ingests a podcast RSS feed, locates published transcripts for each episode, and stores them deterministically so users can archive, compare, or replay runs. This PRD captures functional expectations for the baseline "download what already exists" workflow.

## Background & Context

- The `podcast_scraper` application targets researchers, archivists, and power listeners who need canonical text versions of podcast episodes.
- Many shows publish transcripts via Podcasting 2.0 tags; users want an automated way to resolve those assets and save them locally.
- `docs/ARCHITECTURE.md` describes the technical blueprint and module ownership. This document focuses on *what* needs to happen from a user and product perspective.

## Goals

- Support end-to-end transcript download for every RSS item that exposes a transcript link.
- Ensure reruns are idempotent and resumable without duplicate work.
- Provide operators with clear logging/progress so long-running jobs can be monitored.

## Non-Goals

- Whisper transcription (covered by PRD-002).
- UI/CLI ergonomics beyond must-have parameters (covered by PRD-003).
- Media download optimizations beyond basic HTTP retry and progress visibility.

## Personas

- **Archivist Ava**: Schedules periodic pulls to build a transcript archive for compliance/legal review.
- **Researcher Riley**: Downloads transcripts on-demand to run NLP analysis across multiple podcast series.

## User Stories

- *As Archivist Ava, I can point the tool at an RSS feed and receive a folder of transcripts named deterministically so future runs can diff changes.*
- *As Researcher Riley, I can limit how many episodes to fetch (e.g., first 10) when testing a new feed.*
- *As any operator, I can dry-run a job to preview planned downloads without writing files.*
- *As any operator, I can resume a long job and skip episodes that were already saved.*

## Functional Requirements

- **FR1**: Accept an RSS URL (HTTP/HTTPS) and validate it before executing requests.
- **FR2**: Resolve the feed, parse items safely, and detect transcript URLs via Podcasting 2.0 tags, `<transcript>` nodes, or equivalent.
- **FR3**: Support preference ordering (`prefer_type`) to pick the best transcript when multiple URLs are provided.
- **FR4**: Download transcript assets with retry/backoff and surface failures without halting the entire run.
- **FR5**: Persist transcripts using deterministic filenames `<episode_number> - <title>[ _<run_suffix>].<ext>` in a derived output directory (`output_rss_<host>_<hash>` by default).
- **FR6**: Provide `--skip-existing` semantics so reprocessing avoids already-downloaded episodes.
- **FR7**: Provide `--dry-run` mode that logs planned work (including file destinations) without touching disk.
- **FR8**: Allow `--max-episodes` to cap the number of items processed.
- **FR9**: Emit structured logs + progress updates friendly to terminal usage (leveraging progress abstraction).
- **FR10**: Support optional inter-request delay (`--delay-ms`) for rate-limited feeds.

## Success Metrics
>

- >=95% of episodes with published transcripts complete without manual retry (network errors aside).
- Dry-run output matches real run output (file naming, counts) aside from disk writes.
- End-to-end processing scales to feeds with 1k episodes using default settings without manual intervention.

## Dependencies

- Stable RSS parsing (defused XML) and HTTP client behavior described in `docs/rfc/RFC-003-transcript-downloads.md` and `docs/rfc/RFC-002-rss-parsing.md`.
- Output directory and naming guarantees covered in `docs/rfc/RFC-004-filesystem-layout.md`.

## Release Checklist

- [ ] Integration tests cover happy path, dry-run, skip-existing, and error handling.
- [ ] Logging reviewed for clarity and usefulness during long runs.
- [ ] README updated with usage examples for key flags.

## Open Questions

- Should we support filtering by publish date or keyword in addition to `--max-episodes`? (Future consideration.)
- Do we need per-episode metadata exports (JSON summaries) alongside transcripts? Not in scope for v1.

## RFC-010 Integration

While PRD-001 focuses on transcript downloads, RFC-010 (Automatic Speaker Name Detection) enhances the pipeline by:

- **Metadata Extraction**: During RSS parsing, episode metadata (titles, descriptions) is analyzed to extract speaker names, enriching the episode data model.
- **Metadata Storage**: Detected speaker names are stored alongside transcripts in metadata documents (per PRD-004), enabling downstream search and analysis.
- **Transparent Operation**: Speaker name detection runs automatically during episode processing without affecting transcript download workflows.
- **Language Awareness**: The `--language` configuration affects both NER extraction and future Whisper transcription, ensuring consistent language handling across the pipeline.
