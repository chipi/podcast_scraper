# ADR-177: Content types and transports are extension seams; podcast is the first built-in type

- **Status**: Proposed
- **Date**: 2026-11-11
- **Authors**: Marko Dragoljevic
- **Related**: [RFC-134](../rfc/RFC-134-news-as-the-second-content-type.md) (news as the second
  content type), [ADR-168](ADR-168-installed-extensions.md) (installed extensions),
  [ADR-176](ADR-176-extension-seams-for-the-split.md) (the seams built for release 1)

## Context & Problem Statement

The pipeline knows one kind of item: a podcast episode from an RSS enclosure, with audio and a
transcript. Release 1 put seams around the stages that leave public (ADR-176), but not around the
item itself or the way items arrive. Adding news, or any later type, would mean editing the
podcast path, which is what the seams exist to avoid.

## Decision

1. **A content type is registered, not hard-coded.** `Extension.content_types` adds types. Each
   one names:
   - its id scheme (`item:` ids for new types; existing ids never change, T-8);
   - its stage profile;
   - its prompt profile;
   - its viewer card kind.
2. **A transport is registered.** `Extension.transports` adds transports. A transport lists new
   items for a source definition and returns, for each, either the text with its metadata or a
   URL to fetch.
   - It owns its cursor, which can be a feed position, an API page, or a date and an id.
   - It owns its request headers and its per-host rate limit, so a source that requires a contact
     header or a request budget (SEC EDGAR) fits without changing the interface.
   - The private extension and later product repos register types and transports the same way as
     public does.
3. **Podcast is the first, built-in type, with RSS enclosures its transport.** Its path is
   unchanged, and it is registered the way ADR-176 registers today's code: as the built-in owner.
4. **A text-in entry point runs beside `run_pipeline`.** It takes items that are already text and
   runs the stages their type's profile names. It never reaches audio, transcription, diarization
   or ad cleaning.
5. **News is public's second built-in type** (the operator, 2026-11-11). Its transports,
   extraction and stage profile live in public, and run on a public-only install. The private
   extension adds to it only what it adds to podcasts.
6. **Every item of a new type records its type.** Its metadata carries `content_type`. Podcast
   metadata gains the field only with R3b; until then a document without it reads as
   `podcast_episode`.
7. **One dispatch point, and one item context.**
   - Both multi-feed loops pass each source through one dispatch point, which picks `run_pipeline`
     or the text-in entry point by type.
   - The processing job carries one item context (type, profile, item id, body path).
   - The type's profile decides which stage hooks run; a hook that needs the type reads it from the
     context. Hooks gain no type argument, so podcast calls stay byte-identical.
8. **Shared reads default to podcast items.** The catalog builders and the search query paths take
   a type filter that defaults to `podcast_episode`. A surface sees another type only when it asks,
   so the Player and the MCP tools need no change while new types arrive.
9. **When:**
   - the core release builds decisions 1–4 and 6–8 and proves them with a test-only second type
     in fixtures;
   - decision 5, news, is News 1.0, a release of its own after it.

## Consequences

- **Positive:**
  - a new type is a registration plus its own stages;
  - news proves the seams with a second type and a second transport;
  - the private extension can add types the same way.
- **Negative:**
  - every reader that assumes an episode has to read `content_type` or be limited to podcasts by
    the registry;
  - listing those readers is part of the work.
- **Neutral:** podcast output is unchanged. That is checked by the release's byte-identical
  artifact gate.

## Alternatives considered

- **A news pipeline beside the podcast one, sharing nothing.** Rejected. It proves nothing about
  the core, and doubles every later change to enrichment.
- **News as a podcast with no audio.** Rejected. Every podcast assumption (enclosure, transcript,
  timestamps) would need a fake value, and the fakes would leak into the graph and search.
