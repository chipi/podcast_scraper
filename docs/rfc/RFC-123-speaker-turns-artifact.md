# RFC-123: Speaker Turns as a First-Class Artifact

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Pipeline (diarization, GI), search, player (segments contract), Positions (read-time CIL arc)
- **Related PRDs**:
  - `docs/prd/PRD-047-multilingual-ingest.md` — consumer: translation units and confidence citation
  - `docs/prd/PRD-036-foundation-identity.md` — player `segments.json` contract
  - `docs/prd/PRD-028-position-tracker.md` — consumer: the per-(person, topic) arc
- **Related RFCs**:
  - `docs/rfc/RFC-058-audio-speaker-diarization.md` — upstream: speaker assignment per segment
  - `docs/rfc/RFC-090-hybrid-retrieval.md` — consumer: Tier-1 segment documents
  - `docs/rfc/RFC-124-multilingual-transcription-and-translation.md` — consumer: translation units
  - `docs/rfc/RFC-125-translation-confidence-and-claim-verification.md` — consumer: confidence citation unit
- **Related ADRs**: `docs/adr/ADR-131-speech-normalized-coverage-gate.md`
- **Arc notes**: `docs/architecture/MULTILINGUAL_ARC.md` (§4 slice plan — this RFC is Phase 1, slices S1.1–S1.6, and ships independently)

## Abstract

Speaker turns already exist in the pipeline, but only implicitly. They are the lines of the
diarized screenplay transcript that `format_diarized_screenplay_with_offsets` builds by coalescing
consecutive same-label segments. `gi/speakers.py` later re-derives them by regex over `Name:`
markers, and search chunking ignores them entirely. This RFC proposes a deterministic
`<stem>.turns.json` sidecar that is built once from the segments sidecar and makes turns (and the
sentences inside them) addressable by stable ID, time range and char range. It has immediate value
for English episodes and it is the unit that multilingual translation (RFC-124) and translation
confidence (RFC-125) are built on. It is fully backfillable from existing artifacts.

## Problem Statement

A turn is "one voice, one uninterrupted stretch of speech". It is the natural unit for attribution
("who said this"), for player cues, for position citation ("who took which side"), and for
translation. Today:

- **Attribution parses text.** `build_named_turns` and `build_unverified_named_turns`
  (`gi/speakers.py:96`, `:125`) recover turn boundaries by regex over line-start `Name:` markers in
  the transcript `.txt`. The #2062 fix shows how fragile this is: replaying 7,101 production quotes
  moved 1,739 (24.5%) from a name to nobody, purely from how an unrecognized marker was handled.
  The structural truth, the speaker label per segment, sits one artifact away in `.segments.json`
  and is not used for this.
- **Search chunks straddle speakers.** `build_segment_documents` (`search/segments.py:23`) produces
  200–300-word overlapping windows (`chunk_transcript`, `search/chunker.py:118`) with a single
  optional `speaker_id` for the whole chunk (`search/backend.py:34`). A window spanning a host
  question and a guest answer gets one speaker or none, so "what did the guest say about X"
  retrieves the host's framing as well.
- **No stable citation unit.** Quotes cite char spans (`EvidenceSpan.char_start/char_end`), which is
  correct for verbatim grounding. But nothing can cite "the guest's answer at 41:12" as a unit that
  survives re-rendering the transcript text.
- **Player cues are Whisper fragments.** The `segments.json` contract (`SegmentsResponse`,
  `server/schemas.py:24`) serves raw Whisper segments of 5–15 s, which often cut mid-sentence.
  Subtitles, tap-a-quote-to-hear and highlight export all want complete thoughts.

**Use cases:**

1. **Structural attribution.** Given a quote's char span, return its turn and speaker without
   parsing text.
2. **Speaker-true retrieval.** Every search chunk knows every speaker it contains.
3. **Readable cues.** Sentence-level cues within turns for subtitles and quote playback.
4. **Translation units.** RFC-124 translates sentence groups that never cross a turn.
5. **Confidence citation.** RFC-125 propagates per-unit confidence to claims through turn and
   sentence IDs.

## Goals

1. **Single source of truth**: turns are built once, deterministically, from the segments sidecar
   the transcript was rendered from.
2. **Addressable**: stable `turn_id` and `sent_id`, each with time range, char range and speaker.
3. **Lossless**: turns exactly partition the transcript's speech segments. Every segment belongs to
   exactly one turn.
4. **Backfillable**: buildable from existing `.segments.json` / `.adfree.segments.json` with no
   re-transcription and no GPU.
5. **Forward-compatible** with word-level anchors (forced alignment) without a schema break.

## Constraints & Assumptions

**Constraints:**

- No change to the transcript `.txt` format. Screenplay markers stay, because existing consumers and
  on-disk corpora depend on them.
- Char ranges index into the **same** transcript text the segments sidecar indexes. Turns are
  therefore built **per variant**, because the raw and ad-free texts are two different coordinate
  spaces: `workflow/adfree_transcript.build_adfree_artifacts` re-renders the surviving segments
  through the same formatter, so `adfree.segments.json` carries ad-free offsets and turns built from
  it inherit them.
- A pure function of its inputs. No model calls.

**Assumptions:**

- Diarization quality caps turn quality. If pyannote over-splits one person into two voices, that
  produces false turn boundaries. This RFC does not fix diarization. It makes the boundaries
  explicit and inspectable.
- Whisper punctuation is good enough for sentence splitting in enabled languages. RFC-124's
  per-language gate checks this for non-English languages.

## Design & Implementation

### 1. Artifact

`transcripts/<stem>.turns.json` sits next to each segments sidecar variant (`<stem>.turns.json`,
`<stem>.adfree.turns.json`).

```json
{
  "version": "1.0",
  "episode_slug": "…",
  "language": "en",
  "source": {
    "segments_ref": "transcripts/ep1.adfree.segments.json",
    "segments_sha256": "…",
    "transcript_ref": "transcripts/ep1.adfree.txt"
  },
  "turns": [
    {
      "turn_id": "t0007",
      "speaker_label": "Kevin Roose",
      "speaker": "person:kevin-roose",
      "speaker_role": "host",
      "voice_type": null,
      "start_ms": 2471200,
      "end_ms": 2503900,
      "char_start": 18342,
      "char_end": 19011,
      "segment_idx": [112, 113, 114, 115],
      "backchannel": false,
      "sentences": [
        {
          "sent_id": "t0007.s01",
          "char_start": 18356,
          "char_end": 18471,
          "start_ms": 2471200,
          "end_ms": 2479800,
          "timing": "segment_interpolated"
        }
      ]
    }
  ]
}
```

- `turn_id` is ordinal within the variant (`t0000…`). IDs are stable across rebuilds when the
  inputs are unchanged. When the inputs change, the provenance hash changes and consumers
  invalidate (provenance-keyed invalidation).
- `speaker` / `speaker_role` / `voice_type` are passed through from the enriched segments. The
  formatter already carries exactly these three keys through to the offset segments
  (`formatting.py`, the `for key in ("speaker", "speaker_role", "voice_type")` passthrough), so
  role truth is not re-derived and the guest-as-host bug cannot resurface here.
- The turn's `char_start` points at the first speech character, **after** the `Label:` prefix and its trailing space, so
  the span is pure speech.

### 2. Construction

`build_turns(offset_segments) -> Turns`, in `providers/ml/diarization/turns.py`:

1. Take the offset segments exactly as `format_diarized_screenplay_with_offsets` emits them (same
   start-time sort, same blank-text drop). Turns are then **by construction** the screenplay lines:
   consecutive segments with the same `speaker_label`.
2. **Backchannels are not merged away.** A sub-1.5 s, 1–3-word turn from another speaker between
   two turns of the same speaker ("yeah", "right") stays its own turn with `backchannel: true`.
   Merging would falsify attribution. Flagging lets consumers skip these for chunking and
   translation grouping.
3. **Sentences**: split turn text on `.?!…` followed by whitespace, with a short protected list of
   abbreviations, and never across segment boundaries that end a turn. Sentence times come from the
   containing segments. When a sentence starts or ends mid-segment, the time is interpolated by
   character proportion and `timing` is set to `segment_interpolated`. When both ends coincide with
   segment boundaries, it is `segment_exact`.
4. **Invariants**, asserted at build time:
   - turns are ordered and non-overlapping in char space;
   - every non-blank segment is in exactly one turn;
   - sentences partition their turn's speech span;
   - times are monotonic within a turn;
   - `screenplay_text[turn.char_start:turn.char_end]` is the turn's speech text — the same identity
     the formatter already guarantees per segment.

When the text-only screenplay path is used without offsets (a legacy corpus), turns are not built.
The episode is flagged `turns: unavailable` and consumers fall back to current behavior.

### 3. Where it runs

It runs immediately after each segments sidecar is written — once for the raw variant and once for
the ad-free variant, inside `workflow/adfree_transcript` where the ad-free segments are produced, so
both are built from the exact list the matching text was rendered from. It is cheap (pure Python,
milliseconds per episode) and belongs on the core path, not in an enricher. Turns are a structural
view of core artifacts, not derived intelligence, so they do not carry `derived: true`.

**Which variant a consumer reads** is already decided by
`workflow/adfree_transcript.load_processing_transcript`, the single resolver all NLP consumers use.
Turn-based attribution reads the variant that resolver returned, keyed by its `transcript_ref`.
Nothing in this RFC changes that precedence; RFC-124 extends it.

**Backfill:** `podcast-scraper turns backfill --corpus <dir>` walks existing sidecars. This is
unlike derived intelligence, which cannot be backfilled. Nothing needs to be sequenced ahead of
ingest.

### 4. Consumer migration

Each consumer migrates behind its own flag, in this order:

- **GI speaker attribution** (`gi/speakers.py`): resolve a quote's speaker by char-span lookup into
  turns (binary search on `char_start`). Keep regex attribution as a fallback when `turns.json` is
  absent. Replay the 7,101-quote set from #2062 and report the delta (name→name, name→none,
  none→name) before switching. Note that the existing replay measured direction, not correctness:
  how many of the 1,739 lost names were *wrong* was never measured, so the new replay must report
  the same transition matrix rather than a single "improved" number.
- **Search Tier-1 segments** (`search/segments.py`): chunk within turn boundaries. Adjacent short
  turns may merge into one chunk up to the target size, and the chunk then carries
  `speaker_ids: [...]` (all speakers) plus `turn_ids`. A single long turn is windowed as today.
  `SegmentDocument` gains `speaker_ids` and `turn_ids` as additive fields, which also means the
  LanceDB segment schema (`search/backends/lancedb_backend.py`) gains two columns. This changes the
  index, so it goes through the normal reindex path (RFC-118 delta).
- **Player segments contract**: add an optional `?granularity=sentence` that serves sentence cues
  (`sent_id`, `turn_id`, times, speaker) from `turns.json`. The default remains raw segments for
  contract stability.
- **Positions (read-time CIL arc)**: no migration is required. Positions are not an extraction
  stage — `position_arc` (`server/cil_queries.py:636`) is a read-time query over GI insights and
  their supporting quotes, and those quotes already cite char spans. Nothing in that path waits on
  this RFC. One caveat on "turn-level citation for free": `turn_id`s are **ordinal within a variant**,
  so a span cited in the ad-free text resolves to an ad-free `turn_id` that is not the same number as
  the raw variant's — turns are renumbered from `t0000` after ad segments are dropped. Turn ids are
  therefore variant-scoped and must never be joined across variants by id alone; a consumer that needs
  the correspondence carries the source `turn_id` explicitly (which is what RFC-124 §4.1 does with
  `unit_id`).

### 5. Word-level anchors (forward compatibility)

When forced alignment (wav2vec2) lands, sentences gain `words: [{char_start, char_end, start_ms,
end_ms}]` and `timing` becomes `word_aligned`. This needs no schema version break, only additive
fields. Note for RFC-124: wav2vec2 alignment models are **per-language**, so word anchors for
non-English episodes need per-language aligner checkpoints (pinned per ADR-155) or they stay at
`segment_interpolated`.

## Key Decisions

1. **Turns = screenplay lines.**
   - **Decision**: define turns by the same coalescing rule the transcript renderer already uses.
   - **Rationale**: the `.txt` and the turns can never disagree. There is one definition of a turn
     in the codebase.
2. **Flag backchannels, don't merge them.**
   - **Decision**: keep short interjections as their own turns, with `backchannel: true`.
   - **Rationale**: attribution fidelity beats readability. Readability is a consumer concern.
3. **Sidecar, not a new transcript format.**
   - **Decision**: a new sidecar next to the existing artifacts.
   - **Rationale**: zero migration for existing readers, and it is fully backfillable.
4. **Sentences inside turns, not as a separate artifact.**
   - **Decision**: sentences are nested in their turn.
   - **Rationale**: every sentence belongs to exactly one turn, and nesting makes that invariant
     structural.
5. **One `turns.json` per transcript variant, not one per episode; ids are variant-scoped.**
   - **Decision**: build for the raw and the ad-free text separately, and treat `turn_id` as meaningful
     only within its own variant.
   - **Rationale**: char offsets are only meaningful in one text. A single artifact would have to
     pick a coordinate space and would silently mis-anchor the other. And because the ad-free variant
     drops whole turns and renumbers from `t0000`, the two variants do not share a turn list at all —
     so a cross-variant join by id would silently point at the wrong turn.

## Alternatives Considered

1. **Keep regex-derived turns.** Rejected: fragile (#2062), text-dependent, and unavailable to
   search and the player.
2. **Merge short same-speaker gaps and interjections into long turns.** Rejected: produces cleaner
   cues but silently misattributes the interjection.
3. **Sentence-only artifact, no turns.** Rejected: loses the speaker-exchange structure that
   positions and translation grouping need.
4. **Compute turns on the fly at read time.** Rejected: every consumer would re-implement it, and
   IDs would not be stable enough to cite.
5. **One artifact spanning both variants, with two offset sets per turn.** Rejected: the ad-free
   text drops whole turns, so the two variants do not share a turn list at all.

## Testing Strategy

- **Unit / property**: the invariants in §2.4 over generated segment lists (hypothesis), including
  empty text, identical timestamps, a single-speaker episode, and alternating backchannels.
- **Golden**: 5 fixture episodes (2 diarized English, 1 ad-free re-anchored, 1 single-voice, 1 with
  over-split diarization), with committed `turns.json` goldens.
- **Replay**: the #2062 quote set through turn-based attribution versus regex attribution, reported
  as a transition matrix. Switching requires zero name→different-name transitions.
- **Search**: retrieval eval on the existing query set before and after turn-bounded chunking,
  measuring recall@k and speaker-precision (the fraction of hits whose `speaker_ids` contain the
  queried speaker).

## Rollout & Monitoring

- **Phase 1**: write `turns.json` in the pipeline and run the backfill. Nothing reads it yet.
- **Phase 2**: switch GI attribution after the replay passes.
- **Phase 3**: switch search chunking, with a reindex and eval.
- **Phase 4**: add the player `granularity=sentence` option.

**Monitoring:** the per-episode manifest (RFC-109) gains `turns: {count, backchannels,
median_turn_s, invariant_failures}`. Any invariant failure fails the build for that episode.

**Success criteria:**

1. 100% of episodes with a segments sidecar have a `turns.json` after backfill.
2. The attribution replay shows no wrong-name transitions and a net reduction in unattributed
   quotes.
3. Speaker-precision on search improves, with recall@k unchanged or better.

## Benefits

1. Structural attribution replaces text parsing.
2. Search results know who is speaking.
3. Player cues become complete thoughts.
4. It is the unit RFC-124 and RFC-125 need, at no extra cost to English ingest.

## Open Questions

1. The backchannel thresholds (1.5 s, ≤3 words) are guesses. Tune them on the fixture corpus?
2. Should `turns.json` also be emitted for non-diarized transcripts, as one turn per paragraph with
   `speaker: null`, so consumers have a uniform input? Note that the non-diarized path derives
   offsets by progressive text search (`adfree_transcript._derive_offsets_by_find`) and *skips*
   segments it cannot locate, so a paragraph-turn artifact there would not satisfy the
   every-segment-in-exactly-one-turn invariant as written.
3. Should the player's default granularity eventually flip to `sentence`?

## References

- `src/podcast_scraper/providers/ml/diarization/formatting.py` — `format_diarized_screenplay_with_offsets`
- `src/podcast_scraper/workflow/adfree_transcript.py` — `build_adfree_artifacts`, `load_processing_transcript`, `ProcessingTranscript.transcript_ref`
- `src/podcast_scraper/gi/speakers.py` — `build_named_turns`, `build_unverified_named_turns` (#875, #2062)
- `src/podcast_scraper/search/segments.py`, `search/chunker.py` — Tier-1 chunking
- `src/podcast_scraper/search/backend.py` — `SegmentDocument.speaker_id`
- `src/podcast_scraper/gi/ad_regions.py` — ad-free re-anchoring of `char_start`/`char_end`
- `src/podcast_scraper/server/cil_queries.py` — `position_arc` (read-time Positions)
- `src/podcast_scraper/server/schemas.py` — `TranscriptSegment`, `SegmentsResponse`
