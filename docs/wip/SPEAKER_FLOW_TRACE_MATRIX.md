# Speaker Flow Trace Matrix (#598)

End-to-end trace of how speakers (host/guest) flow through the pipeline.
Validated 2026-04-19. Covers config → NER → GI/KG → bridge.

## Trace: setting → pipeline stage → artifact field

| Stage | Input | Output | Key file | Notes |
| ----- | ----- | ------ | -------- | ----- |
| **Config** | `auto_speakers`, `speaker_detector_provider`, `known_hosts`, `ner_model` | Gates entire speaker pipeline | `config.py` | `auto_speakers=False` → no detection |
| **Feed host detection** | RSS title + description + authors | `Set[str]` host names | `speaker_detection.py:detect_hosts_from_feed` | Priority: RSS authors > NER on feed metadata |
| **Episode guest detection** | Episode title + first 500 chars of description | `List[str]` guest names | `speaker_detection.py:detect_speaker_names` | NER → filter hosts → interview-intent filter |
| **Transcript intro fallback** | First ~300 words of transcript | `Set[str]` host names | `speaker_detection.py:detect_hosts_from_transcript_intro` | Regex patterns ("I'm X", "Welcome to...") |
| **GI Quote nodes** | Whisper diarization segments | `speaker_id` on Quote properties | `gi/pipeline.py` | Only with `screenplay=True` (diarization) |
| **GI SPOKEN_BY edges** | Quote with speaker_id | `Quote → Person` edge | `gi/pipeline.py` | Only with diarization data |
| **KG Person nodes** | `detected_hosts`, `detected_guests` | Entity nodes with `role=host\|guest` | `kg/pipeline.py:_append_pipeline_entities` | MENTIONS edges to Episode |
| **Bridge identities** | GI Person + KG Person nodes | Merged identity with `sources: {gi, kg}` | `bridge_builder.py` | Fuzzy reconciliation at 0.75 cosine |

## Config fields

| Field | Type | Default | Effect |
| ----- | ---- | ------- | ------ |
| `auto_speakers` | bool | True | Master switch for speaker detection |
| `speaker_detector_provider` | str | "spacy" | NER backend: spacy, openai, gemini, etc. |
| `ner_model` | str | en_core_web_sm | spaCy model for NER |
| `known_hosts` | List[str] | [] | Manual host override (skips detection) |
| `cache_detected_hosts` | bool | True | Reuse hosts across episodes |
| `screenplay` | bool | False | Enables Whisper diarization |
| `screenplay_num_speakers` | int | 2 | Expected speaker count for diarization |

## Gaps found and fixed

| # | Issue | Severity | Fix |
| - | ----- | -------- | --- |
| 1 | `DESCRIPTION_SNIPPET_LENGTH=20` truncated guest names | P1 | Raised to 500 |
| 2 | Scoring system over-engineered (1389 lines) | P2 | Simplified to NER → filter → done (935 lines) |
| 3 | No SPOKEN_BY edges without diarization | P2 | By design — needs Deepgram (#597, deferred to 2.7) |
| 4 | Production corpus has 0 persons in artifacts | P1 | Need re-ingestion with latest pipeline |

## Gaps documented (not fixed — by design or deferred)

| # | Issue | Reason |
| - | ----- | ------ |
| 1 | No SPOKEN_BY edges | Needs diarization (Deepgram #597) |
| 2 | NER on markdown transcript headers produces noisy entities | NER runs on RSS metadata, not transcript — non-issue for pipeline |
| 3 | KG uses MENTIONS edge (not PERSON_IN) for persons | Consistent with all entity types — by design |

## Simplification done (#598)

`speaker_detection.py` reduced from 1389 → 935 lines (-33%):

**Removed:**

- Multi-episode heuristic pattern learner (~170 lines)
- Composite scoring system (~190 lines)
- `_has_guest_intent_cue` alias
- 12 unused scoring/position constants

**Simplified `detect_speaker_names` flow:**

```
Before: NER → filter hosts → build candidates → score (NER + heuristic + overlap) → select best → log
After:  NER → filter hosts → interview-intent filter → deduplicate → done
```

**Key behavior change:** all guests that pass the interview-intent filter
are returned (not just the single "best" scored guest). This correctly
handles multi-guest episodes.

## Verified scenarios

| Scenario | Result |
| -------- | ------ |
| Host + Guest (Nora + Daniel) | Both detected correctly |
| Host-only (Alex Morgan solo) | No phantom guest added |
| Multiple guests (panel) | All interview-context guests found |
| Mentioned not guest (Elon Musk) | Correctly filtered out |
| Long description (guest at char 100+) | Found correctly (was truncated before) |
| No description (title only) | Guest from title works |
| Organization as author (NPR, BBC) | Treated as publisher, not host |

## Embedding loader compat (folded into #598)

Three sentence-transformers callsites fixed for 2.x/3.x compat:

| File | API | Fix |
| ---- | --- | --- |
| `embedding_loader.py` | SentenceTransformer | Introspect `__init__` for `local_files_only` |
| `nli_loader.py` | CrossEncoder | Same introspection pattern |
| `model_loader.py` | CrossEncoder (preload) | Same introspection pattern |
