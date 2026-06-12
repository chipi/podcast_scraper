# Tech review — speaker-attribution / diarization-naming pipeline (#876)

**Date:** 2026-06-12. **Status:** review + refactor plan (pre-implementation).
**Why:** the "who said this" logic was built in stages (host detection → guest detection →
diarization → name↔voice mapping → host self-intro → network-author filter). It now spans
several files with overlapping responsibilities. This documents how it's actually wired,
where it's fragile, and a flattening plan with full test coverage.

## TL;DR — the central problem

**Two independent name sets flow through the pipeline and never reconcile:**

1. **Feed-level host detection** (`processing.detect_feed_hosts_and_patterns`) →
   `cached_hosts` → drives **`content.speakers` metadata** *and* **`enrich-edges` SPOKEN_BY**.
   This is where network names ("Colossus") leak in.
2. **Transcript self-intro host** (`extract_self_introduced_host`, re-derived at *diarization*
   time) → drives the **screenplay `Name:` labels** and (transitively) the **GI quote
   `speaker_id`** (`person:patrick-oshaughnessy`).

They are computed in different stages, from different sources, and never cross-checked. That
single split explains the headline bug: the screenplay/quotes say **Patrick O'Shaughnessy**
(correct) while `content.speakers` says **Colossus** (wrong). Every other issue below is a
symptom of the same fragmentation.

## Stage map (file:line)

| Stage | Entry point | Produces | Consumed by |
|---|---|---|---|
| 1. Host detect (pre-transcribe) | `workflow/stages/processing.py:448 detect_feed_hosts_and_patterns` | `cached_hosts` (RSS author → NER title/desc → known_hosts → episode authors) | `content.speakers`, enrich-edges |
| 1b. Self-intro host (post-transcribe) | `speaker_detectors/hosts.py:50 extract_self_introduced_host` (called `diarization/pipeline.py:83`) | `host_name` | screenplay labels only |
| 2. Guest detect (pre-transcribe) | `processing.py:658 _detect_speakers_for_episode` (hosts stripped `:723-728`) | guest-only `detected_speaker_names` | diarization mapping |
| 3. Diarize + map | `diarization/pipeline.py:30 apply_diarization_to_result` → `mapping.py:29 map_speakers_to_names` | per-segment `speaker_label`; `diarization_num_speakers` (`:98`) | screenplay; GI quotes |
| 4. Screenplay | `episode_processor.py:422 _format_transcript_if_needed` → `formatting.py:8` | transcript `.txt` `Name:` lines | GI pipeline, enrich-edges |
| 5. GI quote attr | `gi/pipeline.py:1398-1413` (`speaker_id=person:<slug>`); `content.speakers` at `workflow/stages/metadata.py:64` | quote `speaker_id`; speaker roster metadata | KG, SPOKEN_BY |
| 6. SPOKEN_BY | `cli enrich-edges` → `gi/speakers.py:191` (parses screenplay markers) | Quote→Person edges | KG / viewer |

## Redundancy & coupling (the "built in stages" debt)

- **Host identity computed twice, never reconciled** — feed-level `cached_hosts`
  (`processing.py:448`) vs self-intro `host_name` (`pipeline.py:83`). Different sources, drive
  different outputs.
- **Three name sets, three threads** — guest-only `detected_speaker_names`
  (`processing.py:728`); `host_name` re-derived at diarization (`pipeline.py:83`);
  `detected_hosts` re-derived from `cached_hosts` at metadata (`metadata.py:65`).
- **Two transcript-intro host extractors** — `extract_self_introduced_host` (`hosts.py:50`,
  the one that matters) vs `detect_hosts_from_transcript_intro` (`hosts.py:92`, near-dormant,
  only `detection.py:54`). Same idea, two regex implementations.
- **`is_network_or_org_author` applied 3×** (`processing.py:230,489,519`; `hosts.py:144`); the
  guest/host split is *also* done 3× (`detection.py:67`, `processing.py:723`, `metadata.py:68`).
- **Two SPOKEN_BY producers** — in-pipeline segment-based (`gi/pipeline.py:1404-1423`) vs
  enrich-edges marker-based (`gi/speakers.py`). Different inputs, different name sources.

## Edge cases / fragility

| Edge case | Current behavior | Verdict |
|---|---|---|
| **Co-hosted show (2 hosts)** | only one `host_id` (max intro time, `mapping.py:58`); 2nd host treated as a guest | **Broken** |
| **Panel (>2 speakers)** | guests named by total-talk-time order, not detection order; extras keep raw `SPEAKER_xx` | **Fragile** (mis-pairing with ≥2 guests) |
| **Host doesn't self-introduce** | `host_name=None` → host stays `SPEAKER_00`; quote becomes `person:speaker-00` | **Gap** (no fallback to the feed host name we already have) |
| **Network author tag ("Colossus")** | filtered at host detection, but slips into `content.speakers` via known_hosts/episode-author fallback; never cross-checked vs self-intro | **The reported bug** |
| **`num_speakers` propagation** | computed `pipeline.py:98`, **no metadata field exists** to carry it → surfaces as `None` | **Dead-ends** |
| **#545 length-mismatch guard** | `abs(len(transcript) − Σ len(segment.text)) > 50` → segment speaker_id skipped (`gi/pipeline.py:725`) | **Mis-fires on diarized episodes** — inline `Name:` markers structurally inflate transcript length vs raw segment text, so the guard trips on exactly the episodes that *have* speaker data; enrich-edges then has to recover from markers |
| **Transcript cache hit** | reuses formatted screenplay → re-diar no-op (see RUNBOOK caveat) | Handled by clearing cache |

## Proposed flatten — one "speaker roster" resolution

Replace the three scattered name sets with **a single roster, resolved once, after we have
the most information** (post-transcribe + post-diarize), and thread it everywhere.

```
resolve_speaker_roster(diarization, transcript, feed_hosts, detected_guests, known_hosts)
  -> Roster { voice_id -> SpeakerRole{ name, role∈{host,guest,unknown}, source } },
     num_speakers
```

Resolution order (most-reliable first), per voice:
- **Host voice(s)** = intro-dominant speaker(s). Name by: self-intro (`I'm X`) → `known_hosts`
  config → filtered feed author/NER → else keep raw label + role=host.
- **Guest voice(s)** = remaining by talk-time. Name by detected guest list (de-dup vs host).
- Network/org names filtered once, here.

One roster then feeds **all** consumers: screenplay labels, GI quote `speaker_id`,
`content.speakers`, `diarization_num_speakers`, and enrich-edges. Kills the two-source split,
the triple filtering, and the metadata-vs-screenplay disagreement.

## Refactor plan (incremental, not big-bang)

**P0 — correctness, low-risk (do first):**
1. **Thread `diarization_num_speakers`** into EpisodeMetadata/`content` (add field + writer).
2. **Reconcile `content.speakers` host with the resolved/self-intro host** so metadata matches
   the screenplay (no more "Colossus" while quotes say Patrick).
3. **Fix the #545 guard** for diarized transcripts — compare against the *raw* transcript
   length (pre-marker) or raise the delta tolerance for marker-formatted screenplays, so
   segment-based `speaker_id` isn't silently dropped on diarized episodes.

**P1 — robustness:**
4. **`map_speakers_to_names`: multi-host + panel** — support N host voices (intro-dominant set)
   and stable guest pairing; document the >2-speaker contract.
5. **Host-not-self-introduced fallback** — when `host_name` is None, fall back to the
   (filtered) feed/known_hosts host name instead of leaving raw `SPEAKER_xx`.

**P2 — flatten:**
6. Introduce `resolve_speaker_roster` as the single source of truth; route screenplay,
   metadata, and quote attribution through it. Collapse the two transcript-intro extractors.

## Test plan (success path + variations + edge cases)

**Unit (pure):**
- `resolve_speaker_roster` / `map_speakers_to_names`: 2-speaker host+guest; solo monologue;
  panel (host + 3 guests); co-hosted (2 hosts); host-no-self-intro (fallback to feed name);
  network author tag stripped; guest says "I'm X" late (ignored); more names than voices
  (extras dropped) and more voices than names (extras raw).
- `extract_self_introduced_host`: full surname (O'Shaughnessy), sentence-boundary stop,
  intro-window cutoff, no-match → None.
- `is_network_or_org_author`: person names kept; org markers / mononyms / acronyms rejected.

**Integration:**
- `apply_diarization_to_result`: self-intro names host; guest by talk-time; leftover raw;
  empty-segments → gap-based; #545-style length mismatch still attributes via markers.
- `content.speakers` matches the resolved host (regression for "Colossus").
- `diarization_num_speakers` present in metadata.

**Corpus-level validator** (the e2e gap — `tests/integration/test_corpus_diarization_quality.py`
+ `scripts/tools/diarization_quality_metrics.py`, run on a produced corpus):
- quote attribution rate (quotes with a valid `person:` speaker_id) ≥ threshold;
- no network/org speaker names in `content.speakers`;
- quote timestamp coverage ≥ threshold;
- multi-speaker episodes have ≥2 distinct named speakers;
- `diarization_num_speakers` present;
- SPOKEN_BY coverage (post enrich-edges).
Wired via `make diarization-quality CORPUS_DIR=…` for local full-corpus runs + the e2e
diarization fixture in CI.

**E2E fixtures to add** (extend `tests/fixtures/audio` / `test_diarization_e2e.py`): a
co-hosted episode and a 3-speaker panel, so the multi-host/panel paths are exercised.
