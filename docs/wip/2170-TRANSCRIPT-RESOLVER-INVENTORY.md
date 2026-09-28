# Transcript resolver inventory (#2170 / slice S2.1a)

Every place in `src/` and `scripts/` that decides **which transcript variant to read** for an
episode. The refactor routes the readers through one function; this document is what says which
readers, with which intent, and what is deliberately left alone.

It exists because the claim it replaces was wrong twice. `load_processing_transcript`'s docstring
says it is "the single resolver all NLP consumers (GI, enrich-edges, search) use" — it has **two**
callers. A grep for `load_processing_transcript` finds those two and looks complete; the readers
that resolve independently never mention it.

**Method.** Three repo-rooted sweeps, unioned: every mention of `adfree` / `ADFREE` / `.adfree`; every
use of `transcript_file_path`; every expression joining a directory with a transcript relpath. Then
each hit read in place and classified. Counted per call site, not per file.

## The variants

| File | Written by | Coordinate space |
| --- | --- | --- |
| `<base>.txt` | transcription / diarization | **canonical**, full timeline, ads included |
| `<base>.segments.json` | diarization | times on the canonical timeline |
| `<base>.adfree.txt` | `adfree_transcript.build_adfree_artifacts` | ads excised — **the space GI's `char_start` lives in** |
| `<base>.adfree.segments.json` | same | each segment carries its `char_start`/`char_end` in ad-free text |
| `<base>.adfree.admap.json` | same | the excised ranges, in raw-screenplay space |
| `<base>.cleaned.txt` | `_generate_episode_summary` as a byproduct | the summariser's in-memory `PatternBasedCleaner` output |

`transcript_file_path` in metadata always names the plain `.txt`: three separate guards
(`episode_processor.py:3965`, `run_index.py:503`, `stages/scraping.py:524`) reject `.adfree.txt` and
`.cleaned.txt` as "the transcript". So every variant preference below is a *derivation* from that
one stored path, which is why they could all drift independently.

## Group A — analysis intent (10 sites)

Offsets must match the text. GI computes `char_start` in ad-free space, so a reader that pairs
ad-free offsets with raw text is wrong by however many characters the ads occupied.

| # | Reader | Site | Order today |
| --- | --- | --- | --- |
| A1 | GI build | `workflow/metadata_generation.py:4835` (via `load_processing_transcript`) | adfree → raw |
| A2 | KG build | `workflow/metadata_generation.py:5082` (via `load_processing_transcript`) | adfree → raw |
| A3 | Speaker record from diarized segments | `workflow/metadata_generation.py:1199` | adfree → raw |
| A4 | Search indexer | `search/indexer.py:106` | adfree → raw |
| A5 | GI repair — text | `gi/repair.py:167` | adfree → raw |
| A6 | GI repair — segments | `gi/repair.py:143` | adfree → raw |
| A7 | Recurrent-host scan | `workflow/stages/processing.py:1089` | adfree → cleaned → raw |
| A8 | `m0009` speaker-role backfill | `upgrade/migrations/m0009_backfill_speaker_roles.py:309` | adfree → raw |
| A9 | Capability audit — transcript opening | `capability_audit.py:414` | **raw → adfree → `.transcript.json`** |
| A10 | GI evidence loading (CLI) | `gi/load.py:28` | **raw only** |

**A10 is a live coordinate-space bug.** `_transcript_path_from_artifact_path` builds
`output_dir/transcripts/<base>.txt` unconditionally, then `get_evidence_span` slices it with
`char_start`/`char_end` from the artifact — which GI computed against `.adfree.txt`. Every evidence
span on an episode whose ads were excised is displaced by the ads before it. Bounded to
`gi inspect` and `gi show-insight` (`cli.py:2917`, `:2991`), so CLI-only and no artifact is written
from it. This slice fixes it.

**A9 is an inversion, and it needs a decision, not a silent fix.** Every other analysis reader
prefers the ad-free variant; this one prefers raw. It samples the first
`_TRANSCRIPT_HEAD_BYTES` of a segments sidecar to quote an episode's opening. Raw-first means the
sample can be a pre-roll ad — and openings are exactly where pre-roll lives. But the audit's
subject may legitimately be the canonical artifact rather than what GI saw. **Recorded, not
changed in this slice.** Whichever way it goes, the golden makes the current answer visible first.

A7's ordering has a stated reason worth preserving: ad-free first *because* an unstripped pre-roll
can push a host's self-introduction past the scanned window.

## Group B — timeline intent (2 sites)

The consumer player streams the **original unbridged audio**, ads included. Ad-free segments are
minutes shorter, so pairing them with that audio drifts highlight-follow and tap-to-seek — silently,
because a plausible wrong segment looks like a right one.

| # | Reader | Site | Order today |
| --- | --- | --- | --- |
| B1 | Player segments contract | `server/segments_view.py:35` | **raw → adfree** (deliberate, documented) |
| B2 | Viewer text route | `server/routes/corpus_text_file.py:106` | serves what was asked; degrades adfree → raw → cleaned |

**This is why the resolver takes a `purpose` rather than having one precedence.** B1 and A3 read the
same two candidate filenames in opposite order, and both are correct. Collapsing them would break
one of them, and `segments_view.py`'s docstring exists precisely to stop someone doing that.

## Group C — raw by construction (3 sites)

These join `output_dir` with `transcript_file_path` and read it, with no variant logic and no
fallback. Since that path is always the plain `.txt`, they read raw — not by choice but by omission.

| # | Reader | Site |
| --- | --- | --- |
| C1 | Episode summary | `workflow/metadata_generation.py:2924` |
| C2 | Faithfulness check — content metadata | `workflow/metadata_generation.py:2402` |
| C3 | Faithfulness check — entity reconciliation | `workflow/metadata_generation.py:4268` |

C1 reads the raw text and then applies its own `PatternBasedCleaner` in memory, writing
`.cleaned.txt` as a byproduct (`:3142`, `:3235`). So the summariser has its **own** ad removal and
does not want the ad-free base. That is a third *reason* but not a third file — C1–C3 and Group B
both land on `<base>.txt`.

C2 and C3 check a summary against the transcript. Raw is defensible (the summary was made from raw)
but it is currently an accident rather than a decision. **Routing them through the resolver makes it
a decision**, with no behaviour change: they pass `purpose="timeline"` and keep reading raw.

## Group D — discovery guards, deliberately out of scope (5 sites)

These answer "which file **is** the canonical transcript", by excluding the derived variants. They
are the reason Group A/B/C have a single stored path to derive from — the opposite job from the
resolver's, and every one must keep excluding.

| # | Site | Excludes |
| --- | --- | --- |
| D1 | `workflow/episode_processor.py:2645` | `.adfree.` |
| D2 | `workflow/episode_processor.py:3965` | `.adfree.txt`, `.cleaned.txt` |
| D3 | `workflow/run_index.py:503` | `.adfree`, `.cleaned` |
| D4 | `providers/ml/diarization/pipeline.py:268` | `.adfree`, `.cleaned` |
| D5 | `workflow/stages/scraping.py:524` | `.segments.json`, `.cleaned.txt` |

D3's docstring records a past incident from exactly this confusion: `.adfree.admap.json` — a JSON
ad-map — was once handed back as "the transcript" and fed into the GI/KG cascade, exiting 0.

## Group E — reads both variants on purpose (1 site)

`kg/speaker_coherence.py:573-591` takes `segments` **and** `adfree_segments` and emits
`RAW_VS_ADFREE` when they name different speakers. It must keep receiving both; a resolver that
hands it one variant destroys the check.

## Group F — scripts and audits, inventoried not routed (10 sites)

Out of scope: each is a one-off or an audit with its own reason to pick a variant, and none feeds
the pipeline's coordinate space. Listed so a future sweep does not rediscover them as a surprise.

`scripts/tools/scrub_segments.py:46` · `scripts/tools/scrub_network_speakers.py:183` ·
`scripts/audit/transcript_pairing_audit.py:83` · `scripts/audit/attribution_ceiling.py:90` ·
`scripts/audit/corpus_speaker_audit.py:73` · `scripts/audit/ad_excision_coverage.py:108,125` ·
`scripts/audit/speaker_sync_audit.py:77` · `scripts/backfill/relabel_corpus.py:71` ·
`scripts/measure/2075_speaker_golden_set.py:191` · `scripts/build_synthetic_validation_corpus.py:911`

## Checked and excluded — not transcript readers

Found by the sweeps and confirmed to resolve something else. Recorded so the next sweep's hit count
matches this one.

- `workflow/metadata_generation.py:1032` — `.speakers.diagnostics.json`, same base-derivation idiom,
  one file, no variant choice.
- `search/indexer.py:79`, `:90` and `gi/explore.py:179` — GI and KG artifact paths.
- `server/app_content_source.py:111-118` — translates a run-relative path to a corpus-relative one.
  A path mapper with no variant preference; the resolver may end up using it.
- `utils/corpus_media.py:123-135` — normalizes the relpath for media pairing.

## What the golden is, and what it does not cover

`tests/integration/workflow/test_transcript_resolver_golden.py` records, per reader per
episode, the variant resolved today, into
`tests/fixtures/goldens/transcript_resolver.golden.json` — 80 episodes across two corpora,
chosen because they invert each other:

| Corpus | Episodes | `.adfree.*` | Layout |
| --- | --- | --- | --- |
| `viewer-validation-corpus/v3` | 40 | all | flat (the feed dir *is* the run root) |
| `app-validation-corpus/v3` | 40 | none | real `run_*` dirs |

**Covered:** the 11 resolution sites callable as functions today, on both branches of the
text-variant precedence.

**Not covered, and why it needed a separate test:** no episode in *either* corpus has both
segment sidecars — the viewer corpus ships only `.adfree.segments.json`, the app corpus only
`.segments.json`. So for A6, A8, A9 and B1 the golden records *which file exists*, never
*which one is preferred* — and the preference is the thing that diverges. A synthesized
episode carrying all four artifacts closes it, with explicit assertions rather than a
snapshot row, in `test_precedence_when_both_segment_sidecars_exist`.

**Not covered at all:** A3 (`_build_speakers_from_diarized_segments`, `metadata_generation.py:1199`).
Its precedence is inlined mid-function with nothing to call. Extracting it is part of the
refactor; it joins the golden then.

### Two things the golden exposed

**The viewer-validation corpus mis-shapes the player's input.** It has no raw
`.segments.json` at all, so on all 40 of its episodes `segments_relpaths_for_transcript`
falls through to `.adfree.segments.json` — the variant its own docstring calls "a
last-resort fallback". There it is the only resort. Two consequences: any player test over
that corpus exercises the fallback path rather than the production shape (prod writes both
sidecars), and the ad-free sidecar carries no `id` field, so `to_contract_segments` assigns
positional `seg_NNNN` ids instead of the diarizer's. Recorded, not changed — it is a
fixture-shape question, not this slice's.

**A9's inversion cannot be observed on either corpus.** `capability_audit._transcript_opening`
prefers raw over ad-free, against every other analysis reader. Because no episode has both
sidecars, the golden's `raw` (app) and `adfree` (viewer) rows are both *forced by absence*,
not evidence of a preference. The inversion is real in the source; the corpora cannot
demonstrate it. Stated here so the golden row is not read as more than it is.

## Counts

| Group | Sites | In this slice |
| --- | --- | --- |
| A — analysis | 10 | yes (A9 recorded, not changed) |
| B — timeline | 2 | yes |
| C — raw by construction | 3 | yes, as `timeline`, no behaviour change |
| D — discovery guards | 5 | no |
| E — both variants | 1 | no |
| F — scripts | 10 | no |

**15 readers to route. 16 sites left alone with a reason.** The number that matters for the arc: any
of those 15 is a place a translated episode could read the wrong coordinate space, which is why this
slice goes before the translation work rather than inside it.
