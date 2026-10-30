---
id: 0038_raw_label_speakers_removed
class: full
status: open
issue: none (RFC-131 plan V3, N11 in D1; operator, 2026-10-30: "I would expect migration to clean that up")
touches:
  - src/podcast_scraper/upgrade/migrations/m0038_raw_label_speakers_removed.py
  - src/podcast_scraper/upgrade/registry.py
---

# 0038 — served episodes lose the raw diarization labels their files still credit as speakers

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Part of the D1 deploy
(record [d1_name_consolidation](d1_name_consolidation.md)). Person names are kept out of this public
record; per-episode lists are in the eval-data repo (`data/eval/name-consolidation/n11/`).

## 1. Intake

- **What is wrong.** Before S3 (1802035eb), a segment still carrying a raw diarization label
  (`SPEAKER_07`) and no voice type fell through `gi/pipeline._voice_type_for_char_range` to a
  Person node named after the label, with SPOKEN_BY edges and surfaceable insights. S3 fixed the
  pipeline (such a voice is `unknown`) and the read time (a raw-label Person resolves to no name; an
  insight with no named quote is unattributed). The files of already-served episodes still hold the
  old shape.
- **Measured before (prod report c71876a, 2026-10-30, corpus token 1793330465.0899494):**
  `data_quotes:raw` 107 quotes in 23 episodes, `data_bridge_identities:raw` 28 in 24 episodes.
  Every shown surface is 0 (S3).
- **Reading list:**
  - `gi/pipeline.py`: `_voice_type_for_char_range`, `_voice_flags_for_insight` (an insight's type
    is its FIRST grounded quote's voice), `_apply_voice_flags`, `_quote_props`,
    `_apply_route_and_tag` (routing and salience read `surfaceable`), `VOICE_TYPE_NOT_SURFACEABLE`.
  - `server/app_gi_view.py` (S3's read-time rule); `quality/name_report.py` `_data_rows` (the
    counts above).
  - `builders/bridge_builder.py` (the bridge's identity shape); m0036 (the last bridge migration).
  - `search/indexer.py`: Quote rows carry `speaker_id`, so the index holds the raw id too.
  - `upgrade/migrations/m0029`, `voice_rewrite_derived.py`: the voice-by-voice rewrite. Not reused:
    it rewrites voices from segments, and these episodes' GI is older than their segments (below).
  - Open issues read: #2197 (Person nodes that are not one person), #2200 / #2198 (hidden insights,
    zero-visible episodes), #2352. None covers this shape.

## 2. Outcome and baseline

Snapshot **snapshot-prod-20261030** (release created 2026-10-30T11:48Z; archive sha256
7e366e70…7035f; corpus token mtime 1793330465, newest_kg_count 3069, equal to prod's at the
report runs). Scripts: eval-data `data/eval/name-consolidation/n11/` (to be committed with the
build), run with the branch's `src`.

The change's own numbers, on served episodes:

| # | Definition | Baseline |
|---|---|---|
| N1 | GI Person nodes whose name is a raw label (`name_report.label_kind == "raw"`) | 28 in 24 episodes |
| N2 | Quote nodes whose `speaker_id` is such a Person | 107 |
| N3 | SPOKEN_BY edges to such a Person | 107 |
| N4 | Insight nodes whose FIRST grounded quote (file order of SUPPORTED_BY) is N2 | 53 in 15 episodes, all surfaceable, no voice type |
| N5 | Insight nodes whose `speaker` field is a raw label | 53 |
| N6 | Bridge identities whose `display_name` is a raw label | 28 in 24 episodes |
| N7 | KG Person nodes whose name is a raw label | 0 |
| N8 | Search-index Quote rows whose `speaker_id` is an N1 id | to measure on the clone at step 7 |

Scoreboard (S1, S1n, S1c, S2, S3, S10a, S10b, S12, A1): taken on the clone by
`corpus_change_gate.py` at step 7; not taken yet.

## 3. Diagnosis by loss

One cause, already fixed for new episodes (S3's pipeline change). Every N2 quote, placed on the
segment that holds its `char_start` in the transcript its `transcript_ref` names (the pipeline's
`_segment_char_spans`): raw label and no voice type 77; typed `unidentified` 9, `unknown` 1,
`cameo` 1; not placed 19 (each slices its own text; the placing join finds no segment, cause not
diagnosed). No quote's segment now carries a name, so the migration never has to name anyone.

## 4. Prediction

Dated: 2026-10-31

The rule (the pipeline's own, applied to the stored files):

- an N2 quote loses `speaker_id` and its SPOKEN_BY edge, and takes `speaker_voice_type` from its
  segment's current type, or `unknown` when the segment carries a raw label or cannot be placed;
- an N4 insight takes its first quote's voice type and `surfaceable: false`; its routing tag and
  salience are recomputed by the pipeline's `_apply_route_and_tag`; an N5 `speaker` field is
  removed;
- an N1 Person node is removed (every one is orphaned once its edges go);
- an N6 bridge identity is removed;
- an N8 index row's `speaker_id` is cleared in place (no index rebuild);
- the metadata's GI counts are recomputed.

Integers on snapshot-prod-20261030, after:

| # | After | Population |
|---|---|---|
| N1, N2, N3, N5, N6 | 0 | reached: 24 episodes |
| N4 | 53 insights change: 49 to `unknown`, 4 to `unidentified`, all `surfaceable: false` | 15 episodes |
| N7 | 0 (unchanged) | |
| N8 | 0 | |
| Episodes written | 24; refused 0 | not reached: every other served episode (0 writes) |
| S1 hidden by our defect | +49 exactly (the N4 `unknown` set, by insight id) | trade rule: an accepted loss, below |
| S2 all hidden | +53 exactly (unknown +49, unidentified +4) | trade rule, below |
| S1n named but hidden | unchanged (the 53 carry no name) | must not move |
| S1c consistency | 0 | must not move |
| S12 quotes off their text | unchanged (offsets are not touched) | must not move |
| S10a, S10b episodes with no visible insight | S10a unchanged against the branch's read time (S3 already keeps these 53 off the attributed surfaces); S10b may rise only for an episode whose every insight is in N4: to measure, each listed | |
| S3, A1 | unchanged | must not move |

**Trade-rule moves, by identity:** the 53 insights of N4 (ids in eval-data
`n11/n4_insights.json`, to be committed). They are not "leaving a wrong name": their speaker is a
raw label. They are the operator's decision of 2026-10-30 ("an insight whose speaker is a raw
label is unattributed"), applied to the files. **This needs the operator's explicit acceptance of
S1 +49 and S2 +53 before apply** (method §6: the operator approves predictions that trade numbers).

**What must not move:** every file outside the 24 episodes; in the 24, every node, edge and field
not named above (other quotes, other insights, topics, entities, transcripts, segments, offsets);
organisations; the identity ledger (no N1 id is a join member: to check at step 7).

## 5. Acceptance

_TODO_: the prediction read by a fresh agent (§4a), before any code.

## 6. Build

_TODO_

## 7. Measure

_TODO_

## 8. Verdict

_TODO_

## 9. Operator review

_TODO_

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Not covered

- The 19 quotes the placing join cannot place: why it fails is not diagnosed; they are typed
  `unknown` by the rule above, because their own record credits a raw label.
- Enrichment sidecars keyed by insight id (`insight_sentiment`, `insight_density`): whether they
  read `surfaceable` is checked in the step-7 census, not yet.
