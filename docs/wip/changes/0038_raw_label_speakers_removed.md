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
  - #2304 (episodes whose insight voices carry only `SPEAKER_xx`: touches S10b and S2) and #2294
    (raw ids in quote text, out of scope): added at step 5.
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

Scoreboard baseline, whole served corpus, `scripts/measure/corpus_scoreboard.py --corpus <snapshot>
--label snapshot-prod-20261030` with the branch's `src` (eval-data
`n11/scoreboard_snapshot-prod-20261030_branch.json`): episodes 3069, insights 135,390; S1 17,811;
S1n 7; S1c 0; S2 31,421 (unknown 17,811, unidentified 13,516, commercial 94); S3 9,217; S10a 193;
S10b 2; quotes 262,618; S12 24 (244 with no transcript file); A1 396 hits in 297 episodes.

N8 cannot be measured on the snapshot: the backup does not carry the search index. It is counted on
prod, read-only, at step 10.

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

Integers on snapshot-prod-20261030, after. The scoreboard deltas are exact: the edits above applied
to copies of the 24 episodes and each scored with `corpus_scoreboard.score_episode` before and
after (eval-data `n11/n11_simulate.py`, `n11/sim_deltas.json`).

| # | Before → after | Population |
|---|---|---|
| N1, N2, N3, N5, N6 | 28, 107, 107, 53, 28 → 0 | reached: 24 episodes |
| N4 | 53 insights change: 49 to `unknown`, 4 to `unidentified`; `surfaceable` per the pipeline's `VOICE_TYPE_NOT_SURFACEABLE` (all 53 false: no first quote is `cameo`) | 15 episodes, ids in `n11/n4_insights.json`; 7 of the 53 have an unplaced first quote (ids in `n11/rows5.json`) |
| N2 quotes not first of any insight | 54; their insights keep type and `surfaceable` (21 insights hold only such raw quotes) | must not move |
| N7 | 0 → 0 | |
| N8 | measured on prod at step 10 → 0 | |
| Metadata GI counts | 0 written: none of the 24 carries a `grounded_insights` count block | |
| Episodes written | 24; refused 0; second apply writes 0 | not reached: the other 3,045 served episodes (0 writes) |
| S1 hidden by our defect | 17,811 → 17,860 (+49, the N4 `unknown` set) | trade rule: accepted, below |
| S2 all hidden | 31,421 → 31,474 (+53: unknown +49, unidentified +4) | trade rule: accepted, below |
| S3 principal unidentified | 9,217 → 9,219 (+2, episode 987ebd8e: its 2 new `unidentified` insights are on a principal voice) | trade rule: accepted |
| S10a zero visible | 193 → 194 (+1, episode 631e269e: its last 2 visible insights have a raw-label first quote and a named later one) | cross-check: listed, accepted |
| S10b panel empty | 2 → 2 | must not move |
| S1n | 7 → 7 | must not move |
| S1c | 0 → 0 | must not move |
| S12 | 24 → 24 | must not move |
| A1 | 396 / 297 → unchanged | must not move |
| Insights | 135,390 → 135,390 (no insight removed; ids unchanged) | must not move |
| Served episodes | 3,069 (newest_kg_count 3069) | must not move |

**Trade-rule moves, by identity:** the 53 insights of N4 (ids in eval-data
`n11/n4_insights.json`, to be committed). They are not "leaving a wrong name": their speaker is a
raw label. They are the operator's decision of 2026-10-30 ("an insight whose speaker is a raw
label is unattributed"), applied to the files. Operator accepted: 2026-10-31 S1 +49 and S2 +53, the 53 N4 insights by id (quiz in the
naming session).

Operator accepted: 2026-10-31 S3 +2 (episode 987ebd8e) and S10a +1 (episode 631e269e).

**What must not move:** every search-index row other than the N8 Quote rows (Insight rows carry no
speaker and no `surfaceable`); the enrichment sidecars keyed by insight id (`insight_sentiment`,
`insight_density`: insight ids do not change; whether a sidecar reads `surfaceable` is in the
step-7 census); every file outside the 24 episodes; in the 24, every node, edge and field
not named above (other quotes, other insights, topics, entities, transcripts, segments, offsets);
organisations; the identity ledger (no N1 id is a join member: to check at step 7).

## 5. Acceptance

Prediction read 2026-10-31 by a fresh agent (reviewer, Sonnet, 44 s, read the record, the method and
the tracker only). Its rows, all added above: N8 and the scoreboard rows with values; S10b with an
integer; the baselines of S1 and S2; metadata counts as an integer; the 15 episodes by id; N2 quotes
that are not first quotes as a must-not-move row; insight ids and the enrichment sidecars; index
rows other than N8; the served-episode count; second apply writes 0; which N4 insights have an
unplaced first quote. Reading list: #2304 and #2294 added below.

Spot checks for step 8, each able to fail:

- (a) the N4 insights whose first quote is the `cameo` or the `unknown` one, if any: after, type and
  `surfaceable` as `VOICE_TYPE_NOT_SURFACEABLE` says (the simulation shows none is a first quote);
- (b) 3 of the 19 unplaced quotes' episodes: before, raw `speaker_id`, SPOKEN_BY edge and Person
  node; after, none of the three, voice type `unknown`, quote `text` and offsets byte-equal;
- (c) one N4 insight with a named second quote: before, `speaker` is a raw label; after, `speaker`
  removed and the named quote keeps its `speaker_id`;
- (d) one episode: a count of the raw Person's name in the gi file, the bridge and (on prod) the
  index rows: before > 0, after 0 in each;
- (e) one N1 id: not in the identity ledger.

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
