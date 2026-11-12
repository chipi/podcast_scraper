---
id: s6a_insight_spelling
class: full
status: landed
issue: none (the person-identity design S6, first half; plan in podcast-scraper-eval-data docs/wip/NAME-CONSOLIDATION-ARC-V3-2026-10-30.md)
touches:
  - src/podcast_scraper/identity/display.py
  - src/podcast_scraper/identity/ledger.py
  - src/podcast_scraper/server/app_gi_view.py
  - src/podcast_scraper/server/*.py
  - src/podcast_scraper/server/routes/app_episodes.py
  - src/podcast_scraper/mcp/tools/gi.py
---

# s6a_insight_spelling — one spelling per person on the insight surfaces, from the ledger

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Plan V3 S6 (E5), first half: the
ledger records each joined group's spelling, a display function reads it, and the insight surfaces
apply it for the first time. Search's speaker mapping and D2 item 7 (hidden insights in search) are
S6's next part; the other surfaces of the S1 inventory follow. Person names are kept out of this
record; the measurements are in eval-data `data/eval/name-consolidation/s6/`.

## 1. Intake

- S1's surface inventory (eval-data `s1/SURFACES-2026-10-30.md`): the insight surfaces read the
  speaker's name from the episode's own Person node (`server/app_gi_view.py` `_speaker_name`), and
  apply no identity map.
- the identity ledger, decision 1: the ledger records the display spelling with its evidence.
- Plan V3: the spelling is publisher text first, then frequency. Held-out decides against
  frequency-only.
- S6 step 1 (eval-data `s6/RESULT-2026-10-31.md`): publisher-first is kept.

## 2. Outcome and baseline

Snapshot-prod-20261030, the ledger at S5o (c4c350684), every served episode's surfaced insights
(`s6/insight_surface_r2.py`):

- 99,847 surfaced insights with a named speaker; 14,168 of them in a cross-episode group;
- **R2 on this surface: 42 of the 190 groups that appear are shown under two or more spellings.**
- Truth set, spelling wrong where the show text names the person (R3c): held-out 13 of 33,
  training 19 of 35, under today's frequency spelling.

## 3. Diagnosis by loss

- Each insight shows the spelling its own episode's Person node carries. One person, spelled two ways
  in two episodes, shows two ways, and the group's spelling is decided nowhere.
- Of the spellings still wrong after the rule, 12 of 13 on training have no member spelled right (the
  transcription misspells every episode). That needs a link from the show text to the group, a new
  rule not in this change.

## 4. Prediction

Dated: 2026-10-31

The change:

- **`build_ledger`:** each person cluster records `display_name`, chosen by publisher text first,
  and `display_evidence` (`publisher` with the episode count, or `frequency`). Publisher text first
  means: the member that the most episodes' publisher text names (`guards.EpisodeEvidence.
  publisher_named`), then episode count, then the shorter name, then the name; with no such member,
  the canonical.
- **`identity/display.py`:** a per-corpus view, `person_display(root)`, read from the served
  ledger and cached by the corpus token. `shown(person_id, own_name)` returns the group's
  `display_name` for a joined id, and `own_name` otherwise. With no valid ledger it returns `own_name`
  (no joins means each episode's own name, the identity ledger).
- **`insights_from_gi`** takes `people=`. Every listener surface passes it, and the name report too, so
  R1/R2 measure what is served. Leaving it out keeps today's behaviour.
- `RULE_VERSION` is unchanged at `/3`: no join changes. A ledger without `display_name` (D1's `/2`)
  shows each episode's own name.

Predicted on the snapshot, through the real path (`s6/predict_through_gi_view.py`: `insights_from_gi`
before and after a patch of only `_speaker_name`, every quote's speaker compared). This replaces a
first estimate that read only each insight's first quote:

- **Default path** (recap, board, auto-picks, OG counts, MCP `episode_insights`):
  - 103,845 insights, 201,221 quotes;
  - **2,770 quote speakers change spelling, in 1,559 insights**;
  - R2 (groups shown in two or more spellings) 40 → **0**.
- **Fallback path** (episode panel, moments, captured rail):
  - 113,024 insights;
  - 2,855 quote speakers change, in 1,609 insights;
  - R2 40 → **0**.
- **Both paths:**
  - the insight set and order are identical (0 insights added or dropped);
  - 0 unnamed quotes become named;
  - 0 insights show two spellings of one group after.
- **"The insight's speaker"** is each of its quotes' speakers; `AppInsight` has no speaker field.
  Every quote is mapped, not only the first.
- **Authored `speaker_name`:** 25,157 quotes carry one, though the code comment says the field is
  empty. It wins before any lookup (`app_gi_view.py`), has no id to map, and stays unchanged.
  R2 still reaches 0 with those quotes counted.
- **Raw labels:** `_speaker_name` returns None for them before the display is applied, so none is
  ever spelled through a group.
- **The prototype's spelling and the ledger's `display_name`** must be equal for every group. This is
  checked at step 7; any difference stops the change.
- **Cache:** `person_display` keys on `perf_cache.corpus_mtime`, which includes the ledger file, so a
  rebuilt ledger changes the shown spelling without a restart. It reads the ledger once per token,
  beside `served_id_map`'s own cached read.
- **D1's `/2` ledger on prod during D2's window:** it has no `display_name`. Every changed surface is
  then byte-identical to today, each episode's own name. A test runs one episode against a `/2`
  ledger.
- **`quality/name_report.py`:** passes `people=` on its served counters, so R2 moves by design.
  R1 and every other number must be unchanged (must not move). `name_report._insight_names`
  builds names itself: it gets the same mapping, or the change says why not.
- **R3c on the truth set:** held-out 13 → 9 of 33, training 19 → 13 of 35.
- **Joins:** none added, removed or moved; the ledger's join list is byte-identical to S5o's.
- **Person-page URLs:** unchanged. Routes key on the person id, and an id is never renamed by a
  spelling.
- **Must not move:**
  - insight counts, order, tiers and visibility on every surface;
  - unjoined speakers' names;
  - the unattributed fallback;
  - the raw-label rule (S3).
- **R6:** every join that newly reaches an insight surface is listed. These are the joins inside
  the 190 groups that appear on the panel; their reading is the step-8 list.

## 5. Acceptance

Fresh reader, 2026-10-31, a sonnet reviewer subagent with read-only tools, under a minute: "not
complete". Rows added:

- quote speakers, not insights;
- both paths, measured through the real function rather than my first estimate;
- authored names; raw labels;
- prototype spelling = ledger `display_name`;
- the cache key; the `/2` window;
- name_report's own numbers.

Spot checks for step 8:

- one group member shown in the episode panel and on the board, with the same spelling;
- an authored quote unchanged;
- an insight with quotes from two joined speakers, showing one spelling;
- one episode against a `/2` ledger, byte-identical to today.

## 6. Build

Tests, written first (`tests/unit/podcast_scraper/identity/test_display.py`, invented names):

- the ledger records the group spelling and its evidence;
- every episode shows the group spelling, and without the view each episode's own;
- no ledger: each episode's own name;
- a `/2` ledger without spellings changes nothing;
- a rebuilt ledger is picked up without a restart;
- an unjoined person keeps its name.
They failed before the code: the module did not exist.

Code:

- `identity/ledger.py`: `_add_display_spellings`, run in `build_ledger` with the guards' own
  evidence. Each person cluster gets `display_name` and `display_evidence`.
- `identity/display.py`: `PersonDisplay.shown`; `display_from_ledger`; `person_display(root)`,
  cached by `perf_cache.corpus_mtime`.
- `server/app_gi_view.py`: `insights_from_gi(..., people=)`, applied through `_shown` on the
  quote's `speaker_id` and `SPOKEN_BY` lookups. An authored `speaker_name` still wins unchanged.
- Every listener caller passes `person_display(root)`: board, auto-picks, recap, trending, the
  captured rail (both), episode notes, the episode routes (panel, the second insights read, moments),
  moments' `score_candidates` and `pick_moments`, the goal reel (both), and MCP `episode_insights`.
- The OG card and the quality reports' counts are unchanged; they count insights and show no names.
- **A correction to section 4: `quality/name_report.py` is not changed.** Its `_insight_names` is
  the scan's own resolution, kept so its roster mode reproduces the scan (its docstring). Changing
  it would change what that report measures. S6a's R2 is measured by its gate instead.
- Test stubs of `insights_from_gi` in `test_app_auto_picks.py` and `test_app_trending_episodes.py`
  now accept the new keyword.

## 7. Measure

- **Gate** (eval-data `s6/s6a_gate.py`, the real code over snapshot-prod-20261030, the ledger built
  in memory, nothing written):
  - default path: 2,770 quote speakers changed in 1,559 insights;
  - fallback path: 2,855 quote speakers changed in 1,609 insights;
  - both equal to the prediction;
  - insight sets identical; 0 unnamed quotes named;
  - the ledger's `display_name` equals the prototype's spelling for every group;
  - `S6A GATE: PASS`.
  The served output therefore equals the prototype's, and the prototype measured R2 (groups shown
  in two or more spellings) going 40 → 0 on both paths.
- **Mutants, both killed by the gate:** frequency instead of publisher text; the view ignored.
- **Tests:** identity, server, quality and MCP unit areas, `2136 passed`; flake8 and mypy clean.

## 8. Verdict

MISSED: no fresh verdict on this rule on its own. D2's step-8 reviews were of migrations 0039–0042 (their records, section 8), over the combined ledger.

Operator accepted: 2026-11-03 this step was covered only by D2 as stated above (A1 close-out of feat/ingestion-arc, option 1).

## 9. Operator review

MISSED: no operator review of this record on its own. The operator accepted D2 on 2026-11-03 (`d2_name_consolidation.md`).

Operator accepted: 2026-11-03 this step was covered only by D2 as stated above (A1 close-out of feat/ingestion-arc, option 1).

## 10. Before apply / deploy

MISSED: no plan diff of its own. The rule shipped in the D2 deploy (`sha-8906696`); prod's plan for D2 is in records 0039–0042, section 10.

Operator accepted: 2026-11-03 this step was covered only by D2 as stated above (A1 close-out of feat/ingestion-arc, option 1).

## 11. After apply / deploy

MISSED: not re-measured on prod on its own. D2's prod apply and its after-measurement are in records 0039–0042, section 11.

Operator accepted: 2026-11-03 this step was covered only by D2 as stated above (A1 close-out of feat/ingestion-arc, option 1).

## 12. Retrospective

MISSED: no retrospective of its own.

Operator accepted: 2026-11-03 this step was covered only by D2 as stated above (A1 close-out of feat/ingestion-arc, option 1).

## Not covered

- Search's speaker mapping and hidden insights in search (D2 item 7): the next part of S6.
- The other surfaces in the S1 inventory: transcript segments, people chips, person card,
  perspectives, top persons, trending, OG, MCP relational tools and emails. They come after this.
- Frozen copies (highlights, favourites, sent emails) cannot change at read time.
- The 12 groups where no member is spelled right.
