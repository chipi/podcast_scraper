# 0034: step-5 acceptance criteria

<!-- markdownlint-disable MD029 -- the criterion numbers are identifiers the record cites -->

Written by a fresh step-5 reviewer (an Opus advisor), 2026-10-26, from the goal, the change
record's prose and its dated section-4b prediction, and the method. The reviewer did NOT read
`m0034_talk_floor_names_removed.py` or its tests. It read, as data, the builder's find output
`scratchpad/m0034/list_snap1025.json` (556 entries, 397 distinct episode ids, roles 400 guest / 156
host, entries shaped `[episode_id, raw voice, name, role, null, null, "talk share N% < 5%"]`) and
the baseline scoreboard `scratchpad/m0034/baseline_scoreboard.json`.

Snapshot: `snapshot-prod-20261025` (scratch copy `scratchpad/snap1025/corpus`). Before = that copy.
After = a fresh copy of it with 0034 applied. Every number below comes from a stored script, with
its snapshot id and its join key.

## Definitions (used by every criterion)

- **Voice key:** (`episode_id`, raw diarizer voice id `SPEAKER_NN`). A per-file metadata path is
  allowed too. A run directory is not (D14).
- **Talk share:** the voice's talk seconds divided by the episode's total talk seconds, every voice
  counted, ad voices included. The builder states, in the record and before step 7, which file
  this is read from (`segments.json` durations, ad-free segments, or diagnostics `talk_time_s`) and
  which pipeline function it must equal (`pipeline._apply_talk_floor`). The floor is strict:
  share < 0.05.
- **Named:** the voice's served label is not a placeholder (`SPEAKER_NN`, "Speaker", "Speaker N",
  "Unknown Speaker", empty).
- **L:** the frozen list, built on this snapshot. **K:** L's entries whose verdict is `CORRECT` or
  `CORRECT_MISSPELLED`. **W:** L's entries whose verdict is `WRONG_PERSON`.
- **Insights on a voice:** the S7 raw-voice join. An insight belongs to the voice of the raw
  segment that holds its first SUPPORTED_BY quote's `char_start`. It is never joined by the
  insight's `speaker` field (§3 S3 note).
  - **Iw:** insights on W voices. Iw is split by their before-state:
    - **Iw_vis:** shown;
    - **Iw_unid:** hidden, `unidentified`;
    - **Iw_unk:** hidden, `unknown`;
    - **Iw_com:** hidden, `commercial`.
  - **Iw_none:** insights with no placeable quote, counted apart.
- **Qw:** quotes on W voices, by segment (not by `speaker_id`).

## A. Population and the list

1. **The population, re-derived by the step-8 reviewer's own code.** The pass bar is exactly 556
   voice keys in 397 episodes on this snapshot. The set must equal the frozen list key for key
   (set difference 0 both ways). The reviewer's code recomputes talk share from the file the
   builder names in the definitions.
   - Every voice with share in [0.045, 0.055] is listed with its share, to show the boundary.
   - Any key in one set and not the other is a FAIL until its cause is stated in the record.
2. **Not reached, as counts on this snapshot** (D11). The builder states each integer before step 7:
   - (a) every named voice at or above the floor, and every under-floor voice already unnamed;
   - (b) every placeholder-labelled under-floor voice (the record says 0 on 10-17; restate for
     10-25);
   - (c) the 14 episodes ingested after the snapshot;
   - (d) **D4 of #2094:** episodes where the under-floor W voice took the name of a person whose
     real voice (at or above the floor) is unnamed. 0034 removes the wrong name. It does not give
     the name to the right voice. These voices stay unnamed, and their count is a NOT COVERED line;
   - (e) the one episode in L with one name on two under-floor voices
     (`ef0fb53c-56a7-11f1-823b-6b194b39cee8`, one host's name), plus every episode where a name
     in L is also held by a voice at or above the floor. Count both and list them by voice key.
3. **The list is the snapshot's and is bound.**
   - `upgrade/lists/0034_talk_floor_names_removed.json` is committed. It does not exist in this
     worktree on 2026-10-26.
   - Each entry stores the voice's talk seconds and the episode total, as numbers, so the share can
     be recomputed without the migration. The string "talk share 0.2% < 5%" alone is not enough.
   - The list's sha256 appears in three places: the verdicts header `list_sha256`, the migration
     digest (`list_file`), and the receipts header.
   - An edit to any field of any entry refuses the load. A test on a copy proves this, and the
     test is able to fail.
4. **Registry order:** 0034 is registered before 0035 (0035 leaves under-floor voices to 0034,
   post-deploy plan item 17). The full `upgrade verify` on the after-clone shows 0034 PASS. Every
   other registered migration's line is identical, line for line, to the before-state.

## B. Verdict coverage (D13)

5. **Coverage:** `upgrade/verdicts/0034_talk_floor_names_removed.tsv` holds exactly 556 rows, one
   per key in L. Each row has a verdict in the closed set (`CORRECT`, `CORRECT_MISSPELLED`,
   `WRONG_PERSON`). Pass bars:
   - unreviewed 0, duplicate keys 0, keys outside L 0;
   - |K| + |W| = 556;
   - the header carries migration id, `reviewed_commit`, `list_sha256`, reviewer and date;
   - the verdicts' author is not the builder.
6. **Each verdict carries the line that decides it.**
   - Every row has a segment index and its text.
   - `CORRECT` and `CORRECT_MISSPELLED` need positive identity evidence: a self-introduction, an
     address answered by this voice, a host read in the host's own voice, or a diarization
     fragment of the same person shown by adjacent continuity. "The name occurs in the episode"
     is not evidence of identity. Neither is "this voice is in a promo".
   - Under the operator's rule (a wrong name is worse than a hidden insight), a voice the reader
     cannot decide is `WRONG_PERSON`. Its deciding line starts `undecidable:`, and those rows are
     counted on their own.
7. **Agreement with the labels.**
   - For each of the 10-17 measurement's 92 labelled under-floor names whose labelled run is
     still served on 10-25 and whose key is in L, compare the verdict with the label:
     - right, role error and host-read ad map to K;
     - wrong, spurious and non-own ad / promo map to W.
   - Every disagreement is listed and read by the step-8 reviewer, and each is settled in the
     verdicts file or in a logged gold correction.
   - Pass bar: 0 disagreements left unexplained.
8. **The clip class is settled before the verdicts are written. This is a reviewer finding; see
   the end of this file.**
   - Under the rule as stated, a clip of a public figure named as that figure is the voice's own
     person. It is therefore `CORRECT` and kept.
   - The prediction counted clips with "bad".
   - The record states which reading the operator chose. Kept clips are counted as their own
     class inside K.

## C. The change's own outcome numbers (snapshot-prod-20261025, before → after)

9. **O1, under-floor names served without a keep verdict.** Population: every served named voice
   with share < 0.05, counted with the definitions above.
   - Before: 556 − |K|. After: **0**.
   - Re-running the floor detector on the after-clone lists exactly the K keys: no W key, and no
     new key.
10. **O2 and O3, names off and names kept.** The pass bar is that the counts equal the verdicts
    exactly:
    - names removed: exactly |W| voice keys;
    - names kept byte-unchanged: exactly |K| voice keys;
    - written episodes: exactly the episodes holding a W key;
    - refused: 0 on the snapshot clone (moved-on refusals belong on prod, criterion 27).
    - Predicted K 66 / W 490. A difference from 66 / 490 is reported against the prediction and
      investigated. It does not pass or fail on its own, because the verdicts decide it. The
      prediction is restated as integers once the verdicts exist and before step 7 is measured.
11. **O4, the name is gone from every W voice's surfaces.** After, the removed name appears on 0
    of the W voice's:
    - segment labels and ad-free segment labels;
    - `<name>:` prefixes in `.txt`, `.cleaned.txt`, `.adfree.txt` and `.anon.txt` (if present),
      and turns / ad-free turns speakers;
    - diagnostics `name` / `resolved_name`;
    - quote `speaker_id`, insight `speaker`, GI `SPOKEN_BY` targets;
    - KG / bridge edges from this voice.

    The name's occurrences as speech, and on other voices, are unchanged in count. This is
    #2294's class, so it is checked by grep in both states and can fail.
12. **O5, the insights traded (principle 5; the operator's rule).**
    - Iw_vis go shown → hidden `unknown`;
    - Iw_unid go `unidentified` → `unknown`;
    - Iw_unk stay as they are;
    - **Iw_com stay `commercial`.** A W voice already typed `commercial` must not be retyped
      `unknown`. That would be a move out of a correctly typed non-person class, which principle
      5 does not allow.
    - Each set is listed by insight id in the gate report.
    - **Before the verdicts:** the builder states, as integers, the shown and hidden-by-type
      insights and quotes on all 556 L voices (the upper bound).
    - **After the verdicts:** the builder states the integers on W.
13. **O6, people who lose their last voice.** These are GI Person nodes, KG person nodes and
    bridge identities whose only voice in the episode was a W voice. Each is listed by
    (episode_id, person id) and must be removed or kept by one stated rule.
    - A person still held by another voice in the episode, a K voice included, keeps that voice's
      segments, insights, quotes, `SPOKEN_BY` edges and its KG / bridge entry. Only the edges from
      the W voice change.
    - A Person that other insights' `MENTIONS_PERSON` still target is kept, as in 0033's B5.
    - The roster / `context.json` outcome follows m0025's shared path, and every true fact lost
      with it is listed (compare 0031's lost-fact case, recorded in 0031's change record).

## D. The §3 scoreboard floor

Committed command:
`python scripts/measure/corpus_scoreboard.py --corpus <copy> --json <out>`, run before and after.
The baseline values are from `baseline_scoreboard.json` on snapshot-prod-20261025.

14. **Must not move. Exact integers, and for S12 the same quote ids:**
    - S1c 0 → 0;
    - S12 24 → 24, same ids;
    - `quotes_ref_unresolved` 244 → 244;
    - **S3 9,359 → 9,359.** A voice under 5% cannot hold 30% of the talk, so any move is a defect;
    - S2 commercial 94 → 94;
    - A1 382 hits / 290 episodes / 2 without ad-free text, unchanged;
    - S10b 2 → 2;
    - insights 130,401; quotes 252,651; episodes 2,966; `episodes_no_gi` 0;
      `metadata_unreadable` 0;
    - the ad-map mismatch set (`source_length != len(.txt)`, or an excised text not equal to the
      text at its range), corpus-wide: the same paths before and after.
15. **Must not rise:**
    - **S1n 7.** It may fall only by S1n ids on W voices, each listed. The 7 ids are compared as
      a set.
    - **S7-wrong** on every gold set still served (dev_v1, val_v1, val_v2, and val_v4 if pinned to
      this snapshot), through `scripts/measure/gold_insights.py --snapshot snapshot-prod-20261025`,
      before and after.
    - **S11** (non-person names on shown insights). It is a cross-check, but here a rise is a
      FAIL, because this change only removes names.
16. **May move only as listed:**
    - S1 16,810 → 16,810 + |Iw_vis| + |Iw_unid|, exactly;
    - S2 30,563 → 30,563 + |Iw_vis|, exactly;
    - S2 unidentified 13,659 → 13,659 − |Iw_unid|, exactly;
    - S2 unknown = S1.

    In each case the id-level set difference equals O5's lists, with 0 ids outside them.
    - S10a 190 → 190 + the episodes whose every shown insight was in Iw_vis, listed by episode id.
    - Cross-checks, each reported with its value:
      - S4 falls by exactly |Qw on named voices|;
      - S9 rises by the W voices' share;
      - S5 on prod /metrics only;
      - S8 unchanged.
17. **S7 / S6 on the gold sets.**
    - S7-correct may fall only by insights on W voices that a gold label calls the right person.
      Each such insight is listed and read (criterion 7). Pass bar: 0 left unread.
    - S6 is the number of labelled voices whose published name became worse, before vs after on
      the clone. `naming_gate.py` compares code modules, not corpus states, so the reviewer's own
      script is used, and its path is stated.
      - Pass bar: S6 = 0 after criterion-7 corrections.
      - Otherwise each regression is a verdict error to fix, and fixing it re-runs step 7.
    - Matcher limit: token-subset matches count as correct (§3).

## E. Every surface, the files, the undo

18. **Files written.** Before step 7, the builder states the exact suffix list and the integer
    count. A sha256 census of every file under the copy's `feeds/` shows:
    - the changed files equal the receipts exactly;
    - every changed file lies in an episode with a W key;
    - 0 changes elsewhere;
    - 0 changes in K-only episodes.
19. **Offsets.** In every written episode:
    - every quote slices to its text (episode S12 = 0, or its before value with the same ids);
    - every ad-free segment slices to its text;
    - ad maps carried (`source_length` = len(`.txt`), each excised text equal to the new text at
      its range, ranges and `chars_removed` unchanged);
    - turns `segments_sha256` equals the new segments;
    - metadata GI / KG counts equal a recount.

    A shifted offset FAILS. Removing a name shortens the prefix, so every offset after it moves.
20. **Derived-surface census (D17).** Run
    `python scripts/measure/derived_surface_census.py --corpus <after> --receipts <0034 receipts>`
    over every suffix under `metadata/` and `transcripts/` in the written episodes, and over every
    reader of speaker names and insight visibility.
    - Each surface is ticked written, ticked unaffected (with the reason it holds no speaker name
      or visibility), or listed stale with its ladder rung. The census must at least cover:
      - metadata (`content.speakers`, counts);
      - segments and ad-free segments;
      - the transcript files;
      - turns and ad-free turns;
      - speaker diagnostics;
      - `context.json`;
      - ad maps;
      - GI, KG, bridge;
      - the enrichment sidecars (`insight_sentiment`, `insight_density` and every other enricher
        output carrying a speaker or person id);
      - transcript manifests;
      - the corpus-level enrichments under `enrichments/`;
      - the app / player corpus outputs;
      - **prod's search-index rows** (criterion 28).
    - A surface left off the census is a FAIL.
21. **Undo on the full clone.** On a fresh copy of the after-clone:
    - restored N (N = criterion 18's integer), refused 0;
    - a sha256 census byte-identical to snapshot-prod-20261025;
    - 0034's verify reads pending, and the scoreboard equals the before JSON field for field;
    - apply, undo, apply, undo also ends refused 0 and identical;
    - backups go to `.podcast_scraper/upgrade-backups/0034/` only. Every other migration's backups
      and receipts are byte-identical.
22. **Paths exercised, counted on the clone** (principle 8): removed (person → unknown); removed
    from a voice whose name another voice keeps; removed from a `commercial` voice; removed with a
    Person kept by mentions; removed with the person's last voice; kept by `CORRECT`; kept by
    `CORRECT_MISSPELLED`; refused because moved on (0 on the clone, a test only); refused because
    no verdicts file. A path with 0 clone instances and no test is a NOT COVERED line.
23. **Mutations.** At step 8 the reviewer mutates the floor rule (`<` vs `<=`, the share
    denominator, the placeholder test), the verdict mapping (keep on `WRONG_PERSON`, remove on
    `CORRECT_MISSPELLED`), the list binding, the commercial-type guard, the shared-name guard and
    the offset shift, all against the repo tests.
    - Pass bar: 0 survivors, or an equivalent mutant argued in writing.

## F. Reading (what the step-8 reviewer must read)

24. **All of K** (expected about 66, more if clips are kept). A wrong name kept is the operator's
    worst error.
    - Pass bar: 0 wrong keeps.
    - Each wrong keep is corrected to `WRONG_PERSON`, the list sha and the verdicts are re-bound,
      and step 7 re-runs.
25. **W sample.**
    - A random sample of at least 60 W rows, stratified by feed, with the seed recorded;
    - all `undecidable:` rows;
    - all 156 host-role entries that end in W;
    - every row from criteria 2(e) and 7.
    - Each wrongly removed right name is corrected, and the random sample then grows to 120.
    - **More than 3 wrong removals in 120 sends the whole verdicts file back for a full re-read.**
26. **Spot checks able to fail** (D16). In both states, the reviewer reads the transcript view of
    3 W episodes and 2 K episodes it picks itself, including The Daily
    `SPEAKER_11` (0.7%; its line is a self-introduction inside a house promo, so expected K).
    - Pass bar: each episode reads as predicted in both states.

## G. Prod (steps 10 and 11)

27. **Prod's plan diff.** Prod produces its own per-episode plan, read-only, from the frozen list
    and is diffed against the clone's plan.
    - Every difference is listed with its cause. Expected causes:
      - an episode reprocessed or rewritten on prod since 10-25 (refused as moved on);
      - an episode evicted or deleted.
    - Unexplained differences: 0.
    - The outcome hook's exit 4 is cleared only by an accept file naming each difference and the
      operator's go.
    - The 14 post-snapshot episodes (plus any added since) are checked by the floor detector
      read-only. Every under-floor named voice there is listed with its pipeline trace
      (`kept_host` is allowed). Any other voice found is a pipeline defect, reported before apply.
28. **Search-index rows.** Count prod's rows that carry a removed name or a W person id for W
    voices' insights and quotes, read-only, before apply, and list them by episode. They are fixed
    by targeted row edits (ladder rung 3) on the operator's go, never by a rebuild. After the
    edits: 0 such rows.
29. **After apply.** Prod's apply counts equal the restated prediction minus the criterion-27
    refusals. Pass bars:
    - the prod scoreboard moves exactly as criteria 14-16 restated for prod's population;
    - verify is PASS;
    - the backups and receipts are owned by the corpus owner;
    - the apply runs in daytime, outside the 00:00-04:30Z nightly.

## H. Landing

30. **Binding.** The record's step 8 holds GO. The outcome sha (`build_outcome.py`) is bound to the
    code digest that includes the list file. `make ci-fast; echo MAKE_EXIT=$?` on the committed
    tree ends with MAKE_EXIT=0, and the log is attached.
31. **The MISSED line counts only with an operator line.** The record's `MISSED:` (built before
    steps 4 and 5) counts only once an `Operator accepted: 2026-10-.. ...` line follows it
    (template rule).

## What sends the work back

To step 3 or 4, never as a reworded pass:

- the population differs from 556 / 397 without a stated cause (1);
- a key without a verdict, or a keep without identity evidence (5, 6);
- O1 after ≠ 0, or a name left on any W surface (9, 11);
- any S1 / S2 / S10a movement outside the id lists; any S1c, S3, S12, A1, S2-commercial, S10b,
  insight or quote count move; S1n, S7-wrong or S11 rising (14-16);
- a commercial voice retyped unknown (12);
- a person with another voice losing that voice's attribution (13);
- an offset shift or a new ad-map mismatch (19);
- a changed file outside the receipts (18);
- the undo not byte-identical (21);
- a wrong keep, or more than 3 of 120 wrong removals (24, 25);
- an unexplained difference in prod's plan (27);
- a census surface left unticked (20).

## Intake check (step 1) against the method

Step 1 requires a reading list, one line each, of every open and recent issue, doc, scoreboard and
metric on the topic, with what is already known (§4 step 1, D9). The record's section 1 has none.
It has three bullets of rationale and no issue (`issue: none`). Missing, found by `gh issue list
--state all` and a repo search on 2026-10-26:

- **#2094 D4, "a tiny cluster consumes the guest's name"** (29 episodes: The Daily 10, The Flip
  5, Planet Money 4, Lenny's 3, Hard Fork 3). This is the cause behind the change, and its
  not-reached half (the guest stays unnamed) is not stated.
- **NAMING_LADDER.md v40 and v53.** The pipeline floor now has a host exemption: seated as host AND
  the LLM resolution calls it host, unless the voice speaks of the host in the third person. The
  record cites only v26. Served episodes get a verdict-based keep instead, so new and served
  episodes follow different rules. The record must state that.
- **NAMING-BRANCH-MERGE-PLAN-2026-10-24.md.**
  - Gate 2: the floor took 5 right host names off (1.5-4.4% of the talk).
  - "the talk floor taking self-introduced guests and audience questioners off (5)", among the 18
    harmful labelled changes.
  - Measured but not built: a guest exemption (3 right, 6 wrong).
  - These are the floor's measured costs, and the prediction does not use them.
- **NAMING_GATE_RUNBOOK.md**, `naming_shadow_report.py`: floor removals by LLM role, the proxy
  instrument for this change.
- **0035's record** and post-deploy plan items 16-18 (the ordering, and the overlap of 0035's
  under-floor exclusions).
- **#2364** (Patreon shout-outs published as guests), **#2333** (split clusters: fragments of a
  real person under the floor), **#2367**, **#2105** (role), **#2294** (removed names left in
  transcript text: the derived-surface risk), **#2359** (clip and "Play from" drift: offsets),
  **#2334** (prod record repair and guardrail), **#2200 / #1897** (the hidden-insight side of the
  trade), **#2096** (epic).
- **Numbers without script and snapshot** (D10): "88% of two-voice episodes named by older code";
  the 10-17 "92 / 84 / 1 / 6 / 1" (it cites the eval repo NAMING-RETHINK but not the script); "the
  four labelled segments" (undefined).
- **No 10-25 baseline for the change's own numbers.** `baseline_scoreboard.json` is not cited in
  the record, and the prediction's S1 / S2 / S10a moves are not integers (§4 step 4: "~" and
  "only by" are not predictions). Criterion 12's upper bound fixes this before the verdicts.

## Reviewer findings on the prediction (4b)

1. **66 / 490 is a ratio extrapolation** (11 / 92 × 556) from a sample that is not the list. The 21
   listed beyond the 10-17 measurement are mostly clips, promos and unknowns. L holds 156 host-role
   names, and the v40 evidence shows real hosts under the floor. A K well above 66 is plausible.
   It must be reported, never fitted.
2. **The clip question (criterion 8)** changes K and is an operator decision, not the verdict
   reader's.
3. **"S1 / S2 rise only by the visible insights of the 490"** omits Iw_unid (S1 also rises by
   unidentified → unknown) and the commercial case.
4. **Reprocess durability is not stated.** A K guest under the floor, for example a self-introduced
   guest, loses the name on any later reprocess, because the pipeline floor exempts hosts only.
   This is a NOT COVERED line.

## Not verified by this reviewer

- I did not read the m0034 code or its tests (by design).
- I did not recompute talk shares; only the list file's counts were checked (`jq`: 556 entries,
  397 distinct episode ids, one duplicated (episode, name) pair).
- I did not check whether the 10-17 labelled cases are still served on 10-25.
- I did not count S1 / S2 contributions per L voice, or which list voices are already typed
  commercial.
- I did not check whether `gold_insights.py` accepts this snapshot's gold pins.
- I did not check prod's search index or its plan.
- I checked per-language only by grepping the copy's metadata. Every `"language"` value found is
  `"en"`, so the per-language rows equal the totals on this snapshot. The 3,504 metadata files
  against 2,966 served episodes were not reconciled; I take them to be the served selection, but
  I have not verified that.
