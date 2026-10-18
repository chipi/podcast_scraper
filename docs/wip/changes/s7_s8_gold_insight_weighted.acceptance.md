# S7 / S8 instrument: step-5 acceptance criteria

Written by the fresh step-5 reviewer, 2026-10-18T10:05 UTC, before any code, after its own
reference measurement (`scratchpad/reviewer6/measure2.py` on `lost/clone30`, gold copy, labels_v1;
per-insight output `reviewer6/run2.jsonl`, 30,236 lines, sha256 fcf24dcf…, byte-identical twice).
At step 8 the instrument equals these numbers, or every difference is explained insight by insight.

Found first: 51 stale cases (9 / 39 / 3), all superseded by another served run; 700 served
(92 / 461 / 147); v447 confirmed (wrong 29 hidden, unnamed 3 hidden); every grounded quote cites an
`.adfree.txt`; ref forms relative, `../run_X/` and absolute `/app/output/…/run_A/../run_A/…`; one real
cross-run ref (val_v2 w038: a by-name join loses 41 insights, by path 34 correct / 5 wrong / 2
unnamed); 49 cases with an absolute `transcript_file_path`; speakerless 348 / 1,159 / 160 placed by
the join; every case `en`; `surfaceable` False or absent (val_v1: 74 flag-absent insights still
excluded by `insights_from_gi`); 0031 overlap 19 cases, entry voices before: correct 696 (695 hidden),
wrong 29 (v447 SPEAKER_00).

1. **Join:** the first SUPPORTED_BY edge from the insight in file order; its quote's `char_start`;
   the segment with a <= start < b under `gi.pipeline._segment_char_spans`; that segment's raw
   `speaker`. Equal to the reviewer's voice for all 30,236 insights (diff by set, case, insight id).
2. **Ref resolution by path:** `/app/output/<rest>` -> `<root>/<rest>`, others against the run dir,
   `..` normalised; segments = resolved `.txt` -> `.segments.json` (`x.adfree.txt` ->
   `x.adfree.segments.json`); 0 unresolved on clone30; w038 34 / 5 / 2.
3. **Still served:** `meta_relpath` exactly in `select_served_artifacts`; skipped 51 listed by id,
   included 700; each labelled voice id present in the served segments (misses printed, expected 0).
4. **Label set:** given per set or the newest `labels_vN` by numeric N, printed; `no_label` counted.
5. **Buckets, one per insight:** no_grounded_quote; quote_not_placeable (unresolved ref / no offsets
   / outside all segments / segment without raw speaker / spans None); voice_not_labelled;
   S8_undecidable (`not is_scored`); scored_no_name by role; S7_correct; S7_wrong; S7_unnamed
   (stored = `properties.speaker`, unnamed when empty or `SPEAKER(_\d+)?`; correct by
   `naming_gate._same`). On clone30:

   | set | insights | correct | wrong | unnamed | S8 | no-quote | not placeable | not labelled | scored no name |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | dev_v1 | 4,427 | 2,452 | 380 | 1,313 | 77 | 116 | 3 | 0 | 86 |
   | val_v1 | 19,737 | 11,612 | 2,157 | 4,225 | 930 | 424 | 25 | 0 | 364 |
   | val_v2 | 6,072 | 3,706 | 439 | 1,109 | 650 | 117 | 6 | 0 | 45 |

6. **Shown / hidden** (`surfaceable is False` = hidden): dev 2,340/112, 380/0, 0/1,313, 24/53;
   val_v1 11,128/484, 2,128/29, 279/3,946, 549/381; val_v2 3,607/99, 439/0, 1/1,108, 326/324
   (correct, wrong, unnamed, S8); flag-absent excluded by `insights_from_gi` printed (val_v1 74).
7. **Conservation by insight id,** asserted in code: buckets sum to 4,427 / 19,737 / 6,072 (30,236);
   duplicates 0. (Tightens the 09:50 prediction, which left out not-placeable and not-labelled.)
8. **Matcher tiers in S7 correct:** fold-exact / token-subset / fuzzy-or-title: dev 2,248/164/40,
   val_v1 10,629/441/542, val_v2 3,337/204/165; "strict S7 correct = fold-exact"; both limits stated
   (token-subset counts some wrong as correct; ASR spellings and junk names count as wrong).
9. **Sensitivity, reported:** wrong insights with a later quote on the stored name's voice (dev 33,
   val_v1 100, val_v2 18, val_v2 recounted after w038); stored speaker != joined label among wrong
   (dev 5); speakerless per bucket.
10. **Per language** for 5-7 (only `en`; a synthetic two-language test).
11. **Determinism and pinning:** JSON, no wall clock; snapshot id from the command line, the root,
    the ledger's 0030 entry, label dirs, a sha256 over case and label files, a sha256 over the served
    metadata list; per-insight rows; two runs byte-identical.
12. **Read-only:** nothing newer under clone30 or gold after a run; runs on a read-only 3-case copy.
13. **Tests and mutations:** scenario tests for every path above; mutations killed: join by stored
    label, last quote, sorted quotes, `Path(ref).name`, no `/app/output` mapping, `is_scored` without
    `low`, flag-absent as hidden, `SPEAKER_xx` as named, the matcher swapped either way, served by
    episode id, lexicographic label dirs, no conservation assert, not-placeable folded into no-quote.
14. **Known answers:** v447 before 29 / 3 and after 0031 B.3 wrong 0, unnamed 32 (accepted); 51
    skipped (accepted, ids added); conservation tightened (7); the 19 overlap cases printed per entry
    voice before -> after (wrong 29 before; correct shown/hidden 1/695 before; unnamed).
15. **Docs:** method §3 S7/S8 rows updated to the instrument's numbers with the snapshot id (S8 77 /
    930 / 650).

Out of scope: tightening `same_person`; the undecidable bucket; feed/loss match; val_v1 held-out
status; S6; non-English gold; S7 outside gold; any corpus write; first-quote vs majority (reported in 9).

Reviewer's concerns: the gold sets existed only in the scratch copy (restored 2026-10-18 to
`.test_outputs/naming_gold` in the main checkout, 1,502 files); `corpus_scoreboard._Texts` joins by
file name inside the served run, so it loses cross-run refs (w038) and the scoreboard's S3 join has
the same defect (separate finding, not measured).
