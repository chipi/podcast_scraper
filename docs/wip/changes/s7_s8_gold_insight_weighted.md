---
id: s7_s8_gold_insight_weighted
class: full           # an instrument, not a corpus change; run through every step anyway
status: open
issue: 2380
touches:
  - scripts/measure/gold_insights.py
  - src/podcast_scraper/quality/gold_insights.py
---

# S7 / S8 — insight-weighted gold numbers (method step 0, before 0031's step 10)

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §3 (S7, S8) and §4.
Operator 2026-10-18: 0031 waits for S7/S8 and the migration outcome hook (step 0, option B).

## 1. Intake

- Method §3 S7: insight → its first grounded quote → `char_start` → the segment holding it → that
  segment's RAW `speaker` (the pipeline's `_segment_char_spans`); among insights on scored, nameable
  labelled voices: correct / wrong person / unnamed; insights with no grounded quote counted apart.
  S8: insights on voices labelled `confidence: low` or `role: unknown`.
- The v0.1 prototype (`scratchpad/lost/goldgate.py`, 2026-10-17) joined by stored label, which
  double-counts when one name sits on two voices (795 insights on val_v1): NOT VALID.
- Gold sets (`.test_outputs/naming_gold/`, copied to `scratchpad/gold/naming_gold`): dev_v1 101
  cases, val_v1 500, val_v2 150 (751 labelled cases); 51 no longer the served run (dev 9, val_v1 39,
  val_v2 3); val_v1 is a development-tier set now (read while tuning).
- Matcher: `kg.speaker_coherence.same_person` accepts token subsets ("Josh" = "Josh Elman").
- Speakerless insights (no `speaker`): 348 / 1,159 / 160 (dev / val_v1 / val_v2) fell out of v0.1.
- Reuse: `quality/corpus_scoreboard.py` already has the raw-voice quote join (`_Texts`,
  `_voice_of_quote`); `naming_gate.py` has `is_scored` / `outcome` and the label loader.
- 0031's overlap, measured 2026-10-18 (`scratchpad/lost/gold0031.py`): 19 of its 54 episodes are
  gold cases on the served run; gold agrees with the review on 19 of 19.

## 2. Outcome and baseline

The instrument's own numbers, per gold set and per language, on a snapshot clone: S7 correct /
wrong person / unnamed (each split shown / hidden), S8 undecidable, insights with no grounded quote,
insights on unscored voices, cases skipped as no longer served. Baseline: none valid (v0.1 is not).

## 3. Diagnosis by loss

Not a loss diagnosis: the defect to remove is the v0.1 join (by stored label, so a name on two
voices counts their insights twice) and the speakerless gap (an insight with no `speaker` dropped
from the denominator although its quote places it on a voice).

## 4. Prediction

Dated: 2026-10-18T09:50

Known answers, stated before the instrument exists (on `lost/clone30`, gold copy
`scratchpad/gold/naming_gold`, labels = each set's newest `labels_v*`):

- val_v1 v447 (The Flip, Jessica Espinoza): before 0031, SPEAKER_00 (labelled "Jessica Espinoza",
  high) carries 29 insights stored as "Justin Norman": wrong person 29, all hidden; SPEAKER_01
  (labelled "Justin Norman", medium) carries 3, stored unnamed: unnamed 3. After 0031 revision B.3
  (applied on a fresh clone30 copy): v447 has wrong person 0 and unnamed 32.
- Cases skipped as no longer the served run: 51 (dev 9, val_v1 39, val_v2 3), as in the method.
- Conservation, exact per set: correct + wrong + unnamed + S8 undecidable + unscored-role +
  no-grounded-quote = every Insight node of the included cases; no insight counted twice (the v0.1
  defect), checked by insight id.
- The 19 cases 0031 overlaps: their entry voices hold no wrong-person insight after 0031.
- What it must not do: write anything; read a label set it was not given; join by stored label.

## 5. Acceptance

Dated: 2026-10-18T10:05

The fresh reviewer's criteria, written before code after its own reference measurement: [s7_s8_gold_insight_weighted.acceptance.md](s7_s8_gold_insight_weighted.acceptance.md).

## 6. Build

`src/podcast_scraper/quality/gold_insights.py` (module), `scripts/measure/gold_insights.py`
(CLI), `tests/unit/podcast_scraper/quality/test_gold_insights.py` (18 scenario tests on real-shaped
corpora built per test: first-quote order, one name on two voices, speakerless, cross-run ref,
`/app/output` refs, stale case, S8 / nameless labels, unlabelled voice, outside-segment and
no-offset quotes, numeric label versions, missing label file, two languages, matcher tiers, parity
with `naming_gate.is_scored` / `_same`, read-only and deterministic, conservation refusal).

**After the step-8 FAIL (criterion 13: 10 of the reviewer's 20 mutants survived).** Back to step 6,
as the reviewer ruled: no definition or number changes. 19 tests added (37 in all), one per
survivor and the gaps it named:
- a nameless label that is not scored (unknown role, low confidence) is S8 (R3);
- a served case whose label file is missing (R12);
- the plain-text path: segments without offsets placed by text length, the alignment guard's
  refusal, and no segments (`spans_none`, not `outside_segments`; R5, R19);
- a ref to a missing transcript is `unresolved_ref` (R15);
- the later-quote sensitivity (R8);
- the flag-absent count against `insights_from_gi`'s own exclusions, no text and `drop` (R9);
- a labelled voice missing from the segments (R11);
- `--labels` / `labels=` pins a label directory (R17; the CLI now exposes it);
- the bare `SPEAKER` id is unnamed (R20);
- a quote through an `.adfree.txt` ref uses the ad-free segments;
- `entry_voices`.

Also: a served case with no GI file is counted (`gi_missing`), and `ledger_0030` (always empty: 0030
was applied by a script, not the runner) is replaced by `ledger_applied`, what the ledger records.
The snapshot label is the corpus state: `snapshot-prod-20261016+0030` for `lost/clone30`.

## 7. Measure

On `lost/clone30`, gold `.test_outputs/naming_gold` (labels_v1), snapshot
`snapshot-prod-20261016`: every bucket, shown/hidden split, matcher tier, skipped list (51:
9 / 39 / 3), flag-absent excluded count (val_v1 74), sensitivity counters (later quote on the stored
name's voice 33 / 100 / 18; stored name not the segment label among wrong: dev 5) and speakerless
distribution equal the step-5 reviewer's independent reference measurement (its acceptance file).
Two runs byte-identical; nothing written under the clone or the gold sets. Mutations: the 14 the
criteria list, all killed (`scratchpad/lost/s7mut.py`).

Known answers: v447 before: wrong 29 hidden (SPEAKER_00), unnamed 3 hidden (SPEAKER_01); after 0031
B.4 (applied on a fresh clone30 copy): wrong 0, unnamed 32. 0031's entry voices covered by gold
(23): before correct 766 (695 hidden, 71 shown), wrong 29 (hidden), S8 66; after correct 766 (all
shown), wrong 0, unnamed 29 (hidden), S8 66.

**Re-measured after the step-6 fixes** (`--snapshot snapshot-prod-20261016+0030`, report
`scratchpad/lost/s7_b.json`): per-insight rows equal to the reviewed run's (`lost/s7_a.json`,
`lost/s7cmp.py`); every set summary equal; `gi_missing` 0 in dev_v1, val_v1, val_v2;
`ledger_applied` 0001 to 0029 (no 0030). Mutations: the reviewer's 20 (`reviewer6/mut8.py`, run
unchanged on a scratch mirror `scratchpad/s7mirror`, never the repo; baseline 37 passed): 20 of 20
killed. My earlier 14 (`lost/s7mut.py`) edit the repo in place and were not re-run; that script is
retired for this reason.

**The known-answer population, restated.** 0031's entry voices that gold covers are 23 voices in 22
cases (the 76 keys of `review_names()`), not the 19 cases the dated prediction named: that count
came from the older 54-entry plan. The numbers above are on the 23.

## 8. Verdict

Verdict: NO-GO (2026-10-18, step-8 reviewer)

**Step-8 reviewer (fresh advisor), 2026-10-18: FAIL on criterion 13, back to step 6.** Its file:
`scratchpad/reviewer6/verdicts_s7s8_step8.tsv`. PASS: the join (30,236 rows, 0 differing from its
own, before and after 0031 B.4), ref resolution on read-only real cases, criteria 3 to 10, four
byte-identical reports, read-only, the known answers. FAIL: 10 of its 20 mutants survived the 18
tests. Partial: the snapshot label (clone30 is +0030) and the always-empty `ledger_0030`. Small:
a missing GI file not counted; the file list; cite the diff for "re-derived independently".

Verdict: GO WITH CONDITIONS (2026-10-18, step-8 reviewer, re-verdict)

**Step-8 reviewer re-verdict, 2026-10-18.** Its file: `scratchpad/reviewer6/verdicts_s7s8_step8_r2.tsv`
(its own mirror `reviewer6/mirror2` and its own runs). PASS on criteria 1 to 12, 14 and 15: 30,236
rows, 0 differing from its reference; three reports byte-identical (sha256 dea74e10…, `lost/s7_b.json`
among them); nothing written under clone30 or gold; `snapshot-prod-20261016+0030`, `ledger_applied`
0001 to 0029 (no 0030); `gi_missing` 0; the population restated as 23 voices in 22 cases.
Criterion 13 PASS WITH CONDITION: baseline 37 passed; its 20 mutants all killed; of its 6 new
mutants on the new code (`reviewer6/mut8b.py`), 5 killed, N6 (the CLI drops `--labels`) survived.
Condition, before commit: one test that runs the CLI with `--labels` and catches N6.

**Condition met** (before commit): `test_the_cli_passes_labels_through` runs `main()` with
`--labels g=labels_v1`, without it, and with a malformed value. On a fresh scratch mirror
(baseline 38 passed) the reviewer's `mut8b.py`, run unchanged: 6 of 6 killed, N6 included.

## 9. Operator review

_TODO_

## 10. Before apply / deploy

Not applicable (no corpus write); S7/S8 then run on 0031 (before and after) for its step 10.

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Extension 2026-10-24: S7 on an eval run's output (`--voice-map`)

**Why:** the labels name the labelled run's voice ids and metadata path. An eval run writes a new
run directory, and a rediarize renumbers the voices, so a lookup by the after voice id can name
the wrong person. The first R1-GI S7 (`s7_r1gi_v4.json`, no map) read 390 wrong on 21 cases,
against 122 on the same cases at baseline. Two of those 21 cases (y041 and y042) are in the after
copy but not in the run's index; the map leaves them out, so the mapped R1-GI reading has 19 cases.

**What changed:**

- The eval scorer (`podcast-scraper-eval-data` `scripts/eval/naming/score.py`) also writes
  `<rows>.voicemap.json`. Per case it holds the after metadata path and the after voice of EVERY
  labelled voice; the rows cover scored labels only.
- `gold_insights.py --voice-map` does four things:
  - follows each case to its after path;
  - re-keys the labels to the after voices;
  - refuses a map made for another corpus;
  - skips and names cases the map does not cover.
- **Merged voices:** labels that disagree on one after voice put its insights in
  `voice_map_ambiguous` (y029's collapse: 6 labels on one voice). Labels that agree count as one
  person the labelled run split (y018 and y020, one host labelled twice) and are applied; they are
  listed under `labelled_voices_merged_same_label`.
- Without `--voice-map` the report is the same as before, apart from the new bucket's zero counts.
  `s7_base_v4` was re-run and its rows are identical.

**Tests:** 9 new scenario tests. 7 mutants, all killed: no re-key, never ambiguous, always
ambiguous, agreement ignoring scored-ness, no ambiguous bucket, no after path, no corpus guard.

**First readings** (val_v4, against the baseline on the same cases; the GI reran, so insight
counts differ and the comparison is by bucket, not per insight):

| run | cases | insights b / a | correct b / a | wrong b / a | unnamed b / a | ambiguous |
| --- | --- | --- | --- | --- | --- | --- |
| R1-GI | 19 | 971 / 1,099 | 620 / 915 | 122 / 171 | 223 / 5 | 0 |
| R4 | 20 | 1,023 / 1,161 | 671 / 929 | 122 / 175 | 223 / 5 | 35 |
| R5 | 20 | 1,023 / 1,161 | 671 / 929 | 122 / 175 | 223 / 5 | 35 |

**Where the wrong insights moved** (R1-GI; R4 adds 4 in y017, a host / guest swap):

- **Fixed:** y002 (the guest's 39 insights, named as another person before).
- **New wrong:** y007 (35) and y008 (52). Both are the guest's voice served as the host's name
  by `llm_resolution` after the relabel. Names per case: the eval repo's DGX block plan.
- **Unchanged:** the y019 swap (84).

S7 here measures the SERVED names (the ladder). The shadow namers' names are not on the insights.

**Not covered:**

- shadow namers' S7 (it needs their names swapped onto the insights);
- R1 and R2: their after-states were overwritten;
- R3: its after-state was recorded as copied to `R3_after`, which is no longer in scratch, and
  when it went is not known;
- R6: its 162 cases may still be in the run copy, but that was not checked or measured;
- held-out sets;
- the matcher limits above, which still apply;
- per-insight before/after, impossible because the GI reran.

## Not covered

_TODO_
