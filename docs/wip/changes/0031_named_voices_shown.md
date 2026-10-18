---
id: 0031_named_voices_shown
class: full
status: open
issue: 2380
touches:
  - src/podcast_scraper/upgrade/migrations/m0031_named_voices_shown.py
  - src/podcast_scraper/upgrade/migrations/m0029_naming_arc_voices_repaired.py
  - src/podcast_scraper/upgrade/migrations/m0025_one_person_one_entry.py
  - src/podcast_scraper/upgrade/reviewer_verdicts.py
  - src/podcast_scraper/upgrade/migrations/voice_rewrite_derived.py
  - src/podcast_scraper/upgrade/file_rewrite.py
  - src/podcast_scraper/upgrade/read_recorder.py
---

# 0031 — the voices 0029 named, shown the way a fresh run shows a named person

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. The first change to go
through the change gate ([README.md](README.md)). It was built BEFORE the gate existed, on
2026-10-18, straight from the task, and steps 1, 4 and 5 were skipped. They are marked MISSED
below with what was skipped; the gate lets the change land only once the operator accepts each.

Operator accepted: 2026-10-18 steps 1, 2 (snapshot not fresh), 4, 5 and the 0031 tests written
after the code, as missed: "I accept the missed steps if it makes no sense to do them now after
all is done." Also: "I am ok with having similar names in 31 as long as we plan to deal with it
later since this is a different class of problems" (the misspelled-name class, e.g. "Deborah
Kamen" for "Debra Kamin"). Step 0 (S7/S8 and the migration hook not built) is NOT waived by this.

## 1. Intake

MISSED: no reading list was written before the build. Written after it, 2026-10-18:

- #2380, 0031 (now 0032) is defective: a voice it names keeps its insights hidden; the gain path.
- `docs/wip/EVAL-0030-0031-2026-10-17.md`: 0032 (then 0031) held on S1n +1,085.
- `docs/wip/NAMING-STRATEGY-2026-10-17.md` item 2: fix the rewrite 0031 inherits from m0025.
- `docs/guides/CORPUS_CHANGE_METHOD.md` v0.4: §3 metric definitions, S1n gating.
- `m0029_naming_arc_voices_repaired.py`: 83 voices, frozen list, applied on prod.
- `m0025_one_person_one_entry.py` `_segments_by_voice` / `_gi_by_voice`: the shared rewrite.
- `gi/pipeline.py` `_apply_voice_flags`, `_voice_type_for_char_range`, `_apply_route_and_tag`: how a
  fresh run decides an insight is hidden (the voice of its first grounded quote).
- Prod's report 2026-10-17: 0029 verify FAIL, 1 episode left (Migraine, a 10-13 run).


**Revision B (2026-10-18), read before its code:**

- Step-8 review of 97c551101 (`scratchpad/reviewer4/`, verdicts file `verdicts_0031.tsv`, 78 reads):
  NO-GO, condition A: The Flip "Unlocking Gender-Smart Capital At Scale", SPEAKER_00 is Jessica
  Espinoza, not 0029's "Justin Norman" (re-read by the builder: the voice holds 65% of the talk; the
  host voice introduces "Jessica Espinoza, the CEO of 2X Global"; the co-host says "Jessica, I've
  got a comment"). Conditions E (unguarded mutation paths) and #2380's "verify fails on insights".
- Operator 2026-10-18: no hand-made per-entry exceptions; a repeatable rule. Chosen: B, the step-8
  reviewer's wrong-person verdicts become an input every list migration reads; then C, S7 on gold.

## 2. Outcome and baseline

Targets: S1n (gating), S1, S2 by type, S3, S10a; must not move: S1c, S12, A1, insights, quotes.
Baseline on the 2026-10-16 prod snapshot with 0030 applied (`lost/clone30`, 2,685 episodes):
S1 16,211, S1n 1,786, S1c 0, S2 28,247 (unknown 16,211, unidentified 11,956, commercial 80),
S3 8,236, S10a 190, S10b 2, S12 24, A1 357. Command: `scripts/measure/corpus_scoreboard.py
--corpus <clone30>`.

MISSED: the snapshot is not fresh (method D12). It is from 2026-10-16; prod has been reprocessed
since (prod listed 1 pending 0029 episode where the snapshot shows 2). Step 10's prod plan diff is
where the difference is measured.

## 3. Diagnosis by loss

`scripts/measure/hidden_by_migration.py --corpus <clone30> --receipts
naming_arc_voices_repaired.jsonl --backup-tag 0029`: of the 73 served episodes 0029 wrote, S1n is
0 before 0029 and 1,779 after; S1 1,504 → 1,509; S2 2,235 → 2,240. So 1,779 of the corpus's 1,786
S1n are 0029's, and 0029 hid no new insight: it named voices and left their insights hidden.

Read: the Migraine episode (Peter Attia Drive #406) after 0029 on its 09-05 run: SPEAKER_01 named
Brian Grosberg on 3,268 segments, every one still `voice_type: unknown`; 169 quotes with
`speaker_id: person:brian-grosberg` and `speaker_voice_type: unknown`; 92 insights
`speaker: Brian Grosberg`, `surfaceable: false`, `routing_tag: connect`. Cause: the m0025 gain path
sets the name and never clears the voice type on segments, quotes or insights.

Two of 0029's episodes were reprocessed after it (10-13 Migraine, 10-14 Catherine Liu) and came
back with the guest unnamed (same voice, same words: 66% and 86% of the talk).


**Revision B:** no automatic rule separates this case on the data
(`scratchpad/lost/addr_features.py`, `host_shape.py`, all 76 entries): vocatives score 0 on the
wrong entry and up to 11 on correct guests; "named host, stated guest, dominant voice" fires on 4,
1 wrong and 3 correct (25% precision). One wrong case cannot train a rule (rules must generalise);
the designed rule waits for S7 (C). Reviewer's hand read: 1 wrong of 53 shown voices (95% Wilson
0.3%-9.9%), 29 of 1,925 shown insights.

## 4. Prediction

MISSED: no prediction was written before the build. The numbers in step 7 were measured first,
so a prediction written now would not be one.


Dated: 2026-10-18T08:15

**Revision B prediction (SUPERSEDED at 08:24: the step-5 review changed "withhold" to "unname")**, on `lost/clone30` (2026-10-16 snapshot + 0030), against 97c551101's
measured after-state. The change: a generic reader of the step-8 verdicts file
(`upgrade/verdicts/<migration id>.tsv`); 0031 skips any entry marked WRONG PERSON (counted
`withheld_by_review`, not a refusal); verify also checks that no insight whose first quote is on a
listed named voice stays hidden; tests for the reviewer's surviving mutations.

- Episodes written 53 (was 54), refused 0, withheld by review 1 (episode 5555e937, SPEAKER_00).
- Insights shown 1,896 (was 1,925); voices shown 51, named 2.
- S1 14,841 (+29 vs 14,812); S1n 36 (+29 vs 7); S1c 0; S2 26,351 (unknown 14,841, unidentified
  11,430, commercial 80); S3 7,710; S10a 178 (+1: that episode's only visible insights are the 29
  withheld); S10b 2; S12 24; A1 357; insights 116,595; quotes 224,862.
- Against the 0030 baseline: S1 −1,370, S1n −1,750, S2 −1,896, S3 −526, S10a −12.
- Files written 389 (396 − the episode's 7); undo restores 389, refused 0, feeds byte-identical;
  verify on clone30 fails with 53 to write, on the after-state passes.
- Must not move: S1c, S12, A1, S10b, insights, quotes; nothing shown→hidden; no transcript text
  in the 51 re-given episodes.
- Wrong-person among shown insights: 0 of the reviewer's verdicts (the one WRONG PERSON withheld).

Dated: 2026-10-18T08:24

**Revision B prediction, unname variant** (replaces 08:15). On `lost/clone30`, code = 97c551101 +
revision B. The change: a shared reader of a committed verdicts file
(`src/podcast_scraper/upgrade/verdicts/<migration id>.tsv`, schema in step 5's criteria); 0031
UNNAMES an entry marked WRONG PERSON through m0025's existing unnaming path, REFUSES an entry with no
verdict (`unreviewed`), shows CORRECT and MISSPELLED; verify adds the insight check, the
unname check and unreviewed = 0; tests for every mutation the reviewers listed.

- Branch A (verdicts = the converted reviewer4 file only, Samo Burja unreviewed): written 53
  (50 shown, 2 named, 1 unnamed by review), refused 1 (unreviewed), insights shown 1,896, files
  393, verify FAILS (unreviewed 1). Run only if the step-8 reads are not in first.
- Branch B (verdicts cover all 54; Samo Burja CORRECT): written 54 (51 shown, 2 named, 1 unnamed
  by review), refused 0, unreviewed 0, insights shown 1,896, files 400, verify passes.
- Both branches: S1 14,841; S1n 7; S1c 0; S2 26,351 (unknown 14,841, unidentified 11,430,
  commercial 80); S3 7,710; S10a 178; S10b 2; S12 24; A1 357; insights 116,595; quotes 224,862.
  Against clone30: S1 −1,370, S1n −1,779, S2 −1,896, S3 −526, S10a −12. Against 97c551101: S1 and
  S2 +29, exactly E's 29 insight ids.
- E after: transcripts byte-identical to its 0029 backup; segments, gi, kg, bridge, diagnostics
  parsed-equal to it; 54 quotes at pre-0029 offsets, 0 off text; roster and `context.basic.hosts`
  lose the unplaced "Justin Norman" (m0025's existing rule, a listed loss: a true, voice-less host
  fact in one episode); metadata keeps its later `feed.*` / `episode.licence` values.
- Population (one key, episode_id + raw voice, 0029's 83 entries): 33 named-typed-unknown and 19
  named-typed-unidentified (to show, 52), 2 unnamed again by a reprocess (to name), 22 already done,
  7 unnaming entries (not 0031's): 83. Of the 52, 1 is unnamed by review (E), 51 shown.
- Undo (branch B): restores 400, refuses 0, feeds byte-identical to clone30, verify pending again.
- Must not move: S1c, S12, A1, S10b, insights, quotes; 0 shown→hidden.

Dated: 2026-10-18T08:44

**Revision B.2 prediction** (after the 08:40 measurement, which matched every 08:24 integer). The
derived-surface census (`scripts/measure/derived_surface_census.py`) found what B leaves stale and
the rewrite should carry itself, as 0032 already does: the ad maps of the 3 episodes whose
transcript text changes (`source_length` 39,329 vs 39,290; 28,471 vs 28,519; 122,865 vs 123,377)
and the metadata GI/KG counts where KG persons are added or removed. The change: 0032's
`_carry_admap` / `_recount` and their offset helpers move to a shared module
(`upgrade/migrations/voice_rewrite_derived.py`) that 0031 uses (0032 keeps its copy until its own
record opens it); verify also fails when a written episode's ad map does not match its transcript.

- Scoreboard: unchanged from the 08:40 measurement (S1 14,841; S1n 7; S1c 0; S2 26,351; S3 7,710;
  S10a 178; S10b 2; S12 24; A1 357; insights 116,595; quotes 224,862).
- Counts: written 54, refused 0, unnamed by review 1, insights shown 1,896; `admaps_carried` 3,
  `admaps_left_for_0030` 0; `metadata_counts_synced` 3 (E −1 KG person, the 2 named +1 each).
- Files written 403 (400 + 3 ad maps; the metadata files are already among the 400).
- Census after: ad maps stale 0 and 0 off their transcript; insight_density / insight_sentiment
  stale 54 (refreshed by re-enrichment, not by the migration); manifests 3, anon 2 stale.
- Undo restores 403, refuses 0, feeds byte-identical to clone30.

Dated: 2026-10-18T09:27

**Revision B.3 prediction** (after the 08:55 B.2 measurement and the step-8 NO-GO). The change:
(1) a WRONG_PERSON verdict restores the entry's stored-before-0029 name AND role, so E's SPEAKER_00
loses `speaker_role: host` on its segments and diagnostics returns to role `unknown`; verify's
unname check covers the role; (2) undo counts a receipt whose file already holds its before-state as
restored, not refused (apply, undo, apply, undo on one corpus reports 0 refused); (3) verify fails
when an episode 0031 wrote has an entry without a verdict, and when the verdicts file's
`list_sha256` is not the current list's; (4) the conversion script asks the migration for its list;
(5) tests that kill the reviewer's 5 survivors (verify last-quote, a WRONG_PERSON name mismatch
refused, the unnamed person's GI Person node removed, the diff-path ad-map shift, `_move_offset`'s
gap rule) and a test that keeps 0032's copy of the moved functions AST-equal to the shared module.

- Scoreboard: identical to B.2 (S1 14,841; S1n 7; S1c 0; S2 26,351; S3 7,710; S10a 178; S10b 2;
  S12 24; A1 357; insights 116,595; quotes 224,862): roles enter no scoreboard number.
- written 54, refused 0, unnamed_by_review 1, insights shown 1,896, files written 403, admaps
  carried 3.
- `metadata_counts_synced` 22, these episodes (episode_id, metadata KG nodes/edges before -> after):
  - fbc4b8e7-0333-488d-ab49-604ac46974b9  23/22 -> 24/22
  - b769a5e0-4748-47f6-b9a3-230677e033a4  26/25 -> 27/25
  - 5125f8f2-9724-4f85-9a6d-f0a9c7a5e41a  27/26 -> 28/26
  - e52aca03-ed61-40e1-88e8-7b27006fd4a1  26/25 -> 27/25
  - ccfba8b3-e1ab-4df5-9714-e570baff1961  23/22 -> 24/22
  - 93ccb619-fd1e-4693-89da-da481dc36d3e  18/17 -> 19/17
  - 2cfba12e-e38e-4df9-a3d7-2a20565e1247  27/26 -> 28/26
  - substack:post:159080671  27/27 -> 28/27
  - turkeybooktalk.podbean.com/3fdee49b-80cd-39e2-8217-9ab3cb585c16  28/28 -> 29/28
  - turkeybooktalk.podbean.com/f9412b5a-a368-36ed-ae9d-43146d42f609  25/25 -> 26/25
  - fdfc0bbc-cc66-465c-bfd2-dfee6fdca2b5  26/27 -> 27/27
  - 741c8a11-b0f7-48e0-bf40-76de35bc29ff  27/28 -> 26/26
  - db9dbb14-4257-4c04-936f-359a0a978c67  28/29 -> 28/27
  - 4d691819-3665-4d16-aee1-8655c9322f9b  28/28 -> 27/26
  - f785186b-12b9-4336-95e5-4242cc0b6ddb  24/24 -> 24/22
  - 1f41bb8a-925a-4d3e-95fc-f3790db1619e  21/21 -> 20/19
  - 7ef41fcf-3127-4992-ad89-e3ea46e3ee78  28/28 -> 28/26
  - ef54f007-9c46-4d15-b039-1183f989a26d  24/24 -> 23/22
  - 916c5e0a-b2ab-4935-ae27-7dc4eeb8612a  28/28 -> 27/26
  - 557e200ecdc72f5714593ffe1bde  27/27 -> 28/27
  - 8ff9162f-9f56-46b1-9663-73fe4f90d885  27/27 -> 28/27
  - b36d42b2-42fc-4723-be93-8187189dc793  23/23 -> 24/23
- E after: segments, adfree segments, gi and speaker diagnostics parsed-equal to its 0029 backup
  (no `speaker_role` on SPEAKER_00, diagnostics role `unknown`); transcripts byte-identical to it.
- Undo: 403 restored, 0 refused, feeds byte-identical, verify pending (54); after a second
  apply+undo on the same copy, still 0 refused.

Dated: 2026-10-18T09:29

**B.3 prediction, restated before code** (the step-5 review at 09:27 found the 09:27 text claimed
more than the change does). E after B.3 is parsed-equal to its 0029 backup EXCEPT at 535 known
places, accepted here before measuring: `speaker_label` absent (backup: "SPEAKER_00") on 506
SPEAKER_00 segment rows (253 in each segment file), and `speaker` absent (backup: "SPEAKER_00") on
its 29 insights. m0025's unnaming path removes the key instead of writing the voice id; it is shared
with 0029's and 0032's unnaming entries, so it stays; what the transcript view shows is the same
(SPEAKER_00 253, SPEAKER_01 80, "Unidentified speaker" 66, no "Host"). Everything else in E is
parsed-equal to the backup: 0 SPEAKER_00 rows carry `speaker_role`; diagnostics role `unknown`,
named False, resolved_name SPEAKER_00. The role restore is generic: a WRONG_PERSON verdict sets the
entry's listed old role (None removes it). Undo after apply, undo, apply, undo on one copy: restored
403 (receipts de-duplicated by file, the last receipt per file), refused 0. A changed verdicts file
reopens step 7; its sha256 is in the receipt header and the plan line.

Dated: 2026-10-18T10:05

**Revision B.4 prediction** (after the B.3 measurement at 2026-10-18T09:43:06 and its GO WITH CONDITIONS).
The change: (1) C1, the restore rule completed: a WRONG_PERSON verdict puts back the listed
(old name, old role) in both branches (old name set: the voice is renamed to it with the old role,
or loses the role when it is None), verify expects exactly that state, and the plan treats an entry
already in it as done; (2) the plan contract the outcome hook needs: `dry_run_report` rows carry
every counter (`counts`) and the sha256 of every input file the decision reads (`inputs`), and the
module exposes `verdicts_file()`; (3) verify on a corpus where no listed episode is served passes
with "no listed episode is served here" instead of "checked nothing" (the fixture and fresh corpora).
On `lost/clone30` the outcome is unchanged from B.3, byte for byte: no entry takes the C1 path
(E's old name is None) and the plan rows only gain fields: scoreboard as B.3; written 54, refused 0,
unnamed 1, insights shown 1,896, files 403, ad maps carried 3, `metadata_counts_synced` 22 (the same
ids); the 245 changed files identical to B.3's; undo 403 / 0. New tests: the two C1 cases, the plan
rows' counts and inputs (an input file edited changes the row), verify on an empty corpus passes.

Dated: 2026-10-18T11:10

**Revision B.5 prediction** (after the B.4 step-8 GO WITH CONDITIONS, 2026-10-18). The change is
verify only, plus the record: (1) C-a: verify counts every served metadata file (`served`). Where
it is 0 and the root has no `corpus_manifest.json`, the root is not a corpus and verify FAILS. With
a manifest, it is an empty corpus and verify passes with "an empty corpus: no listed episode is
served here". This is narrower than C-a asked ("FAIL when the total is 0"). The registry's
end-to-end contract (`test_no_faiss_index_is_clean_noop`) runs every migration on a fresh corpus,
which has a manifest and no episodes, and requires verify to pass there. A fresh install is a real
case. (2) C-b: the served selection's inputs, at step 6 below. The plan, the apply and every file
written are untouched, so on `lost/clone30` the outcome is B.4's byte for byte: plan rows 54,
refused 0, files 403, the same 245 changed files, the same scoreboard, undo 403 / 0. Verify on the
after-state passes with `served=2685`, `served_listed=71`, and B.4's other counters
(`insights_checked=2454`, `unnamed_checked=1`, `unreviewed_unchanged=22`, `admaps_checked=71`,
`metadata_unreadable=0`). New tests: a root with no manifest fails, and an empty corpus passes and
says so. Mutants: the manifest check dropped, and `served` never counted, each killed.

Dated: 2026-10-18T12:59

**Revision B.6 prediction** (after the B.5 GO and its clean-commit measurement at 11:46:49; found
by `make ci-fast`). Two defects, no change to what 0031 plans, writes or verifies on a corpus that
serves episodes:
1. `_serves_nothing` (B.5) looks for metadata with a plain `rglob` under `feeds/`, which also walks
   the dot-directories the corpus excludes (snapshots, upgrade backups). The repo's rule
   (`test_corpus_walk.py::test_no_corpus_scanner_walks_without_the_exclusion_rule`, red in
   test-fast) requires `corpus_rglob`. Fixed to `corpus_rglob(root / "feeds", "*.metadata.json")`.
   Metadata only under an excluded dot-directory then counts as none on disk, so such a root with
   a manifest passes as an empty corpus. New test for that case.
2. Docstrings (`make docstrings` requires 100%; 99.7%, 13 names, all from this arc):
   `read_recorder.Recorder.note` / `.inputs`, `reviewer_verdicts.Verdicts.get` / `verdicts_path`,
   `voice_rewrite_derived.DerivedSurfacesEpisode.plan`, `change_gate.Record.touches`, and seven in
   `outcome_hook.py`. Docstrings only, no code.
Plus any change the outcome hook's step-8 review asks of `outcome_hook.py`, restated here before
it is made. None of them changes 0031's plan or apply.

On `lost/clone30L`, from the commit that carries B.6: served is 2,685, so `_serves_nothing` is never
called and nothing it decides changes. Plan rows 54, refused 0, byte-identical to B.5's
(`plan_sha256` e85a2670…); apply writes the same 245 changed files, byte-identical to B.5's
(`diff -rq` against `lost/gateworkB5` = 0); gating moves S1 −1,370, S1n −1,779, S2 −1,896,
S3 −526, S10a −12; verify passes with served=2685, served_listed=71 and B.5's other counters; undo
403 / 0. What changes: the code digest (the files are in 0031's import closure), so the outcome is
rebuilt and its sha changes. Mutant: `corpus_rglob` swapped back to `rglob` is killed by the corpus
walk test.

Dated: 2026-10-18T13:01

**Revision B.6 restated before its test and before the walk's root is set** (the line was edited
to `corpus_rglob(root / "feeds", ...)` as predicted at 12:59; it changes again here). The served
selection walks `corpus_rglob(root, "*.metadata.json")`, the whole root
(`corpus_selection.select_served_artifacts`). B.5's check looked only under `feeds/`, so metadata
in any other layout was invisible to it, and such a corpus with a broken selection passed as "an
empty corpus": the vacuous pass C-a exists to stop. That was B.5's defect. The walk is therefore
`corpus_rglob(root, "*.metadata.json")`: exactly the files the selection sees, with every
dot-directory excluded at any depth. This is the reviewer's third mutant of 12:59 taken as the
design, not killed. Tests:
- metadata only under a dot-directory (`feeds/.x/...` and `.trash/...`) with a manifest → PASS,
  "an empty corpus";
- metadata outside `feeds/` (a run directory at the root) with the selection returning nothing →
  FAIL, "metadata on disk, none served";
- the B.5 case (under `feeds/`) → still FAIL.
Mutants: `corpus_rglob` back to `rglob` (the repo-rule test); the walk rooted at `feeds/` (the
root-run test). On `lost/clone30L` nothing else changes from the 12:59 prediction.

## 5. Acceptance

Dated: 2026-10-18T13:01

Revision B.6 restated: the reviewer's amendment (13:01) replaces item 1 of its 12:59 block (the walk over the whole root, tests a to d).

Dated: 2026-10-18T12:59

Revision B.6: the reviewer's amendment (12:59), appended to the acceptance file. It adds a gate: ci-fast green on the committed B.6 tree.

Dated: 2026-10-18T11:11

Revision B.5: the reviewer's amendment (11:11), appended to the acceptance file. It adds: metadata on disk with none served fails, and a manifest must parse as a JSON object.

Dated: 2026-10-18T10:06

Revision B.4: the reviewer's amendment (10:06), appended to the acceptance file.

Dated: 2026-10-18T09:27

Revision B.3: the reviewer's amendment (09:27), appended to [the acceptance file](0031_named_voices_shown.acceptance-B.md).

Dated: 2026-10-18T08:45

Revision B.2: the reviewer's amendment, appended to the same file (below the 08:22 block).

Dated: 2026-10-18T08:22

Revision B: the fresh reviewer's criteria, written before any revision-B code:
[0031_named_voices_shown.acceptance-B.md](0031_named_voices_shown.acceptance-B.md).

MISSED: no reviewer wrote acceptance criteria before the code. The fresh reviewer was launched
only after the build (step 8).

## 6. Build

- 1bc9d18d0: the m0025 gain path drops the voice type on the named voice's segments and quotes,
  and lifts the hiding on insights whose first SUPPORTED_BY quote is on it, re-routed with
  `_apply_route_and_tag`. Test first: `test_a_voice_that_gains_a_name_is_shown_as_a_fresh_run_would_show_it`
  failed on the defect before the fix; each of the four parts removed alone fails it.
- 97c551101: `m0031_named_voices_shown.py` (0029's list through the fixed rewrite), the old 0031
  renumbered 0032 and unregistered. Tests: `test_m0031_named_voices_shown.py`, 6 cases; three
  decision mutants each fail one. MISSED: these tests were written after the code was drafted.

**Revision B.5 build.** Verify counts `served`. When it is 0, `_serves_nothing` FAILS a root
whose `corpus_manifest.json` is absent or not a JSON object, and a root with `*.metadata.json`
under `feeds/` that none is served from. Otherwise it is an empty corpus and passes, saying so.
Tests: no manifest; an empty corpus; metadata on disk with the selection returning nothing; four
non-object manifests. **C-b, the plan's inputs and the served selection:** `dry_run_report` runs
`select_served_artifacts` (m0031, the loop in `dry_run_report`) outside `recording()`, so the
selection's reads are not hashed. The selection decides only which episodes get a row. A
different served population on prod gives a row on one side only (`clone_only` / `prod_only`),
and that stops the hook's run until each row is accepted with a cause
(`test_outcome_hook.py::test_a_row_on_one_side_only_stops_the_run`, both sides; for the hook's
reviewer to confirm). The verdicts file is the plan-level input, hashed as `verdicts_sha256`.

**Revision B.6 build.**
- `_serves_nothing` walks `corpus_rglob(root, "*.metadata.json")`, the served selection's own walk.
- The 13 docstrings: 7 in `outcome_hook.py`, plus `Recorder.note` and `.inputs`, `Verdicts.get`,
  `verdicts_path`, `Record.touches` and `DerivedSurfacesEpisode.plan`.
- Tests a to c of the 13:01 amendment; d is the existing no-manifest test.
- One test changed: `test_the_moved_functions_stay_identical_to_0032s_copy` now compares the two
  copies' code with each docstring dropped. The docstring on `DerivedSurfacesEpisode.plan` made it
  red, and 0032's copy cannot get the same docstring: its file is gated by 0032's open record
  (steps 1 to 5 still open). Its purpose, that the code does not drift, still holds: a code edit to
  the copy is killed (mutant below).
- `make docstrings`: 100.0%.
- Mutants (`scratchpad/lost/b6mut.py`, the upgrade tests plus `test_corpus_walk.py`), 5 of 5
  killed:
  - plain `rglob`;
  - the walk rooted at `feeds/`;
  - E3, the on-disk check dropped;
  - the exclusion applied to the first segment only;
  - a code edit to `DerivedSurfacesEpisode.plan` (the copy's drift).
- Upgrade, corpus-walk and quality tests: 726 passed, 7 skipped.

## 7. Measure

Clone `lost/clone30` (2026-10-16 snapshot + 0030), code 97c551101, committed scoreboard:

| metric | before 0031 | after 0031 | Δ |
| --- | --- | --- | --- |
| S1 hidden, unknown voice | 16,211 | 14,812 | −1,399 |
| S1n named but hidden | 1,786 | 7 | −1,779 |
| S1c contradictions | 0 | 0 | 0 |
| S2 all hidden | 28,247 | 26,322 | −1,925 |
| S2 unidentified | 11,956 | 11,430 | −526 |
| S3 principal typed unidentified | 8,236 | 7,710 | −526 |
| S10a zero visible | 190 | 177 | −13 |
| S10b panel empty | 2 | 2 | 0 |
| S12 quotes off text | 24 | 24 | 0 |
| A1 sponsor hits | 357 | 357 | 0 |
| insights / quotes | 116,595 / 224,862 | same | 0 |

- Paths: 54 episodes written (52 voices shown under their name, 2 named), 1,925 insights shown,
  0 refused; verify passes; a second plan writes nothing.
- Undo on the full clone: 396 files restored, 0 refused, feeds byte-identical (`diff -rq`), verify
  reads pending again (54 to write).
- Per-episode plan with every counter and input sha256: `lost/three/plan_0031_clone30.jsonl`.
- Files written: 54 each of metadata, gi, kg, bridge, speaker diagnostics; 108 segments; 10 turns;
  2 context; 2 each of `.txt`, `.cleaned.txt`, `.adfree.txt`.
- Gate report: [0031_named_voices_shown.gate.json](0031_named_voices_shown.gate.json).
- Not run: gold gates (S6–S8; S7/S8 are not built), a whole-corpus replay.

**Revision B and B.2 measured** (`lost/clone30`, gate reports: B at 2026-10-18T08:40,
`scratchpad/lost/gate_0031_B.json`; B.2 in [0031_named_voices_shown.gate.json](0031_named_voices_shown.gate.json)):

| metric | clone30 | predicted (08:24 / 08:44) | measured B.2 |
| --- | --- | --- | --- |
| S1 | 16,211 | 14,841 | 14,841 |
| S1n | 1,786 | 7 | 7 |
| S1c | 0 | 0 | 0 |
| S2 (unknown / unidentified / commercial) | 28,247 | 26,351 (14,841 / 11,430 / 80) | 26,351 (14,841 / 11,430 / 80) |
| S3 | 8,236 | 7,710 | 7,710 |
| S10a / S10b | 190 / 2 | 178 / 2 | 178 / 2 |
| S12 / A1 | 24 / 357 | 24 / 357 | 24 / 357 |
| written / refused / unnamed by review | | 54 / 0 / 1 | 54 / 0 / 1 |
| insights shown | | 1,896 | 1,896 |
| files written / undo | | 403 / 403, 0 refused | 403 / 403, 0 refused, feeds byte-identical, verify pending (54) |
| ad maps carried / off in the 54 | | 3 / 0 | 3 / 0; corpus-wide 28 = the same 28 paths as clone30 |
| metadata_counts_synced | | **3** | **22: MISSED** |

- **The miss.** All 22 are episodes whose metadata `knowledge_graph` node/edge counts already
  disagreed with their KG file BEFORE 0031 (e.g. 23 vs 24 nodes); the recount made them true. No
  insight count changed. The prediction assumed the counts were in sync and did not check. Likely
  left by earlier KG writes without a recount (0029; the 2026-10-06 KG person operations), not
  confirmed (`scratchpad/lost/b2/counts2.py`).
- Census after B.2 (`scripts/measure/derived_surface_census.py`): insight_density and
  insight_sentiment stale 54 (re-enrich those episodes), transcript manifests 3, anonymised
  transcripts 2, ad maps 0; context digests are rebuilt and compared by the rewrite itself (only the
  3 that differ are written), so the census's 51 there over-reports; search: chunk and quote rows of
  the 3 episodes whose transcript text changed.
- Verify: `insights_checked=2454, unnamed_checked=1, unreviewed_unchanged=22, admaps_checked=71`.
  2,454 is not reconciled by the builder.
- Mutations: 34 listed by the two reviewers' criteria, all killed (`scratchpad/lost/mutants.py`,
  `b2/mutants2.py`); `_carry_admap`, `_recount` and the offset helpers AST-equal to 0032's.

**Revision B.3 measured** (fresh copy `lost/gateworkB3`, gate report in this folder): every
09:27/09:29 integer held. Scoreboard as B.2; written 54, refused 0, unnamed 1, insights shown
1,896, files 403, ad maps carried 3; `metadata_counts_synced` 22, the same 22 ids moving as listed;
ad maps off corpus-wide 28, the same paths; E against its 0029 backup: transcripts byte-identical,
diagnostics, kg, bridge, context 0 differences, and exactly the 535 stated (`speaker_label` absent
on 253 + 253 segment rows, insight `speaker` absent on 29); undo 403 / 0 refused, feeds identical;
apply, undo, apply, undo on the same copy: 403 / 0 again, identical. Mutations: the reviewer's 5
survivors and its B.3 list killed (`b2/mutants3.py`), the earlier sets re-run, 0 survivors.

**Step 0, operator 2026-10-18: wait for S7/S8 and the migration outcome hook (option B)** before
step 10. Gold check meanwhile (`scratchpad/lost/gold0031.py`): 19 of the 54 episodes are gold cases
on the served run; gold agrees with the review on 19 of 19.

**Revision B.4 measured** (B.3 at 7f7bf2fc6 and B.4 each applied to a fresh `lost/clone30` copy):
feeds byte-identical between the two applies; plan rows 54, refused 0, their counts summing to
apply's totals; gate report rewritten (`0031_named_voices_shown.gate.json`, 245 files changed,
gating moves as B.3); receipts 403; undo 403 / 0, feeds identical; apply again: 54 written, verify
passes with `served_listed=71`; undo again 403 / 0, identical. Mutations: all four sets
(`scratchpad/lost/mutants.py`, `b2/mutants2.py`, `b2/mutants3.py`, `b2/mutants4.py`), 0 survivors.
A process note: a second copy of the undo sequence was run while the first was still running and
its output interleaved; the first run's own log (`b2/runB4.log`) is the record above.

**Revision B.5 measured** (`scratchpad/lost/b2/runB5.sh`, log `runB5.log`, a fresh `clone30`
copy `lost/b5apply`). Plan rows on clone30: 54, refused 0, byte-identical to B.4's (`cmp
plan_B4.jsonl plan_B5.jsonl`). Apply: `diff -rq lost/b3apply lost/b5apply` over the whole tree
exits 0 (B.3 = B.4 = B.5 written). Verify passes: `insights_checked=2454, unnamed_checked=1,
unreviewed_unchanged=22, admaps_checked=71, served=2685, served_listed=71, metadata_unreadable=0`.
Re-plan: 0 episodes. Undo 403 / 0, feeds identical. Apply again and undo again: 403 / 0,
identical. Every integer the 11:10 prediction stated matched. Mutants of 1d: 4 of 4 killed
(`lost/b5mut.py`). Upgrade tests: 622 passed. The reviewer's 65 mutants are not re-run by me.

**Revision B.5 measured from the clean commit** (28a76ce01, `code.dirty: false`) by the gate
script, on `lost/clone30L`: `lost/clone30` with 0030 recorded in its ledger by the store's own
`record_applied` (its `at` is when it was recorded, not when the script applied 0030; nothing
else differs, `diff -rq`), snapshot `snapshot-prod-20261016+0030`. Gate report rewritten
(measured 2026-10-18T11:46:49): computed class full; gating moves S1 −1,370, S1n −1,779, S2
−1,896, S3 −526, S10a −12 (the hook acceptance's D2 numbers); 245 files changed; 54 plan rows
(sha256 e85a2670…); `applied_before` 30 ids, 0030 among them; code digest aaf844b1…; verdicts
f3587fde…. Per-episode deltas: 53 episodes; the 54th (Samo Burja, 0015) writes files (one voice
shown, 592 segments relabelled) and moves no gating metric, so the hook compares it against 0.

**Revision B.6 measured from the clean commit** (ce6a98b64, `code.dirty: false`) by the gate
script on `lost/clone30L`, snapshot `snapshot-prod-20261016+0030`, gate report rewritten (measured
2026-10-18T13:11:34). `diff -rq lost/gateworkB5 lost/gateworkB6` over the whole tree is empty:
every file written, the receipts file included, is byte-identical to B.5's. Plan rows 54,
`plan_sha256` e85a2670… (B.5's); gating moves S1 −1,370, S1n −1,779, S2 −1,896, S3 −526,
S10a −12; 245 files. Code digest aaf844b1… → 653cb12e…, moved by the files B.6 changed, all in
0031's import closure: `m0031_named_voices_shown.py`, `outcome_hook.py`, `read_recorder.py`,
`reviewer_verdicts.py`, `voice_rewrite_derived.py`, `quality/change_gate.py`. The outcome is
rebuilt after the step-8 verdict; its new sha follows the verdict below.

## 8. Verdict

Verdict: NO-GO (revision B.2, 2026-10-18)

**Revision B.2, step-8 reviewer (fresh advisor), 2026-10-18: NO-GO.** Its files:
`scratchpad/reviewer5/verdicts_0031_step8.tsv` (13 reads), `criteria_0031_step8.tsv`.
- PASS: every gating number and the 29 ids with its own scoreboard code; counts 54 / 0 / 1 / 1,896 /
  403 reproduced by its own apply (byte-identical); E's transcripts, quotes and ad map; the 28 ad-map
  set; nothing else moved but the 22 metadata counts; reads 11 of 11 CORRECT; verify's 2,454
  reconciled (1,863 CORRECT + 35 MISSPELLED + 556 on the 22 untouched entries).
- FAIL: H1 the unnamed voice keeps role "host" (shown as "Host" on 253 segments); M1 5 of 45
  mutations survive; M2 the 22-vs-3 miss; M3 undo after apply-undo-apply reports 245 false refusals.
- LOW: verify passes with a written entry's verdict deleted; `list_sha256` never compared; the
  conversion script knows only 0029's list; 0032's copy can drift; verify does not rebuild
  `.adfree.txt`; the record did not say a changed verdicts file reopens step 7 (it does); the roster
  loss needs the operator at step 9.
- Back to step 3 (diagnosis in the verdict), restated at step 4 (B.3, 09:27).


Verdict: GO WITH CONDITIONS (revision B.3, 2026-10-18)

**Revision B.3, the same step-8 reviewer: GO WITH CONDITIONS.** Every criterion of 08:22, 08:45
and 09:27 passed on its own apply of B.3 to a fresh clone30 copy (files:
`scratchpad/reviewer5/criteria_0031_step8_B3.tsv`, `verdicts_0031_step8.tsv`,
`tested_tree.sha256`): E as the 09:29 restatement (535 stated, no "Host"); scoreboard, counts and
the same 22; the same 245 changed files as B.2; undo 403 / 0, twice, and a hand-edited file refused;
verify fails on each of its edits; 55 of 55 mutations killed; the conversion script reproduces
f3587fde; 11 reads CORRECT. Conditions:
- **C1** (generic rule, latent for 0031): a WRONG_PERSON verdict on an entry 0029 RENAMED (old name
  set) is mishandled: with old role "guest" verify demands None and the next plan refuses the
  episode; with old role None the wrong role stays. Fix before any other migration reads verdicts.
- **C2** the gate report names `dirty: true`: commit, then match the files to
  `tested_tree.sha256` or re-run the gate on the commit.
- **C3** the roster/hosts loss (L8) to the operator at step 9.
- For step 0: what S7/S8 and the hook must show for 0031 (the reviewer's list, in its report).


Verdict: GO WITH CONDITIONS (revision B.4, 2026-10-18)

**Revision B.4, the same step-8 reviewer: GO WITH CONDITIONS.** Every criterion of its 10:06
amendment passed with its own code on 7f7bf2fc6 and f1817141e (files:
`scratchpad/reviewer5/criteria_0031_step8_B4.tsv`, `spy2.py`, `influence.py`, `mut4.out`,
`mut4b.out`): B.4 applied byte-identical to B.3 (`diff -r` 0 files); C1 in both old-role cases,
with three mutants killed; 54 plan rows whose counts sum to apply's totals; 657 inputs re-hashed
with 0 mismatches. An independent spy (`sys.addaudithook`) found the opened files equal to the
hashed inputs in all 54 episodes. Removing 85 unhashed sibling files one at a time changed 0
decisions. Verify passes where no listed episode is served and fails on unreadable metadata.
Undo/re-apply/undo gave 403 / 0 with 0 differences. Mutations: 65 of 65 killed; tests: 441 passed.
Conditions (LOW, code or record, no re-measure):
- **C-a** verify on "no listed episode is served" passes on a root that serves nothing (a wrong
  root passes vacuously): print the served total and fail when it is 0; test the empty corpus.
- **C-b** the record names the verdicts file as a plan input, but not the served-selection reads
  (`select_served_artifacts`): list them, or say why they need no hash.
- Both are taken up in revision B.5 (step 4, 11:10).

Verdict: GO (step-8 reviewer, 2026-10-18, revision B.5 = f1817141e + the uncommitted m0031 verify change). Against the 11:11 criteria: C-a met (no manifest or a non-object manifest with 0 served fails; manifest + metadata on disk + 0 served fails; a fresh install passes as "an empty corpus"); C-b met in the record (call site named; test_outcome_hook.py::test_a_row_on_one_side_only_stops_the_run, both sides, passes; the hook's reviewer confirms the hook itself). Outcome on clone30 byte-identical to B.4 (apply diff 0 files, receipts identical, 403 files); verify served=2685, served_listed=71, other counters unchanged; re-plan 0; undo 403/0, feeds = clone30. Mutations 69/69 killed (B.4's 65 + 4 for C-a). Open, unchanged: S7/S8 and the outcome hook before step 10 (operator, step 0); L8 at step 9; snapshot freshness and the prod plan diff at step 10.

The reviewer's files: `scratchpad/reviewer5/ab5.sh`, `ca5.py`, `mutate5.py`, `mut5.out`. Its observation outside 0031: a corpus subdirectory as the root (`clone30/feeds`) plans 58 episodes with 6 refused and verify fails there; whether the runner refuses such a root before apply is the runner's question, not checked.

Shipped outcome: `src/podcast_scraper/upgrade/outcomes/0031_named_voices_shown.json`, sha256
b1546e27c36c56158ebe70a6d29f8b4d24947695362934d2f6f54167c933ae2d (`scripts/measure/build_outcome.py`
from the 11:46:49 gate report and the GO above).

Verdict: GO WITH CONDITIONS (step-8 reviewer, 2026-10-18, revision B.6 = ce6a98b64). Against the 12:59 block with its 13:01 amendment: the verify walk is the selection's walk (corpus_rglob over the whole root; 3,196 = 3,196 files on clone30L); docstrings only elsewhere (docstring-stripped AST equal to fa02ce3c0, make docstrings 100.0%); plan rows identical to B.5 (54, 0 refused); apply byte-identical to gateworkB5 including receipts, 403 files; verify served=2685 served_listed=71 with the other counters unchanged; _serves_nothing called 0 times; undo 403/0; mutations 72/72 killed, the corpus-walk rule test included. Condition: item 5, `make ci-fast` on ce6a98b64 ending EXIT=0 with the log attached, before landing. Open, unchanged: S7/S8 and the outcome hook before step 10; L8 at step 9; snapshot freshness and the prod plan diff at step 10.

The reviewer's files: `scratchpad/reviewer5/criteria_0031_step8_B6.tsv`, `item4.py`, `item4.sh`,
`mutate6.py`, `mut6.out`, `mut6b.out`, `docstrings.out`.

**Measured again after the hook's step-8 conditions** (a6b0f91f5, clean; they change
`outcome_hook.py` and `quality/change_gate.py`, both in 0031's closure, and C3 adds every parent
package to it). Gate report 2026-10-18T13:29:41: `diff -rq lost/gateworkB5 lost/gateworkB6b` over
the whole tree is empty; `plan_sha256` e85a2670…; the same gating moves and 245 files. Code digest
653cb12e… (B.6) → abd29583… (a6b0f91f5). Item 5's ci-fast runs on the commit that carries the
rebuilt outcome, which differs from a6b0f91f5 only in the outcome file and the records.

Shipped outcome (replaces b1546e27…, which is stale since ce6a98b64):
`src/podcast_scraper/upgrade/outcomes/0031_named_voices_shown.json`, sha256
affb5658d317bea113339a07b9e8da5bbc839b6cc307373791885a4cfa72090f, built from the 13:29:41 gate
report (a6b0f91f5, code digest abd29583…) and the GO WITH CONDITIONS above.

## 9. Operator review

_TODO_

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Not covered

- Steps 1, 4 and 5 were skipped (above); the evidence for them was written after the build.
- No gold measurement: whether each shown insight names the right person is only checked by the
  reviewer's hand-read sample.
- The 10-16 snapshot is not prod today; prod's own plan is not yet diffed (step 10).
- The readers of insight visibility (search rows, insight_density / insight_sentiment sidecars,
  insight clusters, corpus enrichers) are listed by the reviewer, not yet by a census script.
- Speaker diagnostics are renamed but their voice type is not touched.
- The 7 S1n insights left are outside 0029's episodes and are not explained.
