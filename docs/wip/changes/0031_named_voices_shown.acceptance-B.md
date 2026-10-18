# 0031 revision B: step-5 acceptance criteria

Written by the fresh step-5 reviewer, 2026-10-18T08:22 UTC, before any revision-B code existed
(amended once at the builder's request, after the design changed from "withhold" to "unname").
Stored here as written; the change record's step 5 points to it. Integers stated by the reviewer
from its own reads of `lost/clone30` and the step-7 plan file.

E = episode 5555e937-4e0a-4e17-abc3-68f5b77e5501, The Flip "Unlocking Gender-Smart Capital At
Scale". Prev = 97c551101's measured after-state.

1. **Gating numbers**, corpus-wide, identical in both branches (Samo Burja carries 0 insights):
   S1 14,841; S1n 7; S1c 0; S2 26,351 (unknown 14,841, unidentified 11,430, commercial 80);
   S3 7,710; S10a 178; S10b 2; S12 24; A1 357; insights 116,595; quotes 224,862. The 7 S1n ids
   equal Prev's 7 (set comparison). Against clone30: S1 −1,370, S1n −1,779, S2 −1,896, S3 −526,
   S10a −12; no gating number rises. E's 29 insights stay in S1 and leave S1n (a wrong-name
   removal, listed by id). Against Prev, S1 and S2 are exactly +29, the diff exactly E's 29 ids.
   S3 re-derived with the raw-voice join. S4 cross-check: −49 named quotes against Prev, all E's.
2. **Counts**, keyed by per-file metadata path:
   - Branch A (only the converted predecessor verdicts; Samo Burja unreviewed): written 53
     (50 shown, 2 named, 1 unnamed by review), refused 1 (`unreviewed`), insights shown 1,896,
     files 393, verify FAILS by design. Cannot land.
   - Branch B (verdicts cover every entry; Samo Burja CORRECT): written 54 (51 shown, 2 named,
     1 unnamed by review), refused 0, unreviewed 0, insights shown 1,896, files 400, verify passes.
   - Any other verdict outcome sends the work back to step 4. Population buckets against 0029's
     83 entries sum exactly. The transcript counter is reconciled with the files written.
3. **E's writes** (the unnaming undoes 0029's binding of the name to SPEAKER_00 through m0025's
   existing path): `.txt`, `.cleaned.txt`, `.adfree.txt` byte-identical to the 0029 backup;
   segments, adfree segments, gi, kg, bridge, speaker diagnostics parsed-JSON-equal to it (each
   deviation listed); all 54 quotes back at their pre-0029 offsets, 0 off their text; its 29
   insights hidden, unknown; metadata keeps its later `feed.*` / `episode.licence` values; the
   roster/context outcome comes from the shared path's rule, not a per-entry choice; sidecars,
   asr, manifest and admap untouched; afterwards "Justin Norman" only in the description (plus
   roster/context if kept).
4. **Nothing else moved**: against Prev, only E's files (branch B; plus Samo Burja's restored in
   A); 0 shown→hidden against clone30; transcript text changes only in speaker labels, only in
   Prev's 2 named episodes plus E.
5. **The generic verdict mechanism**: no episode id, title, voice or name in code (grep 0); one
   shared reader keyed by migration id, proven on a synthetic "0099_x"; a migration with no
   verdicts file behaves as before; the committed schema (header: migration id, reviewed commit
   and list sha, reviewer, date; rows: episode_id, per-file path, raw voice, listed name, verdict,
   deciding line; a closed verdict set); a committed conversion script printing every mapping;
   named refusals, each tested, for an unknown verdict, a missing column or header, a conflicting
   duplicate, a migration id mismatch, a stale name, a key matching no entry; a verdict on an
   unserved entry counted, no write; a missing file for a migration that declares one refuses, and
   missing vs empty are separate tests; an entry with no verdict refused as `unreviewed`; WRONG
   PERSON unnames through m0025's existing path only; MISSPELLED NAME stays shown (unnamed by
   review exactly 1); the file's sha256 in every plan row and the receipt, a changed file reopens
   step 7, the file ships with the migration source.
6. **Verify** passes only if: no insight whose first SUPPORTED_BY quote is on a shown or named
   voice stays hidden unknown/unidentified (commercial exempt); an unnamed-by-review voice carries
   the name on none of its segments, quotes, insight `speaker`, transcript prefixes, diagnostics;
   unreviewed is 0; it prints the insights checked (a pass with 0 checked fails). Shown able to
   fail on copies: one shown insight flipped hidden; a commercial one flipped (still passes); the
   name written back on one of E's segments, quotes or prefixes; Samo Burja's row deleted; clone30.
7. **Mutations**, each killed with no test edits (0 survivors; an equivalent mutant argued in
   writing): the predecessor's 9 survivors; reader mutations (ignore verdicts, every non-CORRECT
   wrong, match by name only, by episode only, by voice only, no stale-name check, unknown values
   accepted, unreviewed shown, unreviewed skipped silently, WRONG PERSON skipped instead of
   unnamed); unname-path mutations (leave quote speaker_id, insight speaker, transcript prefixes,
   unshifted offsets, diagnostics role/named, the Person node); verify mutations (no insight
   check, unknown only, any quote, no unname check, no unreviewed check).
8. **Undo** on a fresh full copy of the after-clone (branch B): restored 400, refused 0, feeds
   byte-identical to clone30, verify pending again.
9. **Must not move**: S1c 0, S12 24, A1 357, S10b 2, insights, quotes, 0 shown→hidden, E's
   untouched files' sha256.
10. **Reads**: before the measurement that lands it, the reviewer's verdicts cover every entry
    0031 touches (all 54, Samo Burja included) in the committed schema; E re-read after; 0 WRONG
    PERSON among shown entries.
11. **Derived-surface census** (D17) produced by a script: every reader of insight visibility and
    speaker names, ticked or stated stale; the unnaming adds roster/context readers.
12. **Not covered**: S7/S8 and gold gates NOT RUN (plan C); the misspelled-name class; an automatic
    wrong-person rule; the 7 S1n insights outside 0029's episodes; E's real host voice stays
    unnamed (no list carries it).

**Reviewer's challenge, decided by the builder:** the roster entry
`{"id":"unplaced_1","name":"Justin Norman","role":"host"}` and `context.basic.hosts` bind no voice
and E's description says "Produced and hosted by … Justin Norman", so they are true. Decision:
m0025's existing unnaming path is used unchanged, which removes them; this is a LISTED LOSS of one
true, voice-less host fact in one episode. Keeping such entries is a change to the shared rule (it
would also apply to 0029's and 0032's unnaming entries) and goes to its own change.

## Amendment for revision B.2 (2026-10-18T08:45, the same reviewer, before any B.2 code)

Measured by the reviewer first (`scratchpad/reviewer5_admap.py`, every `*.adfree.admap.json`):
clone30 has 3,528 ad maps and 28 mismatches (`source_length != len(.txt)` or an excised text not
equal to the text at its range), none in 0031's 54 episodes; after branch B, 31 = the same 28 + 3
that 0031 made.

- **3g replaced:** E's ad map is CARRIED: `source_length` 39,290 = len(.txt), `excised_ranges` [] and
  `chars_removed` 0. The 2 named episodes likewise: every range moved with the text, every excised
  text equal to the new text at its range (only a speaker label inside may differ), range count and
  `chars_removed` unchanged; `.adfree.txt` equals `.txt` with the carried ranges removed (rebuilt and
  compared byte for byte); E's untouched files keep their sha256.
- **Corpus-wide:** after the run exactly 28 mismatches, the SAME 28 paths as clone30; 0 among the 54.
- **Recount:** for each of the 54, the metadata GI/KG counts equal a recount from the written gi/kg;
  unwritten episodes' metadata counts unchanged; `metadata_counts_synced` 3 accepted only if exactly
  3 episodes' count fields change.
- **Files** 403; undo 403/0, `diff -rq` empty, verify pending (54). Scoreboard: the 08:24 integers.
- **The move:** `_carry_admap` / `_recount` AST-equal to 0032's apart from names and imports; 0032's
  tests untouched and passing, 0032 unregistered, its plan output on a fixture identical before and
  after; 0031 tests kill: skip the carry, carry without the shift, drop a range, recompute
  `source_length` but not the ranges, skip the recount, recount GI but not KG and the reverse.
- **Verify** fails on an ad map that does not match its transcript (excised text checked, not only
  the length), for at least the 54; copies with a range off by one, E's old `source_length`, or an
  edited excised text must FAIL; it must not fail on the 28 that existed before unless the record
  says so.
- `insights_checked=2454` is reconciled by id at step 8; until then it is unexplained, not a pass.

Out of scope, own record: the 28 pre-existing ad-map mismatches (15 feeds; length deltas −504 to
+648; not diagnosed).

## Amendment for revision B.3 (2026-10-18T09:27, the same reviewer, before any B.3 code)

1. **E (H1):** parsed-equal to its 0029 backup with zero deviations, or every deviation stated in
   the record before measuring (the builder stated 535 at 09:29: `speaker_label` absent on 506 rows,
   insight `speaker` absent on 29). Role: 0 SPEAKER_00 rows carry `speaker_role`; diagnostics
   role `unknown`, named False, resolved_name SPEAKER_00; the transcript view shows SPEAKER_00 253,
   SPEAKER_01 80, "Unidentified speaker" 66, no "Host". Generic rule (restore the listed old name
   and role); a test with an old role that is not None.
2. **Counts:** the 09:27 integers; `metadata_counts_synced` 22 = the same 22 episode ids, each
   moving exactly as listed and then equal to its KG file; the KG files byte-unchanged; no other
   metadata field moving in those 22.
3. **Against B.2:** only E's two segment files, gi and diagnostics differ; the reviewer's own apply
   is byte-identical to `gateworkB3`.
4. **Undo (shared `undo_from_receipts`):** already-restored only when the current sha equals that
   receipt's `sha_before`; a file in neither state is refused (a hand-edited file gives refused 1);
   single apply+undo 403/0, feeds identical, verify pending 54; apply, undo, apply, undo refused 0
   and identical, the restored count stated before measuring (403 distinct files); every other
   migration's undo tests pass unchanged; `file_rewrite`'s tests gain the neither-state case.
5. **Verify** fails: a written episode's entry with no verdict; a `list_sha256` not the current
   list's (plan and apply refuse too, named); `speaker_role: host` back on one SPEAKER_00 row; E's
   diagnostics role back to host.
6. **Mutations:** 0 survivors across the reviewer's 45 plus V3, R12, U6, D2b, D9, name-but-not-role,
   role-on-segments-not-diagnostics, stale `list_sha256` accepted, undo treating any non-`sha_after`
   file as restored, verify ignoring an unreviewed entry in a written episode; the AST-equality test
   fails on a one-token change in either copy.
7. **Conversion script** asks the migration for its list; tested with a synthetic migration whose
   list differs from 0029's; regenerating 0031's file reproduces its sha (or the new sha is stated).
8. **L7** stated in the record; the receipt header carries the sha.
9. **Not required:** L6 (`.adfree.txt` = `.txt` minus ranges holds for 2,855 of 2,855 ad maps without
   ranges but only 82 of 673 with ranges, so it cannot be a verify rule; the reviewer checks
   "holds after iff held before" for every carried ad map at step 8). L8 is for the operator.
