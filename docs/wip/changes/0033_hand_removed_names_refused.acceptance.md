# 0033: step-5 acceptance criteria

Written by the fresh step-5 reviewer (an Opus advisor), 2026-10-19, before any 0033 code existed,
from its own reads of `lost/gateworkB6c` (E's segments, GI, metadata, diagnostics; the census
re-run). Stored here as written, lightly reformatted; the change record's step 5 points here.
Clone: `lost/gateworkB6c`. Before: the clone as step 4 states it. Every number from a stored script.

## The verdict on SPEAKER_00: CORRECT, it is David Roblin

E's `segments.json`: 3,736 segments, 2 voices; SPEAKER_00 3,055 segments (about 3,290 s),
SPEAKER_01 681 (about 727 s).
- SPEAKER_01 is the host: seg 2 "I'm your host, Luke Timmerman."; seg 3 "Today's guest is David
  Roblin."; seg 4 "David is the CEO of London-based Relation Therapeutics."; seg 18 "…translational
  research at the Francis Crick Institute". Title "Ep176: David Roblin on Building a TechBio
  Company".
- SPEAKER_00 speaks as the guest: seg 30 "Thanks, Luke. It's an absolute pleasure to be on."; seg 43
  "So you may be able to tell me some things about my company."; segs 67–68 "I was key to the
  development of the Crick and was the first COO and Director of Scientific Translation."; seg 151
  "So you said you met my team here."; seg 3699 "I want Relation to be a sector-leading
  biotechnology company."
- Addressed and answering: seg 47 "…a little bit about you, David" → S00 from seg 50; seg 89 "Where
  are you from, David?" → seg 91 (S00) "So I came from a principality of the UK called Wales."; seg
  3691 → seg 3699; seg 3730 "David Roblin, thank you so much for joining me on The Long Run."
- Against, weighed: turn-boundary leaks both ways (S00: seg 29 "David Roblin, welcome to The Long
  Run.", seg 105, seg 125; S01: seg 90 "I'm a Welshman.", seg 3731). Seg 29 is followed on the same
  voice by seg 30 "Thanks, Luke…", so it is a leak, not an address.
- All 58 "Generation" insights read (with up to 2 quotes each): the guest's content. The decision
  trace: `per_voice_self_intro: {"SPEAKER_00": "Generation"}`, `metadata_named: ["David Roblin"]`.

So the repair is **N**.

## A. The repair of E, by N

1. **The verdict as evidence:** a committed verdicts file (D13),
   `upgrade/verdicts/0033_hand_removed_names_refused.tsv`: E, SPEAKER_00, CORRECT, David Roblin, the
   evidence segments above. The migration reads it; with no verdict for E it refuses to name (E
   unchanged, counted).
2. **E's roster** ends with exactly 2 entries: host Luke Timmerman on SPEAKER_01 (unchanged); guest
   David Roblin, placed, `voices: ["SPEAKER_00"]`. No "Generation", no second "David Roblin": 3 → 2.
   The `source` does not read `self_intro`. `normalized_entities` keeps no "generation".
3. **E's GI:** 58 insights `speaker` "David Roblin", 3 Luke Timmerman, 1 none; 109 quotes and 109
   `SPOKEN_BY` edges to one person id, `person:david-roblin` (the id E's KG and bridge already
   use); GI Person nodes stay 6, none "Generation"; `insights_from_gi(gi)` returns 62; no insight
   gains `surfaceable: false`.
4. **E's KG has one David Roblin:** one `person:david-roblin` node, role guest, carrying
   `GUESTS_ON`; no duplicate `MENTIONS`; no Generation node. Expected nodes 29 → 28, edges 30 → 29;
   the builder restates the exact integers before the build.
5. **E's bridge:** identities 27 → 26; `person:david-roblin` `sources {gi: true, kg: true}`; no
   Generation identity.
6. **Derived surfaces:** "Generation" left in E's files only as speech ("Generation X"): 1 each in
   `.txt`, `.adfree.txt`, `.anon.txt`, segments text, ad-free segments text; 0 in `.cleaned.txt`; 0
   as a speaker label, a `<name>:` prefix, a `context.json` field, a diagnostics name or a `turns`
   speaker. Today: `.txt` 62 prefixes, `.adfree.txt` 62, `.cleaned.txt` 32, `turns` 62,
   `adfree.turns` 62, diagnostics (3 `name` + 1 `resolved_name`), `context.json` (`guests`,
   `people`), segments and ad-free segments 3,055 labels each. "David Roblin:" prefixes become 62 /
   62 / 32.
7. **Offsets:** "David Roblin" is 2 characters longer and the 114 quotes cite `.adfree.txt` by
   offset. E's S12 stays 0 (114 of 114 slice to their text); every ad-free segment slices to its
   text; the admap `source_length` and turns `segments_sha256` match the new files or are listed
   with a reason. A shifted offset FAILS.
8. **Files written:** the builder states the exact list and integer before the build (expected 13:
   metadata, segments, adfree.segments, gi, kg, bridge, `.txt`, `.adfree.txt`, `.cleaned.txt`,
   context, speakers.diagnostics, turns, adfree.turns). A sha256 census of every served file shows
   exactly those changed, episode count 1.
9. **Gating scoreboard** (committed command, after-state), all unchanged: S1 14,841; S1n 7; S1c 0;
   S2 26,351; S3 7,710; S10a 178; S10b 2; S12 24; A1 357; insights 116,595; episodes 2,685.
   Hidden-name outcome over all served GI: insights under a hand-removed name 58 → 0; quotes 109 →
   0; episodes 1 → 0.
10. **The full `upgrade verify`:** 0015 FAIL (1 episode) → PASS; every other migration's line as in
    the before-state, line for line (0019, 0022, 0023, 0024, 0025, 0029, 0031).
11. **Receipts isolated:** 0033's own id, receipts file and backup tag;
    `unpublishable_speaker_names_removed.jsonl` and `descriptor_speaker_names_removed.jsonl`
    byte-identical; `.podcast_scraper/upgrade-backups/0015/` and `/0019/` untouched.
12. **Undo on the full clone:** restores every file written, refuses 0, corpus byte-identical
    (sha256) to the before-state; 0015's verify FAILs 1 episode again, the scoreboard equals the
    before rows, 0033 reads pending.

## B. Rule 1: the publish gate refuses 0015's hand-removed names

13. `is_publishable_speaker_name` refuses exactly the 6 `REMOVE` strings (whitespace-normalised,
    case-insensitive, whole-string), with `require_person_shape` True and False, in every language.
    One definition: the gate, m0015 and m0019 read the same object.
14. Old vs new gate over the 2,679 distinct served roster names (all sources, placed and unplaced):
    the names that flip pass → refuse are exactly {"Generation"} (1 entry, 1 episode). "Gene Sykes"
    still passes; "RJ" (KEEP) as before; the other 92 census entries as before.
15. 0015 and 0019: `refused_names(root)` and `plan()` byte-identical old vs new on the before-clone.
16. **Reprocess replay** of the roster step on E's stored inputs with the new code (if impossible
    without the LLM: `MISSED:` with the operator's acceptance). Pass: SPEAKER_01 Luke Timmerman;
    SPEAKER_00 ∈ {David Roblin, unnamed}, never "Generation" or another non-name. If unnamed, the
    known loss is listed: a reprocess hides E's 58 insights until Stage 2 seats the stated guest.

## C. Rule 2: the one-word self-introduction rule

17. The rule is a class, defined in the record before the build; a word list holding census tokens
    fails. It refuses E's "Generation" (seg 93) **with rule 1 off**, through `_self_intros_by_voice`
    on E's voice texts; SPEAKER_01's "Luke Timmerman" is kept.
18. Census (93 entries, 70 distinct). Non-names (4): Generation (E); "Long" (Unhedged, seg 209 "I am
    Long, the Feet news briefing podcast", the finance usage); "Gekoyo" (Made In Africa, segs
    125/181, an ethnic group); "Kanye's" (a16z; out of scope, operator). Real names to keep,
    outside this rule: "Fisher" (Planet Money; placed on the wrong voice, Stage 2), "Joeisenthal"
    (Odd Lots; Joe Weisenthal fused by ASR). Read and judged names: Lee, Guanxin, Lanira, Greg,
    Winnalou, Ed, Christy, Oker, Bernie, Jack, Faith, Konto, Robin, Osbir, Prabh, Chai, Elle, Twiggy;
    the rest judged by form. Pass: Generation refused; 0 of the 89 names refused; Long, Gekoyo,
    Kanye's each refused or listed as not reached. The prediction is restated as integers and a
    named set before the build.
19. Generalisation guards: Faith, Robin, Jack, Fisher, Lee, Elle, Chai pass. A rule refusing the
    bare token "Long" everywhere fails (a Chinese given name).
20. **Held-out 1, beyond the 93:** every one-token candidate `_self_intros_by_voice` yields over all
    served voice texts on the clone (placed or not); old vs new, the newly refused set listed in
    full; the population size and refusal count predicted before the run; the step-8 reviewer reads
    every refusal. Pass: 0 real names refused.
21. **Held-out 2, unseen phrases:** the step-8 reviewer writes at least 20 label phrases and at least
    20 real one-word-name self-introductions the builder does not see (generational and age labels,
    finance positions, ethnic and national group labels, "it's <X>" non-introductions; real names
    including dictionary-word names). Pass: 0 real names refused; label refusals counted, each miss
    a not-covered line.
22. **Held-out 3, gold:** `naming_gate.py` on val_v2: S6 = 0; 0 voices moved is reported as no effect.
23. Per language: held-out 1 by language; any refusal outside English listed and read.

## D. Unchanged corpus-wide

24. Every served file outside criterion 8's list byte-identical (sha256 census).
25. Every §3 row from the committed command equals its before value.
26. `make ci-fast` passes on the committed tree, log attached.

## E. Prod (step 10)

27. Prod's read-only plan diffed against the clone's: the same 1 episode, the same counters, every
    difference with its cause.
28. Prod's search-index rows for E carrying `person:unresolved-generation-…` or "Generation"
    counted and listed (not measured on the clone, which has no index); targeted row edits on the
    operator's go (ladder rung 3).

## The reviewer's findings on step 4

1. "Keeps the other 92" is wrong: Long, Gekoyo and Kanye's are not names; Fisher is a real name on
   the wrong voice.
2. Derived surfaces missing: m0012's rewrite (0015's) does not touch the transcript prefixes (62 /
   62 / 32), turns, diagnostics, `context.json` or `normalized_entities`; m0022–m0024 exist for that.
   Under N the prefix is 2 characters longer, so offsets shift. "Episodes written 1" with no file
   list is not a full prediction.
3. The KG and bridge already hold `person:david-roblin` (KG role mentioned; bridge `gi: false`); N
   merges into it, and its integers are missing.
4. m0019 imports `REMOVE`; the prediction must say m0019 stays unchanged.
5. Reprocess durability is not stated.
6. Step 3 misquotes "I'm a Welshman": seg 90 is on SPEAKER_01; SPEAKER_00's is seg 126.
7. Prod numbers and search-index rows not covered.
8. The other prod FAIL line (0029's) is not named in step 1.

## Not verified by the reviewer

0015's plan counters (it counted 3,055 SPEAKER_00 segments, not the plan); the exact KG edge
integer after the merge; all 109 quotes in full; 47 of the 70 distinct census names (23 read);
whether the roster step for E replays without the LLM; whether E is in a gold set; prod's
search-index rows.

# Revision B (after the step-8 NO-GO)

Written by a fresh step-5 reviewer (Opus advisor), first at 2026-10-19T19:07 UTC, amended at
2026-10-19T19:34 UTC after the operator's answer to its finding 1 ("Exempt only 'I'
(Recommended)"), without seeing any revision-B code. Stored as written, reformatted. "Rev A" is
c71664a6d; "old" is a213d5fa5's hosts.py. Of the original 28, criteria 1–5, 7, 9–15, 17–20 and
22–25, 27, 28 stand and are re-measured on the revision-B commit (B14); 6, 8, 16, 21 and 26 are
amended below.

- **B1 (amends 6, 8), files:** still 14, the step-7 list; 13 sha256-identical to rev A's on the
  same clone; `speakers.diagnostics` differs from rev A's only in `voices[SPEAKER_00].source`
  "self_intro" → "roster" (SPEAKER_01 byte-for-byte rev A's); the decision record
  (`tried.per_voice_self_intro`, `decision_trace`) still holds "Generation" exactly 4 times (the
  accepted deviation from 6, listed); no other diagnostics field holds it. Anything else differing
  from rev A FAILS unless the prediction named it.
- **B2, rule 2 narrowed to "I", as a class:** a lone "I" counts as the label's letter only when
  punctuation or the end of the text follows directly; every other lone capital, "A" included,
  keeps rev A's behaviour and lookahead. Stated in step 6 before the measurement; a word list fails.
- **B3 (amends 21), held-out 2 on `rev33b/heldout2.py`** (35 labels, 35 names), old vs rev A vs B:
  - names: 32 of 35 named, exactly 3 refused ("I'm Will A producer here.", "I'm Bill T, your
    host.", "I'm Grace O, and welcome.", the operator's accepted cost); each of the 32 returns the
    old code's string; "I'm Rose I work on the climate desk." → "Rose", "I'm Mark I host the show."
    → "Mark", "I'm Jay Z." → "Jay", "I'm Malcolm X." → "Malcolm". FAIL on any other refusal, any
    differing string, or Rose/Mark not returned; fewer than 3 refused is stated as a wrong
    prediction.
  - labels: 20 refused, 15 named, identical to rev A phrase by phrase; rule 2 refuses exactly 10 of
    the 25 the old code named (Generation Z, Type A, Type B, Plan, Team, Vitamin, Hepatitis, Class,
    Grade, Category). FAIL on any label differing from rev A. Not-covered line each for the 15 every
    version names: Millennial, Boomer, Zoomer, Xennial, Long Tesla, Long, Short, Bullish,
    Overweight, Nigerian, Yoruba, Kikuyu, Maori, Catalan, Blood Type.
- **B4, held-out 2b,** at least 20 new phrases by the revision-B step-8 reviewer, unseen by the
  builder, at least 3 per class: (a) "<Name> I <word>" run-on, must be named; (b) "<Name> I," /
  "I." / "I" at the end, refused, accepted cost; (c) "<Name> A <word>", refused, accepted cost;
  (d) "<Name> <letter not I/A>," no period, refused, accepted cost; (e) "<Label> I <word>", named,
  a not-covered line each; (f) "<Label> A <word>" and "<Label> <letter>,", must be refused; (g)
  "<Name> <letter>." (initial with period), must be named. FAIL on a refusal in (a) or (g) or a
  named (f).
- **B5, the keep rule** (the builder's, stated before the build): a renamed or unnamed voice's old
  GI Person still the target of any `MENTIONS_PERSON` is kept and those edges are not moved;
  otherwise removed as in rev A. Tests: (a) rename, mentioned: the Person survives (same id, name),
  its mention edges unchanged, no `SPOKEN_BY` to it and no insight's speaker, no dangling or
  duplicate edge; (b) unnamed, mentioned: kept the same; (c) rename, unmentioned: removed, the
  step-7 test green; (d) mutations killed: keep check removed, mentions moved, always kept,
  `SPOKEN_BY` left on the kept Person. On E: 0 mentions of the "Generation" Person (its 5
  `MENTIONS_PERSON` go to charlie-roberts ×2, benjamin-swerner, jake-taylor-king,
  lindsay-edwards), so the GI is byte-identical to rev A's.
- **B6, diagnostics source:** a test failing on rev A's m0025 (renamed to another person:
  `voices[].source` "roster"), and the other branch (same person keeps its source).
- **B7, Person-only map in `_edges_moved_off`:** a test with a non-Person node named as the stale
  person: its edges do not move, and do under a mutation dropping the filter. On the clone 0
  collisions; E's KG byte-identical to rev A's.
- **B8, mutations:** all 24 of `rev33b/mutate.py` killed against a zero-failure baseline, including
  the 10 survivors (R2 lookahead dropped; R2 period allowed; E2; E4; E5; S2; N3; N4; O1; O2), the
  R2 patterns re-expressed for the new regex; plus, killed: I exemption removed; exemption extended
  to A; exemption on every letter; the 4 of B5(d); B6's assignment removed; B7's filter dropped. Log
  attached; FAIL on any survivor.
- **B9 (amends 16), replay:** over the 2,684 replayable served episodes, revision B vs old, exactly
  2 changes (E's SPEAKER_00 → David Roblin; The Daily SPEAKER_01 "Moral" → unnamed), all 6 shards
  (0–5) listed with counts; the record says the replay reuses E's stored LLM answer (`third_person`,
  David Roblin), and a reprocess with another or no LLM answer is not covered.
- **B10, census and held-out 1:** Generation refused, the other 92 unchanged; 1 of 246.
- **B11, gold:** val_v2 and dev_v1 each 0 regressions, 0 fixes.
- **B12, record corrections:** step 3's LLM sentence struck and replaced; the verdicts file's
  `reviewed_commit` 20ec7ab38 (681ef3804 is on no branch); mutations named by their `mutate.py`
  labels; steps 6, 7, 8 filled.
- **B13, ordering:** no revision-B code commit authored before the commit holding this; the final
  gate report's `measured_at` later than this stamp.
- **B14, re-runs on the commit to deploy** (clean tree, digest recorded, one gate report): held-out
  1, 2, 2b; census; replay; gold; full `upgrade verify` (2 FAIL → 0; only 0015's and 0033's lines
  change); undo on the full clone (14 restored, 0 refused, 0 files differ, 0015 FAILs again);
  receipts isolation; scoreboard; the final `corpus_change_gate.py` and `build_outcome.py`, the
  outcome sha256 named in step 8.
- **B15 (amends 26), ci-fast:** `make ci-fast; echo MAKE_EXIT=$?` on the committed tree, log
  attached. Before the step-8 verdict the only allowed failure is the landing test, for 0033's
  step-8 / outcome reasons; black, mypy and every other stage green. After step 8 records GO (or GO
  WITH CONDITIONS) and the outcome sha, re-run on that commit: MAKE_EXIT=0. That commit is deployed.

Reviewer's finding to settle before the build: a kept Person bearing a hand-removed name
(mentioned by another insight) could make 0015's verify FAIL again, if 0015's verify counts
mention-only Persons. The record states which applies, and a test pins it. E is not affected.
