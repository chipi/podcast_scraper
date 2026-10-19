---
id: 0033_hand_removed_names_refused
class: full
status: open
issue: none (the operator opens issues; found by prod's upgrade verify, 2026-10-19)
touches:
  - src/podcast_scraper/speaker_detectors/hosts.py
  - src/podcast_scraper/upgrade/migrations/m0015_unpublishable_speaker_names_removed.py
  - src/podcast_scraper/upgrade/migrations/m0031_named_voices_shown.py
  - src/podcast_scraper/upgrade/migrations/m0033_hand_removed_names_refused.py
  - src/podcast_scraper/upgrade/registry.py
  - src/podcast_scraper/upgrade/verdicts/0033_hand_removed_names_refused.tsv
---

# 0033 — a name a migration removed by hand cannot come back through a reprocess

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Plan:
[INGESTION-ARC-2026-10-19.md](../INGESTION-ARC-2026-10-19.md), Stage 1. A wrong name is worse than a
hidden insight (operator, 2026-10-19), so this goes before the hidden-insight work.

## 1. Intake

- **Found:** prod's `upgrade verify` after 0031's apply (2026-10-19 08:59Z) exits 1. One of its two
  FAIL lines is `0015_unpublishable_speaker_names_removed: 1 episode(s) still carry a refused name`,
  for "Ep176_ David Roblin on Building" (run 20261014-073935). The same line was on prod's
  2026-10-17 verify, and it is identical before and after 0031 on the clone (`lost/clone30L`,
  `lost/gateworkB6c`). So it predates 0031.
- **0015** removed unpublishable speaker names on 2026-10-03. Its `REMOVE` dict is a hand list of
  junk that no gate rule catches safely: "Before Gene", "As Colin", "Super Willing To Be", "Moral",
  "Generation", "Pan-African". It is operator-approved after reading every published host and guest.
- **The episode:** it was reprocessed on 2026-10-14, after 0015 ran. A migration runs once (the
  ledger records 0015 as applied), so 0015 never sees a later run.
- **0015's plan on the clone** (read-only): 1 name ("Generation"), 1 episode;
  `segments_relabelled=6110`, `quotes_unattributed=109`, `insights_unattributed=58`,
  `insights_surface_to_connect=58`, `gi_persons_deleted=1`, `kg_nodes_deleted=1`,
  `kg_edges_deleted=2`, `bridge_identities_removed=1`, `roster_entries=1`.
- **Related, not this change:** the misspelled-name class and the wrong-person rules (Stage 2).

## 2. Outcome and baseline

- **The outcome:** no insight, quote or segment is published under a name 0015 removed by hand,
  now or after any reprocess.
- **Baseline** (clone 10-16 + 0030, after 0031):
  - published under a hand-removed name: 1 episode, 58 insights, 109 quotes, 6,110 segments, all
    "Generation";
  - 0015's verify: FAIL, 1 episode.
- **Prod:** the same episode and line (prod's verify, 2026-10-19 08:59Z). Prod's counts not read yet.

## 3. Diagnosis by loss

- **Cause, measured:** the hand list lives only in the migration; the pipeline's publish gate
  (`speaker_detectors.hosts.is_publishable_speaker_name`) never reads it. Against that gate today
  (`scratchpad/lost/gatecheck15.py`):

  | Name | published with person shape | published, loose |
  | --- | --- | --- |
  | Before Gene, As Colin, Super Willing To Be | no | yes |
  | Moral, Generation, Pan-African | **yes** | yes |

  So any reprocess can publish "Moral", "Generation" or "Pan-African" again; one already did.
- **Loss:** the 58 insights under "Generation" are a wrong name, the worst class. The other names
  publish nowhere today (0015's verify finds 1 episode).
- **The path, read from the episode (clone, `run_20261014-073935`):** the roster entry is
  `{"id": "guest", "name": "Generation", "role": "guest", "voices": ["SPEAKER_00"], "source":
  "self_intro"}`. The self-introduction detector read segment 93, SPEAKER_00: "I'm Generation X, I
  suppose, without giving my age completely away." as "I'm <name>". The episode's stated guest,
  David Roblin (`source: "episode_metadata"`), stays `placed: false`, no voice. There is no LLM
  resolution in the inputs.
- **So it is a wrong person, not only junk.** SPEAKER_00 is very likely David Roblin: the stated
  guest, who talks about himself as the guest ("I'm a Welshman", "as I'm recruiting someone"). The
  host's "David Roblin, welcome to The Long Run." (segment 29) is diarized onto SPEAKER_00 too, so it
  cannot be read as an address. Two defects:
  1. the self-introduction detector accepts an ordinary word ("Generation" from "Generation X")
     as a name;
  2. the publish gate does not refuse 0015's hand-removed names, so nothing downstream catches it.
- **The repair's choice, for step 4 and the reviewer:** remove the name (the voice becomes unnamed,
  its 58 insights hidden), or name the voice David Roblin. Naming him is better for the listener but
  is a decision from evidence, so it goes the way 0031's entries went: a reviewer reads the episode
  and writes a verdict, and the migration applies the verdict. A rule that seats the stated guest
  here is Stage 2's work, not this change's.

## 4. Prediction

Dated: 2026-10-19

**Clone:** `lost/gateworkB6c` (10-16 snapshot + 0030 + 0031, 0031's measured after-state: S1 14,841;
S1n 7; S1c 0; S2 26,351; S3 7,710; S10a 178; S10b 2; S12 24; A1 357; insights 116,595; 2,685
served episodes).

**The episode** (E), "Ep176_ David Roblin on Building" (`run_20261014-073935`), read from its
files: 62 insights, all shown, routing `surface` (58 under "Generation", 3 under "Luke
Timmerman", 1 with no speaker); quotes: 109 on the "Generation" person, 5 on Luke Timmerman. The
roster places "Generation" (guest, `self_intro`) on SPEAKER_00 and leaves David Roblin
(`episode_metadata`) unplaced.

**The cause, reproduced** (`extract_self_introduced_host`, today's code): "I'm Generation X, I
suppose, …" gives `'Generation'`. `is_plausible_mononym` and `is_publishable_speaker_name` both pass
it; they also pass "Millennial", "Boomer" and "Moral".

**Two repairs for E.** The acceptance (step 5) picks one, from a fresh reviewer's verdict on
SPEAKER_00:

| Number (clone) | Before | U: unname SPEAKER_00 | N: name it David Roblin |
| --- | --- | --- | --- |
| insights published under a hand-removed name | 58 | 0 | 0 |
| quotes published under it | 109 | 0 | 0 |
| episodes 0015's verify fails on | 1 | 0 | 0 |
| S1 | 14,841 | 14,899 (+58) | 14,841 |
| S2 | 26,351 | 26,409 (+58) | 26,351 |
| S1n, S1c, S3, S10a, S10b, S12, A1 | as above | unchanged | unchanged |
| insights shown under "David Roblin" | 0 | 0 | 58 |
| episodes written | | 1 | 1 |

- U removes the name the way 0015 does. 0015's own plan for E on the clone: 6,110 segments
  relabelled; 109 quotes and 58 insights unattributed; 1 GI person, 1 KG node and 2 KG edges
  deleted; 1 bridge identity and 1 roster entry removed. E's 4 other insights stay shown, so S10a
  does not move.
- N renames SPEAKER_00 voice by voice (m0025's rewrite, as 0031 does) from "Generation" to "David
  Roblin", guest, and places the metadata entry on that voice. The same 6,110 segments and 109
  quotes change, and the 58 insights stay shown under the new name.

**Two rules for future runs.** They write no corpus file, so no number above moves because of them:
1. The publish gate refuses 0015's hand-removed names. The clone has 1 episode carrying one (E).
2. The one-word self-introduction rule refuses a word that is a label, not a name ("Generation" in
   "Generation X"). The population it must leave alone: the clone's 93 placed one-word
   `self_intro` names, 70 distinct (`scripts/measure/self_intro_names_census.py`). Predicted: it
   refuses exactly 1 of the 93, E's "Generation", and keeps the other 92.

**Seen in the census, outside this change:** "Kanye's" (1 episode, a16z "What Makes a Great
Entrepreneur?", 48 insights; the transcript reads "I'm Kanye's engineer", a retold call; the
stated guest Ben Horowitz is placed on no voice). The operator, 2026-10-19: it stays out of 0033
and goes to Stage 2 as a category (possessives on every extraction path; the stated guest seated
when the self-introduction is refused). 0033's self-introduction rule does not refuse it, so the
census prediction above stands at 1 of 93.

### Restated 2026-10-19, after the step-5 reviewer, before any code

Dated: 2026-10-19T16:30

The reviewer's verdict on SPEAKER_00 is CORRECT, so the repair is **N**. Restated from my own reads
of E's files on the clone (`scratchpad/kg.py`, `grep` over E's transcripts) and the committed
scoreboard (`scripts/measure/corpus_scoreboard.py`, label `gateworkB6c-before-0033`: S1 14,841;
S1n 7; S1c 0; S2 26,351; S3 7,710; S10a 178; S10b 2; S12 24; A1 357; insights 116,595; quotes
224,862; episodes 2,685).

- **Gating rows:** all unchanged (N keeps the 58 insights shown).
- **E, after N:**
  - roster 3 → 2 entries: Luke Timmerman (host, SPEAKER_01, unchanged), David Roblin (guest,
    placed, SPEAKER_00); no "Generation";
  - GI: 58 insights under "David Roblin", 3 Luke Timmerman, 1 none; 109 quotes to
    `person:david-roblin`; 62 visible;
  - KG: nodes 29 → 28, edges 30 → 29 (the "Generation" node merges into `person:david-roblin`,
    which becomes role guest and takes the `GUESTS_ON` edge; the two `MENTIONS` edges become one);
  - bridge: identities 27 → 26, `person:david-roblin` `{gi: true, kg: true}`;
  - transcripts: "Generation:" prefixes 62 / 62 / 32 (`.txt` / `.adfree.txt` / `.cleaned.txt`)
    become "David Roblin:"; "Generation" left once in `.txt`, `.adfree.txt` and `.anon.txt` (the
    words "Generation X"), 0 in `.cleaned.txt`; every quote and ad-free segment offset carried
    (the prefix is 2 characters longer: `.txt` and `.adfree.txt` grow by 124 characters);
  - E's S12 stays 0 of 114.
- **Files written: 14**: metadata, gi, kg, bridge, context, segments, adfree.segments, turns,
  adfree.turns, `.txt`, `.adfree.txt`, `.cleaned.txt`, speakers.diagnostics, adfree.admap (its
  `source_length` follows the longer `.txt`). Every other served file byte-identical.
- **Derived surfaces after the apply** (not written by the migration, from the census on prod):
  E's enrichment sidecars and its search-index rows.
- **0015's verify:** FAIL 1 episode → PASS. Every other migration's verify line unchanged.
- **m0015 and m0019:** `REMOVE` moves to one shared definition they both read; their
  `refused_names` and `plan()` are byte-identical before and after.
- **Rule 1 (gate):** of the 2,679 distinct served roster names, exactly {"Generation"} flips from
  pass to refuse.
- **Rule 2 (self-introduction), the class:** a one-word self-introduction directly followed by a
  lone capital letter that is not an initial ("Generation X", "Gen Z", "Plan B"; not "John F.") is a
  label, not a name. On the census: refuses "Generation"; keeps all 89 names; does not reach
  "Long", "Gekoyo" or "Kanye's" (listed as not reached). Held-out 1 is predicted before its run.
- **Held-out 1, predicted before its run** (criterion 20; dated 2026-10-19T17:10): population
  measured first, 246 one-word self-introductions captured with the rule off, over every voice of
  the 2,685 served episodes, all English (`scripts/measure/self_intro_label_rule_heldout.py
  --population`). Predicted: the rule changes exactly 1 of the 246, E's "Generation".
- **Reprocess replay, predicted before its run** (criterion 16; dated 2026-10-19T18:05): the
  production roster replayed over every served episode of the clone (`scripts/measure/roster_replay.py`,
  hosts.py of a213d5fa5 vs this branch): exactly 1 voice changes its published name, E's
  SPEAKER_00, from "Generation" to David Roblin or unnamed.
- **Reprocess durability:** with both rules, a reprocess of E finds no self-introduced name for
  SPEAKER_00; the roster's next rungs decide, and whether they seat David Roblin is measured at
  step 7 (criterion 16). If they do not, a reprocess hides E's 58 insights (a known loss, until
  Stage 2 seats the stated guest), never a wrong name.
- **Corrections from the reviewer:** "I'm a Welshman" in step 3 is SPEAKER_00's seg 126 (seg 90 is on
  SPEAKER_01). Step 1's reading list adds m0019 (imports `REMOVE`), m0022–m0024 (derived-surface
  resyncs), 0031's record and `upgrade/verdicts/`, and prod's other verify FAIL line (0029's, fixed
  in a213d5fa5).

## 5. Acceptance

Written by a fresh reviewer (Opus advisor), 2026-10-19, before any code: [0033_hand_removed_names_refused.acceptance.md](0033_hand_removed_names_refused.acceptance.md), 28 criteria (A: the repair of E by N; B: the gate rule; C: the self-introduction rule with three held-out checks; D: unchanged corpus-wide; E: prod). The verdict on SPEAKER_00: CORRECT, David Roblin, with segment evidence. Step 4 is restated after it, above.

## 6. Build

_TODO_

## 7. Measure

Clone `lost/gateworkB6c` (snapshot-prod-20261016 + 0030 + 0031; 0031 recorded in its ledger after
its plan read 0 episodes, 0 refused). Final measurement on **c71664a6d** (clean tree, code digest
`246928e62093…`, on top of main b297f2ab0), `scripts/measure/corpus_change_gate.py --migration
0033_hand_removed_names_refused --snapshot "snapshot-prod-20261016+0030+0031"`: report
[0033_hand_removed_names_refused.gate.json](0033_hand_removed_names_refused.gate.json). Class
`full`, `gating_moves {}`, 14 files changed, verify OK.

**Two builds before this one, both measured, both fixed at the cause:**
- **The first run found the rename unfinished.** E's "Generation" Person node stayed in the GI (its
  id was minted, `person:unresolved-generation-…`, never `person_id` of the name), so the bridge
  (27) and context kept it; the KG step dropped its `GUESTS_ON` edge with the node (edges 28, not
  29); the roster entry kept `source: self_intro`; `normalized_entities` kept "generation". Fixed in
  the shared rewrite (`m0025_one_person_one_entry.py`, 40dbb0438), with a test that fails on the
  old code. The same KG defect lost the role edge of 7 of 0029's 10 renames (pre-0029 backups on
  the clone): Stage 2, not this change (the plan's 2.4).
- **0033's plan and verify lines read 0029's wording** ("naming arc voices", "every voice 0029
  names"); each migration now names its own (c71664a6d), 0029's and 0031's strings unchanged.

**The acceptance criteria, measured:**

| # | Check | Result |
| --- | --- | --- |
| 1 | verdict stored and read | `upgrade/verdicts/0033_hand_removed_names_refused.tsv`, CORRECT; no verdicts file: verify FAIL "verdicts file unusable", apply raises, E untouched (tests) |
| 2 | E's roster | 2 entries: Luke Timmerman (host, SPEAKER_01), David Roblin (guest, placed, SPEAKER_00, `source: roster`); `normalized_entities`: no "generation", "david roblin" added |
| 3 | E's GI | 58 insights David Roblin, 3 Luke Timmerman, 1 none; 109 quotes and 109 `SPOKEN_BY` to `person:david-roblin`; 6 Person nodes, no Generation; 0 hidden |
| 4 | E's KG | nodes 29 → 28, edges 30 → 29; `person:david-roblin` role guest with `GUESTS_ON` and one `MENTIONS` |
| 5 | E's bridge | 27 → 26; `person:david-roblin` `{gi: true, kg: true}` |
| 6 | derived surfaces | "Generation:" prefixes 0 in `.txt`, `.adfree.txt`, `.cleaned.txt`; "David Roblin:" 62 / 62 / 40 (the 32 renamed plus 8 already in `.cleaned.txt`); "Generation" left once in `.txt`, `.adfree.txt`, `.anon.txt` and in segment text and words (the words "Generation X"); turns 0; context 0. **Deviation:** the diagnostics keep it 4 times, all in the decision record (`tried.per_voice_self_intro`, `decision_trace`), which records what the reader read; `resolved_name` is David Roblin |
| 7 | offsets | 114 of 114 quotes slice to their text; 3,736 of 3,736 ad-free segments slice to theirs; the ad map carried (`admaps_checked=1`) |
| 8 | files written | 14, exactly the predicted list (metadata, gi, kg, bridge, context, segments, adfree.segments, turns, adfree.turns, `.txt`, `.adfree.txt`, `.cleaned.txt`, speakers.diagnostics, adfree.admap); every other file byte-identical (the gate's census) |
| 9 | gating scoreboard | identical on every key (S1 14,841; S1n 7; S1c 0; S2 26,351; S3 7,710; S10a 178; S10b 2; S12 24; A1 357; insights 116,595); under a hand-removed name: 58 insights → 0, 109 quotes → 0, 1 episode → 0 |
| 10 | full `upgrade verify` | every registered migration's verify on the before-clone and the after-copy (`scratchpad/verify_all.py`): 2 FAIL before (0015: E carries a refused name; 0033: E still to write), **0 after**; those two lines change, the other 30 are identical |
| 11 | receipts isolated | own `hand_removed_names_refused.jsonl` and 14 backups under `0033/`; the 0015, 0019 and 0031 receipts byte-identical, their backups identical (504 / 6 / 403 files), checked directly (`scratchpad/rc.py`: the gate's census skips `.podcast_scraper/` and root receipts) |
| 12 | undo on the full clone | `m33.undo`: restored 14, refused 0; 0 of 112,932 corpus files differ from the before-clone (sha256); 0015's verify FAILs again; 0033's plan writes 1 episode again |
| 13 | gate refuses the 6 | every one, any case and spacing, both shapes, `en` and `es` (tests); one definition, `hosts.HAND_REMOVED_SPEAKER_NAMES`, which m0015 and m0019 read (`is`) |
| 14 | gate old vs new | over the 2,679 distinct served roster names, exactly {"Generation"} flips (a213d5fa5 vs this branch, `scratchpad/gate_cmp.py`) |
| 15 | 0015 and 0019 unchanged | `refused_names` and both `plan()` lines identical old vs new on the before-clone |
| 16 | reprocess replay | the production roster over 2,684 of 2,685 served episodes (`roster_replay.py`, hosts.py old vs new): **2** voices change. E's SPEAKER_00 "Generation" → **David Roblin** (the next rungs seat the stated guest: the repair survives a reprocess). The Daily "The Best TV Shows of the 21st Century" SPEAKER_01 "Moral" → unnamed: a reprocess with today's code would publish 0015's hand-removed "Moral" again; rule 1 stops it. The prediction said 1: **wrong**, it missed this latent case. Not replayed: "AI Sychophancy with Steve Rathje" (no `.segments.json`). My first run launched shards 1–6 of 0–5 and missed shard 0; re-run, 0 changes there |
| 17 | rule 2 is a class | `_LABEL_LETTER_FOLLOWS` (a lone capital letter, not an initial, after a one-word capture); refuses "I'm Generation X" through `extract_self_introduced_host` and `distinct_self_introductions` with rule 1 off (tests) |
| 18 | census | Generation refused; the other 92 entries unchanged; Long, Gekoyo, Kanye's **not reached** (listed) |
| 19 | generalisation guards | Faith, Robin, Jack, Brandon, "Steve Jobs" kept (tests); the cost: "I'm Robin X" is refused too (a real "Malcolm X" would be unnamed, not wrongly named) |
| 20 | held-out 1 | 246 one-word self-introductions over every voice of the 2,685 served episodes (all English), predicted 1 change before the run: **1**, E's "Generation" (`self_intro_label_rule_heldout.py`) |
| 21 | held-out 2 | written and run by the step-8 reviewer (not seen by the builder) |
| 22 | gold | `naming_gate.py`, hosts old vs new: val_v2 0 regressions, 0 fixes; dev_v1 0, 0 (no effect: E is in no gold set) |
| 23 | per language | held-out 1 is all English; no refusal outside English |
| 24–25 | unchanged corpus-wide | the gate's census and scoreboard (criteria 8, 9) |
| 26 | ci-fast | step 8 |
| 27–28 | prod | step 10 |

**Not covered here:**
- held-out 2 (criterion 21): the step-8 reviewer writes and runs it;
- prod (criteria 27–28): prod's plan against the clone's, and E's search-index rows (the clone has
  no index), at step 10;
- E's enrichment sidecars and other corpus-level derived surfaces after the apply: the census on prod;
- the 7 lost role edges of 0029's renames on prod: Stage 2;
- the clone is the 10-16 snapshot; prod has moved since (step 10 diffs the plan).

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

_TODO_
