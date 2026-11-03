---
id: s6h_publisher_ties_and_pluralities
class: full
status: open
issue: none (RFC-131 S6; D2 step 11, the gold measurement of 2026-11-02)
touches:
  - src/podcast_scraper/identity/ledger.py
  - src/podcast_scraper/identity/display.py
---

# s6h_publisher_ties_and_pluralities — no group spelling without a majority or a publisher-text winner

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Names are kept out of this
record; ids are used. The measurements are in eval-data `data/eval/name-consolidation/d2out/`.

## 1. Intake

**Reading list:**
- [s6e_publisher_spelling.md](s6e_publisher_spelling.md): the publisher-text rule this narrows.
- [s6g_tie_spelling_not_chosen.md](s6g_tie_spelling_not_chosen.md): the tie rule this extends.
- [0041_joined_names_respelled.md](0041_joined_names_respelled.md): 0041 writes the group spelling
  into the episode files; its record left the feed-host rule open for the operator.

**What D2's step-11 gold measurement found** (`d2out/gold_after_d2.txt`, `d2out/gold_moves.out`,
D1's after-state against D2's final one, snapshot-prod-20261030):
- **dev_v1 S7_wrong rises by 15.** Principle 5 says it must not rise.
  - 12 insights, episode `urn:bbc:podcast:p0pd1vwf`: group `person:dara-obrey`, shown spelling now
    the group's ASR spelling; gold is the feed's stated co-host.
  - 3 insights, episode `1350c524-d61c-11f0-9faa-57cdf1b79e75`: group `person:john-glenn-hill`, a
    group whose most frequent spelling holds 3 of 9 episodes.
- **50 dev_v1 insights leave the strict tier** (`fold_exact` → `fuzzy_or_title`): group
  `person:mike-munger`, whose publisher text names it two ways, one episode each; the longer won.
- **Causes, read in the code** (`d2out/spelling_evidence_census.out`):
  1. `known_hosts` is not only publisher text. The pipeline adds hosts heard across the feed's
     transcripts (`workflow/stages/processing._recurrent_hosts_from_disk`), in their ASR spelling.
     The publisher-text rule takes them as stated.
  2. A frequency group with no majority still gets a spelling: 8 groups.
  3. Publisher text that ties between two spellings still gets one: 4 groups.
  4. In one episode, an exact name match outvotes a second spelling that also fits, so an ASR
     spelling in a stated field can win. (A reasoned risk: the census's case (b) finds 0 groups.)

## 2. Baseline

D2's final ledger (`/Users/claude/snapshots/d2-final/identity_ledger.json`, after s5d, 4e and s6g):
- joins 631, `blocked_joins` 40;
- person groups by `display_evidence.rule`: publisher 162, frequency 112, tie 75 (s6g; 349 person groups).

0041's plan on base-0041 with this ledger: **501 episodes, 2,525 files** (gate
`0041_joined_names_respelled.gate.json`).

Scoreboard D1 → D2: only S12 moves (24 → 6). Gold: dev_v1 S7_wrong +15, as in section 1.

## 3. Diagnosis

`ledger._publisher_spellings` reads every `publisher_named` id as stated, `known_hosts`'s ASR
recurrences included, takes an exact match over a second fitting spelling within an episode, and
breaks a tie by length. `_add_display_spellings`' frequency branch sets a spelling when the top
spelling holds no majority.

## 4. Prediction

Dated: 2026-11-02T09:30

**The change** (the planned code is eval-data `d2out/s6h_edits.py`, applied verbatim):
- **`identity/guards`:**
  - `EpisodeEvidence.unstated`: the episode's `known_hosts` names that its publisher text (feed and
    episode title, description, author; folded) does not contain.
  - `EpisodeEvidence.stated_named`: `publisher_named` without the unstated names.
  - Joins and guards keep `publisher_named`, so joins do not move.
- **`identity/ledger`:**
  - `_publisher_spellings` skips unstated names;
  - an episode where two folded spellings fit casts no vote (replaces exact-wins);
  - a tie in the publisher counts sets no spelling, `{"rule": "publisher_tie"}`;
  - the frequency branches set no spelling when the top spelling holds at most half of the pool's
    episodes, `{"rule": "plurality"}` (s6g's tie test becomes `_undecided`);
  - the stated-id counter reads `stated_named`.
- **`identity/display`:** `UNDECIDED = {"tie", "plurality", "publisher_tie"}`. Each gives the
  canonical id only its `canonical_name`, as s6g's tie does.
- Rule version stays `cluster-person-no-bridge-guarded/3` (D2's, served nowhere yet).

**Predicted from the planned code** (`d2out/s6h_predict.py` → `s6h_predict.out`: the edited modules
exec'd, the ledger rebuilt on the clone, diffed per group against the served D2 ledger):

| | before | after |
|---|---|---|
| ledger joins / `blocked_joins` | 631 / 40 | **631 / 40** |
| person rule publisher / frequency / tie | 162 / 112 / 75 | **152 / 110 / 75** |
| `publisher_tie` / `plurality` | 0 / 0 | **4 / 8** |
| groups whose display changes | | **26** |
| — of them, spelling removed | | **12** (4 publisher_tie, 8 plurality) |
| — of them, same spelling, evidence changed | | **14** (6 publisher → frequency, 8 fewer episodes) |
| other groups' fields | | **0 change** |
| 0041 episodes / files | 501 / 2,525 | **467 / 2,289** |

- **The 12 with no spelling** (ids in `s6h_predict.out`): publisher_tie `david-krakauer`,
  `mike-munger`, `akarsh-kumar`, `brian-r-brock`; plurality `jeff-hinton`,
  `abelardo-de-la-espriega`, `amanda-aronchik`, `francois-chollet`, `john-glenn-hill`,
  `fiona-fung`, `greg-rosalski`, `yasmin-tayag`.
- **Gold, predicted:**
  - `john-glenn-hill`'s 3 dev_v1 insights go back to S7_correct (each episode's own spelling, D1's);
  - `mike-munger`'s 50 go back to `fold_exact`;
  - **dev_v1 S7_wrong vs D1: +15 → +12**, the 12 being `dara-obrey` (see Not covered).
  - Any other bucket or tier move is measured and listed.
- **Must not move:** joins, members, canonical ids, blocked joins, and every group outside the 26.
- **Cards** of the 12 show `canonical_name`, as s6g's ties.
- **Gates:** `identity/guards` and `identity/ledger` are in the shared code digest, so all four D2
  gates (0039–0042) are rerun; 0039 and 0040 are predicted unchanged in outcome.

### 4b. Revision after the fresh read (before any code)

Dated: 2026-11-02T11:40

**1. Two of section 4's rules are dropped: measured, they change no spelling** (`d2out/s6h_attrib.py`,
one rule at a time, outputs `s6h_attrib_<rule>.out`):

| rule alone | groups whose spelling changes | 0041 episodes / files | evidence-only changes |
|---|---|---|---|
| unstated hosts | 0 | 501 / 2,525 | 13 |
| one-episode no-vote | 0 | 501 / 2,525 | 3 |
| publisher tie | 4 | 494 / 2,466 | 0 |
| plurality | 8 | 475 / 2,352 | 0 |
| all four (section 4) | 12 | 467 / 2,289 | 14 |

- **The unstated-host rule also rested on a false premise (the reader's findings 1 and 4).**
  - `_publisher_text` read an `author` key that the metadata does not have. The feed block has
    `authors`, and the episode block has `people_stated` when the item states anyone.
  - The census of every `known_hosts` name of every served episode
    (`d2out/s6h_unstated_census.py` → `.out`) by where the publisher's own text states it:
    - title or description: 3,159;
    - feed `authors`: 286;
    - neither: 758, across 88 names.
  - The 758 are not all ASR spellings. They include correctly spelled self-introduced hosts and
    parse artifacts (a trailing ")").
  - "Not in the publisher text" is therefore not "an ASR spelling", and the rule would have
    relabelled correct evidence.
- **The no-vote rule** has no measured group behind it (census (b): 0).
- **What remains:** `identity/guards` is not touched. The ledger edits are `s6h_edits.LEDGER[1:7]`:
  - the publisher tie;
  - the `Optional` types;
  - the stated branch's `None`;
  - `_undecided`.
- Together with `DISPLAY`, this is what `d2out/s6h_predict2.py` execs.

**2. The plurality rule, as the code computes it (finding 3).**
- **Pool:** the same pool as s6g, the members with a full name, else all members.
- **Count:** the spellings are folded (`ledger._fold`). A spelling's count is the sum of its
  members' `freq`, their episode counts. An episode that holds two spellings of the group counts
  for both.
- `_undecided` returns:
  - `"tie"` when the top two counts are equal, checked first, as s6g;
  - else `"plurality"` when `2 × top ≤ the sum of all counts`;
  - else `None`, meaning the majority decides.
- **Boundaries:**
  - 2/2 is a tie;
  - 2/1/1 is a plurality (exactly half);
  - 3/2 is a majority;
  - a single spelling is never a plurality.
- It is computed once in the branch (the reader's note).
- **The publisher tie:** the top two publisher-text counts are equal. No spelling is set, and
  `display_evidence` is `{"rule": "publisher_tie"}`. The by-publisher and frequency branches are
  skipped for that group.

**3. Prediction, replacing section 4's table** (`d2out/s6h_predict2.py` → `s6h_predict2.out`; card and
borrow rows read in `s6h_rows_read.out`):

| | before | after |
|---|---|---|
| ledger joins / `blocked_joins` | 631 / 40 | **631 / 40** |
| person groups | 349 | **349** |
| rule publisher / frequency / tie | 162 / 112 / 75 | **158 / 104 / 75** |
| `publisher_tie` / `plurality` | 0 / 0 | **4 / 8** |
| groups losing their spelling | | **12** |
| other groups: `display_name`, `display_evidence`, members, `canonical_name` | | **0 change** (337 groups) |
| 0041 episodes / files | 501 / 2,525 | **467 / 2,289** |
| 0041 folds / refused (episode-scoped, fold needs two full names) | 1 / 173 + 6 | **1 / 173 + 6** |
| respellings D2-final holds in the 12 that 0041 no longer makes | | **28 distinct, 41 node-episodes** |
| cards of the 12 at `canonical_name` | | **12**: 6 fold-equal to prod's card, 6 differ |
| `_group_borrows` bio / photo entries differing (2,386 rows) | | **1 / 0** |

- **The 12:**
  - publisher tie: `david-krakauer`, `mike-munger`, `akarsh-kumar`, `brian-r-brock`;
  - plurality: `jeff-hinton`, `abelardo-de-la-espriega`, `amanda-aronchik`, `francois-chollet`,
    `john-glenn-hill`, `fiona-fung`, `greg-rosalski`, `yasmin-tayag`.
- **The 6 cards that differ from prod's** (`jeff-hinton`, `abelardo-de-la-espriega`,
  `francois-chollet`, `john-glenn-hill`, `fiona-fung`, `greg-rosalski`): prod's card is the
  scan-order pick that s6g describes: a surname alone, a first-name variant or an ASR garble. The s6h card is the
  canonical's own name, which is the group's most frequent spelling in all 6. 9 of the 12 cards
  equal D2's current group spelling (corrected in 4c).
- **The one borrow:** `david-krakauer` no longer borrows the bio of `david-c-krakauer`. Its card
  loses that bio (Not covered).
- **Gold, predicted:**
  - dev_v1: `mike-munger`'s 50 insights go back to `fold_exact`;
  - dev_v1: `john-glenn-hill`'s 3 go back to S7_correct;
  - **dev_v1 S7_wrong against D1: +15 → +12** (`dara-obrey`, accepted);
  - val_v1 and val_v2: no bucket move predicted.
  - Every bucket and tier move D1 → final is listed, as integers, in section 7.
- **Scoreboard, predicted:** S12 stays at 6. S1n and S7-wrong do not rise against the D2 figures,
  except the accepted `dara-obrey` 12.
- **0041's fixpoint:** the ledger rebuilt on 0041's after-state changes nothing. The 12 keep their
  own spellings (the after-check).
- **Must not move, checked by a per-group diff on the real build** (not only the exec):
  - joins, members, canonical ids and `blocked_joins`;
  - every field of the 337 other groups.
- **Other readers (finding 2):**
  - `m0025._ledger_spells`: a non-canonical member of the 12 no longer reads as ledger-spelled,
    so 0025 may respell it to its episode's stated spelling. A node with the canonical id and its
    `canonical_name` still reads as spelled (corrected in 4c).
    - For the 4 publisher-tie groups, that is the publisher's own spelling of that episode.
    - For the 8 plurality groups, no member has publisher text.
    - **Predicted: `upgrade verify` 0025 ok** on the 0041 gate. Any 0025 plan on the after-state is
      counted there as an integer.
  - `m0029.ledger_spellings` skips the 12 (no `display_name`), as it does s6g's ties.
    `upgrade verify` 0029 ok on the gate.
  - `display.py` is the only `src` reader of a rule value (grep). `UNDECIDED` covers all three.
- **Gates:** `identity/ledger` and `identity/display` are in the shared code digest, so all four D2
  gates (0039–0042) are rerun. 0039 and 0040 are predicted unchanged in outcome.
- **Undo:** as s6g. The ledger is rebuilt from the previous code (`identity_ledger.previous.json`
  is kept by the build). 0041's own undo is its gate's undo row, rerun on this digest.
- **Derived surfaces:** the same as 0041's and 0042's records. No new surface reads the ledger.

**4. Tests first (finding 8, narrowed to the kept rules):**
- `_undecided`: 2/2 gives tie; 2/1/1 gives plurality; 3/2 gives a majority; a single spelling
  gives a majority.
- A plurality group gets no `display_name` and `{"rule": "plurality"}`.
- A publisher tie gets no `display_name` and `{"rule": "publisher_tie"}`, and the frequency branch
  does not set one.
- A publisher majority keeps its spelling.
- `display_from_ledger` gives the canonical only `canonical_name` for each of the three rules.
- m0041 `joined()` is true for a plurality and for a publisher-tie member.

**Mutants:**
- the plurality branch off;
- `<` for `≤`;
- tie checked after plurality;
- the publisher-tie branch off;
- `UNDECIDED` without `plurality`;
- `UNDECIDED` without `publisher_tie`.

Each must fail a named test.

**5. Docstrings (finding 10):** `_add_display_spellings` and `_publisher_spellings` document the
publisher tie and the plurality. The no-vote text is unchanged, since that rule is dropped. Black
runs on the edited files before the tests (finding 11).

### 4c. Revision after the second fresh read (before any code)

Dated: 2026-11-02T12:30

The second reader accepted after items 1 to 4. No new measurement except item 1's gold read
(`d2out/s6h_gold_rows.py` → `.out`). Every number below is from an existing output.

**1. Gold rows inside the 12 groups: 66** (`s6h_gold_rows.out`, rows whose stored name, at D1 or now,
is a member name of the 12). These are predicted from the rule, because each episode keeps its own
spelling. They are not computed from an applied after-state; section 7 measures them.

| group | set | rows | D1 | D2 now | predicted after s6h |
|---|---|---|---|---|---|
| `mike-munger` | dev_v1 | 50 | S7_correct `fold_exact` | S7_correct `fuzzy_or_title` | **S7_correct `fold_exact`** |
| `john-glenn-hill` | dev_v1 | 3 | S7_correct `fuzzy_or_title` | S7_wrong | **S7_correct `fuzzy_or_title`** |
| `greg-rosalski` | dev_v1 | 9 | S7_correct `fuzzy_or_title` | S7_correct `fuzzy_or_title` | **S7_correct `fuzzy_or_title`** (D1's spelling) |
| `greg-rosalski` | val_v1 | 1 | S7_correct `fuzzy_or_title` | the same | **unchanged** |
| `akarsh-kumar` | val_v2 | 3 | S8_undecidable | S8_undecidable | **unchanged** |

- **dev_v1 S7_wrong against D1: +15 → +12** (`dara-obrey`, accepted). Every other bucket and tier
  move outside these 66 rows is predicted 0 against D2 now. Section 7 lists every move D1 → final
  as integers.

**2. Scoreboard,** D2 now (`d2out/scoreboard_after_d2.txt`, 3,069 episodes, 135,390 insights,
262,618 quotes), predicted **unchanged** on every row:

| S1 | S1n | S1c | S2 | S3 | S10a | S10b | S12 | A1 hits / episodes |
|---|---|---|---|---|---|---|---|---|
| 17,860 | 7 | 0 | 31,474 | 9,219 | 194 | 2 | 6 | 396 / 297 |

- The scoreboard has no S4, S8, S9 or S11 rows.
- s6h changes spellings only, never visibility or quotes, so no row reaches.

**3. The plurality threshold** (`2 × top ≤ sum`, with exactly half counting as a plurality).
- A majority is more than half. At exactly half, the other spellings hold as many episodes as the
  top one. That is the same no-evidence state as s6g's tie, which the operator ruled on ("Yes",
  2026-11-02: on a tie, D2 does not choose).
- **The count under `<` instead:** 3 groups would lose their spelling, not 8 (`francois-chollet`,
  `abelardo-de-la-espriega`, `john-glenn-hill`).
- **The trade:** five groups sit at exactly half and lose a spelling they held:
  - `jeff-hinton` 5 of 10;
  - `amanda-aronchik` 3 of 6;
  - `fiona-fung` 2 of 4;
  - `greg-rosalski` 2 of 4;
  - `yasmin-tayag` 2 of 4.
- None of the five has a gold harm either way: `greg-rosalski`'s 9 dev_v1 rows are S7_correct
  `fuzzy_or_title` under both.

**4. "Must not move", widened for the real build:**
- the whole cluster dict of each of the 337 other groups, compared for equality;
- the `blocked_joins` contents and the `ledger_id_map` contents, not only their lengths;
- every other top-level key of the ledger except the build stamp.
- `s6h_predict2.py` compared four fields per group and the two lengths. Section 7's diff compares
  everything.

**5. Corrections to 4b:**
- 9 of the 12 cards, not 8, equal D2's group spelling. The 3 that differ are `david-krakauer`,
  `mike-munger` and `akarsh-kumar`.
- `_undecided` is now computed once (`d2out/s6h_edits.py`, `BUILD_LEDGER`, a walrus in the
  `elif`). Its docstring says it counts the sum of the members' episode counts. Both changes are
  behaviour-identical to what was measured.
- m0025: only non-canonical members stop reading as ledger-spelled.
- The bio row: the borrow is keyed to `canonical_name` (Not covered).
- The 28 respellings are those D2-final holds that 0041 no longer makes on base-0041. They are not
  an undo of a prod state.
- Section 4b's attribution row for unstated hosts was measured with the false-premise rule (it
  also flagged names stated only in feed `authors`). Its 0 is from that rule, not from a corrected
  one. The rule is dropped either way.

**6. Other readers:** `m0031` also calls `m0029.ledger_spellings` (`m0031:108`), which skips the 12, as
it does s6g's ties. `upgrade verify` covers 0031 in the gate's verify (every applied migration).

**7. Build set:** `d2out/s6h_edits.BUILD_LEDGER` and `BUILD_DISPLAY`. `LEDGER[0]`, `[7]`, `[8]` and
`GUARDS` are the dropped rules, kept only so `s6h_attrib.py` stays reproducible.

**8. Tests that must stay green:**
- `test_m0041_joined_names_respelled.py::test_0025_leaves_a_voice_the_ledger_spells`;
- s6g's tests in `test_ledger.py`.

**9. Spot checks for the step-8 reader** (each can fail):
- the stored name of `mike-munger`'s gold episodes is their own spelling, at `fold_exact`;
- `john-glenn-hill`'s gold episode shows its own spelling;
- `david-krakauer`'s card shows `canonical_name` and has no bio;
- `jeff-hinton`'s card shows `canonical_name`;
- a `dara-obrey` episode still shows the ASR spelling (the accepted loss).

## 5. Acceptance

**First fresh reader**, 2026-11-02, a sonnet reviewer subagent with read-only tools, one pass before
any code.
- **Verdict:** "Accept after findings 1, 2, 3, 4 and 5 are fixed. Findings 6 to 11 are rows or minor
  edits to add."
- Taken in 4b, which dropped two rules after measuring them.

**Second fresh reader** (a new subagent), on 4b, 2026-11-02.
- **Verdict:** "Accept after findings 1-4 are fixed. There are no blockers."
- Every finding is taken in 4c.

## 6. Build

**Tests, written first** (`tests/unit/podcast_scraper/identity/test_ledger.py`):
- no spelling holding a majority (2/1/1) shows no group spelling, `{"rule": "plurality"}`;
- more than half (3/1/1) is a majority;
- `_undecided`: 2/2 tie, 2/1/1 plurality, 3/2 and a single spelling none, 2/2/1 tie (tie read
  first);
- a tie in the publisher text shows no group spelling, `{"rule": "publisher_tie"}`;
- a publisher majority keeps its spelling;
- the card gets the canonical's name only, for each of tie, plurality and publisher_tie.

Before the code: **9 failed, 34 passed**. The majority, publisher-majority and tie-card tests
already passed, as they should.

**Code** (`d2out/s6h_edits.BUILD_LEDGER`, `BUILD_DISPLAY`, applied verbatim, then black):
- `identity/ledger`: `_undecided` (replaces `_tie`), its branch, and the publisher tie in
  `_publisher_spellings`, with docstrings;
- `identity/display`: `UNDECIDED`.

After the code: 43 passed.

**Mutants** (`mutate.py`, eval-data `d2out/scripts/mut_s6h.sh`): **killed 6 of 6**:
- the plurality branch off;
- `<` for `≤`;
- tie checked after plurality;
- the publisher-tie branch off;
- `UNDECIDED` without `plurality`;
- `UNDECIDED` without `publisher_tie`.

**Full unit suite:** 17,411 passed, 62 skipped, 1 failed. The failure is the landing test,
`test_every_registered_migration_under_the_regime_is_allowed_to_land`: "0039…: step 8: section
missing", red by design until the step-8 verdicts.

isort, flake8 and codespell pass on the changed files. `make type`: "Success: no issues found in
2416 source files".

## 7. Measure

All runs are from eval-data `d2out/scripts/chain5.sh` (log `chain5.log`, "CHAIN_DONE"), on
snapshot-prod-20261030 with 0038. Code digest `bc3c9bb670fd` for all four gates.

**The ledger,** rebuilt on clone-pre0038a (`--no-respell`), diffed against the served one
(`d2out/s6h_group_diff.py` → `.out`, whole clusters, `blocked_joins` and id-map contents, top-level
keys):
- "groups 449 -> 449 | predicted 12, changed as predicted: 12";
- "other changes: 0".
- The 449 include organisation groups.

**Gates,** each on its own base, in order:

| gate | outcome | `upgrade verify` |
|---|---|---|
| 0039 | 187 rows realigned in 1 episode, 0 refused (as before) | ok, every row |
| 0040 | 18 quotes realigned in 8 episodes, 6 left (as before) | ok |
| 0041 | **467 episodes, 2,289 files**, 179 names refused (173 + 6), as predicted | 40 of 40 ok |
| 0042 | 1,835 cluster quotes resynced, 0 unmatched | 41 of 41 ok |

- No gate has gating moves or verify regressions.
- **0042 moved, and it was not predicted:** 1,885 → 1,835.
  - `char_start` changed 1,810 → 1,760 and `char_end` 1,808 → 1,758. `speaker_id` (338) and
    `text` (5) are unchanged.
  - That is 50 fewer quotes whose offsets 0041's respelling moves. It fits s6h's fewer respellings
    inside transcript lines, but it is not traced per quote: the earlier base was replaced.

**After-checks** on the real 0041 apply (`d2out/after_checks_s6h.txt`): **0 must-not-move
breaches**.
- The verdict line reads "AFTER CHECKS: FAIL 0". It prints FAIL because 2 quote spans are off
  their text.
  - Both quotes are among 0040's 6 left, and byte-identical before and after 0041
    (`d2out/quote_spans_both.py` → `.out`: "before 2 after 2").
- **Fixpoint:**
  - joins 631 → 631, 0 changed;
  - blocked 40 → 40;
  - canonical ids gone or new 0, members changed 0;
  - `display_name` changed 0, `display_evidence` changed 0.
- **One episode's shown insight names moved:** 0041's documented fold of a duplicate id
  (`d2out/shown_moved.out`). 65 quotes now show the group's correct spelling.
- 84 transcripts respelled; 19,563 segment labels; 84 episodes with changed chunks (1,273 of
  2,136), re-embedded on the next reindex.

**Gold** (`d2out/gold_after_s6h.json`, D2 final with s6h; moves from `gold_moves.py`):
- against D1: **12 bucket moves**, all `dara-obrey`'s dev_v1 S7_correct → S7_wrong (accepted);
- against D2 before s6h: 3 bucket moves, `john-glenn-hill`'s S7_wrong → S7_correct;
- dev_v1 `fold_exact` 2,201 → **2,251** (`mike-munger`'s 50 back); `fuzzy_or_title` 72 → 25
  (72 − 50 + 3);
- val_v1 and val_v2 unchanged in every bucket and tier.

As predicted in 4c, row by row.

**Scoreboard** (`d2out/scoreboard_after_s6h.txt`): identical to D2's on every row (`diff` of the
two reports: no difference).

## Not covered

- **`dara-obrey`, 12 dev_v1 insights, accepted by the operator as a loss for now (2026-11-02:
  "Accept for now and take a note").**
  - The group is all ASR spellings of the feed's stated co-host. Its group spelling comes from a
    `known_hosts` name that the recurrence scan added.
  - Section 4's unstated-host rule would not have fixed it: the frequency rule picks the same ASR
    spelling (4b, item 1).
  - The fix needs a rule that matches a feed's stated host to a group by first name and the host
    role, not by letters. That rule is the one 0041's record left open.
  - Deferred, not refused. `luisa-lemi` (5 val_v1 insights leave `fold_exact`) has the same shape.
- **`known_hosts` is mixed evidence** (4b, item 1: 758 name-episodes that the publisher text does not
  state). The display rule, the joins, and m0041's and m0025's stated-name reads take it as stated.
  The pipeline cause (the recurrence scan writes ASR spellings into `known_hosts`) is not fixed here.
- **`david-krakauer`'s card loses its borrowed bio.** `_group_borrows` borrows from a row whose
  name is the group's shown name, `display_name or canonical_name`. That is now the canonical's
  name, and the row with the bio carries the middle-initial spelling (corrected in 4c). Its two
  spellings differ only by a middle initial, and `brian-r-brock`'s by an initial. A publisher tie
  between harmless variants removes a group spelling all the same. This follows s6g's
  operator-judged precedent: no evidence, no choice.
- **Five plurality groups hold their top spelling at exactly half** (4c, item 3). They lose a
  spelling a `<` rule would keep.
- **Section 1's cause 4** (an exact match outvotes a second fitting spelling in one episode) is left
  as it is. The census's case (b) finds 0 groups today.
- **The 12 groups show differing spellings across episodes**, as prod does today. One name per
  person is lost for them.
- **Other strict-tier losses D1 → D2 not touched here** (gold, `d1/gold_after_branch.json` against
  `d2out/gold_after_d2.json`):
  - `sarah-gonzalez`-shaped, 16 insights in val_v1 and dev_v1;
  - `erika-barris`, 8;
  - `david-sanger`, 8;
  - one other, 1.
  - They are majority or publisher picks that s6h does not reach.
