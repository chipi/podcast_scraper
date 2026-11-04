---
id: s6g_tie_spelling_not_chosen
class: full
status: landed
issue: none (RFC-131 S6; operator decision, 2026-11-02, after the step-10 read of 2026-11-01)
touches:
  - src/podcast_scraper/identity/ledger.py
  - src/podcast_scraper/identity/display.py
  - src/podcast_scraper/upgrade/migrations/m0041_joined_names_respelled.py
  - src/podcast_scraper/server/app_relational_view.py
---

# s6g_tie_spelling_not_chosen — a tie between two spellings shows no group spelling

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Names are kept out of this
record; ids are used. The measurements are in eval-data `data/eval/name-consolidation/d2i9/`.

## 1. Intake

**Reading list:**
- [s6e_publisher_spelling.md](s6e_publisher_spelling.md): the display rule this changes, publisher
  text first.
- [s5d_evidence_free_fuzzy_joins.md](s5d_evidence_free_fuzzy_joins.md): the ledger this is measured
  on.
- [0041_joined_names_respelled.md](0041_joined_names_respelled.md): 0041 writes the group spelling
  into the episode files.
- [s6f_person_web_and_consensus_scores.md](s6f_person_web_and_consensus_scores.md): bio and photo
  through group ids, which reads `display_name`.
- #2352 and #2353 (the naming gaps): checked; neither covers this.

**What D2's step-10 read found** (`d2i9/respell_pairs.py`, `d2i9/tie_spellings.py` on base-0042,
0041 applied on the s5d ledger):
- 0041 makes 362 distinct respellings, and every surface shows the same spelling since S6.
- **The publisher-text rule** decides 162 groups; **the frequency rule** decides 188.
- **Inside the frequency rule,** many groups have two spellings with equal episode counts and no
  publisher text: a tie.
  - The rule then takes the longer name (`key=(freq, len(name), name)`).
  - In the 82 respellings this produced, about half improve a garbled spelling and about half
    replace a correct one with a garble. The garbles are ASR insertions and deletions.
  - 14 of the worse ones are listed in eval-data `d2i9/tie_spellings.out`.
- A shorter-name tie-break only swaps which half is wrong. A tie carries no evidence either way.
- **Operator decision, 2026-11-02 ("Yes"):** on a tie, D2 does not choose.
  - Each episode keeps its own spelling, in its files (0041 leaves it) and on its pages, as prod
    shows today.
  - The person's card shows the canonical's own name, as today.
  - The majority and publisher-text rules are unchanged.

## 2. Baseline

D2's ledger on the s5d chain (`ledger_s5d4d.json`, base-0042's served ledger, 632 joins), from
`d2i9/s6g_predict.py`:
- person groups with `display_evidence.rule` `publisher` 162, `frequency` 188;
- **of the 188, ties 75** (all in the canonical branch, none in the full-name branch).

0041's plan on base-0041 with this ledger: **556 episodes, 2,805 files** (gate
`0041_joined_names_respelled.gate.json`).

## 3. Diagnosis

`ledger._add_display_spellings`'s frequency branches always set a `display_name`. On a tie, the
spelling they set rests on name length alone.

## 4. Prediction

Dated: 2026-11-02T00:55

**The change,** in `_add_display_spellings`' frequency branches:
- **Tie definition:** the pool the branch picks from (the members with a full name, else all
  members). Spellings folded (`ledger._fold`: case, spacing and accents). A spelling's count is the
  sum of its members' episode counts. A tie is when the top two counts are equal.
- **On a tie,** no `display_name` is set, and `display_evidence` is `{"rule": "tie"}`.
- **What follows from that,** with no other change:
  - `identity/display` gives the group's members no spelling, so every surface shows each episode's
    own name;
  - 0041 respells none of the group's names (`episode_renames`: the spellings are `{None}`);
  - `m0029.ledger_spellings` skips the group.
- **Bio and photo through group ids** (`server/app_relational_view._group_borrows`): it now uses
  `display_name or canonical_name`.
  - For the 75 ties (all in the canonical branch, where the shown name was `canonical_name`), the
    borrowing is unchanged.
  - In a future full-name-branch tie, a member borrows only from members spelled like the
    canonical.
- The publisher-text branches are untouched. The rule version stays
  `cluster-person-no-bridge-guarded/3`, as for s5c and s5d: this is part of D2's `/3`.

**Predicted on the s5d chain** (the ledger rebuilt on base-0041, `--no-respell`):

| | before | after |
|---|---|---|
| ledger joins | 632 | 632 |
| person groups | 350 | 350 |
| `blocked_joins` rows | 39 | 39 |
| groups with a `display_name` (person) | 350 | **275** |
| `display_evidence` `publisher` / `frequency` / `tie` | 162 / 188 / 0 | **162 / 113 / 75** |
| `display_name` changed, other groups | | **0** |
| 0041 episodes / files | 556 / 2,805 | **501 / 2,517** |
| bio / photo borrowings | as before | **unchanged** |

- **Population.** Reached: the 75 tie groups. Not reached: the 162 publisher groups and the 113
  majority groups.
- **Must not move:** joins, members, canonical ids, blocked joins, and every other group's
  `display_name` and `display_evidence`.
- **Cards** for the 75 show the canonical's own KG name. That is the card name prod serves today
  (`s6/s6b_before.json`) for each of the 75 whose canonical is in that file; measured, every
  difference listed.
- **0041's fixpoint** after the change: the ledger rebuilt on its after-state changes nothing. The
  tie groups keep their own spellings, so the counts stay equal.
- **All D2 gates whose code digest changes are rerun.** `identity/ledger` is imported by 0041 and
  0042, so their gates are rerun. 0039 and 0040 are rerun only if their digest moves; that is
  checked, not assumed.

### 4b. Revision after the fresh read (before any code)

Dated: 2026-11-02T01:40

**1. Membership is not spelling (the reader's finding 1).** 0041's `LedgerView.joined()` reads "has a
group spelling" (`display.name_of(i) is not None`), and it decides 0041's folds, its episode-scoped
refusal and its carry of 0025.
- **The change:** `joined()` reads membership. An id has a group spelling, or it is a key or a
  value of the ledger's id map. On today's ledger every member has a spelling, so this is
  behaviour-neutral; a tie group's ids stay joined for those decisions.
- **Predicted:** folds 1 → 1; refused names unchanged (174 "an episode-scoped id", 6 "a fold needs
  two full names"); `d2i9/s6g_predict3.py`.

**2. The card (findings 4, 5).** The person card takes the name from the first episode the scan meets
(`routes/corpus_persons`, `_person_display_name`). With no spelling for the group, a variant's
episode gives `found == pid`, and the route falls back to `name_of(pid) or pid`, which is the raw
id. Prod's card for these groups is therefore that scan-order pick.
- **The change:** for a tie group, the display map gives the canonical id its own name
  (`canonical_name`) and gives the members nothing (`identity/display.display_from_ledger`). So the
  card shows the canonical's name, deterministically and never a raw id. Mentions in episodes keep
  their own spelling; in the canonical's own episodes that is its name.
- **Predicted, cards of the 75:** all 75 show `canonical_name`. Against prod's card
  (`s6/s6b_before.json`): fold-equal **44**, different **31**, ids listed in
  `d2i9/s6g_predict3.out`. Both are picks without evidence. This replaces section 4's "the card
  prod serves today" row, which was wrong.

**3. One definition (finding 2).** Every figure is now on the coded definition, 75 groups:
- 82 distinct respellings undone, 87 node-episodes (`s6g_predict3.out`). `tie_spellings.py`'s 99 used
  every member and unfolded names; it is not used.
- **0041: 556 → 501 episodes, 2,805 → 2,525 files** (v3, the exact
  semantics: canonical-only mapping and membership).

**4. Measured from the code, not only simulated (findings 3, 6).** After the build:
- the ledger is rebuilt on base-0041 (`--no-respell`);
- a per-group diff against `ledger_s5d4d.json` is a gate row. Expected: exactly the 75 lose
  `display_name` and gain `{"rule": "tie"}`, and 0 other groups change;
- the 0041 gate and the after-checks run on a real 0041 apply. Expected: the tie set unchanged
  (the same 75 ids), `display_name changed` 0, `display_evidence changed` 0, joins 632 → 632.

**5. Other readers (finding 7):**
- `m0025._ledger_spells`: a tie group's variant name is no longer "ledger-spelled", so 0025 may
  correct it from the episode's stated spelling. Tie groups have no publisher text in any member's
  episode, so there is no stated spelling. **Predicted: 0025's plan unchanged** (`upgrade verify`
  0025 row, and the 0041 gate's verify).
- `m0029.as_listed`: reads the spelling map. On the old-ledger base-0042 the 75 groups' respelled
  names would no longer read as listed. D2's run applies 0041 on this ledger, so nothing it wrote
  under the old ledger exists on prod. Not a prod concern.
- `cil_queries` takes the episode node's name first. The slug fallback is reached only for an id
  with no named node, which is unchanged.
- `_group_borrows`: `display_name or canonical_name`. **Predicted: 0 entries differ** over the 75.
  In a full-name-branch tie (none on this snapshot) the canonical's name is bare, so the group
  borrows nothing from other full-name members.

**6. Rule version (finding 8).** `/3` is D2's version and is served nowhere yet. D2's deploy rebuilds
the ledger first, so no `/3` ledger with the old spellings is served.

**7. Tests first (finding 9):** tie → no group spelling and `{"rule": "tie"}`; a majority → the
majority spelling; a full-name pool tie; the canonical-only display; `joined()` by membership.

**The table** (replacing section 4's):

| | before | after |
|---|---|---|
| ledger joins / person groups / `blocked_joins` | 632 / 350 / 39 | 632 / 350 / 39 |
| `display_evidence` `publisher` / `frequency` / `tie` | 162 / 188 / 0 | **162 / 113 / 75** |
| groups whose other fields change | | **0** |
| 0041 episodes / files | 556 / 2,805 | **501 / 2,525** |
| 0041 folds / refused | 1 / 180 | **1 / 180** |
| cards of the 75 at `canonical_name` | | **75** (44 = prod's, 31 not) |
| `_group_borrows` entries differing | | **0** |

## 5. Acceptance

Fresh reader, 2026-11-02, a sonnet reviewer subagent with read-only tools, one pass before any code.
Dated: 2026-11-02T01:15
- **Verdict:** "Not complete. Accept after findings 1, 2, 3 and 6 are fixed. The rest are rows to
  add."
- Every finding is taken in 4b.

## 6. Build

**Tests, written first:**
- In `tests/unit/podcast_scraper/identity/test_ledger.py`:
  - a tie shows no group spelling, `{"rule": "tie"}`;
  - a majority keeps its spelling;
  - one spelling written two ways is not a tie;
  - the tie is read among the full names;
  - the card gets the canonical's name only.
- In `test_m0041_joined_names_respelled.py`: a member with no group spelling is still joined.
- Before the code: 4 failed (the majority and same-spelling tests already passed, as they should).

**Code:**
- `identity/ledger._tie` and its branch in `_add_display_spellings`;
- `identity/display.display_from_ledger` (a tie group maps the canonical id only);
- `m0041.LedgerView.joined` (membership);
- `server/app_relational_view._group_borrows` (`display_name or canonical_name`).

**Mutants** (`mutate.py`): **killed 5 of 5**.
- the tie branch off;
- a loose tie test;
- spellings not folded;
- the canonical-only card dropped;
- membership dropped from `joined`.

**Full unit suite:** 17,335 passed, 1 failed: the landing test, red until the step-8 verdicts.
- The suite also found that s5d's commit had broken a server test,
  `test_app_kg_index.TestSpellingVariantsCollapseInTheAppIndex`.
- Its fixture joins two one-episode spellings with no evidence, which s5d refuses. It now gives one
  spelling a second episode.
- I had run only the identity, KG and upgrade areas for s5d.

Black, isort, flake8, codespell and `make type` pass.

## 7. Measure

**The ledger** rebuilt on the clone (`--no-respell`), diffed per group against `ledger_s5d4d.json`
(`d2i9/s6g_group_diff.py` → `s6g_group_diff.out`):
- "groups changed as predicted (tie): 75 | of the 75: 75";
- "other changes: 0";
- blocked joins 39 → 39.

**0041 gate** (`0041_joined_names_respelled.gate.json`, digest `0fc9cb830712`):
- **501 episodes, 2,525 files**, as predicted (v3);
- `upgrade verify` 40 ok, none failing.

**After-checks** on the real 0041 apply (`d2i9/after_checks_s6g.txt`): 0 must-not-move breaches.
- The verdict line reads "AFTER CHECKS: FAIL 0". It prints FAIL because 2 quote spans are off
  their text.
- Both quotes are among 0040's 6 left. They are unchanged by 0041: eval-data
  `d2out/quote_spans_both.out` measured the same 2, byte-identical before and after, on the s6h
  chain.
- This record first quoted the line as a pass without saying so (corrected 2026-11-02).
- joins 632 → 632;
- `display_name` changed 0, `display_evidence` changed 0, so the 75 tie groups are unchanged
  after 0041;
- blocked 39 → 39.

**0039 and 0040 gates** rerun on this digest: unchanged outcomes (18 quotes realigned, 6 left; 187
segment rows in 1 episode).

## Not covered

- **The 82 respellings this undoes** include about 40 that were improvements (a garble replaced by
  the right spelling). Those episodes go back to the garble prod shows today. A later read, publisher
  text or a second episode can decide them.
- **Majority-decided groups** (113) keep the majority spelling, wrong in a few (an ASR spelling
  that occurs in more episodes than the right one). Not changed here.
- **A display of one name per person is lost for the 75:** their members show differing spellings
  across episodes, as today.
