---
id: s5a_person_variants
class: full
status: open
issue: none (RFC-131 S5; plan in podcast-scraper-eval-data docs/wip/NAME-CONSOLIDATION-ARC-V3-2026-10-30.md)
touches:
  - src/podcast_scraper/identity/person_match.py
  - src/podcast_scraper/identity/ledger.py
  - src/podcast_scraper/identity/ledger_cli.py
---

# s5a_person_variants — two same-person spelling variants join: a hyphen read as a space, one middle initial

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Plan V3 step S5 (E4), first
half. The second half (S5b, the per-language vocabulary L1–L5) has its own record. Person names are
kept out of this public record. The pairs are in the eval-data repo under
`data/eval/name-consolidation/s5/`, cited by path.

## 1. Intake

- RFC-131, ADR-166 (the ledger), ADR-167 (guards inside the join): the design.
- D1's record (`d1_name_consolidation.md`): the rule this extends (mode `cluster_person`, linkage
  `no_one_word_bridge`, guards a/b/c) and its numbers.
- Plan V3, S5: only failure classes with counted training cases; every join passes the guards; replay
  from git with one mutant per change; R3a down and R3b zero new on held-out; no tuning on held-out.
- #2056 (one-token drift), #904 (the predicate's tolerant paths), #852 (the map).
- eval-data `s5/INVENTORY-2026-10-31.md`: the training splits under D1's rule, by cause, with the
  step-2 prototype results.

## 2. Outcome and baseline

D1's after-state on the S4 export (eval-data `s4/s4_export_a89e53f.json.gz`, the prod corpus token
1793330465.0899494). The map is rebuilt with `s4_build_map.py` on the branch at 540fb20, and its sha
equals D1's:

- person joins 720, blocked 9, map sha `1f48d22a02b5`;
- truth set S1, ids on both sides: train 34 joined, 11 split, 0 wrong; held-out 19 joined, 14 split,
  0 wrong;
- org joins: outside this change (S5o).

## 3. Diagnosis by loss

The training splits by cause (eval-data `s5/inv.py`, `s5/why.py`, `s5/s4split.py`):

| Cause | Training truth-set pairs | In scope here |
|---|---|---|
| A hyphen against a space in a compound name (word counts differ) | 4 | yes |
| One middle initial on one side | 2 (one also held by guard c) | yes, the exact form only |
| One-word drift | 1, plus 3 in S4's corpus read | no: the show signal fails on train (2 of 5 admitted are two people) |
| Tokens that do not look alike | 4, plus 1 in S4's read | no: the same shape as two people |
| A guard | 1, plus 2 in S4's read | no: guards are not weakened |
| No KG id of that spelling | 5 | not a map decision |
| A title missing from the list | 1 in S4's read | no: it needs a two-word phrase built for one name |

## 4. Prediction

Dated: 2026-10-31

The rule: `cluster_person_variants` accepts a pair if D1's rule accepts it as written. If not, it also
accepts the pair when either of these holds:
- D1's rule accepts the two names with each hyphen read as a space;
- exactly one side carries exactly one one-letter middle initial, and with that initial removed the
  two names are identical, at least two words.

The rule only adds. No pair D1 accepts is refused, and the guards and the linkage run as before.

Predicted on the S4 export (prototype `s5/proto.py`, variant `either_initial_exact`, map sha
`13c1fa0188d5`):

- **Person joins:** 720 → 731. 11 added, 0 removed, 0 moved; blocked stays 9.
- **Truth set S1:**
  - train: 34/11/0 → 39/6/0 (joined/split/wrong);
  - held-out: 19/14/0 → 24/9/0.
- **The 11 added joins, by side** (names in eval-data `s5/proto_diff.txt`):
  - train 4 (hyphen 3, initial 1);
  - mixed 4 (hyphen 2, initial 2);
  - held-out 3 (hyphen 1, initial 2).
  - My read: 9 one person, 2 unclear (one hyphen pair, one initial pair on a common name).
- **Must not move:**
  - org joins (230 on prod, 229 on the snapshot);
  - every person join D1 makes;
  - the blocked list;
  - everything outside the identity map.
- **"Only adds" holds for the pair test, not for the map, and the map numbers are measured.** The map
  build is greedy and order-dependent (`build_entity_canonical_map`, `kg/entity_clusters.py`). An
  added pair can pull a candidate into an earlier group, add a member that a later guard fires on, or
  change a group's canonical through `_pick_canonical`'s order: episode count, then length, then name.
  Measured on the prototype map (eval-data `s5/clusterdiff.py`):
  - **moved** (a D1 variant whose canonical differs from D1's): 0;
  - **D1 canonicals that become variants:** 0;
  - **blocked joins:** identical to D1's 9, compared by content;
  - **the 11 added:** 8 form new groups; 3 land in existing D1 groups (sizes 18, 4 and 2), and the D1
    canonical stays in each. The canonical side of each new group is listed in
    `s5/clusterdiff.txt`. Where the episode counts tie (three groups, 1 against 1), the longer
    spelling wins. This sometimes shows the less correct spelling; choosing the shown spelling is S6.
- **Ledger, the first `/3` build over a `/2` ledger:**
  - the change entry says joins_added 11, joins_removed 0, canonical_changed 0;
  - `identity_ledger.previous.json` holds the `/2` ledger, so `restore_previous` can undo it;
  - a `--dry-run` before it reports `previous_state` `ok_previous_rule`.
- **Ledger state `ok_previous_rule`, by caller:**
  - `read_ledger` returns the ledger with that state;
  - `served_id_map` gives exactly D1's map (sha `1f48d22a02b5` on the export);
  - `write_ledger` and `restore_previous` use it as the previous ledger;
  - `ledger_cli --dry-run` and `--status` name it and say "rebuild";
  - `quality/name_report.py` reports the source as `ledger`;
  - `_observe_state` gets a fifth state, so the gauge reads `ok_previous_rule`=1 and the other four 0.
  A `/1` or unknown rule version stays `version_mismatch` with no joins.
- **Rollback order:**
  - a D1 image reading a `/3` ledger gets `version_mismatch` and serves no joins;
  - so the undo is: first `identity-ledger --restore-previous`, which puts the `/2` ledger back (it
    checks the schema only), then the D1 image;
  - the reverse order leaves a window with no joins.
- **What a listener or reader sees, for the 11 joins:**
  - the two spellings show as one person: one person page, episode counts summed over the group, and
    entity search resolving either id to the canonical;
  - the shown name is the canonical's spelling;
  - nothing else is shown differently. The map is read at serve time, and no corpus file is written.
  - If step 8 reads one of the 2 unclear joins (or any other) as two people, the rule that admitted
    it is narrowed, or the step goes back to the operator. Nothing ships with a join read as two
    people.
- **The real code must give the prototype's map sha.** A different sha is a finding to explain before
  step 7.
- **Ledger:** `RULE_VERSION` moves to `cluster-person-no-bridge-guarded/3`.
  - Today `read_ledger` treats every other rule version as `version_mismatch`, and the api then
    serves no cross-episode joins at all (ADR-166).
  - A bare bump would give D2 a window with zero joins, whichever comes first. With the new image
    first, the api refuses D1's `/2` ledger until the rebuild. With the rebuild first, D1's live api
    refuses `/3`.
  - So this change also accepts a ledger of the immediately previous rule (`/2`). It is served as is,
    with the state `ok_previous_rule`, and the state gauge and `--status` say it needs a rebuild.
  - D2's order on prod is then: deploy, and the api keeps serving D1's joins; rebuild the ledger;
    the api serves `/3`.
  - Predicted: a `/2` ledger is served with exactly D1's map. A `/1` ledger or an unknown rule
    version is still `version_mismatch`, with no joins.

## 5. Acceptance

Fresh reader, 2026-10-31, a sonnet reviewer subagent with read-only tools, about 1 minute: "Not
complete". Rows added to section 4 from its findings:
- "only adds" is about the pair test, not the map; "moved" defined; the 11 joins classified; blocked
  compared by content;
- the `ok_previous_rule` callers: `write_ledger`, `restore_previous`, the CLI, the name report and
  the gauge;
- the rollback order;
- what a listener sees, and what happens to an unclear join judged wrong.

Spot checks for step 8:
1. Read the 2 unclear pairs in the after-state.
2. One initial-form join with an episode-count tie: confirm the canonical name shown.
3. A `/2` ledger: confirm the gauge and `--status`.
4. Confirm `.previous` exists after the first `/3` rebuild.
5. A `/1` ledger: confirm `version_mismatch` with zero joins.

## 6. Build

Scenario tests, written first and failing before the code (8 failed, 56 passed):
- `tests/unit/podcast_scraper/identity/test_person_match.py::test_cluster_person_hyphen_and_middle_initial`:
  12 pairs, both orders, with invented names;
- `test_cluster_person_only_adds_to_the_cluster_rule`;
- `tests/unit/podcast_scraper/identity/test_ledger.py`: a previous-rule ledger is served until the
  rebuild; the rebuild keeps it as `.previous`; an older rule means no joins; the gauge knows all five
  states; `--dry-run` names the state.

Code:
- `identity/person_match.py`: `cluster_person_variants` falls through to `_hyphen_or_initial_variant`
  only when mode `cluster` refuses;
- `identity/ledger.py`: `RULE_VERSION` `/3`, `PREVIOUS_RULE_VERSION` `/2`, `STATE_OK_PREVIOUS_RULE`,
  `LEDGER_STATES`, and `rebuild_needed` and `rule_version` in `ledger_status`;
- ADR-166, decision 3: the one-step-back exception and the undo order.
- `identity/ledger_cli.py` needed no change: `--dry-run` reports `read_ledger`'s state, and
  `--status` prints `ledger_status`.

Checks:
- Identity unit tests: `227 passed`.
- The real code over the S4 export: `joins=731 blocked=9 sha=13c1fa0188d5`, the prototype's sha.
- Replay gate `s5/s5a_gate.sh`, with D1's code from a git worktree at 540fb20: `S5A GATE: PASS`.
- Mutants, all killed:
  - no hyphen retry: 725 joins;
  - fuzzy rest after the initial: 733;
  - initials on both sides: 732;
  - folding before the as-written check: 286;
  - previous rule not accepted: 3 ledger tests fail.
- black, isort and flake8 clean on the changed files; mypy on `identity/`: no issues.

## 7. Measure

_TODO_

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

- The one-word, title and unlike-token classes stay split; they are listed in the inventory.
- Held-out pairs were read only to list the new joins (R6), and no rule was chosen from them. The two
  unsafe merges that ruled out the fuzzy middle-initial version are on training shows.
- The prediction is over the S4 export. It does not include the 22 episodes prod gained since the
  snapshot.
