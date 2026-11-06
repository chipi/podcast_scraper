---
id: s5o_org_drift
class: full
status: landed
issue: none (RFC-131 S5o; plan in podcast-scraper-eval-data docs/wip/NAME-CONSOLIDATION-ARC-V3-2026-10-30.md)
touches:
  - src/podcast_scraper/identity/person_match.py
---

# s5o_org_drift — two organisation names join only when each differing word is a one-letter slip

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Plan V3 step S5o, added by the
operator on 2026-10-31: the viewer shows organisations, and a wrong organisation join is the same
failure as a wrong person join. Names are kept out of this public record. The joins and the read are
in eval-data `data/eval/name-consolidation/s5o/`.

## 1. Intake

- Prod's D1 ledger write (2026-10-31) listed org joins that read as two organisations.
- eval-data `d1/org_joins_read_2026-10-31.txt`: I read the 230 org joins on prod from their names
  only. 61 read as two entities, 15 unclear, the rest one entity. This is not an independent read,
  and no corpus evidence was opened.
- The org rule is mode `cluster` with kind `org` (`identity/person_match.py`). It was not read in
  this arc before; S4 reported "org unchanged" and checked nothing more.
- The operator's choice, 2026-10-31: the one-edit rule, of three options with training numbers.
  The other two were a two-edit rule and no change.

## 2. Outcome and baseline

Snapshot-prod-20261030, the ledger rule at S5b (cf2a4bf28):
- 229 org joins, the same in the ledger and in the read-time map;
- by the name-only read: 61 wrong, 15 unclear, 153 one entity (eval-data `s5o/inventory_snapshot.json`).

## 3. Diagnosis by loss

The wrong joins by shape (`s5o/org_inventory.py`), over all 229:

| Shape | Joins | Wrong | Unclear |
|---|---|---|---|
| one word against one word, fuzzy | 124 | 53 | 13 |
| several words, one differs | 64 | 5 | 2 |
| word count differs | 5 | 2 | 0 |
| several words, two or three differ | 3 | 1 | 0 |
| spacing or hyphen only | 33 | 0 | 0 |

The wrong ones are words that rhyme or share a stem: places onto places, products onto a company
with the same prefix. The org rule allows any differing word that keeps the 0.65 per-word and 0.70
whole-name ratios, and for orgs it puts no limit on how many words may differ.

Chosen on training shows only (94 joins, `s5o/edit_limit.py`). "Each differing word at most one edit
apart" removes 15 of the 16 wrong joins and loses 34 of the 75 right ones. The two-edit limit
removes 9 and loses 17.

## 4. Prediction

Dated: 2026-10-31

The rule: an org pair the cluster rule accepts must also pass one of two tests:
- the cleaned names are equal after removing spaces and hyphens; or
- they have the same number of words with hyphens read as spaces, and every differing word is at
  most one edit apart.

People are not affected: the extra test is for kind `org` only.

Predicted on snapshot-prod-20261030 (prototype `s5o/proto_org.py`, a patch of `_cluster` over the
ledger build and the read-time map):
- **Org joins:** 229 → 97, in both the ledger and the read-time map. 132 removed, 0 added, 4 moved;
  the ledger's org map sha `d05cfbf5f9f3`.
- **By the read:**
  - removed: 58 wrong, 61 one entity, 13 unclear;
  - the 97 after: 90 one entity (same pair as before), 4 moved, 1 unclear, 2 wrong;
  - wrong joins left: two pairs one edit apart.
- **The 4 moved, read:**
  - two move from a wrong canonical to the right one;
  - one moves to a spacing variant of its old canonical;
  - one is unclear.
- **Held-out shows, read once after the choice (`s5o/heldout_check_K1.txt`):** 5 of 5 wrong joins
  removed, 12 of 36 right ones lost, 1 unclear removed. Consistent with training, and nothing tuned
  after it.
- **Must not move:**
  - the person joins: ledger and read-time map byte-identical to S5b's on the snapshot (person part);
  - the blocked list;
  - everything outside the identity map.
- **What a reader sees:**
  - an organisation wrongly merged now shows as its own entity, with its own episodes;
  - a right pair now split shows as two spellings, each with its own episodes;
  - the shown spelling of a joined group is S6's step;
  - nothing in the corpus files changes: the map is read at serve time and in the ledger.
- **Which map:** the 229 → 97 is the map `build_ledger` builds, and equally the in-process
  `build_entity_id_map`; both were measured. The server serves the ledger map
  (`kg/entity_clusters.py` `_served_id_map`), so prod shows 97 only after D2's ledger rebuild.
- **Rule version:** `RULE_VERSION` stays `/3`, bumped once for D2 (S5a). On prod, D1's `/2` ledger is
  served as `ok_previous_rule` after the deploy, and its 229 org joins stay served until the
  rebuild. D2's order: deploy, then rebuild the ledger (`identity-ledger`), then the 97.
- **Other callers, must not move:**
  - the only path where kind `org` reaches `_cluster` is `_JoinTest._match` in
    `build_entity_canonical_map` (ledger and read-time map);
  - `identity/intra_episode_merge.py` passes kind `person`;
  - `search/corpus_graph.py` host matching uses `host_variants`.
  Neither changes.
- **Surfaces fed by the served map:**
  - `search/corpus_graph.py` org nodes;
  - `server/cil_queries.py`;
  - the reference-integrity resolver;
  - `server/routes/corpus_persons.py`.
  `server/app_kg_index.py` keeps only `person:` entries, so the app's org cards don't move.
  Predicted:
  - the graph, CIL and entity resolution show more org nodes: each of the 132 removed joins becomes
    its own org;
  - the 4 moved orgs resolve to their new canonical.
- **Caches:** the map is cached per corpus mtime, and the ledger file is part of that token, so the
  rebuild re-keys every cache without a restart (the cache warm-up re-warms).
- **Reference integrity:** org is not a reference subject kind (`SUBJECT_KINDS` in
  the reference-integrity module is topic, person, theme, storyline). The pass neither re-keys nor
  flags org ids, so no stored listener reference changes target through this change.
- **The real code must give the prototype's org map sha.** Any difference is explained before step
  7.

### 4b. Part 2, the dedup direction (dated 2026-10-31)

- **Inventory** (`s5o/org_splits.py`): org candidates on a shared show that are one name after a light
  normalisation (case, punctuation, spacing, a leading "the", a plural "s"), yet stay apart under part
  1's rule. 17 groups: 8 train, 2 held-out, 7 mixed.
  - On training shows 7 of the 8 are a leading "The" ("The X" / "X").
  - The 8th is a plural with a spacing difference. That one case is not built.
- **The rule:** two org names also join when they are EQUAL (case, spacing and hyphens aside) once
  a leading English article "the" is dropped from either side. There is no fuzzy step on top. A
  first draft compared them through the fuzzy rule and joined "The Nation" onto "Notion"
  (`s5o/proto_article.txt` history), so the comparison is exact.
- **Predicted on snapshot-prod-20261030** (prototype `s5o/proto_article.py`):
  - org joins 97 → 109; 12 added, 0 removed, 0 moved;
  - by side: 7 train, 5 mixed, 0 held-out;
  - by my read: all 12 one entity (newspapers, magazines, podcasts and an institute, each with and
    without "The").
- **The 12, by name, are in `s5o/proto_article.txt`.** Every one has two or more words after the
  article (0 of 12 have one word). On snapshot-prod-20261030 only.
- **One-word guard, added after the fresh read:** the article is dropped only when two or more words
  remain. "The Nation" / "Nation" or "The Times" / "Times" stay apart, because one word carries too
  little, as for people (S4). It costs 0 of the 12 here.
- **Third members:** all 12 are new two-member groups ("was -" for each, 0 moved). No existing group
  gains or loses a member, and the 97 prior joins are unchanged by identity. "The Rest is History"
  and "The Rest is History Club" stay apart, because the match is exact.
- **The 5 groups the rule leaves apart, of the inventory's 17:**
  - a plural "s", 4 groups: one training, two held-out, one mixed. Not built: one training case, and
    a plural can name another thing;
  - an en dash against a hyphen, 1 group, mixed. No training case; listed for later.
- **Must not move:** person joins and the blocked list, and part 1's 97 org joins.
- **Language:** "the" is English. Orgs carry no language in mode `cluster`, so a non-English leading
  article is not dropped. That can only split, never merge.

## 5. Acceptance

Fresh reader, 2026-10-31, a sonnet reviewer subagent with read-only tools, under a minute: "Not
complete". Rows added: the rule version and the apply order; which map; the other callers (must
not move); the surfaces fed by the served map; caches; reference integrity.

Part 2's fresh reader, 2026-10-31, under a minute: "not complete". Rows added: the 5 groups left
apart with causes; the one-word count, and a one-word guard (taken); third members; the 12 listed
by path.

Spot checks for step 8:
- "New Yorker" and "The New Yorker" resolve to one id;
- "The Rest is History" and "The Rest is History Club" stay two ids;
- no served org changes beyond part 1's and the 12;
- the person map and the blocked list byte-identical to S5b on the snapshot;
- the ledger's rule version, and the org map sha `d05cfbf5f9f3`;
- the 2 wrong and the 2 unclear joins left, each named and read once;
- one moved org resolved through entity search;
- a spacing-only org pair still joins.

## 6. Build

Scenario tests, written first (`test_person_match.py::test_org_differing_words_one_edit_apart`,
`test_the_org_limit_does_not_reach_people`):
- two failed before the code: a two-letter one-word pair and "West … / East …";
- a third (a hyphen against a space) failed because today's rule never joined it, and this change
  only narrows. It is recorded as today's behaviour, and the dedup direction is the next part of S5o.

Code: `identity/person_match.py` `_cluster` = the existing rule (now `_cluster_rule`) AND, for kind
`org`, `_org_drift_within_one_edit`.

## 7. Measure

- **Gate** (`s5o/s5o_gate.sh`, the real code over snapshot-prod-20261030, against S5b's ledger joins):
  - person joins identical, 731; blocked identical;
  - org joins 97, org map sha `d05cfbf5f9f3`, equal to the prototype;
  - `S5O GATE: PASS`.
- **Read-time map:** 989 → 857 joins, the 132 org joins removed.
- **Mutants, all killed:**
  - two edits allowed: gate, org joins and sha;
  - the limit applied to people: gate, person joins and blocked;
  - no spacing pass-through: `test_guards.py::TestJoinTest::test_orgs_are_not_guarded`.
- **Tests:** identity, kg, search and server unit areas: `3152 passed`. black, flake8 and mypy
  clean.

### 6b / 7b. Part 2 build and measure

- **Tests, written first** (`test_org_leading_article`, `test_the_article_rule_does_not_reach_people`):
  3 failed before the code.
- **Code:** `_cluster` accepts an org pair when the cluster rule and the one-edit limit both pass,
  or else when `_org_same_without_article` holds.
- **Gate** (`s5o/s5o_gate.sh`, now expecting the final numbers):
  - person joins identical, 731; blocked identical;
  - org joins 109, sha `a71d0a665d8b`;
  - the diff against part 1 is exactly the prototype's 12 added joins;
  - `S5O GATE: PASS`.
- **Mutants:**
  - no one-word guard: killed by a unit test;
  - the article rule applied to people: killed by a unit test;
  - **fuzzy after the article: SURVIVED the gate.** On the snapshot the one-word guard already
    refuses the one pair the fuzzy step would add. It is killed by the unit scenario
    `The Velmora Daily` / `Delmora Daily`, so the guard against it is a test, not the corpus.
- **Tests:** identity, kg, search and server areas pass; flake8 and mypy clean.

## 8. Verdict

MISSED: no fresh verdict on this rule on its own. D2's step-8 reviews were of migrations 0039–0042 (their records, section 8), over the combined ledger.

Operator accepted: 2026-11-03 this step was covered only by D2 as stated above (A1 close-out of feat/ingestion-arc, option 1).

## 9. Operator review

- 2026-10-31: the operator chose the one-edit rule from the three options, on training numbers.

## 10. Before apply / deploy

MISSED: no plan diff of its own. The rule shipped in the D2 deploy (`sha-8906696`); prod's plan for D2 is in records 0039–0042, section 10.

Operator accepted: 2026-11-03 this step was covered only by D2 as stated above (A1 close-out of feat/ingestion-arc, option 1).

## 11. After apply / deploy

MISSED: not re-measured on prod on its own. D2's prod apply and its after-measurement are in records 0039–0042, section 11.

Operator accepted: 2026-11-03 this step was covered only by D2 as stated above (A1 close-out of feat/ingestion-arc, option 1).

## 12. Retrospective

- S4 reported orgs as "unchanged" without reading their joins. The change: a kind is never reported
  unchanged without its joins read (memory `org-pass-in-d2`).

## Not covered

- The read is mine, from the names, with no corpus evidence. The 13 unclear removed joins and the 2
  unclear in the after-state are not decided.
- The dedup direction is not done yet: same-org variants the rule never joined (for example a
  spacing variant on two shows). It is the next part of S5o.
- The 61 right joins now split depend on S6 for which spelling is shown.
