---
id: s5o_org_drift
class: full
status: open
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
  - `server/reference_integrity.py`'s resolver;
  - `server/routes/corpus_persons.py`.
  `server/app_kg_index.py` keeps only `person:` entries, so the app's org cards don't move.
  Predicted:
  - the graph, CIL and entity resolution show more org nodes: each of the 132 removed joins becomes
    its own org;
  - the 4 moved orgs resolve to their new canonical.
- **Caches:** the map is cached per corpus mtime, and the ledger file is part of that token, so the
  rebuild re-keys every cache without a restart (`server/app_cache_warm.py` re-warms).
- **Reference integrity:** org is not a reference subject kind (`SUBJECT_KINDS` in
  `server/reference_integrity.py` is topic, person, theme, storyline). The pass neither re-keys nor
  flags org ids, so no stored listener reference changes target through this change.
- **The real code must give the prototype's org map sha.** Any difference is explained before step
  7.

## 5. Acceptance

Fresh reader, 2026-10-31, a sonnet reviewer subagent with read-only tools, under a minute: "Not
complete". Rows added: the rule version and the apply order; which map; the other callers (must
not move); the surfaces fed by the served map; caches; reference integrity.

Spot checks for step 8:
- the person map and the blocked list byte-identical to S5b on the snapshot;
- the ledger's rule version, and the org map sha `d05cfbf5f9f3`;
- the 2 wrong and the 2 unclear joins left, each named and read once;
- one moved org resolved through entity search;
- a spacing-only org pair still joins.

## 6. Build

_TODO_

## 7. Measure

_TODO_

## 8. Verdict

_TODO_

## 9. Operator review

- 2026-10-31: the operator chose the one-edit rule from the three options, on training numbers.

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

- S4 reported orgs as "unchanged" without reading their joins. The change: a kind is never reported
  unchanged without its joins read (memory `org-pass-in-d2`).

## Not covered

- The read is mine, from the names, with no corpus evidence. The 13 unclear removed joins and the 2
  unclear in the after-state are not decided.
- The dedup direction is not done yet: same-org variants the rule never joined (for example a
  spacing variant on two shows). It is the next part of S5o.
- The 61 right joins now split depend on S6 for which spelling is shown.
