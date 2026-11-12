---
id: s5c_scoped_names_not_pooled
class: full
status: landed
issue: none (the person-identity design D2; operator decision 1B, 2026-11-01)
touches:
  - src/podcast_scraper/kg/entity_clusters.py
  - src/podcast_scraper/identity/ledger.py
---

# s5c_scoped_names_not_pooled — two episode-scoped bare names are never joined to each other

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Names are kept out of this
record; the measurements are in eval-data `data/eval/name-consolidation/d2i9/`.

## 1. Intake

- **Found while measuring 0041** (`d2i9/scoped_to_scoped.py`, snapshot-prod-20261030): the ledger
  joins 366 episode-scoped bare-name ids (`person:unresolved-*`):
  - 361 to a full-name person of the same show, in episodes that have no full name sharing the
    token. That is the later canonicalisation `identity/bare_name_scope` anticipates.
  - **103 groups (179 joins) contain only scoped ids**, e.g. one bare first name over 12 episodes
    of a show. Nothing says they are one person but the bare name.
- **Operator decision 1B, 2026-11-01:** stop joining two scoped ids to each other. **1C later,**
  after the voice-fingerprint shadow validation: re-allow such a join with voice evidence only
  (memory and plan V3, "Not covered / open").
- Scoped ids never surface as cards, follows or derived interests (`is_unresolved_speaker_placeholder`).
  So the visible effect is small: names read at read time, S6's spellings.

## 2. Baseline

Joins 840, person groups 475; groups whose every member is a scoped id: 103 (179 joins).

## 3. Diagnosis

The cluster rule compares names. Two bare first names are equal names, so the rule joins them; no
guard looks at the ids' scope.

## 4. Prediction

Dated: 2026-11-01

The change: a guard in the ledger's join test (`identity/guards`, the join guards's guards list). A
candidate does not join a cluster member when both ids are episode-scoped (`is_scoped_person_id`).
The refusal is recorded in `blocked_joins` with the guard's name `scoped_pair`. A scoped id still
joins a full-name member exactly as today.

Predicted on the snapshot (the ledger built before and after the change):

- the 103 scoped-only groups dissolve: **179 joins fewer**;
- joins of a scoped id to a full-name member: unchanged (361 before; the count after is measured,
  and every difference is explained, since single linkage may route through a scoped member);
- full-name joins: unchanged;
- canonical ids of the surviving groups: unchanged (the served ledger keeps them);
- `display_name` of surviving groups: unchanged;
- read time: a scoped id leaves its group, so its bare name shows as written.

### 4b. Rows added after the fresh read

- **Rule version:** D2 already moves the ledger to `cluster-person-no-bridge-guarded/3` with D1's
  `/2` served as the previous rule, and the plan bumps the version once per deploy. This guard is
  part of D2's `/3`, which is not on prod yet: no second bump. The deploy order is D2's (the ledger
  rebuild with the new image before the api reads it).
- **The guard's shape:** a separate branch in `candidate_guard`, not inside `guard_fires`. It is
  id-based and needs no shared episode (the scoped ids of the 103 groups share none). It fires when
  candidate and member are both episode-scoped. It is listed in `GUARDS` with its verdict. Its
  `blocked_joins` row records the candidate's first episode.
- **Inside one episode:** two scoped ids are refused by this guard too, whatever `co_voice` or the
  mention guard would say.
- **Single linkage:** a scoped id still joins a full-name cluster. A second scoped id is then refused
  against the first (the guard runs against every member), so which one gets the full-name cluster
  depends on the visit order. That order is deterministic (frequency, then name: `entity_clusters`).
  The count of scoped-to-full joins after is measured, not assumed.

### 4c. A correction to 4's change and 4b's guard (before any code)

Dated: 2026-11-01T12:38

The guard as 4 and 4b describe it is wrong, found while writing 4b's single-linkage row.

- **The defect:** a pairwise guard between two scoped ids fires inside a legitimate group too. With a
  full name and 30 scoped mentions of its surname, the second mention is refused against the first,
  so only one mention per full name would join. The 361 legitimate joins would collapse.
- **The change instead,** in the cluster's join test (`kg/entity_clusters._JoinTest`), for people: a
  scoped id may join a cluster only by matching a member that is **not** scoped. A match between two
  scoped ids does not count. No guard fires between scoped members, so every scoped mention still
  joins its full name. A cluster can no longer form from scoped ids alone. No `blocked_joins` row is
  written: nothing is refused, two bare names are simply not evidence of one person.
- **The order:** a full name usually seeds its cluster first (it has more episodes). When a scoped id
  seeds a cluster first, a full name can still match it and join, and later scoped ids then match
  that full name.
- **Predicted:** scoped-only groups after **0** (103 before). Scoped-to-full joins after: measured,
  expected 361 apart from order effects, each difference listed. Full-name joins: unchanged.

## 5. Acceptance

Fresh reader, 2026-11-01, a sonnet reviewer subagent with read-only tools, one pass over this record and two others, before any code:
Dated: 2026-11-01T12:35
"Not complete". Every row is taken in 4b.

## Not covered

- 1C (voice evidence), deferred by the operator.

## 6. Build

Tests, written first (`tests/unit/podcast_scraper/identity/test_ledger.py`): two scoped bare names
are never joined (failed before the code); scoped bare names still join their full name. Code:
`kg/entity_clusters._JoinTest._matches`, in the guarded ledger build only: a scoped candidate matches
only members that are not scoped. Mutant (the pool not narrowed): killed. Identity and KG unit areas:
`932 passed`.

## 7. Measure

Snapshot-prod-20261030, the ledger built with no served ledger, before (46667141a) and after
(`d2i9/s5c_measure.py`):

| | before | after |
| --- | --- | --- |
| joins | 840 | 658 |
| groups of scoped ids only | 103 | **0** |
| scoped-to-full-name joins | 187 | 184 |
| full-name joins | 365 | 365 |
| organisation joins | 109 | 109 |
| non-scoped joins lost / gained | | 0 / 0 |

- **A correction to 1 and 4:** they said 361 scoped-to-full joins. That figure counted the scoped ids
  joined in any way, scoped-only groups included. The right baseline is 187.
- **The 3 scoped-to-full joins lost** are scoped mentions that reached their full name only through
  another scoped mention: 4c's order effect, listed for the step-8 read.
