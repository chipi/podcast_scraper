---
id: s6f_person_web_and_consensus_scores
class: full
status: open
issue: none (RFC-131 D2; operator decisions 2A and 3A, 2026-11-01)
touches:
  - src/podcast_scraper/server/app_relational_view.py
  - src/podcast_scraper/search/operators.py
---

# s6f_person_web_and_consensus_scores — a person's bio and photo found through any of their ids; consensus scores read

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Two small read-time fixes.

## 1. Intake

- **3A, `person_web`** (`server/app_relational_view._person_web`, `hosted_photo_urls`): the bio,
  description and photo are looked up by the exact person id. The person card asks with the
  canonical id; a joined person whose bio sits under another member id shows none. Key voices, the
  person card and topic Top voices take photos from `hosted_photo_urls`, keyed by raw id the same way.
- **2A, consensus operator** (`search/operators._pair_dicts`): it reads `contradiction_score` and
  `cosine_similarity`; the `topic_consensus` enricher writes `contradiction` and `cosine`
  (`enrichment/enrichers/topic_consensus.py`). Both come back as 0.0 and None. The enricher is off
  since 2026-08-24.

## 2. Baseline

The clone (snapshot-prod-20261030, 0038, 0039, the D2 ledger), `d2i9/person_web_reach.py`:
- 475 person groups, 2,386 `person_web` rows;
- **38 cards** whose canonical id has no bio while a member id has one;
- 30 cards that would gain a hosted photo; 205 member ids without a photo of their own in groups
  that have one.
- **The enricher looked each row up by its member's own spelling**, so a member's row can describe
  someone else. Of the 38: in **19** the row's name is the group's spelling; in **19** it is another
  spelling. Some of those are the right person better spelled. At least two look like **wrong joins
  in the ledger** (two different people in one group). Those are reported to the operator, not fixed
  here.
- Consensus: 0 pairs served (the enricher is off).

## 3. Diagnosis

Both read by one key where the data keys another way.

## 4. Prediction

Dated: 2026-11-01

The change:
- **`_person_web(root, person_id)`:** the canonical's own row first. Else a member's row, **only
  when the row's name is the group's spelling** (folded), the member with the most episodes first.
  A row under another name may describe another person and is never borrowed.
- **`hosted_photo_urls(root)`:** a hosted photo is offered under its group's canonical id and the
  group's other ids **under the same rule** (the row's name is the group's spelling). A member's
  own photo still wins for that member.
- **`_pair_dicts`:** reads `contradiction_score`, else `contradiction`; `cosine_similarity`, else
  `cosine`.

Predicted on the clone: **19 cards gain a bio**; the cards gaining a photo are counted at step 7 under
the same rule (at most 30). 0 cards change a bio they already have; 0 unjoined people change; 0 rows
borrowed under another name. Consensus: unit tests only.

### 4b. Rows added after the fresh read

- **The group's ids and spelling come from the served ledger** (`served_id_map`,
  `identity.display.person_display`), at read time. With no valid ledger there are no groups, and
  the behaviour equals today's.
- **An empty bio on the canonical's row** falls through to a member's row under the same name rule.
  A test covers it.
- **Photos:** `_photo_route(person_id)` builds the route from an id, so a borrowed photo is offered
  under the route of the id that hosts it. The serving route (`routes/app_relational`) is unchanged.
  All ten callers of `hosted_photo_urls` receive the wider map; they still look up by their own id.
- **Consensus keys by presence, not truth:** `contradiction_score` if present, else `contradiction`;
  `cosine_similarity` if present, else `cosine`. A real 0.0 is kept. A test covers it.
- **The two likely wrong joins** are among the 19 rows under another name, so they are excluded from
  the 19 that gain a bio. Reported to the operator for the join reads.

## 5. Acceptance

Fresh reader, 2026-11-01, a sonnet reviewer subagent with read-only tools, one pass over this record and two others, before any code:
Dated: 2026-11-01T12:35
"Not complete". Every row is taken in 4b.

## 6. Build

Tests (`tests/unit/podcast_scraper/server/test_app_relational_view.py`, `search/test_operators.py`):
- a joined person's bio and photo come from another id under the group's spelling, never from a
  row under another name, and the photo routes to the id that hosts it;
- with no ledger nothing is borrowed;
- an empty own bio falls through to the group;
- consensus scores under the enricher's keys, a real 0.0 kept.

Code:
- `server/app_relational_view._group_borrows`, used by `_person_web` and `hosted_photo_urls`;
- `search/operators._pair_dicts` (keys by presence).

Mutants, all killed: the name rule off; borrowed photos dropped; the bio borrow off; the keys read by
truth.

## 7. Measure

The clone, the real code, before (no ledger) and after (`d2i9/s6f_measure.py`):

| | before | after |
|---|---|---|
| canonical cards with a bio | 109 | 128 (**19 gained**, 0 lost) |
| ids with a hosted photo | 1,897 | 2,080 |
| canonical cards gaining a photo | | 15 |
| existing own photos changed | | 0 |

Both match the prediction: 19 bios; photos at most 30 cards, measured 15.

