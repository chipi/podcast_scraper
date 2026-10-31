---
id: s6b_person_level_spelling
class: full
status: open
issue: none (RFC-131 S6, the person-level surfaces; operator scope 2026-10-31)
touches:
  - src/podcast_scraper/identity/display.py
  - src/podcast_scraper/server/app_relational_view.py
  - src/podcast_scraper/server/routes/corpus_persons.py
  - src/podcast_scraper/server/cil_queries.py
---

# s6b_person_level_spelling — one spelling per person on the person-level surfaces

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Plan V3 S6, after S6a (insight
surfaces) and S6c (search). The operator's scope (2026-10-31) is the surfaces that show one person
across episodes. The transcript is out: it keeps each episode's names, and the files are fixed by D2
item 9's migration instead.

## 1. Intake

- S1 inventory (eval-data `s1/SURFACES-2026-10-30.md`):
  - the person card shows the first-seen variant spelling under a canonical id
    (`app_relational_view.build_person_card`: `label` is the first episode's name);
  - top persons falls back to the id string when no node of the canonical id is in an episode
    (`routes/corpus_persons._person_display_name`, "defect 3");
  - perspectives read the Person node's `name` only (`cil_queries.topics_perspectives`).
- The OG person card shows the person card's `label`, and the OG topic byline the perspectives'
  `person_name`, so both follow.
- S6a's display view (`identity.display`), and the ledger's `display_name`.

## 2. Outcome and baseline

On snapshot-prod-20261030, with the ledger built in memory and served through the real read path
(`s6/person_card_surface.py`): every joined group's person card, measured. The counts are in
section 4.

## 3. Diagnosis

Each of these surfaces picks one member's spelling by its own rule: the first episode read, the
episode's node, or the id. None reads the group's spelling.

## 4. Prediction

Dated: 2026-10-31

The change:
- `PersonDisplay.name_of(person_id)` gives the group spelling of a joined id, or None.
- **Person card:** `label` = the group spelling when the person is joined; related people's names
  go through `shown`.
- **Top persons:** a found name goes through `shown`. When no node is found, the group spelling
  replaces the raw id fallback, and the id stays only for an unjoined person with no node.
- **Perspectives:** `person_name` goes through `shown`, and `name_of` comes before the id-slug
  fallback.
- Each takes the view from `person_display(root)`; with no ledger, today's output.

Predicted on the snapshot:
- **Person cards:** the count of joined groups whose card name changes, measured on all 475 groups
  (pending; filled in below). In the sample of 5, 1 changes.
- **Top persons and perspectives:** a joined person's shown name is their group spelling on both. The
  gate compares the real functions with and without the view, and counts the changed rows.
- **Must not move:**
  - ids, links, counts, rankings and roles on every surface;
  - unjoined people's names;
  - the episode-scoped "Host" rule.

### 4b. Rows added after the fresh read

- **Perspectives:**
  - applied inside `topics_perspectives`, so every caller follows: topic, theme and storyline
    perspectives, the operator route, the trending, captured and another-view rails, and the OG
    topic byline;
  - grouped by the **canonical id**, not the raw speaker id. Otherwise two ids of one person would
    become two rows under one spelling, a visible duplicate. **This moves counts by design:** a
    joined person's two rows become one, with the insights summed.
- **The KG index** (`build_kg_index`) is where the person card's label, its related people and the
  entity refs get their name. The canonical rewrite now takes the group spelling there, so all three
  follow.
- **Top persons** and the MCP `top_people` share `_rank_all_persons`. The rows showing a raw id are
  counted.
- **The OG image cache** (`max-age=3600`) can show the old spelling for up to an hour after a ledger
  rebuild. Accepted.
- **No change needed:** `person_profile` and `position_arc` return ids only; every cache keys on
  `corpus_mtime`, which includes the ledger.

## 5. Acceptance

Fresh reader, 2026-10-31, a sonnet reviewer subagent with read-only tools, about 1 minute: "not
complete". Rows added in 4b:
- the perspectives callers and grouping;
- the KG index as the home of the card's names;
- top persons' raw-id rows;
- the entity refs;
- the OG cache.

Spot checks for step 8:
- a ledger rebuild without a restart changes the card, top persons and perspectives;
- one variant-pair topic's perspectives read;
- one formerly raw-id top-persons row read.

## 6. Build

Tests, written first: `test_the_person_card_shows_the_group_spelling` and
`test_name_of_gives_a_joined_ids_group_spelling`. Both failed before the code.

Code:
- `identity/display.py`: `PersonDisplay.name_of`.
- `server/app_kg_index.py`: the canonical rewrite carries the group spelling. It runs whenever there
  are joins or spellings.
- `server/routes/corpus_persons.py`: names through `shown`, and the group spelling instead of the raw
  id.
- `server/cil_queries.py` `topics_perspectives`: grouped by canonical id, and the name through
  `shown`.

## 7. Measure

Gate (eval-data `s6/s6b_gate.sh`, `s6b_measure.py` and `s6b_check.py`): S6c's code (a worktree at
670a44e86) against this change, over snapshot-prod-20261030, with the ledger served in memory.

| | before | after |
|---|---|---|
| person cards at the group spelling | 326 of 475 | **475 of 475** (149 changed, as measured before the code) |
| top-persons rows showing a raw id | 17 of 2,308 | **0** of 2,308 |
| perspectives rows, 30 most-discussed topics | 74 | 73 |
| perspectives rows sharing a shown name | 0 | 0 |

- The one merged perspectives row is two ids of one person in the ledger, summed: 45 + 40 = 85
  insights, with the topic's total unchanged at 129.
- One respelling, from the show's text.

Mutants, both killed:
- perspectives grouped by raw id: 74 rows, and 1 sharing a name, which is the duplicate the
  grouping prevents;
- the KG index without the group spelling: 326 cards.

Unit tests: server, identity, search, MCP and quality areas, 2,636 passed; flake8 and mypy clean.

**Read list (step 8, R6):** the 149 cards whose name changes (`s6/person_card_surface_snapshot.txt`
lists the first 20, and `s6b_before.json` / `s6b_after.json` hold all). Some read as corrections. At
least one needs checking against the show text: a first name swapped to its anagram spelling.

## Not covered

- Transcript segments (operator: names only; D2 item 9 fixes the files).
- People chips, key voices, the people rail, trending people and Your Week refs: episode-scoped or
  enrichment-built names, not in the operator's person-level scope.
- MCP relational and connectivity tools: they read the corpus graph's node text. Not measured here.
