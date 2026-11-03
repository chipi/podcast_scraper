---
id: s6d_remaining_name_surfaces
class: full
status: landed
issue: none (RFC-131 S6, every remaining name surface but the transcript; operator 2026-10-31)
touches:
  - src/podcast_scraper/identity/display.py
  - src/podcast_scraper/server/app_kg_view.py
  - src/podcast_scraper/server/*.py
  - src/podcast_scraper/server/og/build.py
  - src/podcast_scraper/server/routes/*.py
  - src/podcast_scraper/server/app_momentum.py
  - src/podcast_scraper/search/corpus_graph.py
  - src/podcast_scraper/search/cil_queries.py
  - src/podcast_scraper/search/operators.py
  - src/podcast_scraper/mcp/tools/enrichment.py
  - src/podcast_scraper/server/app_another_view_rail.py
---

# s6d_remaining_name_surfaces — one spelling per person on every remaining name surface

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Plan V3 S6. The operator
excluded only the transcript, whose files D2 item 9 fixes. Every other surface in the S1 inventory
shows a joined person under the group spelling. S6a (insights), S6b (person card, top persons,
perspectives) and S6c (search) are done. This change does the rest, and closes S6c's one open item:
the consensus operator's visibility.

## 1. Intake

- S1 inventory (eval-data `s1/SURFACES-2026-10-30.md`):
  - surfaces 7, 10 and 15–24 build person names with `entities_from_kg` (the KG node name), with no
    identity map;
  - trending people read the enricher's `person_label` (`app_momentum._labels_from_content`);
  - the MCP relational and connectivity tools read the corpus graph's node text;
  - Your Week refs come from `app_graph_refs.refs_for_slug`.
- S6c record, not covered: the `topic_consensus` operator's pairs need the visible filter once the
  enricher is re-enabled.

## 2–3. Baseline and diagnosis

Each surface shows the episode's own spelling, or the enricher's, or the graph's first-seen one. The
group's spelling, recorded in the ledger since S6a, is read only by S6a–S6c's surfaces.

## 4. Prediction

Dated: 2026-10-31

The change:
- `entities_from_kg(artifact, people=)` spells each person through the display view. Every caller
  with a corpus root passes `person_display(root)`:
  - episode chips (`routes/app_episodes`), key voices, the people rail, the captured rail (both),
    the board brief, episode notes, OG (both), discover, rail turns, the another-view rail, the
    user corpus, graph refs (Your Week) and the KG index;
  - `name_report` keeps the scan's own names (as in S6a);
  - **a correction:** an earlier version listed the digest and the recap here. Their callers
    (`app_digest_sections`, `app_recap_view`, `routes/app_discover`) keep only topics and discard
    the persons, so they are unchanged.
- Trending people: `person_label` through `shown`.
- The corpus graph: a person node's name through `shown`, for the MCP relational and connectivity
  tools.
- The consensus operator drops a pair whose insight the episode view hides (S6c's visible set).

### 4b. Rows added after the fresh read

- **The consensus operator read the wrong keys.** `topic_consensus` writes `person_a_name` /
  `person_b_name`; `search/operators._pair_dicts` read `person_a_label` / `person_b_label`, so its
  labels were always None on real data, and S6d's first test used the wrong keys too. It now reads
  `*_label`, then `*_name`, and the test uses the enricher's keys.
- **Co-appearance and consensus names** (the enrichers' `person_a_name` / `person_b_name`, and a
  co-appearance community's `community_label`) reach four surfaces: the viewer's person card and
  enrichment panels (`/corpus/entity-signals`, the ops envelope route
  `/corpus/enrichments/{id}`), the consumer `/app/corpus/enrichment`, and MCP
  `corpus_enrichment_signals`. One helper, `identity.display.spell_enrichment_people`, spells
  them on all four; it copies, so a cached envelope is never changed.
- **Found while measuring that: the person card's co-appearance rows are split by id.** The
  enricher counts raw episode ids, so a joined person's rows sit under several ids. After the
  spelling, a partner with two ids would be listed twice under one name (12 partner pairs on the
  snapshot), and one pair is a person with themselves. The card route
  (`app_enrichment._person_rows`) now matches the person by group, turns each row so its own side
  carries the id asked for (the clients orient on it) and the partner its canonical id, merges
  one partner's rows with the episode counts summed, and drops a person paired with themselves.
  The summed count is an **upper bound**: the envelope carries no episode ids, and an episode that
  lists either side under two ids counts twice (D2 item 8's 5 episodes are this shape; item 9
  folds them).
- **A community label** is spelled through its anchor; when several members carry the label (one
  person's ids), it is spelled when they all spell it alike.
- **The another-view rail's agreeing pairs** (`app_another_view_rail.agreeing_pairs`) compared the
  enricher's raw ids with the perspectives' canonical ids (S6b), so a joined person's agreement
  was missed. They now take canonical ids, and a person agreeing with themselves is dropped. An id
  defect, not a name, found by the second reader.
- **`app_gi_view` needs no change:** the speaker name comes from `_speaker_name` through `_shown`
  (S6a); the raw `speaker_label` read beside it is only the raw-label check (S3).

Predicted on snapshot-prod-20261030, with the ledger served in memory:
- **Every joined person** shows the group spelling on each changed surface. The gate counts, per
  surface, the person names that change, and requires every joined name to equal the group
  spelling after.
- **Must not move:**
  - ids, links, counts and order;
  - unjoined people's names;
  - orgs and topics;
  - the placeholder and episode-scoped rules.
- **Consensus:** 0 pairs served on the snapshot, because the enricher is off. A unit test covers the
  filter.

## 5. Acceptance

Fresh reader, 2026-10-31, a sonnet reviewer subagent with read-only tools, run on the written
change (late: the code was built before the read, which this record states). "Not complete".
Taken, in 4b:
- the consensus operator's wrong keys, and the test that used them;
- the caller row's digest and recap;
- the `app_gi_view` readers, noted.

A second fresh reader, on 4b's added rows (sonnet reviewer, about 1 minute): "not complete". Taken:
- the community label with several anchors;
- the summed count stated as an upper bound, for both sides;
- the another-view rail's raw ids;
- the consensus operator's visible filter lacked a test for a hidden first insight (a mutant
  removing that check survived); the test now has one.

Spot checks for step 8:
- one joined person's card: one row per partner, each at the group spelling;
- the ops envelope route's co-appearance names spelled, its ids unchanged;
- one MCP `corpus_enrichment_signals` call on a corpus with a ledger.

## 6. Build

Tests, written first or beside the code:
- `tests/unit/podcast_scraper/server/test_name_surfaces_follow_the_ledger.py`: chips, the corpus
  graph and trending labels follow the view, and nothing changes without it;
- `test_operators.py::TestConsensusVisibility`: a pair with a hidden insight is dropped, and labels
  are spelled (with the enricher's keys);
- the same file: co-appearance and consensus names follow the view, a cached envelope is
  untouched, and the card joins a person's ids (one row per partner, counts summed, no self pair,
  rows turned to the id asked for).

Code:
- `server/app_kg_view.entities_from_kg(people=)`;
- `person_display(root)` passed at 15 listener callers. The three that discard persons, `name_report`
  (the scan's own names) and the topic-only callers, are unchanged;
- `server/app_momentum._labels_from_content` (trending people);
- `search/corpus_graph`: `CorpusGraph(people=)`, `_spell_people` after any host merge, and the cached
  builder passes the view whenever it canonicalises;
- `search/operators.consensus_pairs_for_hits`: the visible filter (`_visible_insight_ids`) and spelled
  labels, read from the enricher's own `*_name` keys;
- `identity/display.spell_enrichment_people`, applied in `routes/app_enrichment` (both routes),
  `routes/corpus_enrichments.get_corpus_enrichment` and MCP `corpus_enrichment_signals`;
- `routes/app_enrichment._person_rows`: the card's rows joined by group.

Test stubs updated to the new keyword: `test_app_key_voices.py`; and `test_app_relational_view.py`'s
card test patches `app_kg_index.person_display`.

## 7. Measure

- **Gate** (eval-data `s6/s6d_gate.py`, the real code over snapshot-prod-20261030, the ledger in
  memory):
  - people chips: 17,324 over every served KG, and 839 change name;
  - every joined person's chip shows the group spelling;
  - 0 unjoined names changed; orgs, topics and person ids identical;
  - the corpus graph: all 475 joined person nodes show the group spelling;
  - `S6D GATE: PASS`.
  - co-appearance (the real helper over the whole envelope): 5,314 names, 109 change; every
    joined name at the group spelling; 0 unjoined names and 0 ids changed; 1 pair is two ids of
    one person;
  - the person card (the real route, for each of the 197 joined ids in a pair): 1,112 rows; 0
    partners listed twice, 0 self pairs, every row turned to the id asked for, 0 partner names
    shared, every name at the group spelling;
  - communities: 45; none whose anchor is joined, so 0 labels change on the snapshot;
  - `S6D GATE: PASS`.
- **Mutants, all killed** (each failed a test or the gate):
  - chips ignoring the view; the graph skipping the spelling;
  - the operator reading only `*_label`; the helper not spelling pair names; the card route not
    spelling; the community label not spelled;
  - the card keeping self pairs; not merging a partner's rows; matching by raw id;
  - a community label with several anchors left unspelled; the rail's agreeing pairs by raw id;
  - the operator's visible filter without its first or its second insight check.
- **Unit:** the earlier full run's 5 failures were these stubs. After the enrichment work, the
  server, search, identity, enrichment, quality and MCP areas: `2673 passed, 7 skipped`; flake8
  and mypy clean on the changed files.
- **Consensus:** 0 pairs on the snapshot (the enricher is off), and the filter is unit-tested.

## Not covered

- **The full co-appearance envelope keeps raw ids** (`/corpus/enrichment`, `/corpus/enrichments/{id}`,
  MCP): only its names are spelled, so a joined partner with two ids is two rows with one name
  there (12 pairs on the snapshot; consensus is off). The viewer's graph overlays draw its edges between the graph's
  own node ids, which are per episode. Joining the envelope's ids is not done here.
- **The consensus operator's scores** read `contradiction_score` / `cosine_similarity`, while the
  enricher writes `contradiction` / `cosine`, so both come back as 0.0 / None. Not a name; the
  enricher is off since 2026-08-24. Reported to the operator, not changed.
- **`person_web` rows key on raw ids:** a joined person's bio and photo are found only under the id
  that carries them. Not a name surface; reported.
- The transcript: D2 item 9.
