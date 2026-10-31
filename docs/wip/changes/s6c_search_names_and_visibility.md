---
id: s6c_search_names_and_visibility
class: full
status: open
issue: none (RFC-131 S6, the search part, and D2 item 7)
touches:
  - src/podcast_scraper/search/corpus_search.py
  - src/podcast_scraper/search/cli_handlers.py
  - src/podcast_scraper/search/transcript_chunk_lift.py
  - src/podcast_scraper/server/app_gi_view.py
---

# s6c_search_names_and_visibility — search shows only what the episode view shows, under one spelling per person

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Plan V3 S6 ("search maps
`speaker_id` through the ledger at read time") and D2 item 7 ("search returns hidden insights").

## 1. Intake

- S1 inventory, finding 5: search applies no identity map. Insight hits carry each supporting
  quote's raw `speaker_id` and no name; transcript hits lift a speaker from the bridge.
- The viewer's search card shows `speaker_name` when present, else a name made from the id's slug
  (`web/gi-kg-viewer/src/components/search/ResultCard.vue`). The player shows no speaker on search
  hits. MCP `search_corpus` and `briefing_pack` return the rows as they are.
- No search module reads `surfaceable`.
- The operator's decision on D2 item 7 (2026-10-31): search hides every insight the episode view
  hides by default, 31,545 of 135,390 on the snapshot.

## 2. Outcome and baseline

Snapshot-prod-20261030 (`s6/hidden_in_search.py`): 135,390 insights, all indexed; the episode view
serves 103,845 by default.

The 31,545 it does not serve:
- not surfaceable because the voice is unnamed: 31,327 (unknown 17,811, unidentified 13,516);
- adverts: 94; routed drop: 80; raw-label only: 44.

Today search can return every one of them.

## 3. Diagnosis

Search builds its rows from the index and the GI file, with its own code. It shares neither the
episode view's gate nor its speaker names.

## 4. Prediction

Dated: 2026-10-31

The change:
1. **Visibility:** `corpus_search._filter_and_enrich` drops an insight hit whose `source_id` is not
   in `insights_from_gi(gi)` for its episode (the default path, no fallback), one GI read per
   episode. It drops it inside the filter loop, so `top_k` still fills with visible hits.
2. **Names:** each supporting quote of an insight hit gains `speaker_name`. It comes from the
   episode view's own speaker chain, factored out of `insights_from_gi` into one helper with no
   behaviour change: authored name, then the quote's `speaker_id`, then `SPOKEN_BY`. It is spelled
   through `identity.display.person_display(root)`. `speaker_id` stays the episode's own id.
3. **Speaker scope:** `speaker=` matches a hit's speaker id or its group's canonical id, so the
   group's other ids are found too.
4. **Lifted transcript speaker:** its display name is spelled through the same view.

**Revised after the fresh read (section 5):**
- **One home for visibility:** `_InsightScope` gains `visible`. It is computed by the episode view's
  own default call (`insights_from_gi(gi)`, no unattributed fallback: "what the episode view shows"
  means the default, which differs from the panel's fallback for episodes where nobody is named).
  It is built once per corpus generation in `_cached_insight_scopes`, which is keyed on the GI path
  map; that map is cached by `corpus_mtime`, which includes the ledger.
- **Every reader of it:**
  - the result filter drops an invisible insight hit;
  - a transcript hit's lift drops `lifted.insight` when the lifted insight is invisible, and the
    transcript hit itself stays;
  - the `consensus` operator drops a pair with an invisible insight.
- **The speaker scope** is widened in BOTH the prefilter (`_tier_scope` through
  `_InsightScope.speakers`, which gains each speaker's canonical id) and the result filter, so the
  prefilter is never stricter than the result filter (its stated invariant).
- **Cost:** the visible flags cost one `insights_from_gi` per episode per corpus generation, not per
  query. `_enrich_hit` reuses one parsed artifact per episode within a query, where today it parses
  the GI on every hit.
- **`top_k`:** hidden insights are not prefiltered, because an id list of 31,545 on every query is an
  unmeasured cost. Dropped after retrieval, a query can return fewer than `top_k` only when more
  than 5 of its 6 candidates per slot (`hybrid_search`'s `top_k*6`) are hidden. Measured on the
  fixture index's queries at step 7; prod's index after deploy.
- **Outputs that move, through `run_corpus_search`:** capability search, MCP `search_corpus`,
  compare, briefing_pack, cluster_search, explore, corpus similar-episodes and the digest's topic
  bands. Each loses hidden insights.
- **Not in S6c, moved to S6b (other surfaces):** the relational routes
  (`search/relational_queries.py`) and MCP connectivity (`KGProximitySearch`), which list insights
  from the graph.

Predicted:
- **Visibility,** measured with the real filter over every GI insight treated as a hit: 31,545
  insights become unsearchable and 103,845 stay. The difference must match the episode view's
  served set exactly.
- **Names,** measured with the real `_enrich_hit` over every served insight: the supporting quotes
  whose `speaker_name` differs from today's episode-own name equal S6a's default-path changes,
  2,770 quotes in 1,559 insights. A different count is explained before step 7.
- **The episode view's own output** is byte-identical after the helper is factored out; S6a's gate
  is re-run.
- **Speaker scope and lifted speaker:** the snapshot has no index, so these are judged by unit
  tests and the fixture. The fixture has no joins, so its scoped results are unchanged there.
- **Must not move:**
  - transcript, quote and KG hits other than the lifted speaker's spelling;
  - ranking and scores;
  - `speaker_id`;
  - the search eval for visible insights.
- **What a reader sees:**
  - search returns no insight from an unnamed voice, an advert, a drop or a raw-label-only speaker;
  - the viewer's result card names the speaker with the group's spelling;
  - MCP search rows carry `speaker_name`.

## 5. Acceptance

Fresh reader, 2026-10-31, a sonnet reviewer subagent with read-only tools, about 1.5 minutes: "not
complete". Taken:
- the lift leak;
- the speaker scope widened in the prefilter as well;
- the consensus operator;
- the relational and connectivity listings, moved to S6b;
- the moving outputs, listed;
- the `top_k` underfill, stated and to be measured;
- the default call, defined;
- the per-query artifact cache.

Spot checks for step 8:
- a fixture transcript hit whose quote supports a `surfaceable: false` insight has no
  `lifted.insight`;
- a speaker search by an alias id returns the group's insights through both filters;
- the digest and similar-episodes outputs on the fixture are unchanged, or the change is explained.

## 6. Build

_TODO_

## 7. Measure

_TODO_

## 8. Verdict

_TODO_

## 9. Operator review

- 2026-10-31: hide all 31,545 in search (chosen from three options).

## 10–12.

_TODO_

## Not covered

- Quote hits (`doc_type` quote) of hidden insights are not filtered. Item 7 is about insights; a
  quote hit with an unnamed speaker shows no name.
- The ranking effect on visible results, through the freed `top_k` slots, needs prod's index. It is
  measured after deploy.
