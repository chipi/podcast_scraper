# Handover — player 1.0.4 branch (2026-10-10)

Branch `feat/player-1.0.4`, no PR yet. Living document: updated as each open item lands.

## Moments now matches the Brief

- Moments draws on exactly the insights the Brief shows (`app_moments.score_candidates` →
  `insights_from_gi(..., unattributed_fallback=True)`). Naming no longer decides whether an
  episode has moments; an unnamed moment shows no speaker.
- The Brief's fallback (#2198) no longer fails because of one quoteless insight: an ungrounded
  insight has no speaker, so it does not count as "named" (`app_gi_view.insights_from_gi`).
- Every "▶ Moments" door reads `has_moments` (episode payloads) / `episode_has_moments` (search
  hits), the reel's own rule, computed by `app_moments.episode_has_moments` and cached per GI file
  (~1 ms cold per episode).
- Measured on a read-only copy of the prod corpus (2,958 served episodes): Brief vs Moments
  disagreed on 289 episodes before, 2 after. Neither of the 2 has a time on disk.

## For someone else to fix: the insights upstream of Moments

**#2304** has the full root-cause analysis:
- 230 episodes where speaker naming resolved nobody (212 have an unnamed `host` voice; 62 have the
  guest's name in metadata but never placed on a voice);
- 3 where the value gate made every insight tier 0;
- 1 transcript with no timestamps;
- 1 episode whose shown insights are all quoteless.

## Speaker and topic scopes go into the search query

- Before, they were result filters: a speaker search ranked the whole corpus, kept a few hundred
  rows, and lost that speaker's insights whenever other people's ranked above them.
- Now `corpus_search._tier_scope` resolves the matching insight ids from the GI files (the same
  rules as the result filters, checked by test) and passes them per table
  (`SearchQuery.tier_filters`):
  - insights: `episode_id IN … AND source_id IN …`;
  - a topic: only `kg_topic` rows whose id contains it from aux, and no transcript table read;
  - a speaker: transcript and aux tables unchanged.
- The exact result filter still runs.
- Cost: an `IN` of 8,000 ids took 74 ms against 18 ms unscoped (synthetic 60,000-row table, local).
  The insight index is built lazily, once per corpus generation: about 3 s at prod size on the
  first speaker/topic search after an ingest. It is deliberately NOT in the cache warmer, because
  a CPU-bound warmer starved searches into 504s on 2026-10-06. Only the operator viewer and MCP
  send these scopes.
- Not changed: quote rows in the two-tier index carry no `speaker_id` (the aux table has no such
  column), so a speaker scope still drops every quote hit, as it did before.

## Open on this branch

- Limitation 3: prod's search index has no time on transcript passages (read-time workaround only).
- Limitation 4: word forms / misremembered words found 89–92% of the time.
- The phone still needs the newest dev build.
- Prod duplicate-milestone cleanup needs operator OK.
