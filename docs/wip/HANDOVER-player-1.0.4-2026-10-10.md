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

## Transcript passages in the search index carry their time

- **Cause, from the code:** the indexer timed chunks only from `metadata.summary.timestamps`. The
  pipeline does not write that, so every transcript passage was indexed at 0 ms while the timed
  transcript (`*.segments.json`) sat beside it.
- **Indexer:** `indexer.body_timestamps` now times every chunk from the segments of the body it
  chunks.
  - Analysis body: `load_transcript(..., ANALYSIS)`.
  - Source-language layer: its own `<base>.segments.json`.
  - Spans come from the segments' own char offsets or cumulative lengths (GI's mapping); failing
    that, each segment's text is found in order in the body.
  - On the local corpora every segment was found: 2,796 of 2,796 in raw transcripts, 730 of 730 in
    source-language ones.
- **Rows already indexed:** migration **0028_transcript_chunk_times** times them in place from the
  text on disk. No re-index, no model.
  - It uses a partial `merge_insert` on `id` (time columns only), then `compact()`, so the existing
    indexes take the rewritten rows.
  - A row is written only if its stored text equals the body slice it maps to, so a transcript
    changed since indexing is left alone.
  - The undo record is written first (`search/transcript_chunk_times_undo.json`, `m0028.undo`),
    and the sidecar is updated to match.
  - On a copy of the e2e corpus: 284 of 284 rows timed (77 of them source-language), verify
    passes, undo restores all 284.
- The app's read-time timing stays: it narrows a hit to the exact sentence, not only the chunk.

## The Brief finds other word forms and one misremembered word

- `exact_passages` gained two read-time tiers: `forms` (inflection key) and `most` (all but one of
  three or more words). Each passage carries `marks`, the words it matched, which the Brief
  highlights.
- 94 real episodes, `scripts/eval/brief_search_eval.py --brief --synonyms` (cached synonyms):
  - word form: top 10 89.5% → 93.9%, first 20.2% → 72.1%;
  - one word misremembered: top 10 92.2% → 98.8%, first 17.8% → 86.5%;
  - no other row dropped.
  - Detail: `docs/wip/BRIEF-SEARCH-EVAL-2026-10-10.md`.

## After the deploy: the prod agent's list

**`docs/wip/HANDOVER-prod-agent-player-1.0.4-post-deploy.md`** is the step-by-step list for
whoever deploys this branch. Each prod step needs the operator's go. In order:
1. Confirm the images.
2. Migration **0028**: dry-run, read the counts, apply, verify. Undo is documented.
3. The duplicate playback milestones: `scripts/ops/dedupe_playback_milestones.py`, dry-run then
   `--apply`, with undo.
4. Check Moments against the known examples.
5. Check the speaker/topic and Brief search changes.
6. Check the first nightly writes timed transcript rows.

## Next session: the competitive analysis findings

The operator starts the next session on the findings of the discovery competitive scan
(2026-10-09). Start from:

- **`docs/wip/discovery-competitive-analysis-2026-10-09.md`**: the scan, per app, with sources
  inline. Its summary:
  - explaining *why* a recommendation fits is the best-evidenced lever;
  - discovery is moving inside episodes (chapters, mentions);
  - recommenders optimise "not your usual habit";
  - nobody recommends from what a listener highlighted or noted;
  - stacked popularity surfaces risk lower diversity.
- **The scan as a page:** https://claude.ai/artifact/HRXJkuKB6sqZdiFjma4pez
- **`docs/adr/ADR-162-mine-means-the-listeners-own-world.md`**: the "Mine" decision made from it.
- **Step 0 done (2026-10-10):** the decisions are [ADR-163](../adr/ADR-163-the-jobs-of-discover.md); the plan is `docs/wip/DISCOVER-ARC-PLAN-2026-10-10.md` (Step 1: the "why" contract, per-rail measurement, momentum on captures and completion).
- Discover 2, the preview built on this branch, was removed (operator 2026-10-10, UI only; its APIs stay: `GET /trending/episodes`, `/interests/derived`). The arc starts from the scan, not from it.

## Open

- The PR's CI must be green before merge.
- The phone still needs the newest dev build (mobile session).
- Naming (#2304) is someone else's.
