# Public's served surfaces for s-harness (feat/corpus-database-release ed1bf9962)

Built 2026-11-26 by the public session from the route decorators under
`src/podcast_scraper/server/routes/` and the mounts in `server/app.py`. Read-only; no code run.
Spot-checked by hand: `builtin_read_hooks.ROUTERS()` returns `()`; `init_corpus_engine` is called
only at `server/app.py:177-179` outside `db/`; the decorator lines of corpus_library, search and
corpus_news match.

## Scope findings first

1. The player's consumer corpus routes (`/api/app/episodes`, podcasts, persons, topics, search,
   trending, artwork ...) are not in public. They come from the installed extension (Common),
   mounted by `_mount_extension_routers` (`server/app.py:288`). Public's `/api/app/*` is app state
   only (auth, admin, preferences, profile, graph-events) plus `/api/app/version`. Public's in-repo
   helpers those routes call: `server/segments_view.py`, `kg_view.py`, `relational_view.py`,
   `app_episode_search.py`, `app_moments.py`, `app_umbrellas.py`, `app_audio_bridge.py`,
   `content_source.py`, `slugs.py`, `kg_index.py`, `corpus_access.py`.
2. No public server read path reads corpus rows yet: every route below reads files.
3. No MCP server in public; the MCP tools are Common's. `search/capability.py:49`
   (`structured_corpus_search`) and `search/relational_capability.py` are shared by `/api/search`,
   `/api/relational/*` and the MCP.
4. Extension read hooks public's routes call (bodies not in public): `read_hooks.graph_artifacts`,
   `lift_transcript_hits`, `entity_id_map`, `bridge_node_episodes`, `gi_integrity`,
   `insights_from_gi`, `groupings.search_operators`, `groupings.load_theme_payload`,
   `extensions.share_card_contributions`, `extensions.enrichment_contributions`.

## Postures

- P = on the public operator surface (`PODCAST_SERVE_OPERATOR_PUBLIC=1`,
  `_OPERATOR_PUBLIC_READ_ROUTES`, `app.py:218-234`, viewer gate) and on the tailnet operator.
- F = tailnet operator only. J = tailnet and the jobs/feeds flag (`PODCAST_SERVE_ENABLE_*`).
- App-only (`PODCAST_SERVE_APP_ONLY=1`, the player backend) mounts none of `/api/*` operator reads:
  only health, version, `/api/app/*`, `/og/*`, `/internal/*` and extension routers.
- Prod's env is not in public; which postures prod runs is not confirmed from here.
- Most routes take `?path=` (validated against `app.state.output_dir`); default corpus otherwise.

## Corpus reads

### Library (corpus_library.py, P)

| Route | Line | Answers | Reads |
| --- | --- | --- | --- |
| GET /api/corpus/feeds | 336 | feeds, unique episode counts | cumulative catalog over `*.metadata.json`; corpus-art |
| GET /api/corpus/feed-signals | 382 | a show's top topics/people, guests, themes, trending, grounding (`feed_id`, `top_k`, `max_episodes`) | catalog; `*.kg.json`; enrichments topic_theme_clusters, temporal_velocity, topic_cooccurrence_corpus, grounding_rate; search/topic_clusters.json |
| GET /api/corpus/episodes | 410 | paginated episodes (`feed_id`, `q`, `topic_q`, `since`, `until`, `has_gi`, `has_kg`, `topic_cluster_only`, `sort`, `cursor`, `limit`, `with_cil_topics`) | cumulative catalog; episode enrichment probes; topic_clusters.json + `*.bridge.json` |
| GET /api/corpus/episodes/detail | 537 | one episode's metadata, summary, transcript path, bridge counts, topic pills | last-run catalog; metadata; bridge; topic_clusters.json |
| GET /api/corpus/episodes/similar | 700 | semantic peers (`metadata_relpath`, `top_k`) | last-run catalog; LanceDB |
| POST /api/corpus/resolve-episode-artifacts | 186 | episode ids to GI/KG/bridge paths | last-run catalog |
| POST /api/corpus/node-episodes | 278 | episodes (and news items with `content_types`) naming a node | `bridge_node_episodes` hook or kg_index over `*.kg.json`; catalog; news via topic_clusters.json and topic_theme_clusters |

Membership differs: cumulative-unique for feeds, episodes, coverage, stats; last-run for detail,
similar, resolve and digest. Rows must reproduce both.

### Search and index (P)

| Route | Line | Answers | Reads |
| --- | --- | --- | --- |
| GET /api/search | search.py:118 | hybrid search (`q`, `type`, `feed`, `since`, `speaker`, `topic`, `episode_id`, `grounded_only`, `content_types`, `top_k`, `embedding_model`, `dedupe_kg_surfaces`, `enrich_results`, `operator`) | LanceDB; GI map and metadata by scope; GI and transcripts (`lift_transcript_hits`); entity_id_map; news index + served items; topic_similarity.json; side effect: appends search/query_log.jsonl |
| POST /api/search/compare | search.py:345 | two subjects compared (briefing packs + judge summary) | LanceDB scoped searches (inner reads not traced) |
| GET /api/index/stats | index_stats.py:48 | index stats and staleness | LanceDB; artifact mtimes |
| GET /api/index/timeseries | index_stats.py:149 | indexed docs per month by type | LanceDB |
| GET /api/corpus/query-activity | query_activity.py:28 | daily search counts | query_log.jsonl (runtime data, not corpus) |

### Relational graph (relational.py, all GET, P)

All build `get_corpus_graph(root, derive_speaker_links=True, reconcile_hosts=True)` from
`*.kg.json` plus `graph_artifacts` (GI, bridge), entity_id_map, host_variants, person_display.
positions, insights-about, related-insights also rerank on LanceDB (`hybrid_insight_scores`).

| Route | Line | Answers |
| --- | --- | --- |
| /api/relational/positions | 78 | insights a person stated (`person`) |
| /api/relational/insights-about | 93 | insights mentioning an entity |
| /api/relational/topic-entities | 108 | entities in a topic by mentions |
| /api/relational/entities-in | 123 | people/orgs an insight mentions |
| /api/relational/episodes | 137 | a show's episodes |
| /api/relational/related-insights | 152 | insights sharing topic/entity with a seed |
| /api/relational/insight-detail | 167 | one insight's text, quotes, topics, entities |
| /api/relational/episode-insights | 197 | insights related to an episode's insights |
| /api/relational/who-said | 212 | per-person insights on a topic |
| /api/relational/cross-show | 226 | top insight per show for a topic (`per_show`) |
| /api/relational/topics | 246 | topics a person engages with |
| /api/relational/co-speakers | 267 | people on the same topics as a person |
| /api/relational/related-topics | 283 | topics sharing insights with a topic |

### Artifacts and files (P)

| Route | Line | Answers | Reads |
| --- | --- | --- | --- |
| GET /api/artifacts | artifacts.py:66 | per-episode artifact list | corpus walk for kg/gi/bridge; membership filter; metadata publish date |
| GET /api/artifacts/{path} | artifacts.py:126 | one artifact JSON | the file; news item files 404 unless served (its kg.json excepted) |
| GET /api/corpus/text-file | corpus_text_file.py:149 | text/JSON file (`relpath`) | the file; adfree/cleaned fallbacks; speakers.diagnostics never served; news files 404 |
| GET /api/corpus/media | corpus_media.py:72 | audio with ranges | media/* |
| GET /api/corpus/binary | corpus_binary.py:33 | artwork | .podcast_scraper/corpus-art/* |

### Dashboard (P)

| Route | Line | Answers | Reads |
| --- | --- | --- | --- |
| GET /api/corpus/digest | corpus_digest.py:209 | windowed digest + topic bands (`window`, `since`, `compact`, `include_topics`, `max_rows`) | last-run catalog; topic_clusters.json; bridge pills; LanceDB per topic; node_episodes. Rolling windows use now(): compare only `all`/`since` |
| GET /api/corpus/coverage | corpus_coverage.py:38 | GI/KG coverage by month and feed | cumulative catalog |
| GET /api/corpus/stats | corpus_metrics.py:145 | month histogram, episode/feed/topic counts | cumulative catalog |
| GET /api/corpus/documents/manifest | corpus_metrics.py:186 | corpus_manifest.json | the file |
| GET /api/corpus/documents/run-summary | corpus_metrics.py:221 | corpus_run_summary.json | the file |
| GET /api/corpus/runs/summary | corpus_metrics.py:252 | newest 150 runs' metrics | run.json walk |

### News (corpus_news.py, P)

| Route | Line | Answers | Reads |
| --- | --- | --- | --- |
| GET /api/corpus/news/items | 37 | served items newest first (`publisher_id`, `since`, `limit`, `offset`) | news metadata under sources/news_article; publisher registry; serveable=none dropped |
| GET /api/corpus/news/item | 62 | one item with KG nodes | served items; item kg.json |
| GET /api/corpus/news/summary | 82 | item/publisher/teaser/paywalled counts | served items |

### Enrichments (corpus_enrichments.py, P)

| Route | Line | Answers | Reads |
| --- | --- | --- | --- |
| GET /api/corpus/enrichments | 100 | envelope catalog | enrichments/*.json; run_summary.json |
| GET /api/corpus/enrichments/{id} | 211 | one corpus envelope (person pairs re-spelled) | the envelope; identity ledger |
| GET /api/corpus/episode/enrichments/{id} | 256 | one episode envelope | `<metadata_dir>/enrichments/<stem>.<id>.json` |
| GET /api/corpus/entity-signals | 299 | enrichment rows for a person or topic | guest_coappearance, topic_consensus; temporal_velocity, topic_similarity, topic_cooccurrence_corpus |

### Share cards (always mounted, unauthenticated, app-only too)

| Route | Line | Answers | Reads |
| --- | --- | --- | --- |
| GET /og/{kind}/{ident}.png | app_og.py:26 | share card PNG (topic, person, org, episode, show, storyline, theme) | catalog; kg/gi; relational card builders; artwork; groupings and share_card_contributions |
| GET /og/page/{route}/{ident} | app_og.py:59 | og-tag HTML for bots | same |
| SPA og injection | spa.py:46-147 | og tags on document routes | same |

### Tailnet-only and flagged reads (F/J)

GET /api/ops/corpus/integrity (ops.py:236, gi_integrity hook), /api/ops/corpus/preprocessing
(ops.py:267), /api/ops/corpus/usage (ops.py:294, disk bytes: not comparable), /api/usage
(usage.py:38, P and F, llm cost files), /api/feeds (feeds.py:91, feeds.spec.yaml),
/api/feeds/overrides (feed_overrides.py:84), /api/enrichment/{status 247, health 263, metrics 348,
run-summary 385, stats 401, events 444}. Always mounted with corpus-derived fields: /api/health
(health.py:130) and /api/version, /api/app/version (version.py:140).

## Not compared (writes, jobs, app state, ops)

Writes and jobs: index rebuild, topic-clusters rebuild, run/episode rollback deletes, feeds and
overrides PUT/DELETE, operator-config and enrichment-config PUT, jobs POSTs, enricher re-enable.
App state: `/api/app/auth/*`, `/me`, `/admin/*`, `/preferences`, `/profile/*`, `/graph-events*`,
`/internal/outbox/*`. Ops: resilience, ops summary/cache-stats/secrets/gateway, llm-gateway,
`/metrics` (multi-worker only).

## Not covered

- Common's extension routes and MCP tools: the real player surface; not read here.
- Prod's serve env and postures.
- Not traced to files: search.compare internals, corpus_similar beyond its first lines,
  hybrid_insight_scores, corpus_graph._ingest, usage_rollup_snapshot, preprocessing.audit,
  enrichment_staleness, the jobs/operator-config/scheduled-jobs GETs, catalog internals beyond the
  metadata scan, whether member_episode_artifacts drops news files from /api/artifacts.
- CLI read paths (cli.py, cli_handlers.py) not enumerated.
- Parity noise to expect: rolling digest windows; query_log side effect; disk usage; build info in
  health/version; `str(root)` in many responses; catalog caches keyed on corpus mtimes.
