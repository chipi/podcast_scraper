# HTTP API Reference

The optional **FastAPI** viewer server exposes a JSON HTTP API under the `/api` prefix, consumed by the Vue 3 Workbench. It is **not** part of the programmatic `podcast_scraper` library surface (see [API index](index.md)); install the `[dev]` extra and run it with `make serve` (see the [Server Guide](../guides/SERVER_GUIDE.md) for running, architecture, and extension).

This page is the source-of-truth catalogue of endpoints and response models. The live, always-current spec is the server's own OpenAPI at **`/docs`** (Swagger UI) and **`/openapi.json`**.

## Authentication

Local **dev** server: no auth.

**Production is login-first: every `/api/app/*` route requires a session** and answers
`401` without one. The complete set of exceptions is the allow-list in
`tests/integration/server/test_login_first_matrix.py`, which fails if any other route is
reachable anonymously — treat that test as the source of truth over this page:

Of those, this build serves:

- `/api/app/auth/{login,callback,logout,dev-users,status}`
- `/api/app/auth/email/{request,verify,code}` — the magic-link front door (#2272) and the store
  reviewer's code sign-in; they create the session
- `/api/app/profile/{user_id}/avatar` — deliberately open (#2109): an `<img src>` cannot send a
  bearer header, and the id is an opaque token that is never displayed
- `/api/health`

The test's list also names routes a downstream build mounts (the consumer app's discover, artwork,
unsubscribe, person photo, organisation logo, app-exit and version routes, and an MCP OAuth
server); a public-only server does not serve them.

### The admin surface and cross-site writes

- **Admin routes are restricted to a private network, not on the public edge.** `/api/app/admin/*` (users, access policy),
  `/api/app/ranking-config` (served when the app is installed) and the `/api/app/graph-events/*`
  readout require an admin session
  (`get_admin_user`: anonymous 401, non-admin 403) AND are never proxied by the public player
  nginx — it answers 404 for them (operator decision 2026-10-05). `POST /api/app/graph-events`
  itself is the player's event ingestion and stays public-facing.
- **Cross-site cookie writes are refused** (`server/app_csrf.py`). A `POST`/`PUT`/`PATCH`/`DELETE`
  that carries the session cookie (whatever else it carries) is answered **403** when its
  `Origin` — or, without one, its `Referer` — is neither this host (by name; the port is not
  compared) nor a trusted web origin (`PODCAST_SERVE_CORS_ORIGINS`; unset, the local dev servers on
  a dev box and NOTHING on a deployed one). The native-shell origins are CORS-allowed but never
  trusted for a cookie write: native auth is a Bearer token. `Origin: null` is refused.
  A request stating no origin passes (browsers always send one cross-site). Exempt: the OAuth
  callback, which Sign in with Apple posts from appleid.apple.com by design.
- **Audit records say who.** Operator-API writes are audited with `via`
  (`operator_key` / `operator_read_key` / `admin_session` / `none`) and, for a session, `by` = the
  user id. Admin writes (`/api/app/admin/*`, ranking config) and operator overrides record who and
  the value before and after. All in the JSONL audit log at `<app_data_dir>/audit.jsonl`.

**NOT covered by this page:** the operator/viewer surface (`/api/corpus/*`, `/api/jobs/*`) has its
own gating, and nothing here describes rate limits, CORS, or the reverse proxy at the edge.

Two rows on this page claimed otherwise until 2026-09-18 — `episodes/{slug}/stats` was marked
"public, no auth" and `episodes/{slug}/segments` said access "becomes auth-gated in later Epic-1
tasks". Both have carried `Depends(get_current_user)` for some time. An integration written against
either would have been built for anonymous reads and met a wall of 401s.

## Endpoints

| Method | Path | Tag | Description | Key query params |
| ------ | ---- | --- | ----------- | ---------------- |
| GET | `/api/health` | health | Liveness and **capability flags**: `status`; core viewer `artifacts_api`, `search_api`, `explore_api`, `index_routes_api`, `corpus_metrics_api`, `cil_queries_api` (default **true** when mounted); the insight stage's `explore_api`, `cil_queries_api` and `corpus_persons_api` are true only when its routers are mounted in this serve posture; `artifact_kinds` lists the per-episode artifact kinds served (`kg`, plus `gi` and `bridge` with the insight stage; absent on older builds); catalog `corpus_library_api`, `corpus_digest_api`, `corpus_binary_api`. **RFC-077 (default false):** `feeds_api`, `operator_config_api`, `jobs_api` when those route groups are mounted. Omit digest flag on older builds → clients treat digest as unavailable. | — |
| GET | `/api/version` | version | **What is running.** Public: `release` (`CL_RELEASE`, the infra release id; null when unset), `image_release` (`PODCAST_RELEASE`, the public image ref), `api` (`version`, `git_sha`, `built_at` from the image's `/app/VERSION.json`), `database` (`configured`; `schema_at_head`), `corpus` (`code_version`, `produced_by`, as `/api/health`). An admin session also gets `modules` (each installed extension's `version_info`: `name`, `version`, `git_sha`, refs) and, in `database`, `plane`, `current_revision`, `head_revision`, `server_version`. Cached per process for 5 s; no corpus walk. Also served at `/api/app/version` for the player's edge. | — |
| GET, PUT | `/api/feeds` | feeds | Read/write structured **`feeds.spec.yaml`** under the resolved corpus root (JSON **`{ "feeds": [...] }`** on PUT). | `path` |
| GET, PUT, DELETE | `/api/feeds/overrides` | feeds | Operator overrides of feed and episode fields ([#2283](https://github.com/chipi/podcast_scraper/issues/2283)), stored in **`overrides.json`** at the corpus root (beside `feeds.spec.yaml`) and applied by the pipeline; `path` must name that root (400 otherwise). **GET** `/api/feeds/overrides` lists them; **PUT/DELETE** `/api/feeds/overrides/feed?url=` and `/api/feeds/overrides/episode?url=&guid=` set or remove one (PUT replaces that entry's fields). Feed fields: `title`, `description`, `authors`, `image_url`, `language`, `hosts`; episode fields add `published_date`, `guests`, `speaker_renames`. An unknown field, a non-ISO language, a non-http image or a blank key is **422**; a language is stored as its ISO code. Same mount and guard as `/api/feeds` (admin session or operator key, reads included); every change is audited with who and the value before and after. | `path`, `url`, `guid` |
| GET, PUT | `/api/operator-config` | operator_config | Read/write **viewer-safe** operator YAML at the server-resolved path. **GET** returns `content`, `operator_config_path`, and **`available_profiles`** (union of packaged `config/profiles/*.yaml` names from **cwd** and **repo** roots, same as `Config` preset resolution; excluding `*.example.yaml`). When the file is **missing or whitespace-only** and the packaged preset **`cloud_balanced`** exists, **GET** **writes** a minimal `profile: cloud_balanced` file first (idempotent if already that content). **PUT** rejects forbidden **secret** keys and top-level **feed** keys (`rss`, `rss_url`, `rss_urls`, `feeds`) — use `/api/feeds` / **`feeds.spec.yaml`** for feeds. See [RFC-077](../rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md) (Phase 1b) and [PRD-030](../prd/PRD-030-viewer-feed-sources-and-pipeline-jobs.md). | `path` |
| GET, POST | `/api/jobs` | jobs | List (`GET`) or enqueue (`POST`) pipeline jobs for the corpus; **`GET /api/jobs/{id}`**, **`POST /api/jobs/{id}/cancel`**, **`POST /api/jobs/reconcile`**. `POST` supports per-feed scoping (P1.4): `feed` (RSS URL or feed-workspace-dirname slug; omit for full batch — resolved via `feeds.spec.yaml`, **404** on unknown slug), `skip_existing` (bool; skip episodes already present, keyed on stable episode_id/GUID), `append` (bool; episode_id-validated resume into a stable `run_append_*` dir), `max_episodes` (int ≥ 1; cap episodes this run), `episode_offset` (int ≥ 0; skip the newest N before selecting), `episode_order` (`newest` or `oldest`; **400** on any other value). All per-feed knobs require `feed` to be set. | `path`, `feed`, `skip_existing`, `append`, `max_episodes`, `episode_offset`, `episode_order` |
| GET | `/api/scheduled-jobs` | scheduled-jobs | List in-process cron schedules from `viewer_operator.yaml` `scheduled_jobs:` ([#708](https://github.com/chipi/podcast_scraper/issues/708)). Each row carries `name`, `cron`, `enabled`, `next_run_at` (UTC ISO; `null` when disabled or invalid cron). Mounts only when **`enable_jobs_api`** is on. Operators add/remove schedules by editing the YAML via **`PUT /api/operator-config`**, which triggers a scheduler reload in-process. | `path` |
| GET | `/api/artifacts` | artifacts | List `*.gi.json`, `*.kg.json`, and `*.bridge.json` (recursive); each item includes `mtime_utc` (#507) and `publish_date` (YYYY-MM-DD from episode metadata when present, else UTC calendar day from file mtime). | `path` (required) — corpus output directory |
| GET | `/api/artifacts/{path}` | artifacts | Load and return a single artifact JSON by relative path. | `path` (required) — corpus root for the relative lookup |
| GET | `/api/index/stats` | index | LanceDB index stats, staleness heuristics, and rebuild job flags (`rebuild_in_progress`, `rebuild_last_error`; #507). | `path`, `embedding_model` (optional; compare index to this id, else `Config` default) |
| POST | `/api/index/rebuild` | index | Queue background `index_corpus` (202); mutex per corpus. Poll `GET /api/index/stats`. | `path`, `rebuild`, `embedding_model`, `vector_index_path`, `vector_index_types` (comma-separated) |
| GET | `/api/search` | search | Corpus search — **hybrid** (two-tier BM25 + dense via RRF over LanceDB, with compound results) when a `search/lance_index` exists (RFC-090), with sentence embeddings; reports `no_index` when no usable index is present (FAISS retired in #995, ADR-099). **Transcript** hits may include optional **`lifted`**: overlapping **Quote** → **Insight** plus **speaker** / **topic** display names from **`bridge.json`** when present ([Semantic Search Guide — lift](../guides/SEMANTIC_SEARCH_GUIDE.md#chunk-to-insight-lift-and-offset-verification)). **Insight** hits may include **`supporting_quotes`** (from indexer enrichment): quote **`speaker_id`** / **`speaker_name`** mirror **`.gi.json`** — often **`null`** / absent when segments lack diarization labels (GitHub [#541](https://github.com/chipi/podcast_scraper/issues/541); canonical rules: [Development Guide — GI quote `speaker_id`](../guides/DEVELOPMENT_GUIDE.md#gi-quote-speaker-id)). Successful responses include optional **`lift_stats`**: **`transcript_hits_returned`** and **`lift_applied`** for the returned page (after `top_k`). Each hit carries **`source_tier`** (`insight` / `segment` / `aux`, derived from `doc_type` — PRD-033 FR1.1); the response carries **`query_type`** (detected intent: `entity_lookup` / `raw_evidence` / `temporal_tracking` / `cross_show_synthesis` / `semantic`, from the RFC-090 rules router — FR1.4). Each call also appends a privacy-light event (timestamp + `query_type`, no raw text) to `search/query_log.jsonl` for the Dashboard activity chart. | `q` (required), `path`, `type`, `feed`, `since`, `speaker`, `grounded_only`, `top_k`, `embedding_model`, `dedupe_kg_surfaces` (default `true`: merge same-text `kg_entity` / `kg_topic` rows) |
| GET | `/api/relational/*` | relational | **Relational-query layer** (RFC-094 / [#882](https://github.com/chipi/podcast_scraper/issues/882)) — read-only traversals over the typed cross-layer `CorpusGraph` (#874) that ground the PRD-033 Detail/Digest/Dashboard surfaces. All return a `RelatedNode` projection (`{id, type, text, show_id, episode_id}`); never raise (empty results on a missing id). **Flat lists:** **`/positions`** (`person` → insights they STATED, FR4.1), **`/insights-about`** (`entity` → insights that MENTION it), **`/entities-in`** (`insight` → entities it mentions), **`/topic-entities`** (`topic` → mentioned entities, ranked by frequency, FR4.2), **`/episodes`** (`podcast` → its episodes, HAS_EPISODE), **`/related-insights`** (`insight` → topic/entity siblings), **`/episode-insights`** (`episode` → siblings of its insights, FR4.3). **Grouped** (`{key: [RelatedNode]}`): **`/who-said`** (`topic` → insights per person, FR4.2), **`/cross-show`** (`topic` → top insight per distinct show — the corpus differentiator, FR3.2). `positions` / `insights-about` / `related-insights` re-rank results by an entity-scoped **hybrid** query over structural order (best-effort; degrades to structural order without an index). | `path`; one of `person` / `entity` / `insight` / `topic` / `podcast` / `episode` (per route); `k` (cap); `cross-show` also `per_show` |
| GET | `/api/corpus/query-activity` | dashboard | Daily **search-volume** counts (zero-filled, oldest→newest) from the append-only `search/query_log.jsonl` that `/api/search` writes (timestamp + intent only, no raw query text). Powers the Dashboard search-activity chart (PRD-033 FR6.2; honest scope — volume over time, not query-by-topic). | `path`, `days` (1–365, default 30) |
| POST | `/api/corpus/resolve-episode-artifacts` | corpus | Map logical **`episode_id`** values (from metadata) to corpus-relative **`gi_relative_path`** / **`kg_relative_path`** / **`bridge_relative_path`** from one catalog scan. Rows without GI on disk are listed in **`missing_episode_ids`**. The Workbench’s topic-cluster **sibling auto-load** builds candidate ids from **`topic_clusters.json`** → **`clusters[].members[].episode_ids`** (union per touched cluster). | JSON body: `episode_ids` (required), `path` (optional if server default set) |
| POST | `/api/corpus/node-episodes` | corpus | **Progressive graph expansion** (cross-episode): for a canonical **`node_id`** (`person:`, `org:`, or `topic:`), scan **`*.bridge.json`** bundles (no full triple-read), then resolve sibling **`*.gi.json`** / **`*.kg.json`** by stem. Returns **`episodes`** (metadata-relative paths when present, optional **`episode_id`**, GI/KG flags) and **`truncated`** when the cap is hit. The Workbench uses this on eligible nodes (**double-tap** expand). | JSON body: `node_id` (required), `path` (optional if server default set), `limit` (optional cap) |
| GET | `/api/corpus/feeds` | corpus | Aggregate feeds from episode metadata under the corpus root. | `path` (optional if server default set) |
| GET | `/api/corpus/episodes` | corpus | Paginated episode list (newest-first scan); optional filters. Each item includes **`cil_digest_topics`** as an **empty** array (CIL pills are returned on **digest** rows and **episode detail** only, to avoid per-row bridge reads in the list). GI/KG path fields support graph loads after opening an episode. When **`topic_cluster_only=true`**, the server keeps rows whose **`bridge.json`** topic appears on a multi-member topic-cluster **member** that lists this episode's **`episode_id`** in **`members[].episode_ids`** (reads **`search/topic_clusters.json`** per request; pipeline-built clusters include those ids). | `path`, `feed_id`, `q` (title substring), `topic_q`, `since` (`YYYY-MM-DD`), `topic_cluster_only` (boolean), `limit` (1–200), `cursor` |
| GET | `/api/corpus/episodes/detail` | corpus | Episode row + summary bullets + GI/KG/bridge paths + **`cil_digest_topics`** (same shape as list / digest). | `path`, `metadata_relpath` (required) |
| GET | `/api/corpus/episodes/similar` | corpus | LanceDB semantic peers for an episode; **200** with `error` when index missing. | `path`, `metadata_relpath` (required), `top_k` (1–25) |
| GET | `/api/corpus/digest` | corpus | Feed-diverse **recent episodes** (metadata + GI/KG flags), optional **`cil_digest_topics`** per row (bridge + topic-cluster order; omitted when `compact=true`), and optional **semantic topic** bands. `compact=true` forces 24h, smaller cap, no topic bands and no CIL pill enrichment. | `path`, `window` (`24h` / `7d` / `1mo` / `since`), `since` (required if `window=since`; `1mo` = previous **calendar month**, UTC), `compact`, `include_topics`, `max_rows` |
| POST | `/api/corpus/topic-clusters/rebuild` | operator | **Operator-gated**: admin session or operator key whenever auth is configured (a creator is refused). (Re)build `search/topic_clusters.json` off the current index without a full reindex; reuses the per-corpus rebuild gate. **202** accepted / **409** a rebuild is already running / **503** LanceDB unavailable. Poll `GET /api/corpus/topic-clusters`. | `path`, `threshold` (default 0.75) |
| DELETE | `/api/corpus/runs/{run_id}` | operator | **Operator-gated** rollback (fails **closed** without a credential). Move a whole run's episodes to `<corpus>/.trash/<ts>/` (recoverable), re-aggregate `corpus_manifest.json` cost, then full `index_corpus(rebuild=True)` via the shared gate (topic-clusters recompute after). **202** with a summary + `/api/index/stats` poll pointer / **400** invalid run_id or missing `confirm` / **404** unknown run_id / **409** a rebuild is already running. | `path`, `confirm` (must equal `run_id`), `dry_run` |
| DELETE | `/api/corpus/episodes/{episode_id}` | operator | **Operator-gated** finer-grained rollback (for `--append` runs that share one `run_append_<hash>` dir): move just that episode's files (transcript/metadata/media/sidecars, all subdirs) to `.trash/`, re-aggregate, reindex. Same status codes as the run rollback. **Note:** `GET /api/corpus/episodes` (the consumer Library list) stays open — only the DELETE is operator-gated. | `path`, `confirm` (must equal `episode_id`), `dry_run` |
| GET | `/api/corpus/text-file` | corpus | Inline file under the corpus root for browser viewing (`.txt`, `.md`, `.vtt`, `.srt`, `.json`). Graph **Quote** node detail uses this for the in-app transcript viewer and the **Open raw transcript in new tab** header link. Pipeline runs that **direct-download** WebVTT/SubRip normalize to **`transcripts/… .txt`** (plus `*.segments.json` for GI timing); metadata usually points at that `.txt`. If `relpath` ends with `.txt` (and is not already `*.cleaned.txt`) and that file is missing, the server tries the sibling `stem.cleaned.txt` (metadata often still references the raw Whisper path). Post-#974, GI `transcript_ref` points at the **ad-free base** (`stem.adfree.txt`); when that is requested but absent (pre-#974 corpus), the server degrades to the raw `stem.txt`. | `path` (optional if server default set), `relpath` (required) |
| GET | `/api/corpus/stats` | corpus | **Publish-month** histogram (`YYYY-MM` → episode count) from one catalog scan; GI/KG Dashboard. | `path` |
| GET | `/api/corpus/documents/manifest` | corpus | Return `corpus_manifest.json` at corpus root (`null` when the corpus has none). | `path` |
| GET | `/api/corpus/documents/run-summary` | corpus | Return `corpus_run_summary.json` at corpus root (**404** if missing). | `path` |
| GET | `/api/corpus/runs/summary` | corpus | Discover `run.json` under the tree (capped), compact metrics per file for Dashboard. | `path` |
| GET | `/api/corpus/coverage` | corpus | GI/KG sibling-file presence per episode; aggregates by publish month and feed (Dashboard). | `path` |
| GET | `/api/app/auth/login` | app | **Consumer platform auth** (#1063): begin the single-provider OAuth code flow — **307** redirect to the provider with a signed CSRF state cookie. **503** when auth is unconfigured (no `APP_SESSION_SECRET` / OAuth creds). | — |
| GET | `/api/app/auth/callback` | app | **Consumer platform auth**: complete OAuth — verify state, exchange code, upsert the user (plain per-user files), set an HMAC-signed session cookie, **307** to `/`. **400** invalid state; **403** account not on the sign-in allowlist ([#1064](https://github.com/chipi/podcast_scraper/issues/1064)); **502** exchange failure. | `code`, `state` |
| POST | `/api/app/auth/logout` | app | **Consumer platform auth**: clear the session cookie (**204**). | — |
| GET | `/api/app/me` | app | **Consumer platform auth**: the signed-in user's `{user_id, email, name}`; **401** when not authenticated. Gate dependency `get_current_user` for per-user routes. | — |
| GET, PUT | `/api/app/playback/{slug}` | app | **Consumer platform** (auth, per-user files): playback position `{slug, position_seconds, updated_at?}`; GET returns 0 when unset ([#1065](https://github.com/chipi/podcast_scraper/issues/1065)). | — |
| GET, PUT | `/api/app/queue` | app | **Consumer platform** (auth): the user's play queue — ordered episode slugs `{items[]}` ([#1065](https://github.com/chipi/podcast_scraper/issues/1065)). | — |
| GET, POST, DELETE | `/api/app/library` (+ `/{feed_id}`) | app | **Consumer platform** (auth): subscribed podcasts — list / subscribe (idempotent on `feed_id`) / unsubscribe ([#1065](https://github.com/chipi/podcast_scraper/issues/1065)). | — |
| GET, PUT, DELETE | `/api/app/favorites` (+ `/{kind}/{ref}`) | app | **Consumer platform** (auth): saved items grouped by kind — `{episodes[], entities[{kind, ref, label, sublabel?, color?}]}` (`AppFavoritesResponse`). `PUT` saves `{kind: episode\|show\|topic\|person\|storyline\|theme, ref, …}` (idempotent on `kind`+`ref`); `DELETE /{kind}/{ref}` removes (`ref` URL-encoded). `GET` pages only when `limit` is sent (1.0.3): one filtered, sorted page plus `total` and per-kind `counts`; without it the response is unchanged. | `kind`, `q`, `color`, `sort`, `offset`, `limit` |
| GET, PUT | `/api/app/interests` | app | **Consumer platform** (auth): the user's interest token list `{items[]}` (`InterestsResponse`) — a **mixed token set** (`tc:` clusters / `topic:` / `person:`); `PUT` replaces it (`InterestsUpdate`). Drives flag-gated personalized discovery (`rank_discover`). [#1098](https://github.com/chipi/podcast_scraper/issues/1098). | — |
| POST, DELETE | `/api/app/interests/{token}` | app | **Consumer platform** (auth): follow / unfollow a single interest token (cluster `tc:` / topic `topic:` / person `person:`) — the entity-card `Follow` toggle. Idempotent; `DELETE` is a no-op when absent. Returns `{items[]}`. [#1097](https://github.com/chipi/podcast_scraper/issues/1097). | — |
| GET | `/api/app/admin/access-policy` | app-admin | Who may sign in — the persisted policy when there is one, else the startup env policy. | — |
| PUT | `/api/app/admin/access-policy` | app-admin | Replace the sign-in access policy. Takes effect on the NEXT sign-in — no redeploy. | — |
| GET | `/api/app/admin/release` | app-admin | One app's released version, served to its native update prompt (admin only). | `app` |
| PUT | `/api/app/admin/release` | app-admin | Set an app's released version at RUNTIME — no deploy, no restart (operator 2026-10-07). | — |
| GET | `/api/app/admin/users` | app-admin | List every platform user (admin-only). | — |
| POST | `/api/app/admin/users` | app-admin | Pre-provision a user with a role. 409 if that identity already exists, 422 on a bad role. | — |
| PATCH | `/api/app/admin/users/{user_id}` | app-admin | Change a user's role and/or active state (admin-only, self-lockout-guarded). | — |
| DELETE | `/api/app/admin/users/{user_id}` | app-admin | Hard-delete a user (admin-only). An admin cannot delete their own account. | — |
| GET | `/api/app/auth/dev-users` | app | Predefined dev identities for the sign-in picker — only when the MOCK provider is active. | — |
| POST | `/api/app/auth/email/code` | app | Sign in the store-review address with its code (see ``app_review_signin``). | — |
| POST | `/api/app/auth/email/request` | app | Send a sign-in link to ``email``. ALWAYS answers 202, whatever the address is. | — |
| GET | `/api/app/auth/email/verify` | app | Consume a sign-in link: create or sign in the account, then redirect. | `token`, `platform`, `return_to` |
| GET | `/api/app/auth/status` | app | Whether platform auth is *configured*, plus the signed-in user (if any) — never 401s. | — |
| GET | `/api/app/graph-events/session/{session_id}` | app | One session's ordered event timeline (admin only) — the step-by-step view + replay source. | — |
| GET | `/api/app/graph-events/sessions` | app | List analytics sessions (admin only), most-recent first — one row per session to inspect. | — |
| GET | `/api/app/graph-events/summary` | app | Aggregate graph analytics across all users (admin only) — usage / size / breakage. | — |
| GET | `/api/app/preferences` | app | Return the current user's preferences payload ({} when unset). | — |
| PUT | `/api/app/preferences` | app | Replace the ENTIRE preferences payload with the request body. | — |
| PATCH | `/api/app/preferences` | app | Shallow-merge the request body into the stored payload, then apply its item-level ``ops``. | — |
| POST | `/api/app/profile/avatar` | app | Store the signed-in user's uploaded avatar (overrides the OAuth one). 415/413/400 on a bad | — |
| POST | `/api/app/profile/name` | app | Set the signed-in user's display name. 400 when nothing usable remains after trimming, when | — |
| GET | `/api/corpus/entity-signals` | corpus_enrichments | Corpus enrichers pre-filtered to ONE graph node's entity (``?path=``-scoped viewer plane). | `kind`, `id`, `path` |
| GET | `/api/enrichment/config` | enrichment-config | Resolved enrichment config view: profile block + operator override + merged. | `path` |
| PUT | `/api/enrichment/config` | enrichment-config | Validate + persist the operator-side ``enrichment:`` block. | `path` |
| GET | `/api/enrichment/config/admission` | enrichment-config | Per-enricher accuracy-gate admission status (the UI leg of the cascade). | — |
| GET | `/api/enrichment/config/schema` | enrichment-config | Full JSON Schema for the ``enrichment:`` block (data-driven UI). | — |
| GET | `/api/enrichment/events` | enrichment | Return a slice of the JSONL event stream (most recent first). | `path`, `enricher_id`, `event_type`, `limit` |
| GET | `/api/enrichment/health` | enrichment | Per-enricher health: failure/circuit state PLUS whether the enricher is switched on. | `path`, `enricher_id` |
| POST | `/api/enrichment/health/{enricher_id}/re-enable` | enrichment | Operator manual recovery — same effect as ``podcast enrich --re-enable``. | `path` |
| GET | `/api/enrichment/metrics` | enrichment | Per-enricher metrics: the last run's summary, and a rollup over ``window``. | `path`, `enricher_id`, `window` |
| GET | `/api/enrichment/provider-types` | enrichment-config | Catalogue of registered provider types grouped by protocol. | — |
| GET | `/api/enrichment/run-summary` | enrichment | GET /api/enrichment/run-summary — last completed run summary. | `path` |
| GET | `/api/enrichment/stats` | enrichment | Corpus enrichment freshness — the enrichment mirror of ``GET /api/index/stats``. | `path` |
| GET | `/api/enrichment/status` | enrichment | Return the current live status (``{}`` when the file is absent / corrupt). | `path` |
| POST | `/api/jobs/enrichment` | enrichment | Enqueue a ``corpus_enrichment`` job (202 + optional queue position). | `path` |
| POST | `/api/jobs/resume` | jobs | Release the hold the stop endpoint set — a brake that cannot be released is a defect. | `path` |
| GET | `/api/jobs/running` | jobs | What is executing right now (#1785) — the view that used to need ``docker ps`` over SSH. | `path` |
| POST | `/api/jobs/stop` | jobs | The emergency brake (#1785): hold the queue, SIGTERM running work, verify, report. | `path`, `verify_seconds` |
| GET | `/api/ops/cache-stats` | ops | Per-namespace hit/miss/size + build-time-saved for the central in-process perf caches | — |
| GET | `/api/ops/corpus/integrity` | ops | GI integrity for every corpus-member episode — counts, defect lists, verdict. | — |
| GET | `/api/ops/corpus/preprocessing` | ops | Preprocessing damage assessment; the repair worklist is returned AS JSON, never written. | — |
| GET | `/api/ops/corpus/usage` | ops | Corpus disk footprint by top-level directory (bytes actually on disk under the root). | — |
| GET | `/api/ops/gateway/auth` | ops | Does the credential THIS CONTAINER actually holds authenticate against the gateway? | — |
| GET | `/api/ops/llm-gateway` | ops | Per-key spend / budget / burn for the prod LiteLLM gateway. | — |
| GET | `/api/ops/secrets/status` | ops | Presence/size/hash-prefix of every staged secret — VALUES NEVER CROSS THIS BOUNDARY. | — |
| GET | `/internal/outbox/pending` | internal | Pending envelopes for a channel — current-consent-filtered, non-expired, oldest-first. | `channel`, `limit` |
| POST | `/internal/outbox/{envelope_id}/status` | internal | Record a terminal status (idempotent per id); suppresses the channel on bounce/complaint. | — |
| GET | `/og/page/{route}/{ident}` | og | The og/twitter tags for one shareable document, for a link-preview bot. | — |
| GET | `/og/{kind}/{ident}.png` | og | Render ``(kind, ident)`` to a PNG card. 404 when the entity can't be resolved. | — |

Design and response field semantics: [Corpus Digest](../rfc/RFC-068-corpus-digest-api-viewer.md). Topic strings: repo `config/digest_topics.yaml`.

## Response models

Pydantic response schemas are defined in
[`schemas.py`][schemas-py]:

- `HealthResponse`
- `ArtifactListResponse` / `ArtifactItem`
- `IndexStatsEnvelope` / `IndexStatsBody` / `IndexRebuildAccepted`
- `CorpusSearchApiResponse` / `SearchHitModel` (optional **`lifted`** on transcript rows when lift applies; optional **`supporting_quotes`** on insight rows — quote speaker fields follow GI segment/diarization rules, issue **#541**)
- `CorpusResolveEpisodesRequest` / `CorpusResolveEpisodesResponse` / `CorpusResolvedEpisodeArtifact`
- `CorpusFeedsResponse` / `CorpusFeedItem`
- `CorpusEpisodesResponse` / `CorpusEpisodeListItem`
- `CorpusEpisodeDetailResponse`
- `CorpusSimilarEpisodesResponse` / `CorpusSimilarEpisodeItem`
- `CorpusDigestResponse` / `CorpusDigestRow` / `CorpusDigestTopicBand` / `CorpusDigestTopicHit`
  (digest rows: **`summary_bullet_graph_topic_ids`** parallel to **`summary_bullets_preview`**;
  topic bands: **`graph_topic_id`** — `topic:{slug}` from the band label for Graph focus, empty
  when no episode names that topic; **`rows_in_window`** — the episodes in the window, of which
  `rows` is a per-show sample)
- `CorpusStatsResponse` / `CorpusRunsSummaryResponse` / `CorpusRunSummaryItem`
- `RelatedNodeModel` / `RelationalListResponse` / `RelationalGroupedResponse` (relational-query layer, RFC-094 / #882)
- `QueryActivityResponse` / `QueryActivityBucket` (search-activity, PRD-033 FR6.2)

[schemas-py]: https://github.com/chipi/podcast_scraper/blob/main/src/podcast_scraper/server/schemas.py

## Additional shipped endpoints

Rich catalog with response models, auth, and params — extracted from
handler decorators and function signatures. Response-model class names
resolve against [`server/schemas.py`][schemas-py].

| Method | Path | Response model | Auth | Params | Purpose |
| --- | --- | --- | --- | --- | --- |
| GET | `/api/corpus/binary` | binary stream | open | `path`, `relpath` | Serve a corpus-relative binary artifact (audio-thumbnail bytes etc.) with the right Content-Type. |
| GET | `/api/corpus/enrichments` | envelope list | open | `path` | List available corpus-scope enricher envelopes (id + status metadata). |
| GET | `/api/corpus/enrichments/{enricher_id}` | envelope | open | path: `enricher_id`; query: `path` | Full envelope for one corpus-scope enricher. |
| GET | `/api/corpus/episode/enrichments/{enricher_id}` | envelope | open | path: `enricher_id`; query: `metadata_relpath`, `path` | Per-episode enricher output; identify the episode by its `.metadata.json` relative path. |
| GET | `/api/corpus/feed-signals` | `CorpusFeedSignalsResponse` | open | `path`, `feed_id`, `top_k`, `max_episodes` | Feed-level signals for one show (topic distribution, cadence). |
| GET | `/api/corpus/media` | binary stream | open | `path`, `relpath` | Locally-persisted audio bytes (RFC-096). |
| GET | `/api/index/timeseries` | `IndexTimeseriesResponse` | open | `path` | Historical index-rebuild + staleness stats over time. |
| GET | `/api/jobs/subprocess-log` | plain text | open | `job_id`, `path` | Full stdout+stderr of a pipeline job's subprocess. |
| GET | `/api/jobs/subprocess-log-tail` | `PipelineJobLogTailResponse` | open | `job_id`, `path`, `max_bytes` | Tail-slice + optional live polling of a running job's subprocess log. |
| GET | `/api/jobs/{job_id}/log` | plain text | open | path: `job_id`; query: `path` | Full log of the parent job (wraps subprocess log with job envelope). |
| GET | `/api/jobs/{job_id}/log-tail` | `PipelineJobLogTailResponse` | open | path: `job_id`; query: `path`, `max_bytes` | Tail-slice of the parent job log. |
| GET | `/api/operator-config/profiles` | `OperatorProfilesResponse` | open | — | List packaged `config/profiles/*.yaml` presets available for the `profile:` key in `viewer_operator.yaml`. |
| GET | `/api/ops/summary` | ops summary JSON | open | — | Ops dashboard summary — costs, model spend, provider health. |
| POST | `/api/ops/resilience/reset` | 200 OK | open | `scope` (query) | Force-reset a resilience breaker before its cooldown (ADR-113). |
| GET | `/api/resilience` | resilience-state JSON | open | — | Current resilience-layer state per provider (open breakers, blown fuses). |
| POST | `/api/search/compare` | `SearchCompareResponse` | open | JSON body: `SearchCompareRequest` | Two-subject briefing pack (Search v3 compare operator). |
| GET | `/api/usage` | usage rollup JSON | open | `window`, `provider` | Provider usage rollup — token / spend counters over the current window. |

**Relational sub-routes** (the `/api/relational/*` row above is the
catch-all; each is real and used): `/positions`, `/insights-about`,
`/topic-entities`, `/entities-in`, `/episodes`, `/related-insights`,
`/insight-detail`, `/episode-insights`, `/who-said`, `/cross-show`,
`/topics`, `/co-speakers`, `/related-topics`. All return a
`RelatedNode` projection (`RelationalListResponse` for flat lists,
`RelationalGroupedResponse` for grouped shapes); each takes `path` +
its scope key (`person` / `entity` / `insight` / `topic` / `podcast` /
`episode`) plus an optional `k` cap.
