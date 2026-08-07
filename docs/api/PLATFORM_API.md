# Consumer Platform API (`/api/app`)

The end-user **Learning Platform** API — a slug-addressed, consumer-shaped surface that lets
a signed-in listener browse episodes, follow the synced transcript, play audio **streamed
from the original host**, search grounded passages, and keep a personal library / queue /
resume position. It is **separate from the operator API** (`/api/corpus/*`, `/api/relational/*`,
the GI/KG viewer) — different audience, its own auth, its own namespace.

- **Specs:** PRD-035–041, RFC-098 (foundation), RFC-100 (audio bridge), RFC-101 (corpus).
- **Mounted unconditionally** under `/api/app` by `create_app` (the legacy `enable_platform`
  flag is a no-op).
- **Decisions baked in:** shared corpus + per-user overlay · bridge-never-rehost · **no
  request-time LLM** (answers are extractive grounded retrieval) · per-user state as **plain
  files, no DB** · minimal OAuth multi-user.

---

## Auth & sessions

OAuth (single provider, Google to start) → a stateless **HMAC-signed session cookie**
(`lp_session`); no server-side session store. Per-user state lives under
`<APP_DATA_DIR>/users/<user_id>/` as plain JSON files.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/auth/login` | Begin OAuth — **307** to the provider with a signed CSRF `state` cookie. **503** when unconfigured. |
| GET | `/api/app/auth/callback?code=&state=` | Verify state, exchange code, upsert user, set session, **307** to `/`. **400** bad state · **403** not on the allowlist · **502** exchange failure. |
| POST | `/api/app/auth/logout` | Clear the session cookie (**204**). |
| GET | `/api/app/me` | `{user_id, email, name}` for the signed-in user; **401** otherwise. |

**`get_current_user`** is the FastAPI dependency gating every per-user route: it resolves the
signed cookie → `User`, rejecting missing/forged/expired cookies **and disabled users** with
**401**.

### Access control (allowlist)

Default **deny**: only allow-listed emails/domains may create an account.

| Env | Meaning |
| --- | --- |
| `APP_SIGNUP_MODE` | `allowlist` (default) or `open`. |
| `APP_ALLOWED_EMAILS` | Comma-separated emails (allowlist mode). |
| `APP_ALLOWED_DOMAINS` | Comma-separated domains (allowlist mode). |

> With `allowlist` mode and an empty list, **nobody** can sign in until you add emails/domains
> (or set `APP_SIGNUP_MODE=open`).

### Access boundary

**Read** routes (episode detail/segments/insights/entities, search) are currently **open**
(anonymous read — RFC-098 OQ1, pending a decision). **Per-user state** routes
(playback/queue/library) and `me` require a session.

---

## Catalog (episode lists)

Episode lists are served through a pluggable **`ContentSource`** (`#1078`). The MVP backend
(`LocalCorpusSource`) enumerates the already-processed local corpus, newest-first; a
`DiscoverySource` (`#1069`) can later implement the same contract with no API change.
Lightweight by design — per-artifact depth counts (`insight_count`, `speaker_count`) are read
lazily from the per-episode endpoints, not the list.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/episodes?page=&page_size=&status=&feed_id=` | Catalog across the corpus, newest-first. `{items[{slug, title, feed_id, podcast_title, publish_date, duration_seconds, episode_image_url, feed_image_url, artwork_url, status, summary_preview, summary_bullets[], topics[], has_transcript, has_summary, has_gi, has_kg, has_bridge}], page, page_size, total, has_more}`. `summary_preview` = short clean lede; `summary_bullets[]` = full summary (card expand-on-demand). `page≥1`, `1≤page_size≤100` (**422** otherwise). `status` ∈ `ready`\|`pending`. |
| GET | `/api/app/podcasts/{feed_id}/episodes?page=&page_size=&status=` | Same shape, scoped to one feed. |
| GET | `/api/app/podcasts` | Distinct shows in the corpus (Home "Your shows" + show-page header): `{items[{feed_id, title, artwork_url, image_url, description, episode_count}]}`. |

`status`: `ready` when a transcript exists (playable), else `pending`. Local-content MVP yields
`ready`; richer states (not-scraped/processing) arrive with scrape-on-demand (`#1069`).

---

## Episodes

Addressed by a stable, URL-safe **slug** (`{feed-slug}-{hash(feed_id,episode_id)}`), derived
deterministically and stable across re-scrapes.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/episodes/{slug}` | Detail: `{slug, title, feed_id, podcast_title, publish_date, duration_seconds, episode_image_url, feed_image_url, summary_title, summary_bullets, summary_text, has_transcript, has_summary, has_gi, has_kg, has_bridge}`. **404** unknown slug. |
| GET | `/api/app/episodes/{slug}/segments` | The frozen `segments.json` contract: `{version, episode_slug, segments[{id, start, end, text, speaker?}]}`. **404** when no transcript/segments. |
| GET | `/api/app/episodes/{slug}/insights` | Grounded GIL insights: `{episode_slug, insights[{id, text, grounded, insight_type?, confidence?, position_hint?, quotes[{text, speaker?, char_start?, char_end?, start_ms?, end_ms?}]}]}`. Empty list when no GI. |
| GET | `/api/app/episodes/{slug}/entities` | KG entities: `{episode_slug, persons[], orgs[], topics[]}`. Empty when no KG. |
| GET | `/api/app/episodes/{slug}/related?top_k=` | "More like this" — semantic peer episodes (vector similarity), as an `AppEpisodesResponse`. **200 + empty** when the index is unavailable (graceful). |
| GET | `/api/app/episodes/{slug}/stats` | **Public** (no auth) cross-user reach — anonymous aggregate counts only: `{slug, listeners, opens, insights, daily[{date, count}]}` (`EpisodeStatsResponse`). Distinct listeners + total opens come from scanning every user's listen log; `insights` is the grounded-insight count; `daily` is a 14-day opens sparkline. Zeroed when no `APP_DATA_DIR` is configured. |

---

## Search (extractive grounded retrieval — no request-time LLM)

Reuses the hybrid index (RFC-090); answers are real ranked passages, never generated prose.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/episodes/{slug}/search?q=&top_k=` | Episode-scoped: over-fetch by feed, narrow to this episode. |
| GET | `/api/app/search?q=&top_k=&grounded_only=&scope=` | Library-wide grounded search. `scope=all` (default) spans the shared corpus; **`scope=mine`** (P3 #1120, auth-gated — **401** signed out) is grounded **recall** over the user's heard∪captured set, with honest zero-coverage (empty, no global fallback). Each hit's `metadata` is enriched with `episode_slug` / `episode_title` / `podcast_title` / `episode_artwork` (thumb) so the client can jump to the episode + moment (`/episode/{slug}?t=`) and render results like library cards. |

Both return the standard search shape (`{query, results[{doc_id, score, metadata, text,
source_tier, supporting_quotes?, lifted?}], query_type, lift_stats?}`) and carry
`error:"no_index"` (HTTP 200, empty results) when no index is built.

---

## Artwork (serve our stored copy, never re-fetch from origin)

The counterpart to _bridge-never-rehost_ for audio: cover art is small and downloaded **once
at ingest** into the corpus-art store, so the app serves **our copy** and never re-fetches
graphics from the origin host. Two sizes, both derived from the local original (downscale
only): `large` (the original — ≥1400² at source, fits the player hero) and `thumb` (≤320px
for lists, generated on first request and cached). Content-addressed → served `immutable`,
so the browser + PWA service worker keep it on-device after one fetch.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/artwork?ref=&size=` | Serve stored art. `ref` = corpus-relative path under the corpus-art store (**400** otherwise); `size` ∈ `large`\|`thumb` (default `large`). **404** when the file is absent. `Cache-Control: immutable`. |

Episode summary/detail carry **`artwork_url`** (our local copy — `thumb` in lists, `large`
in detail) plus the remote `episode_image_url`/`feed_image_url` as **fallback only**. Clients
use `artwork_url` when present.

---

## Audio bridge (play from origin, never rehost)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/episodes/{slug}/audio-source?validate=` | `{episode_slug, url, mime?, duration_seconds?, media_id?, strategy:"direct", resolved_url?, verified?, content_length?}` from `content.media_url`. The client plays `url` directly. With `validate=true`, a HEAD follows redirects and reports `resolved_url`/`verified`/`content_length` (falls back to `verified:false` + the original URL on failure). **404** when no origin URL. |

> The server never stores or proxies third-party audio. A no-store pass-through proxy
> (for hosts that block direct play) is a documented, deferred follow-up.

---

## Per-user state (auth required; plain files, no DB)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/playback` | All saved positions, newest-updated first (Home "Continue"): `{items[{slug, position_seconds, updated_at?}]}`. |
| GET, PUT | `/api/app/playback/{slug}` | Resume position `{slug, position_seconds, updated_at?}`; GET returns 0 when unset. |
| GET, PUT | `/api/app/queue` | Play queue `{items: [slug, …]}`. |
| GET, POST, DELETE | `/api/app/library` (+ `/{feed_id}`) | Subscriptions — list / subscribe (idempotent on `feed_id`) / unsubscribe. |

### Favorites & interests

The favorites bucket is **polymorphic** (episodes + insights, grouped by kind). Interests are a
**mixed token set** — clusters (`tc:`), topics (`topic:`) and people (`person:`) — fed by two
entry-points: the Home cluster picker (writes `tc:` ids via `PUT`) and the `Follow` toggle on a
person/topic entity card (single-token `POST` / `DELETE`). They drive flag-gated personalized
discovery (`rank_discover`, which scores cluster + topic + person overlap; see PRD-043 / RFC-102).

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/favorites` | Saved items grouped by kind: `{episodes[{…EpisodeSummary}], insights[{ref, text, episode_slug?, podcast_title?, start_ms?}]}` (`AppFavoritesResponse`). Episodes are hydrated from the corpus; insights are a stored snapshot (no global detail route). |
| PUT | `/api/app/favorites` | Save an item (idempotent on `kind`+`ref`); body `{kind: episode\|insight\|person\|topic, ref, label?, sublabel?, slug?, start_ms?}` (`FavoriteAdd`). Returns the updated favorites. |
| DELETE | `/api/app/favorites/{kind}/{ref}` | Remove a saved item by `kind`+`ref` (`ref` URL-encoded; no-op if absent). Returns the updated favorites. |
| GET, PUT | `/api/app/interests` | The user's interest token list `{items: [token, …]}` (`InterestsResponse`); `PUT` replaces it `{items}` (`InterestsUpdate`). Tokens are a mixed set (`tc:` / `topic:` / `person:`). |
| POST | `/api/app/interests/{token}` | Follow one token (cluster `tc:` / topic `topic:` / person `person:`), idempotent; returns `{items[]}`. |
| DELETE | `/api/app/interests/{token}` | Unfollow one token (no-op if absent); returns `{items[]}`. |
| GET | `/api/app/clusters?limit=` | **Top interest clusters** for the picker, by corpus prevalence: `{items[{id, label, size}]}` (`AppInterestClustersResponse`). `1≤limit≤50` (default 12). |

### Listening analytics

Computed from per-user files (playback + an append-only listen-events log) — **no DB, no LLM**.
`/me/stats` is the signed-in user's own summary; `/episodes/{slug}/stats` (in **Episodes** above) is
the **public + anonymous** cross-user reach.

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/app/listen/{slug}` | Append one "episode opened" event to the user's listen log (`<data_dir>/users/<id>/listen_events.jsonl`) for analytics. **204**; best-effort, never blocks playback. |
| GET | `/api/app/me/stats` | The signed-in user's own listening summary: `{episodes, shows, listening_seconds, active_days, day_streak, daily[{date, count}]}` (`UserStatsResponse`). `daily` is a 14-day opens sparkline; `StatPoint` = `{date, count}`. |

---

## Knowledge cards (person / topic)

KG-grounded person/topic cards (PRD-043; RFC-102). `scope=mine` (P3 #1122, auth-gated — **401**
signed out) is the **"your corpus" lens**: the guest/topic restricted to the episodes the user has
heard∪captured (the _appears-in_ list + `episode_count` are filtered).

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/persons/{id}?scope=` | Person card `{id, label, episode_count, episodes[…Summary], related_people[], related_topics[]}` (`AppPersonCard`); **404** when the person is in no KG. |
| GET | `/api/app/topics/{id}?scope=` | Topic card `{id, label, cluster_*, sibling_topics[], episode_count, episodes[…], related_people[]}` (`AppTopicCard`); **404** when absent. |
| GET | `/api/app/topics/{id}/conversation-arc` | Topic conversation arc (ADR-108) — `{topic_id, weeks[{week, volume, negative, neutral, positive, avg_compound}]}` (`AppTopicConversationArcResponse`): ISO-week buckets of insight volume × VADER sentiment mix, oldest first. **200 + empty `weeks`** when the topic has no dated insights (never 404). Drives the consumer topic-card weekly-bar surface. |
| GET | `/api/app/entities/search?q=` | Resolve a query to a person/topic card (exact/near-exact); `{query, entity}` or `entity:null`. |

---

## Capture — highlights & notes (P2; PRD-040 / RFC-098 §7)

Per-user files (`highlights.json`, `notes.json`); all **auth-gated** (**401** signed out). The route
mints opaque ids + timestamps; the timestamp is the stable anchor (re-anchors on re-scrape, never
dropped).

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/highlights?episode=` | The user's highlights (`{items[Highlight]}`), optionally scoped to one episode slug. `Highlight` = `{id, episode_slug, kind(span\|moment\|insight), start_ms?, end_ms?, char_start?, char_end?, segment_ids[], quote_text?, speaker?, source_insight_id?, color?, created_at, anchor_status?}`. |
| POST | `/api/app/highlights` | Capture a highlight (**201**); body `HighlightCreate`. |
| PATCH | `/api/app/highlights/{id}` | Edit `color` / `quote_text` (`exclude_unset` — explicit `color:null` clears it); **404** if absent. |
| DELETE | `/api/app/highlights/{id}` | Remove; returns the remaining `{items[]}`. |
| GET, POST, PATCH, DELETE | `/api/app/notes` (+ `/{id}`) | Free-text notes targeting `highlight\|insight\|episode`. `GET ?target=&target_id=` scopes; `POST` (**201**) `{target, target_id, text}` (`text` min length 1 → **422**); `PATCH {text}`; `DELETE`. |
| GET | `/api/app/highlights/export.md` | Markdown export of all highlights + attached notes (grouped by episode; `text/markdown` attachment). |

---

## Collections — curated highlight sets (PRD-046 / RFC-111)

Named, ordered sets of the user's highlights (the curation surface). Per-user files; **auth-gated**.

| Method | Path | Description |
| --- | --- | --- |
| GET, POST | `/api/app/collections` | List (`CollectionsResponse`) / create (**201**, body `{name}` → `Collection`). |
| GET, DELETE | `/api/app/collections/{id}` | Detail with hydrated items (`CollectionDetail`) / delete (returns remaining). |
| POST | `/api/app/collections/{id}/items` | Add a highlight `{highlight_id}` (idempotent) → the updated `Collection`. |
| DELETE | `/api/app/collections/{id}/items/{highlight_id}` | Remove a highlight from the collection. |

---

## Delivery — digest consent + web push (PRD-046 / RFC-110)

The "Your Week" recap + push nudges. The **in-app** view is the primary surface and is _not_
consent-gated (a user's own data); the **email + push** _delivery_ is consent-gated — nothing is
delivered without an explicit opt-in, and email additionally needs a **verified** email. No
request-time LLM (D6). **Auth-gated** except the unsubscribe GET (one-click, token-bearing).

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/your-week` | The in-app **"Your Week"** view (`YourWeekResponse` `{sections[{kind, items[]}], period_label, generated_at}`, #1412) — the SAME rollup the email sends, served live and **decoupled from email consent** (visible in-app even with the digest email off; the `comms.digest.enabled` toggle governs only the outbound email edge). Items are enriched **in-app** with local artwork (`image_url`) + a backfilled `episode_title` for topic-centric items — route-local fields, **not** part of the `DeliveryEnvelope` contract. Empty `sections` when nothing is due yet. |
| GET, PUT | `/api/app/comms` | Delivery settings `{digest{enabled, cadence(weekly\|daily), day_of_week, hour, paused}, push{enabled}, email_verified, unsubscribe_ref}` (`CommsSettings`). `PUT` a whole section (server fills defaults — never send a partial). |
| GET | `/api/app/comms/unsubscribe?ref=` | One-click unsubscribe landing (HTML) — the `ref` is the opaque per-user token from the digest footer (RFC-110). |
| POST | `/api/app/comms/unsubscribe` | Confirm unsubscribe (turns the digest off). |
| GET | `/api/app/push/vapid-key` | The public VAPID key for the browser Push subscription (`VapidKeyResponse`). |
| POST, DELETE | `/api/app/push/subscribe` | Register / remove a Web Push subscription `{endpoint, keys{…}}` → `{count}` of active subscriptions (`PushSubscriptionsResponse`). |

> The delivery **outbox** is an internal, channel-agnostic seam (`/internal/outbox`, tailnet-only,
> `INTERNAL_OUTBOX_TOKEN`-gated) — the last-mile sender drains it; it is not part of the public
> consumer surface.

---

## Consolidation — recall, enrichment, resurfacing (P3; PRD-041 / RFC-101)

Read-time projections over the user's heard∪captured corpus + the RFC-088 enrichment envelopes — **no
request-time LLM (D6)**, **read-only over enrichments (ADR-104)**. The heard set = ≥30% played ∪ any
capture (RFC-101 §1).

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/episodes/{slug}/enrichment` | Per-episode enrichment signals `{slug, signals{<enricher_id>: data}}` for the viewed episode (RFC-088 envelopes; only OK enrichers). **404** unknown slug. |
| GET | `/api/app/corpus/enrichment` | Corpus-scope signals `{signals{<enricher_id>: data}}` (temporal velocity, topic similarity, …). |
| GET | `/api/app/resurfacing` | Highlights due to resurface, most-overdue first: `{items[{highlight, reflection_prompt}], paused}`. Read-time ladder (2d/1w/1mo/3mo on `created_at`/`last_surfaced`); empty when paused. **Auth-gated.** |
| POST | `/api/app/resurfacing/{id}/surfaced` | Record a resurfaced highlight as seen (advances its ladder). **204.** |
| GET, PUT | `/api/app/resurfacing/settings` | Pacing `{paused}` (`PUT` to pause/resume). |
| GET | `/api/app/interests/derived` | Implicit interests ranked by occurrence across the user's corpus: `{items[{token, kind, label, count}]}` — `person:`/`topic:` tokens, beside explicit follows. **Auth-gated.** |

> Recall itself is `GET /api/app/search?scope=mine` (see **Search**) — grounded retrieval over the
> heard set, not a separate endpoint.

---

## Personal corpus — faceted membership + revision log (RFC-114 / #1470)

The read-time definition of _what the user has engaged with_, split into two facets: **`experienced`**
= heard (≥30% played) ∪ captured (highlights, notes, saved-**insights**), and **`saved`** = whole-episode
favorites (save-for-later, deliberately **not** counted as experienced). A per-user **revision counter +
change log** (reconcile-on-read: recompute membership, diff, append add/remove events, bump the
revision) lets a consumer poll deltas incl. **tombstones**. No request-time LLM (D6). All **auth-gated**.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/corpus` | Summary `{revision, experienced_count, saved_count, top_entities[]}` (`CorpusSummary`). |
| GET | `/api/app/corpus/episodes?facet=` | Episode slugs in one facet `{facet, slugs[]}` (`CorpusFacetEpisodesResponse`). `facet` ∈ `experienced`\|`saved`. |
| GET | `/api/app/corpus/changes?since=` | Delta since a revision `{revision, since, truncated, events[{seq, kind: added\|removed, facet, ref}]}` (`CorpusChangesResponse`). `truncated=true` when `since` predates the retained window → the consumer must do a full re-read. `since=0` = a fresh consumer. |
| GET | `/api/app/corpus/ranked` | Experienced episodes ranked by **strength** `{items[{slug, strength}]}` (`CorpusRankedResponse`) — weighted heard-fraction + captures + favorited + relistens (RFC-114 Phase 2). |

> Distinct from `/api/app/corpus/enrichment` (RFC-088 corpus signals, under **Consolidation**) — same
> prefix, different surface. The change log is the episode-granular primitive; RFC-113's export keeps
> its own finer-grained content-hash snapshot instead (it must catch highlight edits, not just
> membership).

---

## Knowledge export — graph-aware Obsidian vault (RFC-113 / #1472)

Serializes the personal corpus as a **connected** Obsidian vault: each highlight becomes a note that
**wikilinks** to id-keyed `[[People/…]]` / `[[Topics/…]]` / `[[Episodes/…]]` under `closelistening/`.
Extractive, **bridge-only** (transcript quotes + `/player/{slug}?t=` deep links, never audio), **no
LLM** (D6). **Incremental**: a server-side content-hash vault snapshot + cursor — `since` matching the
last export returns only changed notes + a `removed` tombstone list; a mismatch (new device / behind)
returns a **full** export with `replace_namespace: true` (replace the whole folder). **Auth-gated.**

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/export?format=obsidian&since=` | Zip of the changed `closelistening/…` notes + `manifest.json` (`mode`, `revision`, `written[]`, `removed[]`, `replace_namespace`). Response headers `X-Export-Mode` (`full`\|`incremental`), `X-Export-Revision`, `X-Export-Written`, `X-Export-Removed` (all `Access-Control-Expose-Headers`) advance the client cursor without unzipping. `since=0` (or a mismatch) → full. **400** for a non-`obsidian` format. |

---

## MCP access — remote agents (bring-your-own-model; RFC-112 / #1471)

Lets an entitled user connect their **own** AI agent (claude.ai custom connector, Claude Code, Cursor)
to the platform's remote MCP server and search/read the shared corpus **as them** — no platform-side
LLM (D6). Two auth paths: **OAuth 2.1 + PKCE** (per-user connectors like claude.ai) and **personal-access
tokens** (CLI clients). Everything here is gated by the **`mcp_access`** entitlement (an orthogonal
boolean grant; **403** without it) on top of the session. The remote MCP server itself is a **separate
service on its own vhost** (`mcp.<domain>/mcp`) — see `docs/guides/PLAYER_PUBLIC_LAUNCH.md` §MCP.

**Token management + connector config** (`mcp_access`-gated):

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/mcp/config` | Connector wiring `{connector_url, authorization_server, oauth_enabled}` (`McpConnectionConfig`). `connector_url` = the MCP endpoint a client pastes (`https://mcp.<domain>/mcp`); null when unconfigured. |
| GET | `/api/app/mcp/tokens` | List the user's PATs (metadata only) `{items[{id, label, created_at, last_used_at}]}` (`McpTokensResponse`) — the secret is never returned after creation. |
| POST | `/api/app/mcp/tokens` | Mint a PAT `{token, meta}` (`McpTokenCreated`); body `{label}`. The `clp_mcp_…` plaintext is shown **once**, stored SHA-256-hashed. **201.** |
| DELETE | `/api/app/mcp/tokens/{id}` | Revoke a PAT; returns the remaining list. |

**OAuth 2.1 authorization server** (we host it — DCR + PKCE, public clients only). Discovery is public;
`/authorize` is session-gated (+ `mcp_access`); `/register` + `/token` are cookie-less server-to-server:

| Method | Path | Description |
| --- | --- | --- |
| GET | `/.well-known/oauth-authorization-server` | RFC 8414 metadata (root-mounted). **503** when `APP_MCP_ISSUER_URL` unset. |
| POST | `/api/app/mcp/oauth/register` | Dynamic Client Registration — a public client self-registers its `redirect_uris`. **201.** |
| GET | `/api/app/mcp/oauth/authorize` | Consent screen (discloses the redirect origin + a Deny) → single-use code; **silent 302** when consent is remembered. Session + `mcp_access` required (**401**/**403**). |
| POST | `/api/app/mcp/oauth/authorize` | Approve → remember consent → mint a code → 302 back with `code` + `state`. |
| POST | `/api/app/mcp/oauth/token` | Exchange an auth code (PKCE-S256) or refresh token (rotating) for `{access_token, refresh_token, token_type, expires_in, scope}`. `invalid_grant` **400** on failure. |

> The RFC 9728 protected-resource metadata (`/.well-known/oauth-protected-resource`) is served by the
> **MCP server** on `mcp.<domain>`, not the app — it points a cold client back to the authorization
> server here.

**Internal verify seam** (service-to-service, **tailnet-only**, NOT on the public edge):

| Method | Path | Description |
| --- | --- | --- |
| POST | `/internal/mcp/verify` | The `mcp` server resolves a presented bearer (PAT **or** OAuth access token) here → `{authenticated, user_id, mcp_access, scope}` (`McpVerifyResponse`). Gated by the shared `INTERNAL_MCP_TOKEN` (`X-Internal-Token`; **503** unconfigured, **401** mismatch). Re-checks `mcp_access` **live**, so a revoked grant denies an already-minted token at connect time. |

---

## Operator API — admin-only (separate surface)

The operator API backs the viewer's **admin-only** surfaces (Dashboard, Ops, Configuration). It
covers `/api/feeds`, `/api/operator-config`, `/api/ops`, `/api/jobs*`, `/api/scheduled-jobs`,
`/api/enrichment/config`, `/api/index/rebuild` — **reads and writes** (#1071, admin gating #1128):

- **Access rule:** a request is allowed with a valid **admin session** (the shared `lp_session`
  cookie → `role == admin`) **OR** a valid operator **key** (`X-Operator-Key` matching
  `APP_OPERATOR_API_KEY`). Otherwise **403**. Either credential grants access — browser admins use
  their session; headless automation can use the key.
- **Enforced only when it can be:** the gate activates when platform auth is configured (a session
  secret + per-user data dir) **or** a key is set. A bare deployment with neither keeps the prior
  network-only (Tailscale, RFC-082) behavior — no lockout on upgrade.
- Every mutating operator request is appended to `<APP_DATA_DIR>/audit.jsonl` (best-effort).
- Roles: `listener` (player only) < `creator` (viewer base) < `admin` (+ Dashboard/Ops/Config/users).
  Bootstrap admins via `APP_ADMIN_EMAILS`.

---

## Configuration (env)

| Env | Default | Purpose |
| --- | --- | --- |
| `APP_DATA_DIR` | `<corpus>/.app` | Per-user files + audit log (outside the shared corpus tree). |
| `APP_SESSION_SECRET` | _(unset → auth inert)_ | HMAC key for the session cookie. |
| `APP_SESSION_COOKIE_SECURE` | `false` | Set `true` behind HTTPS. |
| `APP_OAUTH_PROVIDER` | _(unset → Google)_ | Set `mock` to use the local network-free provider for **dev/e2e only** (never prod); logged loudly. |
| `APP_OAUTH_GOOGLE_CLIENT_ID` / `_SECRET` | _(unset → login 503)_ | Google OAuth app credentials. |
| `APP_OAUTH_MOCK_EMAIL` / `_SUBJECT` / `_NAME` | `dev@localhost` / `dev-local` / `Dev User` | Override the mock provider's dev identity (only when `APP_OAUTH_PROVIDER=mock`). |
| `APP_SIGNUP_MODE` / `APP_ALLOWED_EMAILS` / `APP_ALLOWED_DOMAINS` | `allowlist` / — / — | Access control. |
| `APP_ADMIN_EMAILS` | _(unset → no bootstrap admins)_ | CSV of emails granted `admin` on login. |
| `APP_OPERATOR_API_KEY` | _(unset)_ | Operator API key — an alternative to an admin session for the admin-only operator routes. |

### Remote MCP (RFC-112)

These configure the OAuth authorization server + the internal verify seam on the **app**, and the
separate **`mcp`** server. `INTERNAL_MCP_TOKEN` unset ⇒ the whole MCP surface is **inert** (verify
returns 503 → every agent connect 401), which is the safe default.

| Env | Read by | Default | Purpose |
| --- | --- | --- | --- |
| `INTERNAL_MCP_TOKEN` | app + mcp | _(unset → MCP inert)_ | Shared secret gating `/internal/mcp/verify` (the app checks it; the `mcp` server sends it as `X-Internal-Token`). Also the master on/off switch. |
| `APP_MCP_ISSUER_URL` | app + mcp | _(unset → OAuth 503)_ | The OAuth authorization-server origin (the player apex, e.g. `https://<domain>`). Builds the RFC 8414 metadata + endpoint URLs. |
| `APP_MCP_RESOURCE_URL` | app + mcp | _(unset)_ | The MCP server's public origin (e.g. `https://mcp.<domain>`). The connector URL a client pastes is this + `/mcp`; also the RFC 9728 `resource` identifier. |
| `APP_MCP_VERIFY_URL` | mcp | _(unset → auth fails closed)_ | Where the `mcp` server verifies bearers (compose-network URL of the app's seam, e.g. `http://api:8000/internal/mcp/verify`). |
| `APP_MCP_ALLOWED_ORIGINS` | mcp | _(unset → no browser Origin gating)_ | Comma-sep browser `Origin` allow-list (DNS-rebind guard). claude.ai connects server-side (no Origin) so empty is safe; set e.g. `https://claude.ai` to lock the browser surface. |
| `MCP_PORT` | mcp (compose) | `8009` | Loopback port the `mcp` container publishes for the Caddy `mcp.<domain>` vhost. |

---

## Tooling

- **Reference client** — `python -m podcast_scraper.server.app_reference_client --base-url
  <url> --session <lp_session cookie> --slug <slug>` walks the whole spine end-to-end (a
  contract proof; the product PWA is RFC-099).
- **Operator user admin** — `python -m podcast_scraper.server.app_users_cli
  {list,disable,enable,delete,export} --data-dir <APP_DATA_DIR>`.

## Not yet (deferred)

- **Consumer scrape-on-demand** (`POST /api/app/scrape`, #1069 **phase 2**) — deferred to the
  self-serve epic (Podcast Index `DiscoverySource` + guardrails), gated on real persistence + the
  PWA. Operator-side corpus growth already works via the pipeline itself
  (`--feeds-spec` / `--rss`, single-feed over #807), so curated growth is available today. See
  `docs/wip/player/1069-SCRAPE-ON-DEMAND-SCOPE-ANALYSIS.md`.
- **No-store audio proxy** (#1070) — until a host blocks direct play.
- **Consumer PWA** (RFC-099) — the actual front-end app, a separate workstream.

See the per-route detail in [HTTP_API.md](HTTP_API.md).

## Additional shipped endpoints

Rich catalog with response models, auth, and params. Response-model
class names resolve against `server/schemas.py`.

**Discovery + home feed**

| Method | Path | Response model | Auth | Params | Purpose |
| --- | --- | --- | --- | --- | --- |
| GET | `/api/app/discover` | `AppEpisodesResponse` | optional session | `limit` | Home discovery feed. When signed in AND `APP_PERSONALIZED_RANKING=true`, interest-ranked via the user's followed clusters; otherwise recency. |
| POST | `/api/app/discover/click` | 204 | optional session | JSON body: `slug`, `position` | Fire-and-forget click telemetry for ranking feedback. Silent no-op signed out or on network error. |
| GET | `/api/app/theme-clusters` | `AppStorylinesResponse` | open | `limit` | Home "Storylines" — theme clusters (topics discussed together). |
| GET | `/api/app/trending` | `AppTrendingResponse` | optional session | `kind`, `scope`, `limit` | RFC-103 momentum — trending entities of a given `kind`, corpus-wide or `scope=mine`. |
| GET | `/api/app/ranking-config` | ranking-config JSON | open | — | Discovery-ranking weights + toggles (admin surface; write-gated). |
| PUT | `/api/app/ranking-config` | ranking-config JSON | open | JSON body | Persist ranking-config changes. |

**User preferences (USERPREFS-1)**

| Method | Path | Response model | Auth | Params | Purpose |
| --- | --- | --- | --- | --- | --- |
| GET | `/api/app/preferences` | `UserPreferencesResponse` | session (401 signed-out) | — | Full USERPREFS-1 payload. |
| PUT | `/api/app/preferences` | `UserPreferencesResponse` | session (401 signed-out) | JSON body: full payload | Replace stored payload. |
| PATCH | `/api/app/preferences` | `UserPreferencesResponse` | session (401 signed-out) | JSON body: partial (null values delete a key) | Shallow-merge — preferred for single-key writes. |

**Podcast signals + topic perspectives**

| Method | Path | Response model | Auth | Params | Purpose |
| --- | --- | --- | --- | --- | --- |
| GET | `/api/app/podcasts/{feed_id}/signals` | `AppPodcastSignalsResponse` | open | path: `feed_id`; query: `top_k` | Show-level signals: topics, key people, recurring guests, dominant themes, trending topics. |
| GET | `/api/app/topics/{topic_id}/perspectives` | `AppTopicPerspectivesResponse` | optional session | path: `topic_id`; query: `scope` | Multi-perspective synthesis — each speaker's grounded insights on a topic (#1146). `scope=mine` restricts to the user's heard∪captured set. |

**Capture (deferred CRUD)**

| Method | Path | Response model | Auth | Params | Purpose |
| --- | --- | --- | --- | --- | --- |
| PATCH | `/api/app/notes/{note_id}` | `Note` | session (401 signed-out) | path: `note_id`; JSON body: text | Edit note text. |
| DELETE | `/api/app/notes/{note_id}` | `NotesResponse` | session (401 signed-out) | path: `note_id` | Remove a note; returns the remaining list. |

**User state**

| Method | Path | Response model | Auth | Params | Purpose |
| --- | --- | --- | --- | --- | --- |
| DELETE | `/api/app/library/{feed_id}` | `LibraryResponse` | session (401 signed-out) | path: `feed_id` | Remove a show from the user's Library. |

**Auth surface (probe endpoints)**

| Method | Path | Response model | Auth | Params | Purpose |
| --- | --- | --- | --- | --- | --- |
| GET | `/api/app/auth/dev-users` | dev-users JSON | open | — | Mock identities for the sign-in picker; populated only when the `MOCK` provider is configured. |
| GET | `/api/app/auth/status` | auth-status JSON | open | — | Which provider is active + whether signup is open. Used by the login page. |

**Graph-event telemetry**

| Method | Path | Response model | Auth | Params | Purpose |
| --- | --- | --- | --- | --- | --- |
| POST | `/api/app/graph-events` | 204 | optional session | JSON body: event array | Client-side ingest — graph interaction events. |
| GET | `/api/app/graph-events/summary` | summary JSON | open | — | Operator rollup (aggregate counters by event type). |
| GET | `/api/app/graph-events/sessions` | sessions JSON | open | — | List distinct browsing sessions. |
| GET | `/api/app/graph-events/session/{session_id}` | session-detail JSON | open | path: `session_id` | Full event stream for one session. |

**Admin users** (admin-only; surface parity with `app_users_cli`)

| Method | Path | Response model | Auth | Params | Purpose |
| --- | --- | --- | --- | --- | --- |
| GET | `/api/app/admin/users` | `list[UserOut]` | admin | — | List all users. |
| POST | `/api/app/admin/users` | `UserOut` (201) | admin | JSON body | Create a user. |
| PATCH | `/api/app/admin/users/{user_id}` | `UserOut` | admin | path: `user_id`; JSON body | Update user attributes (role, enabled flag, etc.). |
| DELETE | `/api/app/admin/users/{user_id}` | 204 | admin | path: `user_id` | Delete a user. |
