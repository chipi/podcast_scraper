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
| GET | `/api/app/episodes?page=&page_size=&status=&feed_id=` | Catalog across the corpus, newest-first. `{items[{slug, title, feed_id, podcast_title, publish_date, duration_seconds, episode_image_url, feed_image_url, status, summary_preview, topics[], has_transcript, has_summary, has_gi, has_kg, has_bridge}], page, page_size, total, has_more}`. `page≥1`, `1≤page_size≤100` (**422** otherwise). `status` ∈ `ready`\|`pending`. |
| GET | `/api/app/podcasts/{feed_id}/episodes?page=&page_size=&status=` | Same shape, scoped to one feed. |

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

---

## Search (extractive grounded retrieval — no request-time LLM)

Reuses the hybrid index (RFC-090); answers are real ranked passages, never generated prose.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/app/episodes/{slug}/search?q=&top_k=` | Episode-scoped: over-fetch by feed, narrow to this episode. |
| GET | `/api/app/search?q=&top_k=&grounded_only=` | Library-wide (whole shared corpus for now; scoped to the user's library once that lands). |

Both return the standard search shape (`{query, results[{doc_id, score, metadata, text,
source_tier, supporting_quotes?, lifted?}], query_type, lift_stats?}`) and carry
`error:"no_index"` (HTTP 200, empty results) when no index is built.

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
| GET, PUT | `/api/app/playback/{slug}` | Resume position `{slug, position_seconds, updated_at?}`; GET returns 0 when unset. |
| GET, PUT | `/api/app/queue` | Play queue `{items: [slug, …]}`. |
| GET, POST, DELETE | `/api/app/library` (+ `/{feed_id}`) | Subscriptions — list / subscribe (idempotent on `feed_id`) / unsubscribe. |

---

## Operator API hardening (separate surface)

The operator write routes (`PUT /api/feeds`, `PUT /api/operator-config`, `POST /api/jobs*`,
`POST /api/index/rebuild`) gain optional **API-key auth** + an **audit trail** (#1071):

- `APP_OPERATOR_API_KEY` set → those routes require a matching `X-Operator-Key` header (else
  **401**); unset → key check skipped (Tailscale-only model, backward-compatible).
- Every mutating operator request is appended to `<APP_DATA_DIR>/audit.jsonl` (best-effort).

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
| `APP_OPERATOR_API_KEY` | _(unset → no key check)_ | Operator write-path API key. |

---

## Tooling

- **Reference client** — `python -m podcast_scraper.server.app_reference_client --base-url
  <url> --session <lp_session cookie> --slug <slug>` walks the whole spine end-to-end (a
  contract proof; the product PWA is RFC-099).
- **Operator user admin** — `python -m podcast_scraper.server.app_users_cli
  {list,disable,enable,delete,export} --data-dir <APP_DATA_DIR>`.

## Not yet (deferred)

- **Scrape-on-demand** (`POST /api/app/scrape`, #1069) — blocked on a pipeline enhancement to
  target a specific feed/episode (the pipeline currently runs the whole corpus).
- **No-store audio proxy** (#1070) — until a host blocks direct play.
- **Consumer PWA** (RFC-099) — the actual front-end app, a separate workstream.

See the per-route detail in [HTTP_API.md](HTTP_API.md).
