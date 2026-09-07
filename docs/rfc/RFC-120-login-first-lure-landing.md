# RFC-120: Login-first access with a logged-out lure landing

- **Status**: Draft
- **Authors**: Marko, Claude (Opus 4.8)
- **Stakeholders**: Consumer App (learning-player), Server API, Infra/Edge (Caddy)
- **Related RFCs**:
  - `docs/rfc/RFC-108-operator-public-gated-surface.md` (reuses the player pre-launch gate pattern)
- **Related Documents**:
  - `infra/caddy/player.caddy` (the coming-soon gate; issue `#1262`)
  - `web/learning-player/docs/capacitor-build-runbook.md` (mobile build/ship)

## Abstract

- **What:** Move the learning-player from "fully anonymous-browsable app behind a coming-soon
  curtain" to a **login-first** model: a logged-out visitor sees a **lure landing** (a curated
  teaser of the corpus with all actions removed and a "create a free account" CTA); all real
  content and actions require a free account. Web and mobile behave identically.
- **Why:** Convert visitors into free accounts ("without an account, nothing is there"), and give
  the native TestFlight build a clean auth model that bakes **no shared secret** — each user's own
  login is the key.
- **Approach:** A dedicated logged-out landing view + login-first route guard (frontend), a curated
  anonymous **teaser** surface (backend/edge), and one additive edge rule so a logged-in client
  (native) reaches the API via its own Bearer with no baked cookie.

**Architecture Alignment:** Keeps the existing HMAC session model (`app_sessions`), the OAuth
bypass (`@oauth_public`), and the layered gate; it adds a login-first UX layer on top rather than
replacing auth.

## Problem Statement

Today the app is **not** login-first. `HomeView` renders for anonymous visitors and fetches
`/discover` + `/corpus/trending-topics` on mount; the router only gates `/queue`, `/library`,
`/profile`; and **dozens** of `/api/app/*` read endpoints serve content to anonymous users by
design (`get_optional_user`): `/episodes`, `/episodes/:slug*`, `/search`, `/discover`, `/trending`,
`/topics/:id`, `/persons/:id`, `/podcasts`, `/theme-clusters`, `/clusters`, etc. The only thing
hiding this pre-launch is the Caddy coming-soon curtain (`#1262`) — a **shared** secret
(`cl_preview` cookie / `marko`+`guest` basic-auth).

Two problems:

1. **Product:** we want a freemium funnel — a public *taste* that entices sign-up, with real
   content/actions behind a free account. The current "anonymous sees everything" defeats that.
2. **Mobile secret:** to ship a TestFlight build that works while pre-launch, the current design
   bakes the shared `cl_preview` cookie into the IPA (extractable; breaks on rotation). We want the
   native build to carry **no** shared secret — the tester's own login should be the key.

**Use Cases:**
- Invited beta user (web) → passes the invited-only gate → sees the lure landing → creates a free
  account → full app.
- TestFlight tester (mobile) → opens app → lure landing (or sign-in) → signs in with their own
  Google account → full app. Nothing secret shipped in the app.
- Public launch → drop the invited-only gate; the lure landing becomes the public front door;
  account signup flips from allowlist to open.

## Goals

1. Logged-out users (web + mobile) get a **lure landing**: curated teaser content, **no** actions
   (play/save/queue/follow), a prominent "create a free account" CTA.
2. All real content + actions require a free account — enforced consistently across web and mobile.
3. The native build bakes **no shared secret**; a logged-in client reaches the API with its own
   Bearer.
4. Preserve the **invited-only** posture during beta (the layered model below), and make public
   launch a small, well-defined flip.
5. Keep web's existing basic-auth doorman working (operator explicit requirement).

**Non-Goals:**
- Editorial hand-curation of the teaser (v1 reuses an existing recency/trending feed).
- Paid tiers / entitlements (this is free-account-only).
- Dropping the coming-soon gate now (that is the public-launch step, out of scope here).

## Constraints & Assumptions

- **Layered auth model (decided):** during beta, the Caddy invited-only gate stays as the outer
  layer; the lure landing + free-account content live behind it; the gate is dropped at public
  launch. (Operator decision, this session.)
- Account creation is backend-gated by `APP_SIGNUP_MODE` (default `allowlist`) + `PLAYER_ALLOWED_EMAILS`.
  Beta = allowlist; public launch = `open`.
- Native auth already works (RFC-108 / `#1310`): `ASWebAuthenticationSession` (iOS) / `@capacitor/browser`
  (Android) → signed token in `@capacitor/preferences` → `Authorization: Bearer`.
- Browsers never attach `Authorization` to top-level navigations, so an additive "Bearer passes"
  edge rule does **not** open the web coming-soon document — it only helps the native client.

## Design & Implementation

### 1. Frontend — lure landing + login-first (web + mobile)

- **New `LandingView`** (route `/` for logged-out; or a `/welcome` the guard redirects to). Shape
  DECIDED = **"marketing lure" (slim)**, not a mirror of `HomeView`:
  1. Value-prop hero ("Understand any podcast in minutes…") + primary **"Create your free account"** CTA + secondary "Sign in".
  2. ONE curated **"Featured this week"** rail — ~6–8 episode cards, **read-only** (no play/save/queue/follow).
  3. A row of topic chips ("Explore: #ai #markets …") — read-only.
  4. A short **"How it works"** 3-step strip.
  5. Repeat **"Create your free account"** CTA at the foot.
  Every card/interaction funnels to signup — clicking a card does NOT open the (auth-gated) detail.
  This keeps the anonymous surface to the small teaser set (§3) and optimizes first-impression/conversion.
- **Login-first guard:** flip the router so all content routes require auth (remove the per-route
  `requiresAuth` exceptions; keep `/login`, `/landing`, and static/legal public). Logged-out →
  `LandingView`. The existing `?redirect=<path>` post-login return is already wired.
- `HomeView` becomes authenticated-only (its logged-out branches are removed; the lure lives in
  `LandingView`).

### 2. Teaser surface (curated anonymous content)

- v1: reuse **`GET /discover`** (already anonymous, recency-ordered) and/or `GET /corpus/trending-topics`
  as the teaser feed, capped to a small N and stripped of anything requiring auth.
- The teaser is the **only** anonymous content surface. Its endpoint(s) are explicitly enumerated so
  the edge/backend can treat them as the public lure and everything else as account-only.

### 3. Enforcement depth (DECIDED: backend-enforce now)

Enforce login-first at **both** the frontend (router/UI) **and** the backend now. **Correction from
review (B2):** most content endpoints today have **no auth dependency at all** (not merely
`get_optional_user`) — e.g. every GET in `app_episodes.py` (`/episodes`, `/podcasts`,
`/podcasts/{id}/episodes|signals`, `/episodes/{slug}` + `/related|/insights|/stats|/entities|/segments|/audio-source|/search`),
`/clusters`, `/theme-clusters` (`app_discover.py`), `/entities/search`, `/topics/{id}/perspectives`
(`app_relational.py`). So the sweep is **"add an auth dependency to ~30 endpoints that never had
one,"** not "swap `get_optional_user` → `get_current_user`" (which is only ~12 call sites).

**This must be mechanical, not by-hand:** build an **anonymous-surface inventory** — every `@router`
GET/POST vs. its `Depends` — and an explicit **allow-list of anonymous survivors** checked in as the
single source of truth for both backend and Caddy. Everything not on the allow-list gets
`Depends(get_current_user)`.

**Anonymous survivors allow-list (the ONLY anonymous surface):**
- `GET /discover` — featured-episode taste (server-clamped, below).
- `GET /corpus/trending-topics` — topic chips (server-clamped).
- `GET /artwork` — teaser card images (else the landing renders broken images — B3). Path-validated
  via `safe_artwork_target`; accept that refs are enumerable from teaser payloads.
- `GET/POST /comms/unsubscribe` — token-based email unsubscribe (must stay anonymous — I6).
- `/auth/*`, `/.well-known/*`, MCP OAuth, health — already gate-exempt (`@oauth_public`).

**Server-side anonymous clamp (I1):** `/discover` (max `limit` 50) and `/corpus/trending-topics`
(max 100) must **force a small cap for anonymous callers in the endpoint** (e.g. anon → 8, ignore
the query param). The cap cannot live in the client — gate-exempt means the whole internet can call
these pre-launch. State explicitly: pre-launch we accept leaking ~8 recent episode
titles/slugs/artwork + top topic labels as the intended public lure.

**Telemetry (I4):** `POST /graph-events` and `POST /discover/click` are anonymous today. The landing
must **suppress** them when logged-out (not gate-exempt them — an anonymous world-writable endpoint
is an abuse surface).

Everything else requires auth, so a logged-out user clicking a teaser card is funnelled to signup.

### 4. Edge — no baked secret for mobile

Add one additive rule to `player.caddy`, above `@authed`. **Path-scoped to `/api/app/*` (B2)** so a
`Bearer` header does NOT bypass the coming-soon **document** curtain — only API calls (which is all
the native app needs; native prod loads a local bundle):

```
# A logged-in client (native app) reaches the API with its own Bearer — no shared
# cl_preview cookie. Path-scoped so the web document stays curtained; backend still
# verifies the token + PLAYER_ALLOWED_EMAILS.
@bearer {
  header Authorization Bearer*
  path /api/app/*
}
handle @bearer { reverse_proxy 127.0.0.1:8092 }
```

**Teaser exemption** uses exact paths (not a glob — a glob like `/api/app/discover*` would also
exempt `POST /discover/click`, M1):

```
@teaser path /api/app/discover /api/app/corpus/trending-topics /api/app/artwork
handle @teaser { reverse_proxy 127.0.0.1:8092 }
```

Order invariant (tested in the curl matrix): `@teaser` and `@bearer` sit **above** `@authed` (which
today 401-challenges a native `Bearer` via `basic_auth`).

Keep `@preview` / `@preview_ok` / `@authed` / `@oauth_public` / `@static` / coming-soon fallback
**unchanged** — web is untouched. The native build drops `VITE_PREVIEW_COOKIE` +
`VITE_PREVIEW_BASIC_AUTH` (no `initGateCookie`, no Basic fallback).

**Teaser is gate-exempt (DECIDED, D4).** The two teaser endpoints (§3) are added to the Caddy
bypass (alongside `@static` / `@oauth_public`), so the logged-out landing renders on **both** web
and mobile with no secret. Everything else stays gated (web) / auth-required (backend).

## Key Decisions

- **KD1 — Layered gate, not replaced.** Basic-auth invited-only stays for beta; login-first + lure
  sit behind it. Dropping the gate is the public-launch flip, not part of this RFC.
- **KD2 — Reuse `/discover` as the teaser (v1).** No new curated endpoint until there's a reason;
  avoids editorial infra now.
- **KD3 — No baked secret on mobile.** The native build ships zero gate credential; the tester's
  own Bearer is the key (edge `@bearer`).

## Alternatives Considered

- **Bake the shared `cl_preview` cookie in the IPA (status quo).** Rejected: extractable secret; a
  gate rotation bricks every installed build (documented failure mode).
- **Client-signal bypass for all native traffic (`Origin: capacitor://localhost`).** Softer curtain
  (spoofable); considered only for the teaser surface (D4), not the whole API.
- **Frontend-only enforcement (Caddy gate protects the API).** Rejected as the primary model — it's
  a lie the curtain hides, and it would not survive the public-launch gate drop. Backend enforcement
  is in scope now (D1).

- **Enforcement-completeness gate (the release gate):** a `curl` matrix asserting **every**
  `/api/app/*` route returns 401 when anonymous **except** the allow-list (§3). This is the
  invariant that must be green before the edge `@bearer` deploys.
- **Existing-test migration (I3 — the largest chunk):** ~11 integration files browse `/api/app`
  anonymously (`test_app_episodes.py`, `test_app_search.py`, `test_app_discover.py`,
  `test_app_catalog.py`, `test_app_routes_consumer.py`, `test_app_artwork.py`,
  `test_app_episode_stats.py`, `test_app_podcast_signals.py`, `test_filler_topics_never_reach_http.py`,
  `test_app_multiuser_concurrency.py`, `test_viewer_corpus_library.py`) plus Playwright stack-tests
  that browse without login — all migrate to an authed-client fixture.
- Frontend unit/e2e: logged-out → landing (no action controls); gated route → login; post-login
  `?redirect` returns to the intended path (I2); cold-start awaits session restore so a logged-in
  user never flashes the landing (M2); 401 interceptor routes stale-shell PWAs to the landing (M3).
- Edge `curl` matrix: anon web `/` → coming-soon; `Bearer` on `/` → **still** coming-soon (B2);
  `Bearer` on `/api/app/*` → app; teaser paths → 200 anonymous; `/comms/unsubscribe` → 200 anonymous.
- Mobile: device build with **no** baked cookie → sign in → content loads (validates `@bearer`);
  verify CORS covers anonymous teaser fetches from `capacitor://localhost` (M4).

## Rollout & Monitoring

**Corrected order (B1) — `@bearer` must NEVER reach prod while any content endpoint lacks an auth
dependency:**

1. RFC sign-off.
2. **One branch:** anonymous-surface inventory + backend enforcement (auth on ~30 endpoints) +
   server-side teaser clamp + frontend login-first + `LandingView` + existing-test migration.
   (Backend-first alone would 401 the current anonymous HomeView for invited beta users;
   frontend-first alone is unenforced — they ship together.)
3. **Release gate:** the enforcement-completeness curl matrix is green (every `/api/app/*` → 401
   anonymous except the allow-list).
4. **Edge (gated prod deploy — explicit approval):** teaser exemption + path-scoped `@bearer`; drop
   the baked cookie from the mobile build. Only after step 3 is green.
5. Device validation on the operator's phone → TestFlight (fastlane `beta`).
6. (Public launch, later) `APP_SIGNUP_MODE=open` + drop the coming-soon gate. Confirm no deployment
   shares `PODCAST_SERVE_OPERATOR_PUBLIC=1` (boot guard `_guard_operator_public_open_signup`,
   `app.py`) before the flip (I5).

Rollback: revert the `player.caddy` rule + redeploy; the frontend/backend changes are a normal app
revert. **Abuse (open signup → unlimited scrape accounts) is a Non-Goal here** — no rate-limiting in
this RFC (I5).

## Resolved Decisions

All resolved by the operator this session:

- **D1 — Enforcement depth:** **Backend-enforce now** (frontend + backend), not frontend-only.
  Content endpoints require auth except the teaser set; survives the public-launch gate drop.
- **D2 — Teaser source:** **reuse `/discover`** (recency) + `/corpus/trending-topics` for v1; no new
  curated endpoint yet.
- **D3 — Landing shape:** **dedicated `LandingView`** for logged-out (not an adapted `HomeView`).
- **D4 — Mobile teaser reachability:** **gate-exempt the teaser endpoints** at the edge, so the lure
  renders on web + mobile with no baked secret.
- **D5 — Account signup:** keep `APP_SIGNUP_MODE=allowlist` during beta (add tester emails), flip to
  `open` at public launch.

## References

- `infra/caddy/player.caddy` — coming-soon gate (`#1262`)
- `web/learning-player/src/router/index.ts` — route guard
- `web/learning-player/src/stores/auth.ts`, `src/services/api.ts`, `src/services/tier.ts`, `src/services/native.ts`
- `web/learning-player/src/views/HomeView.vue` — current adaptive home
- `server/app_discover.py`, `server/app_auth.py` — discover feed + session verification
