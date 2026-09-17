# Learning player — E2E surface map

This document is the **Playwright automation contract** for the consumer learning player
(`web/learning-player`) — the sibling of the operator viewer's
[E2E_SURFACE_MAP.md](../../gi-kg-viewer/e2e/E2E_SURFACE_MAP.md). It lists surfaces, entry paths,
owning specs, and the selectors / roles / labels tests rely on. Contributors and agents also use it
when **debugging** the app or driving it via tools that consume the **accessibility tree**
(Playwright, Playwright MCP, Chrome DevTools MCP snapshots): it records expected roles, labels, and
`data-testid`s, not only test selectors.

It complements — does not replace — the design docs:
[PRD-042](../../../docs/prd/PRD-042-home.md) (Home / Learning Hub),
[PRD-043](../../../docs/prd/PRD-043-knowledge-layer.md) (knowledge layer + personalization),
[PRD-041](../../../docs/prd/PRD-041-consolidation.md) (consolidation),
[UXS-011](../../../docs/uxs/UXS-011-consumer-learning-app.md) (shell / IA),
[UXS-012](../../../docs/uxs/UXS-012-consumer-home.md) (Home),
[UXS-013](../../../docs/uxs/UXS-013-knowledge-clusters.md) (clusters / storylines),
[UXS-014](../../../docs/uxs/UXS-014-interaction-patterns.md) (card / modal interaction patterns).

**Key distinction from the operator viewer.** The operator specs are still mostly **route-mocked**
(`page.route(**/api/**)`) — 33 of 38, which is drift from the intended architecture, not the target
(see [#1619](https://github.com/chipi/podcast_scraper/issues/1619)). The player specs run against
the **real API** over the **committed validation corpus**
(`tests/fixtures/app-validation-corpus/v3`) — the Playwright `webServer` boots a real backend on
`:8011` and the built app on `:4174`. So a player spec exercises the actual server surface (search,
discover ranking, capture, consolidation), and fixtures live in the corpus, not in per-spec route
handlers.

## Where the data comes from — read this before changing any fixture

Nothing in this suite is invented per-spec. Everything has a home, and all of it lives in the
**Python half of the repo**, which is why it is easy to miss from in here:

| What | Where | Notes |
| --- | --- | --- |
| Corpus (episodes, transcripts, GI/KG, search index) | [`tests/fixtures/app-validation-corpus/v3`](../../../tests/fixtures/app-validation-corpus/README.md) | Committed, deterministic, schema-current. The API boots against this. |
| **Episode audio** | `tests/fixtures/audio/<FIXTURES_VERSION>/` — currently **`v3`** | One `.mp3` per episode id, covering every episode in the corpus. **Check [`tests/fixtures/FIXTURES_VERSION`](../../../tests/fixtures/FIXTURES_VERSION) first** — the folder is versioned. |
| RSS / transcript fixtures | `tests/fixtures/{rss,transcripts}/` | Same versioning rule for `transcripts/`. |
| Mock podcast host (host loopback) | [`make serve-e2e-mock`](../../../docs/guides/E2E_TESTING_GUIDE.md) → `127.0.0.1:18765` | Serves `/audio/<episode_id>.mp3`, RSS and transcripts. Simulates a real podcast host. |
| Mock podcast host (compose network) | [`docker/mock-feeds/`](../../../docker/mock-feeds/README.md) | Nginx sidecar, same fixtures, for `make stack-test-*`. |
| The whole picture | [`docs/guides/E2E_TESTING_GUIDE.md`](../../../docs/guides/E2E_TESTING_GUIDE.md), [`tests/fixtures/README.md`](../../../tests/fixtures/README.md) | Start here when something looks absent. |

> **If you conclude a fixture "doesn't exist", you are probably in the wrong tree.** Every asset
> this app plays or renders already exists somewhere above. Incident 2026-08-13: an agent searched
> only `tests/fixtures/app-validation-corpus/v3`, found no `.mp3`, concluded the repo had no audio,
> and hand-built an MP3 encoder to synthesise some. `tests/fixtures/audio/v3/` had real audio for
> all 36 episodes, one directory up. Read `tests/fixtures/README.md` — its title is the answer.

### Audio: real files, no interception

`content.media_url` is a **relative** `/audio/<episode_id>.mp3` — the same convention the RSS
fixtures use — and the app's `/audio` proxy forwards it to the mock podcast host, which serves the
real fixture audio from `tests/fixtures/audio/<FIXTURES_VERSION>/`. Playwright starts that host as a
`webServer` alongside the API.

It used to be an undecodable data URI with a route stub (`routeLoadableAudio`) substituting a
synthetic WAV, so every transport assertion tested the stub rather than the player (#1618). Both are
gone. `fixture-audio.spec.ts` asserts the corpus audio decodes in a real browser, so this cannot
regress quietly.

## Setup invariants

Rules the suite depends on that are **not** visible from any single spec:

| Invariant | Where | Why it matters |
| --------- | ----- | -------------- |
| `globalSetup` wipes `e2e/.app-state` | [globalSetup.ts](globalSetup.ts) | `signInIsolated` ids are **stable** per (spec, project) and the state dir is gitignored but persists between local runs. A leftover `resurfacing_settings.paused = true` breaks later honest-empty assertions. Rebuilt empty-state specs pass in CI and flake locally without this. |
| `globalSetup` builds the LanceDB index | [globalSetup.ts](globalSetup.ts) | The index is gitignored; several routes branch on `has_index`. Absent, index-dependent specs (search, perspectives) silently assert against a different result set. |
| `openTranscript` clicks only if visible | [helpers.ts](helpers.ts) | The transcript is a toggle on mobile and an always-visible column on desktop. The helper makes one spec pass under **both** Playwright projects; a spec that just clicks fails on desktop. |

> **This map is a living contract.** When you add a surface, rename a `data-testid`, or change an
> entry path, update the matching row **in the same PR**. See the [coverage gaps](#coverage-gaps)
> section for surfaces that currently have **no owning spec**.

## Runtime

| Item | Value |
| ---- | ----- |
| Config | [playwright.config.ts](../playwright.config.ts) |
| `baseURL` | `http://127.0.0.1:4174` (built app via `vite preview --strictPort`) |
| Projects | `mobile-chrome` (Pixel 7) + `desktop-chrome` (Desktop Chrome) — phone-first primary target (UXS-011) |
| Backend | Real API on `:8011` over `tests/fixtures/app-validation-corpus/v3` (Playwright `webServer`, **no mocks**) |
| Specs | `e2e/*.spec.ts`, shared [helpers.ts](helpers.ts), fixtures under [validation/](validation/) |
| Sign-in | [`signInIsolated(page, who, testInfo)`](helpers.ts) — dev-auth a fresh isolated user per test |

## App shell + routes

Header brand (→ **home**) + `<nav>` of [NavIconLink](../src/components/NavIconLink.vue): **Browse**
(catalog), **Library**, and a profile link when signed in; **Sign in** / **Sign up** links when
signed out.

> **The profile link's accessible name is dynamic**: `auth.user?.name || t('profile.title')`
> ([App.vue](../src/App.vue)), i.e. the signed-in user's name, falling back to **"Your profile"** —
> never the literal string "Profile". Match on the user name your spec signed in as, or on
> "Your profile".

| Route | Name | View | Auth | Notes |
| ----- | ---- | ---- | ---- | ----- |
| `/welcome` | `landing` | [LandingView](../src/views/LandingView.vue) | **public** | **Logged-out lure landing (RFC-120)** — hero + "Create your free account" CTA, read-only Featured teaser (4 distinct shows) + topic chips + how-it-works; every card/chip funnels to signup with `?redirect` threaded. The signed-out entry to the app (UXS-012 "Access model"). Testids: `landing-cta-primary`, `landing-cta-signin`, `landing-featured`, `landing-card`, `landing-chip` (+ `landing-cta-foot`). |
| `/login` | `login` | [LoginView](../src/views/LoginView.vue) | **public** | Dev sign-in |
| `/` | `home` | [HomeView](../src/views/HomeView.vue) | auth | Learning Hub — adaptive hero, discovery. **Authed-only under login-first (RFC-120)** — signed-out visitors get `/welcome`, not this |
| `/catalog` | `catalog` | [CatalogView](../src/views/CatalogView.vue) | auth | "Browse" — episode catalog |
| `/search` | `search` | [SearchView](../src/views/SearchView.vue) | auth | Corpus semantic search + KnowledgePanel |
| `/podcast/:feedId` | `podcast` | [PodcastView](../src/views/PodcastView.vue) | auth | Show page → its episodes |
| `/episode/:slug` | `player` | [PlayerView](../src/views/PlayerView.vue) | auth | Transcript + playback + capture. Zone D (the live insight panel over the artwork) has two states: `data-testid="player-zone-d-live"` while an insight is surfacing, `data-testid="player-zone-d-rest"` between them. Zone D is deliberately **not** an ARIA live region — it restates what the listener is already hearing, and PlayerView owns exactly one live region (`src/__checks__/live-regions.test.ts`) |
| `/queue` | `queue` | [QueueView](../src/views/QueueView.vue) | auth | Play queue + reorder |
| `/library` | `library` | [LibraryView](../src/views/LibraryView.vue) | auth | Saved (episodes/insights) + highlights |
| `/profile` | `profile` | [ProfileView](../src/views/ProfileView.vue) | auth | Stats + interests entry |
| `/topic/:id` | `topic` | [TopicView](../src/views/TopicView.vue) | auth | Standalone topic page (#1261-6) — `data-testid="topic-view"` |
| `/person/:id` | `person` | [PersonView](../src/views/PersonView.vue) | auth | Standalone person page (#1261-6) — `data-testid="person-view"` |
| `/storyline/:id` | `storyline` | [StorylineView](../src/views/StorylineView.vue) | auth | Storyline page (F4.5) — theme cluster derived from the anchor topic id; `data-testid="storyline-view"`. Replaced the old bottom-sheet |
| `/browse` | `browse` | [BrowseView](../src/views/BrowseView.vue) | auth | Discover hub — `data-testid="browse-view"`. A trending-shows tile row on top (`trending-shows-rail`, "See all →" `trending-shows-seeall` → the Shows tab pre-sorted by Trending), then a **"Trends"** heading over the shared `DiscoveryExplorer` (`data-testid="discovery-explorer"`; same tabbed `DiscoveryList` as Home, capped at 10 rows, with a per-kind "See all →" `discovery-see-all` → `/trends?tab={kind}`), then a content band with two tabs (`data-testid="browse-tab-{tab}"`, where `{tab}` is `episodes` or `shows`). Each panel is kept mounted (`data-testid="browse-panel-episodes"`, `data-testid="browse-panel-shows"`). (operator 2026-09-14, replaced the top-3 `DiscoveryDashboard`.) |
| `/trends` | `trends` | [TrendsView](../src/views/TrendsView.vue) | auth | Full entity-trends page — `data-testid="trends-view"`, back link `data-testid="trends-back"`. Three kind tabs (`data-testid="trends-tab-{kind}"`, where `{kind}` is `topic`, `storyline`, or `person`), a Rising⇄Trending sort toggle (`data-testid="trends-sort"`) and a Corpus⇄Mine scope toggle (`data-testid="trends-scope"`); each tab renders `DiscoveryList`. Reached from the Discover explorer's per-kind "See all →" link (`discovery-see-all`). |
| `/settings` | `settings` | [SettingsView](../src/views/SettingsView.vue) | auth | Settings/About (#8) — version/build/platform, help, Config (offline-mode/clear-cache/clear-downloads), About & legal links, `data-testid="settings-view"` |
| `/about/:page` | `about-page` | [AboutPageView](../src/views/AboutPageView.vue) | auth | Placeholder About/legal pages — third-party / privacy / terms; `data-testid="about-page"`, empty content for now |
| `/browse/shows` | `browse-shows` | — (redirect) | auth | **Redirects to `/browse?tab=shows`** (#2004). Rendered as the hub's tab panel via [ShowBrowseView](../src/views/ShowBrowseView.vue) `embedded`, `data-testid="show-browse-view"`. It used to render standalone — no tab strip, own heading, its own back-to-Home — so the same content had two presentations depending on how you arrived |
| `/browse/topics` | `browse-topics` | — (redirect) | auth | **Redirects to `/browse?tab=topics`** (#2004). Rendered as the hub's tab panel via [TopicBrowseView](../src/views/TopicBrowseView.vue) `embedded`, `data-testid="topic-browse-view"`. It used to render standalone — no tab strip, own heading, its own back-to-Home — so the same content had two presentations depending on how you arrived |
| `/browse/people` | `browse-people` | — (redirect) | auth | **Redirects to `/browse?tab=people`** (#2004). Rendered as the hub's tab panel via [PersonBrowseView](../src/views/PersonBrowseView.vue) `embedded`, `data-testid="person-browse-view"`. It used to render standalone — no tab strip, own heading, its own back-to-Home — so the same content had two presentations depending on how you arrived |
| `/:pathMatch(.*)*` | — | → `home` | — | Catch-all redirect (a signed-out visitor then bounces to `/welcome`) |

**Login-first (RFC-120 #2009):** the guard denies by default — only `landing` (`/welcome`) and
`login` are reachable signed-out; **every other route** redirects a signed-out visitor to the
**landing** with `?redirect=<fullPath>` ([router/index.ts](../src/router/index.ts)). The `auth`
column reflects this; the per-route `meta.requiresAuth` flags are legacy no-ops now.

## Surfaces and owning specs

| Surface | Intent (short) | Typical entry | Spec files |
| ------- | -------------- | ------------- | ---------- |
| **App shell / nav** | Header brand → home; `<nav>` NavIconLink **Browse** / **Library** / **Profile**; **Sign in** / **Sign up** when signed out | Every page | `smoke.spec.ts` (+ implicit in all) |
| **Home** | Adaptive hero — signed-in with in-progress history: **"Continue listening"**; otherwise kicker **"Ask across every episode"** + title **"Find any moment you've heard."**; search bar (`#home-search`); **topic chips** under the search field (`home-topic-chips` / `home-topic-chip`, up to **2** from `getTrendingTopics()`, tapping one runs that search — absent, not stubbed, when the corpus has no velocity data); dismissible **set-your-interests** card → picker; **What's new** (a featured `01` hero + a numbered "chart" of rows `02–06` — rank numeral + small square artwork + title/show, the per-row `EpisodeActions` cluster dropped so the title has room on a phone; restored operator 2026-09-14); a tabbed **discovery switcher** (`discovery-tab-topic`/`storyline`/`person`, labels **Topics**/**Storylines**/**People**) with a Rising⇄Trending sort switch + a Corpus⇄Mine scope switch; **Recommended**. ("Your shows" was removed from Home operator 2026-09-14 — the shows you follow already live in Library › Following.) | `goto('/')` | `home-search.spec.ts`, `smoke.spec.ts`, `full-listen.spec.ts` (entry) |
| **Discovery (#4, one list)** | Home folds discovery into ONE tabbed switcher (`home-discovery`) over three entity KINDS — `discovery-tab-topic`/`storyline`/`person` (Topics default). Two little switches drive it: a Rising⇄Trending **sort** (`discovery-sort`; Rising = velocity, Trending = volume) and a Corpus⇄Mine **scope** (`home-trending-scope`), plus a 1M/3M/6M/1Y window — all on one control row. Every kind renders the SAME `DiscoveryList` row (label + optional subtitle + trend-hued sparkline + trailing metric that follows the sort + follow), from `GET /api/app/trending` (EWMA velocity + volume, anchored to `APP_TRENDING_NOW`). Topics/people open the entity card; storylines open the storyline overlay. Replaced MomentumRail / TrendingTopics / Storylines. | Home, below What's new | `trending.spec.ts` |
| **EntityCard (person/topic)** | Overlay (from Search/Home) or inline (from Insights) card: **Follow**, **Your corpus** scope (all/mine), cluster identity (**Theme** + **Similar**), theme members, **Follow storyline**, **Perspectives**, **Signals**, related people/topics; re-entrant back stack. **The open overlay carries a `?card=kind:id` query (#1594)** so it has a history entry: browser/Android-hardware Back pops it and CLOSES the card instead of navigating the page underneath, and closing by Escape/backdrop pops that entry so the next Back is not absorbed by stale state. Entity ids are already kind-namespaced, so the query is the id itself. Nothing reads the query back yet — it is a history marker, not a deep link. | Trending/Storyline chip, Search entity hit, KnowledgePanel | `perspectives.spec.ts` (Perspectives), `entity-signals.spec.ts` (Signals), `home-rails.spec.ts` (Back + Escape history behaviour) |
| **Interests picker** | Modal: **Topics** (semantic `tc:`) + **Storylines** (`thc:`) sections; Save replaces only the offered subset (preserves `topic:`/`person:` follows) | Home interests card **or** Profile → **Choose interests** | ⚠️ **UI: none** — `recommendation.spec.ts` drives `/api/app/interests` directly |
| **Catalog (Browse)** | Episode catalog / browse-all | `goto('/catalog')` (nav **Browse**, Home **Browse all →**) | ⚠️ **none dedicated** |
| **Search** | Corpus semantic search; passage hits + **KnowledgePanel** (entity chips → card); entity-in-search resolution | `goto('/search?q=…')`, Home search submit | `home-search.spec.ts`, `consolidation.spec.ts` (`?q=index`) |
| **Player (episode)** | Transcript (paragraph-grouped; **opt-in on mobile** via the controls-panel `transcript-toggle`, always-visible side column on desktop), floating/sticky controls on mobile, **capture** ([CaptureMoment](../src/components/CaptureMoment.vue) — `data-testid="capture-moment"` idle/failed, `data-testid="capture-receipt"` for the 4s followable receipt linking to Library → Saved; rendered TWICE at complementary breakpoints — masthead on `lg:`, sticky transport corner below it — so exactly one is on screen at any width), summary region, insight **density** strip, **reach chip** (withheld below the k-anonymity floor — see #1957). Manual sync controls are currently hidden. | `goto('/episode/:slug')`, via Podcast/Library/Queue/Home rows | `transcript.spec.ts`, `transcript-toggle.spec.ts`, `transcript-paragraphs.spec.ts`, `full-listen.spec.ts`, `capture.spec.ts`, `entity-signals.spec.ts`, `player-reach.spec.ts` |
| **Podcast (show)** | Show page → episode list, **show signals band** (`podcast-signals` + `ps-theme` / `ps-distinctive-topic` / `ps-topic` / `ps-person` rows — the corpus-wide "Trending here"/`ps-trending` row was removed, its velocity being corpus-wide not show-specific), publishing-cadence chart (`show-activity`), the **feed by-line** (`podcast-byline` — RSS author/host names, #2043) + **feed meta** (`podcast-feed-meta` — language badge + last-updated), and the **Follow show** toggle (`follow-show`, `aria-pressed`) — a *feed subscription*, distinct from interest follows | `goto('/podcast/:feedId')` (e.g. `p05`) | `follow-show.spec.ts`; also reached by `auth-queue`, `capture`, `consolidation`, `perspectives`, `entity-signals`, `transcript*` |
| **Follow show (feed subscription)** | `POST`/`DELETE /api/app/library` — optimistic toggle, reverts on failure. Feeds Your Week's "new in your follows". **Not** the same store as interest tokens (`topic:`/`person:`/`thc:`) | Show page header, signed-in only | `follow-show.spec.ts` |
| **Your Week** | In-app personal digest (`your-week`, expand via `yourweek-toggle`) — self-hides when every section is empty. "New in your follows" needs ≥1 followed show with unheard graph-carrying episodes | Home, when due | `your-week.spec.ts`, `follow-show.spec.ts` |
| **Topic / Person pages (#1261-6)** | Standalone routable entity pages (`topic-view`, `person-view`) — the non-modal counterpart to EntityCard. Reached by tapping a trending topic chip (`trend-spark-row` → `router.push` to `/topic/:id`; it is a button, not an anchor). | `goto('/topic/:id')`, `goto('/person/:id')`, Browse hub topic chip | `browse-and-topic-pages.spec.ts` |
| **Browse hub (#14)** | The three corpus indexes folded into ONE tabbed hub (`browse-view`, `browse-tab-{tab}` — `episodes` default), each panel the standalone index in `embedded` mode (`v-show`-mounted so tab switches never refetch); `?tab=` deep-links a tab and re-syncs live (kept-alive `watch`). The standalone `/browse/topics` · `/browse/people` routes still resolve but are now reached via the hub. Home's compact `home-browse-nav` "Discover" strip (`home-discover-topics` · `home-discover-storylines` · `home-discover-people`) deep-links into the `/trends` see-all page (`?tab=topic`/`storyline`/`person`), not the hub (operator 2026-09-14). | `goto('/browse?tab=…')`, Home `home-browse-nav` | `browse-and-topic-pages.spec.ts` |
| **Trend-window selector (RFC-103 R2)** | A segmented control on the Topics/People trending sections (and Home "Rising now") that picks the window the velocity is measured over — `trend-window-tabs`, `trend-window-{window}` (`1m`/`3m`/`6m`/`1y`, **3m** default). Changing it refetches `GET /api/app/trending?window=…`. The velocity is monthly, corpus-latest-month-anchored, floored at `min_total`, ranked by velocity × volume. | Browse Topics/People, Home Rising | `browse-and-topic-pages.spec.ts` |
| **Search listener features** | Also-about chips (`related-topic-chips`), matched-fields kicker (`matched-fields`), save-query (`save-query-button` → `saved-searches-section` in Library), more-like-this rail (`related-episodes-rail`), search-scope switch (`search-scope`, a single compact toggle **button** with `aria-pressed`, labelled "Search scope"; tapping flips all↔mine and re-runs the query into one results region + reflects `?scope=` in the URL — player review 2026-09-13, was a radiogroup) | `/search?q=…` | `search-listener-features.spec.ts`, `consolidation.spec.ts` |
| **Knowledge Panel dialog (S9)** | The learning panel is a native `<dialog>` — `knowledge-panel`. **Mobile: modal** (`showModal()`) — focus trapped, background inert, Escape closes, focus returns to `player-open-insights`. **Desktop (≥1024px): non-modal** (`show()`) — a docked rail beside the player, deliberately NOT trapping focus, because nothing is covered and the transcript must stay keyboard-reachable. Mode follows the viewport live, so rotating a phone re-modes it. | player route | `knowledge-panel-a11y.spec.ts` |
| **Mini-player (#1587)** | Persistent transport bar — `mini-player`, artwork+title link `mini-player-open`, play/pause `mini-player-toggle`, queue `mini-player-queue` (opens the Queue panel). Visible whenever an episode is loaded and you are NOT on that episode's player page. Sits above the bottom nav on mobile. | any route, once playback starts | `audio-continuity.spec.ts`, `queue-panel.spec.ts` |
| **Queue & Recent panel (#1838)** | Bottom-sheet modal (`queue-panel`, close `queue-panel-close`) opened FROM the transport — `player-queue` on the full player (next to the speed pill) and `mini-player-queue` on the mini-player. Two sections: **Up next** (the play queue, reusing QueueView) and **Recently played** (`queue-panel-recent` — tapping a row resumes, does not re-queue). Moved here from the old Library Queue/Recent tabs. | Full player or mini-player queue button | `queue-panel.spec.ts` |
| **Audio continuity (#1587)** | The `<audio>` element is owned by the **player store**, appended to `<body>`, and outlives every view — so client-side navigation no longer stops playback. `document.querySelector('audio')` still works (`app-audio`). A full `page.goto` reload DOES stop it; that is a page load, not navigation. | — | `audio-continuity.spec.ts` |
| **Bottom nav (#1594)** | Mobile tab bar — `bottom-nav`, tabs `bottom-nav-{home\|search\|library\|profile}`. `sm:hidden`, so specs that must work on BOTH projects should click the header nav instead. | mobile viewports | covered incidentally; no dedicated spec |
| **Mobile invariants (#1312)** | Sticky transport stays pinned (`player-controls-sticky`), MediaSession metadata + playbackState, dark-canvas no-white-flash | `mobile-chrome` project | `mobile-invariants.spec.ts` |
| **Trending shows rail (RFC-103)** | Cover-art carousel with cadence sparkline (`trending-shows-rail`, `trending-show-card`) → show page. Also mounted at the TOP of Discover (above the entity dashboard, operator 2026-09-14) with a `see-all` header (`trending-shows-seeall`) that drops into the Shows tab. | Home, below Momentum; Discover top | ⚠️ **none** — see [gaps](#coverage-gaps) |
| **Queue** | Play queue; reorder via `↑`/`↓` chevrons; QueueButton add/remove; the ITEM routes (`POST/DELETE /queue/items`) vs the whole-list PUT that only `move` still uses; the cached-queue notice and its disabled reorder arrows | `goto('/queue')` (auth) | `auth-queue.spec.ts`, `queue-reorder.spec.ts`, `queue-offline-surface.spec.ts` |
| **Library** | Tabs: **Saved** (per-kind **Episodes** / **Insights** + folded Highlights section), **Following**, **Collections** (RFC-119, its own first-class tab). Queue + Recent moved OUT to the player-surface Queue panel (#1838); Collections is no longer nested under Saved. | `goto('/library')` (auth) | `library-saved.spec.ts`, `capture.spec.ts`, `consolidation.spec.ts` |
| **Collections (RFC-119 / #1839)** | Typed pinboards — mixed items `{kind: highlight\|episode\|show\|search\|topic\|person\|link}`; create/open/delete, add external link (URL-only), **Play-all** queues the collection's episodes. Empty state "No collections yet". | Library **Collections** tab | ⚠️ **none dedicated e2e** — unit (`CollectionsView.test.ts`) + integration (`test_app_collections_routes.py`); empty state touched by `library-saved.spec.ts` |
| **Profile** | User stats; **interests** section → picker; resurfacing settings | `goto('/profile')` (auth) | ⚠️ **none dedicated** |
| **Login** | Dev sign-in — user list + custom subject | `goto('/login')`, auth-guard redirect | `auth-queue.spec.ts` + every authed spec (via `signInIsolated`) |
| **Listening recap** | Profile's "Your listening" panel (window toggle, day bars, the coverage line that says how much of the window was recorded, what kept coming up, the saved line) and the Home prompt that points at it. The fabricated `Hours` tile is asserted ABSENT. | `goto('/profile')`, `goto('/')` (auth) | `recap-and-deep-links.spec.ts` |
| **Deep links** | `?t=<seconds>` opens an episode AT that moment, overriding the remembered resume for that load; a malformed `t` still opens the episode | `goto('/episode/<slug>?t=42')` | `recap-and-deep-links.spec.ts` |
| **PWA / offline** | Service-worker registration, manifest + icons, `__buildInfo`; offline behaviour of Library/Queue (audio is **not** SW-cached; per-user API is **not** cached) | `goto('/')` then offline | `pwa.spec.ts`, `offline.spec.ts` — **the update toast itself is NOT covered**, see [gaps](#coverage-gaps) |
| **Capture / consolidation** | Mark-moment capture → highlights; consolidation suggestions (derived interests) | Player mark-moment; Library | `capture.spec.ts`, `consolidation.spec.ts`, `full-listen.spec.ts` |
| **Tab strips (#1594 item 7)** | [Tabs](../src/components/Tabs.vue) — the only tab strip in the app, replacing seven hand-written ones. Two patterns, chosen by what the control does: `tabs` (`role="tablist"`, `aria-selected`, `aria-controls` → panel) where it switches between distinct panels — Library (`library-tab-*`), Browse (`browse-tab-*`), Home discovery (`discovery-tab-*`); and `radio` (`role="radiogroup"`, `aria-checked`, no `aria-controls`) where it re-parameterises ONE region — trend window (`trend-window-*`) and Your Week layout. (Search scope became a standalone `aria-pressed` toggle button, and the entity-card corpus scope was removed — neither is a Tabs radiogroup any more.) Both carry a roving tabindex and arrow-key movement with selection following focus. Panel ids come from `panelAttrs()` so the tab↔panel pair cannot drift. | Library, Browse, Home, Search, entity card, Profile | `Tabs.test.ts` (22), `tabs-single-implementation.test.ts` (guard against an eighth); surface specs exercise the strips they own |
| **Collapsible spine (Knowledge Panel)** | [CollapsibleSection](../src/components/CollapsibleSection.vue) — native `<details>`, so keyboard, disclosure role and expanded-state announcement come from the element. Wraps Key points, Topics & People, Insights and More-like-this (`kp-section-{key}`, toggle `kp-section-toggle-{key}`). The SUMMARY is deliberately not collapsible: it is why the panel was opened and it is one paragraph. Sections are OPEN by default and remember the user's choice in `lp.kp.<key>` — per USER, not per episode, since "don't show me related" is a preference about the panel. The count rides in the header (`Insights · 8`) so a folded section still says what it holds; storage failure falls back to OPEN, never to hidden. | Player → Insights panel | `CollapsibleSection.test.ts` (default open, count in header, persistence, per-section keys, storage-failure fallback, native element) |
| **Episode tile (rails)** | [EpisodeTile](../src/components/EpisodeTile.vue) — an episode in a HORIZONTAL rail: artwork on top at full slot width, then show kicker + a full-width title clamped to three lines, and the shared `EpisodeActions` row pinned to the tile BOTTOM (`mt-auto`). Used by the player's "More like this" (`related-episodes-rail`, 176px slots). Distinct from `EpisodeCard`, the row for vertical lists (Podcast, Queue, the Queue panel's recently-played). The action set is the SAME shared row everywhere — favourite + queue inline + a `⋯` overflow carrying download + add-to-collection (count must not change by view), a uniform three top-level controls across web and native; no summary at this width (a truncated summary is just the shape of a summary). | Player → below the transcript | `EpisodeTile.test.ts` (stacking order, clamped full-width title, actions at the bottom and never absolute, shared set, shape without artwork) |
| **Stale notice (Home, #1909)** | [StaleNotice](../src/components/StaleNotice.vue) — one line at the top of Home when any rail is showing what it had last time rather than fresh data (`stale-notice`), with the retry those rails no longer carry (`stale-retry`). #1909 scoped hydrate-then-revalidate for the Home rails and it never landed, so with no network Home was five identical "Couldn't load this right now" cards — the app saying one thing five times, in the place the content should have been. Rails now keep their content and report `stale`; this states the page-level fact once, above them. It MUST carry the retry: a stale section renders no error card and so offers none of its own, and trading a wall of noise for a page with no way to refresh would be quieter and worse. Deliberately muted — no icon, no alarm colour — over content that is perfectly readable. Retry remounts the four rails that own their own fetches (`:key="railKey"`) and reloads the sections HomeView owns. | Home → offline / server unreachable | `useSectionState.test.ts` (hydrate, stale-not-error, race, staleness tally); NOT covered end-to-end — the browser tier has no offline-rail spec |
| **Insight type marks (#2004 item 8)** | [InsightTypeMark](../src/components/InsightTypeMark.vue) — the mark that tells one insight from another in the Knowledge Panel list, which can hold 36 of them. Four SVG shapes at a fixed size (diamond / ring / triangle / square) so all four weigh the same, plus a neutral dot for a type outside the closed vocabulary. Colour rides on the MARK only, from `--lp-insight-*`, which alias `--lp-topic` / `--lp-grounded` / `--lp-warning` / `--lp-person` so every visual direction adapts them for free; the label stays mono + muted (`.lp-kicker`, #2013) and none of it spends `--lp-accent`. Shape is the primary channel and colour the second — the set is separable in greyscale. The old green "grounded" dot that preceded it is REMOVED: it rendered on the same condition as the row's `▶ mm:ss` button, so it distinguished nothing and diluted the mark beside it. `data-testid="insight-type"` | Player → Insights panel | `KnowledgePanel.test.ts` (shape distinct, colour distinct, no accent, mark is first in the row) |
| **Destructive confirms (#1594)** | [ConfirmDialog](../src/components/ConfirmDialog.vue) — a native `<dialog>` opened with `showModal()`, so the browser supplies the focus trap, Escape and an inert background. Focus lands on `data-testid="confirm-cancel"`, never on `data-testid="confirm-accept"`, so muscle memory cannot delete. Fronts the three deletes that destroy authored content and cannot be undone (the create endpoints mint new ids): delete a collection (`data-testid="collection-delete"` → `data-testid="collection-delete-confirm"`), delete a highlight (`data-testid="highlight-delete"` → `data-testid="highlight-delete-confirm"`), delete a note (`data-testid="note-delete"` → `data-testid="note-delete-confirm"`). **Removing an item from a collection is deliberately NOT confirmed** — the item survives, only a membership row goes, and friction there is what teaches people to click through the ones that matter. | Library → Collections; Library → Saved → highlights | `collections.spec.ts` (confirm + Escape in a real browser); `ConfirmDialog.test.ts`, `CollectionsView.test.ts`, `HighlightsView.test.ts` (wiring) |
| **44px touch targets (#1594)** | The minimum-target rule, enforced as measured geometry rather than reviewed by eye. Card rows expose `data-testid="episode-card"`; the action controls inside (Favourite / Queue / Download / Add-to-collection) keep a 32px ring and carry the `.lp-tap` class, which grows the HIT area to 44px via a centred pseudo-element — four real 44px circles would be 176px of a 375px phone. The row gap is `gap-[12px]` — a literal px, deliberately not the rem-scale spacing class: the rem scale renders 11.4px at this app's root size, which left the targets overlapping by 0.6px. Saved-item colour swatches could not use `.lp-tap` — a 24px pitch has no room — so the button itself is 44px with the coloured dot as an inner span: the SET picker is `SavedColorControl` (`data-testid="saved-swatch"`, opened from `saved-color`), the colour FILTER is `SavedFilterBar` (`data-testid="saved-filter-swatch"`). | `goto('/browse?tab=episodes')`; `goto('/library?tab=saved')` with ≥1 highlight | `design-invariants.spec.ts`; class-level guard in `src/__checks__/touch-affordances.test.ts` |
| **Discovery ranking** | Personalized `/api/app/discover` responds to followed-interest levers (PRD-043 #1098) | API-level (`PUT /api/app/interests`) | `recommendation.spec.ts` |
| **Design invariants** (cross-cutting, not a surface) | The rules the redesign is made of, enforced against the RENDERED page: the accent is spent only on things a finger can act on — in text, background, border, outline or SVG paint, alpha included; every kicker carries the instrument voice (mono, never accent). Swept over Home, Browse, Library, Profile, Catalog, Search **and the Player** (reached by navigation, since its route needs a slug). Both vacuity holes are guarded: each surface asserts it rendered real content, and a sweep asserts the accent is reachable at all. Stands in for pixel baselines, which macOS-vs-Linux rasterisation makes unmaintainable here — layout/spacing drift is deliberately NOT covered (#1946) | `goto('/')`, `/browse`, `/library`, `/profile`, `/catalog`, `/search`; Player via `/podcast/p05` → first episode | `design-invariants.spec.ts` |
| **Collections** (RFC-119) | Pin any typed item into a collection, and read them back. The control is `data-testid="add-to-collection"` — on browse rows (`EpisodeCard`, `kind: episode`), the show page (`PodcastView`, `kind: show`), search results, entity cards, **and the player masthead** (added #2013 follow-up: you could pin from a list or pin a whole show, but not the episode you were listening to). It opens `data-testid="add-to-collection-menu"` holding `data-testid="add-to-collection-pick"` rows and an inline create; failures render `data-testid="collection-error"`. Library → Collections lists `data-testid="collection-open"` rows, an opened one renders `data-testid="collection-items"`, and a failed load is `data-testid="collections-load-error"` (NOT the empty state — that distinction is the bug this surface shipped with) | `goto('/browse')` → row control; `goto('/library?tab=collections')` | `collections.spec.ts` |
| **Visual direction switch** (cross-cutting, not a surface) | `?direction=<name>` repaints the app from `theme/directions.css` and persists for the session; `?direction=` (empty) turns it back off. Asserted through the real boot path — that `main.ts` runs the resolver, that the attribute is the one the stylesheet keys on, and that the choice survives a FULL page load. Checks the resolved `--lp-canvas`, not just the attribute, since a typo'd name sets an attribute and changes no colour. Signed-out, on `/login`, because the switch is applied before the app decides anything about a user | `goto('/login?direction=ember')`, `/login`, `/login?direction=` | `theme-direction.spec.ts` |
| **Zone D at the smallest phone** | The live insight panel must not clip or leave the viewport at **375 x 667** (iPhone SE/mini — the design harness runs at Pixel 7's 412px, 9% wider, and nothing tested the narrow end). Sweeps the whole episode timeline and asserts the invariant for every insight the panel actually renders, rather than pinning a timestamp. Vacuity-guarded: fails if the panel never appears, or if the longest insight it showed is under 150 chars. Corpus measurement behind the number: 124 Insight nodes, 108 renderable in Zone D, longest renderable 200 chars | `goto('/podcast/p06')` → "More Drift, Less Signal"; playhead driven via the `<audio>` element | `zone-d-small-viewport.spec.ts` |

## Coverage gaps

The 2026-08-12 audit listed thirteen surfaces that rendered with **no owning Playwright spec**. All
thirteen were closed on 2026-09-03 — see the table below for where each now lives. What remains is
listed after it, with the reason it is not automatable rather than merely undone.

| Surface | Now covered by |
| ------- | -------------- |
| **Discovery list** — `DiscoveryList`, one shared list for topics / storylines / people (rows `data-testid="discovery-row"` + `data-testid="discovery-follow"`, hint `data-testid="discovery-hint"`, expand `data-testid="discovery-expand"`); ranked by the sort switch `data-testid="discovery-sort"` (Rising = velocity / Trending = volume). Replaced the old MomentumRail / TrendingTopics / Storylines rails. | `home-rails.spec.ts`, `trending.spec.ts` |
| **Your Week** (`your-week`) | `home-rails.spec.ts`, `your-week.spec.ts` |
| **Catalog / Browse** — `CatalogView`, `ShowBrowseView` (`browse-view`, `browse-tab-*`, `show-browse-grid`, `show-browse-search`, `show-browse-sort`) | `browse-and-profile.spec.ts` |
| **Profile** — `ProfileView` (`profile-settings-link`, `profile-edit-interests`) | `browse-and-profile.spec.ts`, `recap-and-deep-links.spec.ts` |
| **Interests picker (UI)** | `browse-and-profile.spec.ts` — skips cleanly when the corpus offers no entry point |
| **Podcast signals band** — `PodcastSignalsBand` (`podcast-signals`, `ps-distinctive-heading`, `ps-distinctive-topic`, `ps-topics-heading`, `ps-theme`, `ps-topic`, `ps-person`) | `knowledge-bands.spec.ts` |
| **Show activity chart** (`show-activity`, `show-activity-bar-*`) | `knowledge-bands.spec.ts` |
| **Insight density** (`player-insight-density`, `player-density-*`) | `knowledge-bands.spec.ts` |
| **Knowledge panel** (`knowledge-panel`, `kp-*`) | `knowledge-bands.spec.ts` |
| **EntityCard storyline link** (`ec-storyline-link`) — the topic card's single "Part of a storyline" link; opens the storyline overlay (`storyline-card`) ON TOP, where Follow-storyline (`storyline-follow`) now lives | `entity-and-rails-invariants.spec.ts` |
| **Topic conversation arc** (`topic-conversation-arc`, `tca-bar-*`) | `entity-and-rails-invariants.spec.ts` |
| **Trending shows rail** (`trending-shows-rail`, `trending-show-card`) | `entity-and-rails-invariants.spec.ts` (invariant — see below) |
| **Storyline page** — `StorylineView` (`storyline-view`, `storyline-follow`, route `storyline`). Reached by opening a storyline from the Discover explorer (`/browse`, `discovery-row`) or from the Trends page (`/trends?tab=storyline`, `discovery-row`). | `storyline.spec.ts` |
| **Episode action row** — `EpisodeActions` (`episode-actions`) | `episode-actions.spec.ts` |
| **Overflow menu** — `OverflowMenu` (`overflow-trigger`, `overflow-menu`, `mark-played`) | `overflow-menu.spec.ts` |
| **Note composer** — `NoteComposer` (`note-composer`, `note-input`, `note-save`, `note-item`, `note-delete`) | `note-composer.spec.ts` |
| **Notes kind filter** — Library → Boards → "Your notes" (`collections-notes`, `collections-note`, `collections-notes-empty`) topped by `TypeFilterBar` (`notes-type-filter`, `notes-type-all`, and one chip per kind — `notes-type-show`, `notes-type-storyline`, `notes-type-person`, and likewise for highlight / insight / episode / topic). Chips are the ENTITY a note is attached to and render only for kinds that have notes, so the spec writes one note on a show and one on a storyline through the shared `NoteComposer` before the strip exists at all — and asserts `notes-type-person` is ABSENT. The tab's `collections-search` box narrows the same section; combined with a chip it is how an empty result is reached, which must keep the strip and show `collections-notes-empty`. | `boards-notes-filter.spec.ts` |
| **Sparkline** — `Sparkline` (`sparkline`, `sparkline-line`, `sparkline-area`) | `sparkline.spec.ts` (asserts a real path from data on trend rows) |
| **Trend momentum** — `TrendMomentum` (`trend-momentum`) — one shared velocity badge/sparkline for topic cards, the storyline page, and the storylines rail (BT.4/F4.2) | unit `TrendMomentum.test.ts`; exercised in-surface via `knowledge-bands.spec.ts` (topic card), `storyline.spec.ts`, `home-rails.spec.ts` |

Three of those are asserted as **invariants** rather than as presence: a rail whose data the fixture
corpus does not produce is *supposed* to omit itself (UXS-012), so demanding it be visible would
test the corpus rather than the app, and would fail for the right behaviour. What is asserted is the
contract a listener can actually be hurt by: **when present, it has content; it is never an empty
shell.**

### What is still NOT covered, and why

| Surface | Why not |
| ------- | ------- |
| **PWA update toast** (`pwa-update-*`) | Needs a service-worker UPDATE to occur mid-session — a second build installed behind a running page. Playwright can install a SW but cannot cheaply produce a genuine update event, and faking it would assert the mock rather than the toast. Unit-tested (`PwaUpdateToast.test.ts`). |
| **Downloads, Downloaded list, Device settings** — `DownloadButton`, `DownloadedList`, `DeviceSettings` | Behind `isNative()` — they render nothing in a browser, by construction. Covered by the DEVICE tier (`make test-app-ios-journey`). |
| **Listening recap** — `ListeningRecap`, `RecapPrompt` | Covered — `recap-and-deep-links.spec.ts` and `recap-and-offline-writes-real-corpus.spec.ts` (Tier-3). Listed so the components are findable by name. |
| **Highlights view** — `HighlightsView` | Export (`Export Markdown` link → `/api/app/highlights/export.md`) and notes are covered by `capture.spec.ts`; the share-card control (`highlights.share`) hands off to the OS share sheet, which a browser cannot drive. |
| **Resurfacing inbox** — `ResurfacingInbox` | The pacing control (pause/resume) and the fresh-user empty state are covered by `consolidation.spec.ts`. A genuinely DUE item cannot be produced deterministically here — it needs a highlight captured far enough in the past, which the version-pinned fixture corpus does not (and should not) synthesize; forcing it would test the clock, not the app. |

## Spec tiers that do NOT run on pull requests

`e2e/*.spec.ts` runs on every PR. Three other tiers live in subdirectories and run elsewhere —
which is precisely why they rot unnoticed, and why they belong in this map.

One UI change has now broken two of them in sequence. The Discover redesign (`be58472b0`, #2072)
desynced the live smoke — resynced in `cf56a1e91`, whose message names the cause: the guard
*"globs `e2e/*.spec.ts`, not `e2e/live/`, so the drift went unseen"* — and then broke the Tier-3
walk, which ran red nightly from 2026-09-16 until `2678f77b2`. The guard
(`src/__checks__/surface-map.test.ts`) now globs `e2e/**/*.spec.ts`, so every tier below is held
to the same contract as the PR suite.

| Spec | Covers | Runs |
| ---- | ------ | ---- |
| `live/smoke.live.spec.ts` | Deployed `closelistening.app` reachable + shell renders (#43) | post-deploy |
| `live/surfaces.live.spec.ts` | Public read-only consumer surfaces (incl. `home-search-input`) | post-deploy |
| `live/account.live.spec.ts` | Per-user surfaces — Collections / Library / Queue | post-deploy |
| `live/offline-arc.live.spec.ts` | The offline arc (#1905, #1906, #1914, #1925) | post-deploy |
| `live/privacy-floor.live.spec.ts` | k-anonymity floor on cross-user reach (#1923) | post-deploy |
| `live/trending.live.spec.ts` | RFC-103 R2 trending — window selector + corpus-anchored denoising | post-deploy |
| `validation/listen-through-real-corpus.spec.ts` | Full listen-through vs a real backend + corpus | nightly (Tier-3) |
| `validation/recall-real-corpus.spec.ts` | Recall — search `scope=mine` vs `scope=all` | nightly (Tier-3) |
| `validation/recap-and-offline-writes-real-corpus.spec.ts` | Recap honesty, queue item-writes, `?t=` deep link | nightly (Tier-3) |
| `validation/consolidation-real-corpus.spec.ts` | Library Revisit resurfacing ladder | nightly (Tier-3) |
| `validation/multi-perspective-topic-real-corpus.spec.ts` | Multi-perspective topic card (#1146) | nightly (Tier-3) |
| `validation/offline-shell-real-corpus.spec.ts` | Offline shell survives a REAL network drop | nightly (Tier-3) |
| `design/surfaces.design.spec.ts` | Screenshots every redesigned surface for the critic loop (#1944, #1945) — asserts only that the surface LOADED; input, not a test | manual / design runs |

`search-result-actions` is the `EpisodeActions` cluster on a SearchView result row; it is covered
by unit (`SearchView.test.ts`) rather than by a spec, and named here so the map accounts for it.

**A Tier-3 spec must not anchor on an optional surface.** These run against an operator-supplied
corpus, so anything a corpus may legitimately not produce — the trending rails above all — is
absent by design, not broken. `validation/recap-and-offline-writes-real-corpus.spec.ts:81` clicked
the first `a[href*="/podcast/"]` on Home, whose only source is the trending-shows rail, and that
rail is *allowed* to render nothing (see the invariant note above). Reach a show episode-first.

## Shared components & shell — naming index

The reusable widgets and app-shell pieces the surface specs drive indirectly (via the view that
hosts them) rather than by a dedicated file. Named here so the map accounts for every rendered
component — a name is the contract "this exists and here is where it is exercised", per the
`surface-map` guard. Closes the 2026-09-03 `KNOWN_GAPS.components` seed.

| Component | What / where | Exercised by |
| --------- | ------------ | ------------ |
| `AddToCollectionButton` | Pin any item into a collection (RFC-119); EpisodeCard, EntityCard, Player, Search | `collections.spec.ts`, `capture.spec.ts` |
| `AppSplash` | Web launch overlay under the native splash handoff; App shell | shell overlay — no dedicated spec; unit-adjacent via App boot |
| `AppUpdateBanner` | Native-only "update available" banner (`app-update-banner` / `app-update-action` / `app-update-dismiss`) when the server's `player_version` outruns this baked build (wave-I.6); App shell. Web updates flow through `PwaUpdateToast` (service worker) | unit via `useAppUpdate.test.ts` (version compare + native gating); no store URL pre-launch |
| `BottomNav` | Mobile bottom tab bar (`sm:hidden`); App shell | `mobile-invariants.spec.ts`, `zone-d-small-viewport.spec.ts` |
| `BrandGlyph` | Close Listening ember-waveform mark; header / login / empty states | decorative identity mark — no dedicated spec |
| `CardRail` | Horizontal swipe/snap carousel with desktop chevrons; Your Week, Player | `your-week.spec.ts` |
| `ConnectedAgents` | MCP connector URL + PAT wiring for entitled users (RFC-112); Settings | `SettingsView.test.ts` (unit); native/settings surface |
| `EpisodeRow` | The one compact episode list-row — thumbnail + title + show kicker linking to the player (`episode-row`), top-aligned, with a `#trailing` slot for a row action; entity card, storyline page, Knowledge Panel "More like this" | exercised via `entity-and-rails-invariants.spec.ts`, `storyline.spec.ts`, `knowledge-bands.spec.ts` |
| `EntityCardBody` dismiss (`ec-dismiss`) | The entity card's single left control: ✕ when the card IS the destination (overlay sheet, standalone page), ‹ back when it is a drill-down inside a panel — `dismissAtRoot` decides. The ✕ is a drawn `CloseIcon`, not the character U+2715, which is a tofu box on iOS. | `icon-rendering.spec.ts` (asserts a drawn icon with a non-zero box, and that no tofu-prone character is rendered) |
| `PersonCardContent` | The person-specific BODY of the entity card (bio + signals, hosted shows, episodes, related people/topics, notes), rendered by the `EntityCardBody` shell | exercised via the entity card — `perspectives.spec.ts`, `entity-signals.spec.ts` |
| `TopicCardContent` | The topic-specific BODY of the entity card (momentum badge, similar topics, storyline link, strongest shows, episodes, conversation arc, perspectives, top voices, notes), rendered by the `EntityCardBody` shell | exercised via the entity card — `entity-and-rails-invariants.spec.ts`, `perspectives.spec.ts` |
| `OrgCardContent` | The organization BODY of the entity card (#2031): mentioned-in episodes + co-occurring people / other orgs / topics, rendered by the `EntityCardBody` shell. Lean by design, plus an org_web block (description + logo + attribution, #2035) when the enricher matched. | ⚠️ **no e2e** — the `app-validation-corpus` fixture contains **no organizations** (`MENTIONS_ORG` count 0), so no e2e can open an org card against it. Unit-tested in `OrgCardContent.test.ts` + the projection in `test_app_relational_view.py`. e2e follows once the fixture (or an org-bearing corpus) carries orgs. |
| `ShareMenu` | Share affordance (#2036, `share-menu`) — **Share card** (editorial PNG via `entityShareCard`), **Share link** (canonical URL), **Share text** (caption); bridge-only. Wired on the entity card, episode (PlayerView), show (PodcastView) and storyline (StorylineView). Link unfurl is a **server-rendered `og:image`**: `GET /og/{kind}/{id}.png` (`routes/app_og.py`) + `server/spa.py` OG-meta injection, unauthenticated, tested server-side (`tests/unit/podcast_scraper/server/test_app_og.py`, `test_og_card.py`). | ⚠️ **no e2e yet** — unit-tested client (`entityShareCard.test.ts`, `EntityCardBody.test.ts`, `PodcastView.test.ts`, `StorylineView.test.ts`, `ShareMenu.test.ts`) + the server OG suite (`test_og_card.py`, `test_app_og.py`, `test_artwork_symlink_safety.py`). e2e (menu opens, actions fire) follows when the e2e corpus grows the fixtures. |
| `EpisodeRecapPanel` | Post-episode recap (#2038 / RFC-122, `episode-recap-panel`) — replaces the transport in place on PlayerView when the episode finishes (the player store's `justFinished`, on `ended` or ≥95%): summary key points, the single strongest attributed quote, top insights, key-topic chips, and storyline links. Kept short enough to sit on a phone with **no internal scroll**; "more like this" is the related rail on the page, not repeated here. When a next episode is queued it is an **end-card**: a countdown (`recap-countdown`) + `recap-play-next` continue the queue, `recap-stay` cancels — the recap drives the advance, not `onEnded` (the store holds auto-advance on this surface via `setAdvanceHold`). Empty queue → recap-then-stop; dismiss (`recap-dismiss`/`recap-back`) restores the player; new episode clears it. Bridge-only. Served by `GET /api/app/episodes/{slug}/recap` (`routes/app_episodes.py`). UXS-014 §Post-episode recap. | `recap-panel.spec.ts` (finish → recap → dismiss with an empty queue; queued-next → countdown end-card → continue). Unit: `EpisodeRecapPanel.test.ts` (content + countdown), `player.test.ts` (finish signal + advance-hold + `playNext`), server `test_app_episodes.py` (recap projection). |
| `FavoriteButton` | Heart save toggle (`.lp-fav`), shared everywhere (UXS-014) | `follow-show.spec.ts`, `capture.spec.ts` |
| `FollowButton` | The one show-follow pill (`follow-show`), inline on the show header + overlay on `ShowTile` (F2.4). Topic/person (`ec-follow`) and storyline (`storyline-follow`) pills are hand-rolled with their own static testids — see the EntityCard / StorylineView rows | `follow-show.spec.ts` |
| `FollowedInterests` | Followed topics/people/storylines, unfollow inline; Library | exercised via `LibraryView` — no dedicated spec |
| `KeyVoicesRail` | Home rail of the user's most-present people (`key-voices-rail` / `key-voice`) linking to person cards — the per-user "key voices" (wave-G); Home, authenticated | unit via `KeyVoicesRail.test.ts`; self-hides when empty |
| `ListToolbar` | The one search / filter / sort / view header for big lists (UXS-014), shared by Browse › Episodes (`CatalogView`) and Browse › Shows (`ShowBrowseView`) so they cannot drift. A wide search fills the row; filter, sort and view are compact `ToolbarMenu` controls (operator 2026-09-14). Each surface supplies its own selectors via props (Episodes `list-toolbar-sort` + per-option `list-toolbar-sort-opt-az`, Shows `show-browse-*`). Shows adds a Shows-only **Trending** sort (velocity, RFC-103) that blends a per-row sparkline into the list — operator 2026-09-14; Episodes keeps the four shared sorts. | exercised via `browse-and-profile.spec.ts`, `browse-and-topic-pages.spec.ts`, `ShowBrowseView.test.ts`, `ListToolbar.test.ts` |
| `MiniPlayer` | Persistent mini transport with progress; App shell | `audio-continuity.spec.ts`, `mobile-invariants.spec.ts` |
| `NotificationsBell` | Header bell + unread badge (`notifications-bell` / `notifications-badge`) opening the in-app inbox dropdown (`notifications-panel`, `notification-item`, `notifications-mark-all`) — the `in_app` channel (wave-I); App shell, authenticated | exercised via the header when signed in; unit-adjacent via `notifications` store |
| `OfflineBanner` | App-level "Offline — showing saved" bar (`offline-banner`) when `navigator.onLine` is false (F1.2); App shell, under the masthead | `offline.spec.ts` |
| `PlayerControls` | Scrubber, skip, speed, insight-density ticks; Player | `player-reach.spec.ts`, `transcript.spec.ts`, `full-listen.spec.ts` |
| `CloseIcon` | The shared dismiss ✕, drawn as an inline SVG. It was the CHARACTER U+2715, which is absent from the iOS UI font and rendered as a tofu box on device — every sheet, panel and modal showed "?" where its close control belongs (2026-09-16). Font-stack fallbacks did not fix it; drawing it did. | `icon-rendering.spec.ts` — asserts an `<svg>` with a non-zero box AND that no tofu-prone character is rendered. The earlier claim here that Playwright could not see it was WRONG: Playwright queries the DOM, not the a11y tree, and reports a hidden `<svg>` as visible with a real box. XCUITest genuinely cannot (it reads the a11y tree), so the NATIVE tier still relies on screenshot review. `aria-hidden` is correct and stays — the button owns the name. |
| `ProfileAvatar` | Account picture (`profile-avatar`) — initials on a name-derived hue until the server exposes an OAuth photo; Profile header + masthead | exercised via Profile / the header avatar; unit-adjacent |
| `AvatarCropModal` | Square crop-on-upload modal (`avatar-crop-modal`; `avatar-crop-zoom` / `avatar-crop-confirm` / `avatar-crop-cancel`) — pan + zoom a picked photo to a circle before upload, emitting a cropped PNG; Profile header | exercised via Profile avatar upload; unit-adjacent |
| `PlayerSkeleton` | Player loading skeleton (`player-skeleton`) reserving artwork/title/controls/transcript on a cold uncached load (F1.3) | `PlayerSkeleton.test.ts` (unit); cached episodes skip it |
| `QueuePanel` | Up-next + recently-played sheet/panel; from MiniPlayer | `queue-panel.spec.ts`, `queue-reorder.spec.ts` |
| `SavedColorControl` | The one collapsed colour control (RFC-121 ph. 4 / #2042): a current-colour dot (`saved-color`) that opens a 44px-swatch popover (`saved-swatch`). Shared by the Highlights list and, since colour reached class-A favorites, saved episodes + entities; Library → Saved | exercised via `LibraryView` / `HighlightsView`; unit `SavedColorControl.test.ts` |
| `SavedFilterBar` | The Saved tab's lifted filter bar (RFC-121 ph. 3 / #2042, `saved-filter-bar`): type chips (`saved-type-all` / `saved-type-*`, none = all), a colour filter (`saved-filter-swatch`), and a sort select (`saved-sort`); Library → Saved | exercised via `LibraryView`; unit `SavedFilterBar.test.ts` |
| `EpisodeGroupCard` | An episode-grouped result block (`episode-group`, override per surface): the shared `EpisodeCard` as the header — artwork, show, title, date — with the group's rows collapsible beneath it via `episode-group-toggle` / `episode-group-body` ("Hide/Show {noun}", `aria-expanded`, `v-show` so inner state survives a collapse). Groups start EXPANDED. The per-group COUNT rides the card's `#aside` slot (under the artwork), so the toggle never repeats it. Two callers: Search (episode groups, `noun` = matches, rows = folded/plain hits, FULL header card with actions) and Library → Revisit (`revisit-group`, `noun` = moments, rows = due captures, `slim` header — 80px artwork, no action cluster, tight padding — plus a `revisit-listened` "Listened {date}" line in `#meta` from `getPlaybackList`). Extracted 2026-09-17 because Search had the card-with-artwork treatment and Revisit had a bare text heading for the same object. | exercised via `SearchView.test.ts` (`an episode group collapses its matches and starts expanded`), `ResurfacingInbox.test.ts` (`collapses an episode group and restores it`); e2e via `search.spec.ts` |
| `ShowActivityChart` | Episodes-per-month bar sparkline (`show-activity`); Show page | `knowledge-bands.spec.ts` |
| `TypeFilterBar` | The multi-select "kind" chip strip (`<prefix>-filter` / `<prefix>-all` / `<prefix>-<key>`, none selected = All), shared by every surface that filters by kind. Extracted 2026-09-17 because the markup existed verbatim in `SavedFilterBar` and `SearchView` and the Boards tab wanted a third copy. Three callers, each passing its own testid prefix: Library → Saved (`saved-type-*`), Search (`search-type-*`), Library → Boards notes (`notes-type-*`, one chip per entity kind a note is attached to — only kinds actually present). | exercised via `SavedFilterBar.test.ts`, `SearchView.test.ts`, `CollectionsView.test.ts` (`notes kind filter`); no dedicated spec of its own |
| `ShowAllToggle` | The "Show all (N) / Show less" control (`show-all-toggle`) under a capped Library section (#2042 follow-up) — each per-type section shows the top N and expands in place; a search lifts every cap. Library → Following + Saved; Collections (Boards tab) → boards + notes | exercised via `LibraryView`, `CollectionsView`; unit-adjacent |
| `ShowRow` | A show as a LIST ROW (`show-row`), built to `EpisodeCard`'s proportions: 128px artwork with the episode count and the surface's own controls beneath it, the name and the description filling the right via the shared `lp-media-*` clamp, with `show-row-read-more` when the description overflows. Replaces two divergent representations — Discover → Shows (list view) used a 44px thumbnail, Library → Saved used a bare line of text — so a show stops reading as a different kind of thing from the episode rows beside it. The `#actions` slot is what differs per surface: Discover passes Follow + save + the trending sparkline, Saved passes the colour picker + save, Following passes Follow + save. Reached via `goto('/browse?tab=shows')` → list view, or `goto('/library?tab=saved')` | **coverage gap** — no dedicated spec; exercised indirectly via `follow-show.spec.ts` and `LibraryView` |
| `ShowTile` | Square-artwork show tile with follow overlay; Home/Library/Browse | `home-rails.spec.ts`, `follow-show.spec.ts` |
| `SkipLink` | Keyboard skip-to-`#main` (UXS-011 a11y); App shell | keyboard a11y — exercised by the axe sweeps |
| `StorylineCard` | Teleported overlay wrapping `StorylineView` (embedded) — opens the storyline ON TOP from a topic card's link (`storyline-card`, `storyline-card-close`, `?storyline=` history); focus trap + Back-to-close via `useModalSheet` | `entity-and-rails-invariants.spec.ts` |
| `TierSwitch` | Dev↔prod target pill, internal build only (`tierSwitchEnabled()`) | internal build only — never rendered on web |
| `TopicConversationArc` | Weekly stacked-bar conversation shape (`tca-bar-*`); Entity card | `knowledge-bands.spec.ts` |
| `TranscriptList` | Synced, paragraph-grouped transcript with tap-to-seek; Player | `transcript.spec.ts`, `transcript-paragraphs.spec.ts`, `capture.spec.ts` |
| `TrendWindowTabs` | 1M·3M·6M·1Y window control (RFC-103); trending rails/browse | `trending.spec.ts`, `browse-and-topic-pages.spec.ts` |
| `TrendingShowsRail` | Full-width show slices with sparkline horizon (`trending-show-card`); Home + Discover top (the Discover copy adds a `see-all` header `trending-shows-seeall` → Shows tab, operator 2026-09-14) | `home-rails.spec.ts`, `trending.spec.ts` |
| `TrendingSparkChips` | Trending topics as sparkline rows (`trend-spark-row`); Home/browse | `trending.spec.ts`, `browse-and-topic-pages.spec.ts` |
| `DiscoveryExplorer` | The shared discovery section (`discovery-explorer`) — the tabbed `DiscoveryList` (topics/storylines/people) + the Rising⇄Trending sort (`discovery-sort`) and Corpus⇄Mine scope (`home-trending-scope`) switches. Used by BOTH Home (capped 5) and Discover (capped 10, with a `discovery-see-all` link). Extracted from HomeView so the two cannot drift (operator 2026-09-14, replaced the top-3 `DiscoveryDashboard`). | `browse-and-profile.spec.ts`, `trending.spec.ts` |
| `TrendsView` | Full entity-trends page (`data-testid="trends-view"`, `data-testid="trends-back"`, `data-testid="trends-tab-{kind}"`, `data-testid="trends-sort"`, `data-testid="trends-scope"`); reached from Discover dashboard | `trending.spec.ts` |
| `ToolbarMenu` | Compact toolbar control (operator 2026-09-14) — a small trigger (an icon circle for sort/view, or a chip showing the current label for the filter facet) that opens a vertical option menu with the active one ticked; replaces the native `<select>` / two-button toggle so sort + view collapse to little circles and the search keeps its width. Each option carries a per-value selector derived from the trigger's own. Used by `ListToolbar`. | `ToolbarMenu.test.ts`; exercised via `ListToolbar.test.ts`, `ShowBrowseView.test.ts` |
| `YourWeekCard` | One Your-Week digest card (quote- or title-forward); Home Your Week | `your-week.spec.ts`, `home-rails.spec.ts` |

## Stable selectors and hooks (contract)

Prefer updating this section when Playwright assertions (or the components) change. Views mostly rely
on **roles / accessible names / RouterLinks**; reusable widgets carry `data-testid`.

### Home ([HomeView](../src/views/HomeView.vue))

| Element | Hook |
| ------- | ---- |
| Search input | `#home-search` (label `home.askKicker`) |
| Home topic chips | `home-topic-chips` (container), `home-topic-chip` (each) |
| Discovery section | `data-testid="home-discovery"` |
| Interests card CTA | button `interests.cardCta` → opens `InterestsPicker` |

### Discovery ([DiscoveryList](../src/components/DiscoveryList.vue))

| Element | Hook |
| ------- | ---- |
| Kind section | `data-testid="discovery-list-{kind}"` where `{kind}` ∈ `topic`, `storyline`, `person` — e.g. `data-testid="discovery-list-topic"`, `data-testid="discovery-list-storyline"`, `data-testid="discovery-list-person"` |
| Kind tab | `discovery-tab-{key}` where `{key}` ∈ `topic`, `storyline`, `person` — rendered by the Tabs component from the `testid` prop |
| Rows | `data-testid="discovery-row"` + `data-testid="discovery-follow"` (collapsed to top 5 + `data-testid="discovery-expand"` "Show N more") |
| Sort switch | `data-testid="discovery-sort"` — Rising (velocity) ⇄ Trending (volume) |
| Metric hint | `data-testid="discovery-hint"` — explains the trailing ×/count on screen (#1595) |

One list for all three kinds (topics / storylines / people), from `GET /api/app/trending` (carries both velocity + volume); the sparkline hue encodes trend direction. Storylines merge in `getStorylines` for the "N topics" subtitle + the anchor topic the overlay opens on.

### Storylines follow token (`thc:`)

Storylines surface in `DiscoveryList` when `kind="storyline"` — each row is `data-testid="discovery-row"` with a `data-testid="discovery-follow"` toggle. Clicking a row opens the `StorylineCard` overlay (see StorylineCard below). The old chip component (storyline-chip) is removed.

| Element | Hook |
| ------- | ---- |
| Storyline row | `data-testid="discovery-row"` (inside `data-testid="discovery-list-storyline"`) — clicking opens the StorylineCard overlay |
| Follow toggle | `data-testid="discovery-follow"` (`aria-pressed`; follows the `thc:` id) |

### EntityCard ([EntityCardBody](../src/components/EntityCardBody.vue) + [EntitySignals](../src/components/EntitySignals.vue) / [TopicPerspectives](../src/components/TopicPerspectives.vue))

| Element | Hook |
| ------- | ---- |
| Follow (this entity) | header `data-testid="ec-follow"` — text `Follow` / `Following` (`aria-pressed`; token = the entity id) |
| Topic momentum | `data-testid="ec-topic-momentum"` — the "↑ Rising" badge (`TrendMomentum`) leading the topic card, gated to genuinely-rising topics (moved here from EntitySignals) |
| Storyline | `data-testid="ec-storyline-link"` — one link that opens the storyline overlay (`storyline-card`) on top; Follow-storyline lives there now (`storyline-follow`). A topic with no cluster shows `ec-single-topic` |
| Similar topics | the cluster-members chips (`ec-similar-topic`) — drill in place via the back stack |
| Perspectives | `data-testid="topic-perspectives"`, per-take `topic-perspective` |
| Signals | `data-testid="entity-signals"` — PERSON-only now, rows `es-coappears` / `es-consensus` / `es-consensus-row` (grounding removed #1927 → per-EPISODE, operator-only; topic momentum moved to `ec-topic-momentum` on the card; similar + storyline render on the card itself, not here) |

### Interests picker ([InterestsPicker](../src/components/InterestsPicker.vue))

| Element | Hook |
| ------- | ---- |
| Topics section | `data-testid="interests-topics"` (semantic `tc:` chips) |
| Storylines section | `data-testid="interests-storylines"` (`thc:` chips) |
| Chip pressed state | `aria-pressed` per chip; **Save** / **Cancel** buttons (`interests.save` / `interests.cancel`) |
| Modal | `role="dialog"` `aria-modal="true"`; backdrop click / **Esc** / **✕** dismiss (focus trap) |

### Player ([PlayerView](../src/views/PlayerView.vue) + [EpisodeDensity](../src/components/EpisodeDensity.vue))

| Element | Hook |
| ------- | ---- |
| Transcript toggle (mobile) | `data-testid="transcript-toggle"` (in the controls panel; `aria-expanded`). Transcript is **opt-in on mobile** — closed by default, this opens/closes it. Hidden on desktop (transcript is the always-visible side column). |
| Insight density | `data-testid="episode-density"` / `player-insight-density`; bands `player-density-band`, ticks `player-density-tick`, segments `density-{early,mid,late,peak}` |
| Capture | `aria-label` `capture.markMoment` → `capture.marked` |
| Sync controls | **Hidden** (`SHOW_SYNC_CONTROL=false`) pending a better sync fix — the `player.syncEarlier`/`syncLater`/`syncReset` UI is off; the offset machinery still applies any stored value. |
| Summary region | `role="region"` `player.summaryRegion` |
| Insights entry | `data-testid="player-open-insights"` — a LABELLED control ("✦ N insights"), not the old `💡 N` chip (#1595) |
| Reach chip | `data-testid="player-reach"` — cross-user listeners/opens + sparkline. **Renders only when there is something to show** (#1957): `/episodes/{slug}/stats` withholds `listeners`/`opens` below the k-anonymity floor (#1923) and returns `daily: []` with them, so all three children go false together by design. The wrapper used to render regardless and painted an empty pill. |

### Queue ([QueueView](../src/views/QueueView.vue))

| Element | Hook |
| ------- | ---- |
| Reorder | `aria-label` `queue.up` / `queue.down` (chevrons in the card icon row) |

### Login ([LoginView](../src/views/LoginView.vue))

| Element | Hook |
| ------- | ---- |
| Dev user list | `data-testid="dev-user-list"`, per-user `dev-user-{hint}` |
| Custom subject | `data-testid="dev-custom-input"` + `dev-custom-submit` |
| Sign in | `data-testid="signin-button"` |

### Knowledge panel ([KnowledgePanel](../src/components/KnowledgePanel.vue))

| Element | Hook |
| ------- | ---- |
| Insights list | `data-testid="kp-insights"`, fold control `kp-insights-show-all` |
| Entity chips | `data-testid="kp-topic-chip"` / `kp-person-chip` → opens the entity card in-panel |
| Save an insight | `aria-label` "Save to highlights" — the ONE save per insight since #1593; the `.lp-fav` heart is no longer on insight rows |

### Section states ([SectionStatus](../src/components/SectionStatus.vue))

Shared by every data-backed Home section (#1591). See UXS-012's state contract for when each shows.

| Element | Hook |
| ------- | ---- |
| Loading skeleton | `data-testid="section-loading"` (`aria-busy="true"`) |
| Error + retry | `data-testid="section-error"` (`role="status"`), button `section-retry` |

> **Empty is NOT this component's job.** Whether an empty section hides or renders depends on
> *why* it is empty — hide when the system is empty, render (with the action) when the user is.
> The caller owns that; see UXS-012.

### Your Week first run ([YourWeek](../src/components/YourWeek.vue))

| Element | Hook |
| ------- | ---- |
| First-run rows | `data-testid="yourweek-firstrun"`, one `li` per digest section |
| Compact/full toggle | `data-testid="yourweek-toggle"` — **absent** when there is no content to expand |

### PWA ([PwaUpdateToast](../src/components/PwaUpdateToast.vue))

| Element | Hook |
| ------- | ---- |
| Update toast | `data-testid="pwa-update-toast"`, `pwa-update-reload`, `pwa-update-dismiss` |

## Shared helpers

- [`signInIsolated(page, who, testInfo)`](helpers.ts) — dev-auth a fresh, test-isolated user (so
  per-user state — interests, queue, favorites, playback — never bleeds across specs running the
  shared real backend).
- Specs assert against the **committed** `app-validation-corpus/v3` fixtures; when a spec needs a
  specific KG shape (e.g. theme clusters for storylines, perspectives), that shape must exist in the
  corpus, not be route-mocked. Adding a surface that needs corpus data → extend the fixture corpus.
- [`openTranscript(page)`](helpers.ts) — reveals the transcript on mobile, no-ops on desktop. Use it
  rather than clicking `transcript-toggle` directly, or the spec passes on one project and fails on
  the other.
- Audio needs no setup: the corpus points at the mock podcast host and Playwright starts it (#1618).
  Any spec may play audio and assert on the transport with no interception at all.

## Corpus anchors

The suite pivots on specific fixture content. These are the anchors specs have standardised on —
**check here before regenerating the corpus**, because changing them breaks specs for reasons that
look like product bugs:

| Anchor | Used for |
| ------ | -------- |
| Episode "Index Investing Without the Myths" | search / consolidation (`?q=index`) |
| Passage `/Index funds are not a strategy/` | grounded-passage assertions |
| Episode "Risk Is a Systems Property" | graph-carrying episode for Your Week / follows |
| Episode "The Risk Panel: Diversify or Concentrate?" | multi-perspective topic |
| Shows "Long Horizon Notes", "Below the Surface" | show-page + follow-show flows |
| Topic "risk management" | topic card, perspectives, signals |
| Speakers "Daniel Cho", "Scott Bessent" | speaker attribution, person card |
| "10 perspectives" | perspectives count assertion |

> **Selector hygiene.** Fixed: the Insights-panel entity chips were selected via the CSS classes
> `button.text-topic` / `button.text-person` — a styling-coupled selector that would break on any
> restyle, and the cause of two flaky specs. They now carry `kp-topic-chip` / `kp-person-chip`.
> Do not reintroduce class-based selectors for behaviour.
