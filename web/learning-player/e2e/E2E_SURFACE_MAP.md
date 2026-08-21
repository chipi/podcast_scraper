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
| `/` | `home` | [HomeView](../src/views/HomeView.vue) | public | Learning Hub — adaptive hero, discovery |
| `/catalog` | `catalog` | [CatalogView](../src/views/CatalogView.vue) | public | "Browse" — episode catalog |
| `/search` | `search` | [SearchView](../src/views/SearchView.vue) | public | Corpus semantic search + KnowledgePanel |
| `/podcast/:feedId` | `podcast` | [PodcastView](../src/views/PodcastView.vue) | public | Show page → its episodes |
| `/episode/:slug` | `player` | [PlayerView](../src/views/PlayerView.vue) | public | Transcript + playback + capture |
| `/queue` | `queue` | [QueueView](../src/views/QueueView.vue) | **requiresAuth** | Play queue + reorder |
| `/library` | `library` | [LibraryView](../src/views/LibraryView.vue) | **requiresAuth** | Saved (episodes/insights) + highlights |
| `/profile` | `profile` | [ProfileView](../src/views/ProfileView.vue) | **requiresAuth** | Stats + interests entry |
| `/login` | `login` | [LoginView](../src/views/LoginView.vue) | public | Dev sign-in |
| `/topic/:id` | `topic` | [TopicView](../src/views/TopicView.vue) | public | Standalone topic page (#1261-6) — `data-testid="topic-view"` |
| `/person/:id` | `person` | [PersonView](../src/views/PersonView.vue) | public | Standalone person page (#1261-6) — `data-testid="person-view"` |
| `/browse/topics` | `browse-topics` | [TopicBrowseView](../src/views/TopicBrowseView.vue) | public | Topic index (#1261-6) — `data-testid="topic-browse-view"` |
| `/browse/people` | `browse-people` | [PersonBrowseView](../src/views/PersonBrowseView.vue) | public | People index (#1261-6) — `data-testid="person-browse-view"` |
| `/:pathMatch(.*)*` | — | → `home` | — | Catch-all redirect |

`meta.requiresAuth` routes redirect a signed-out visitor to `login` with `?redirect=<fullPath>`
([router/index.ts](../src/router/index.ts)).

## Surfaces and owning specs

| Surface | Intent (short) | Typical entry | Spec files |
| ------- | -------------- | ------------- | ---------- |
| **App shell / nav** | Header brand → home; `<nav>` NavIconLink **Browse** / **Library** / **Profile**; **Sign in** / **Sign up** when signed out | Every page | `smoke.spec.ts` (+ implicit in all) |
| **Home** | Adaptive hero — signed-in with in-progress history: **"Continue listening"**; otherwise kicker **"Ask across every episode"** + title **"Find any moment you've heard."**; search bar (`#home-search`); dismissible **set-your-interests** card → picker; **What's new** (featured `01` + ranked rows `02–06`); **Trending topics**; **Storylines**; **Recommended**; **"Your shows"** — the shows the signed-in user FOLLOWS (`getLibrary()`, capped at 11 + a "See all" tile to Library). Absent when signed out; renders an explanatory empty state rather than self-hiding when a signed-in user follows nothing. | `goto('/')` | `home-search.spec.ts`, `smoke.spec.ts`, `full-listen.spec.ts` (entry) |
| **Trending topics** | Corpus "heating up" (`temporal_velocity`) as **sparkline rows** — coloured by **storyline** (theme cluster), grouped by storyline, collapsed to top 5 (`trend-spark-expand`); rows open the topic card + one-tap follow. The four-way view switcher was removed in #1589 (it was an unresolved operator A/B lab). | Home, below What's new | ⚠️ **none** — see [gaps](#coverage-gaps) |
| **Storylines** | Theme clusters (topics discussed together) as a browsable rail; chip opens the anchor topic card, `＋`/`✓` follows the `thc:` cluster | Home, below Trending | ⚠️ **none** — see [gaps](#coverage-gaps) |
| **Momentum rail (RFC-103)** | Read-time "Rising now" (`GET /api/app/trending`, EWMA momentum anchored to `APP_TRENDING_NOW`) — generic per-kind chips: label + weekly sparkline + `↑` velocity + follow (interest-token kinds). `momentum-rail-{kind}`, `momentum-chip`, `momentum-follow`. Wired for `kind=topic` (opens topic card) | Home, below Storylines | `trending.spec.ts` |
| **EntityCard (person/topic)** | Overlay (from Search/Home) or inline (from Insights) card: **Follow**, **Your corpus** scope (all/mine), cluster identity (**Theme** + **Similar**), theme members, **Follow storyline**, **Perspectives**, **Signals**, related people/topics; re-entrant back stack | Trending/Storyline chip, Search entity hit, KnowledgePanel | `perspectives.spec.ts` (Perspectives), `entity-signals.spec.ts` (Signals) |
| **Interests picker** | Modal: **Topics** (semantic `tc:`) + **Storylines** (`thc:`) sections; Save replaces only the offered subset (preserves `topic:`/`person:` follows) | Home interests card **or** Profile → **Choose interests** | ⚠️ **UI: none** — `recommendation.spec.ts` drives `/api/app/interests` directly |
| **Catalog (Browse)** | Episode catalog / browse-all | `goto('/catalog')` (nav **Browse**, Home **Browse all →**) | ⚠️ **none dedicated** |
| **Search** | Corpus semantic search; passage hits + **KnowledgePanel** (entity chips → card); entity-in-search resolution | `goto('/search?q=…')`, Home search submit | `home-search.spec.ts`, `consolidation.spec.ts` (`?q=index`) |
| **Player (episode)** | Transcript (paragraph-grouped; **opt-in on mobile** via the controls-panel `transcript-toggle`, always-visible side column on desktop), floating/sticky controls on mobile, **capture** (mark moment), summary region, insight **density** strip. Manual sync controls are currently hidden. | `goto('/episode/:slug')`, via Podcast/Library/Queue/Home rows | `transcript.spec.ts`, `transcript-toggle.spec.ts`, `transcript-paragraphs.spec.ts`, `full-listen.spec.ts`, `capture.spec.ts`, `entity-signals.spec.ts` |
| **Podcast (show)** | Show page → episode list, **show signals band** (`podcast-signals` + `ps-theme` / `ps-topic` / `ps-trending` / `ps-person` rows), publishing-cadence chart (`show-activity`), and the **Follow show** toggle (`follow-show`, `aria-pressed`) — a *feed subscription*, distinct from interest follows | `goto('/podcast/:feedId')` (e.g. `p05`) | `follow-show.spec.ts`; also reached by `auth-queue`, `capture`, `consolidation`, `perspectives`, `entity-signals`, `transcript*` |
| **Follow show (feed subscription)** | `POST`/`DELETE /api/app/library` — optimistic toggle, reverts on failure. Feeds Your Week's "new in your follows". **Not** the same store as interest tokens (`topic:`/`person:`/`thc:`) | Show page header, signed-in only | `follow-show.spec.ts` |
| **Your Week** | In-app personal digest (`your-week`, expand via `yourweek-toggle`) — self-hides when every section is empty. "New in your follows" needs ≥1 followed show with unheard graph-carrying episodes | Home, when due | `your-week.spec.ts`, `follow-show.spec.ts` |
| **Topic / Person pages (#1261-6)** | Standalone routable entity pages (`topic-view`, `person-view`) — the non-modal counterpart to EntityCard | `goto('/topic/:id')`, `goto('/person/:id')`, Home `home-browse-nav` | `browse-and-topic-pages.spec.ts` |
| **Browse indexes (#1261-6)** | Topic and people indexes (`topic-browse-view`, `person-browse-view`) | `goto('/browse/topics')`, `goto('/browse/people')`, Home `home-browse-nav` | `browse-and-topic-pages.spec.ts` |
| **Search listener features** | Also-about chips (`related-topic-chips`), matched-fields kicker (`matched-fields`), save-query (`save-query-button` → `saved-searches-section` in Library), more-like-this rail (`related-episodes-rail`), search-scope switch (`tier-switch`, tablist "Search scope") | `/search?q=…` | `search-listener-features.spec.ts` |
| **Knowledge Panel dialog (S9)** | The learning panel is a native `<dialog>` — `knowledge-panel`. **Mobile: modal** (`showModal()`) — focus trapped, background inert, Escape closes, focus returns to `player-open-insights`. **Desktop (≥1024px): non-modal** (`show()`) — a docked rail beside the player, deliberately NOT trapping focus, because nothing is covered and the transcript must stay keyboard-reachable. Mode follows the viewport live, so rotating a phone re-modes it. | player route | `knowledge-panel-a11y.spec.ts` |
| **Mini-player (#1587)** | Persistent transport bar — `mini-player`, artwork+title link `mini-player-open`, play/pause `mini-player-toggle`. Visible whenever an episode is loaded and you are NOT on that episode's player page. Sits above the bottom nav on mobile. | any route, once playback starts | `audio-continuity.spec.ts` |
| **Audio continuity (#1587)** | The `<audio>` element is owned by the **player store**, appended to `<body>`, and outlives every view — so client-side navigation no longer stops playback. `document.querySelector('audio')` still works (`app-audio`). A full `page.goto` reload DOES stop it; that is a page load, not navigation. | — | `audio-continuity.spec.ts` |
| **Bottom nav (#1594)** | Mobile tab bar — `bottom-nav`, tabs `bottom-nav-{home\|search\|library\|profile}`. `sm:hidden`, so specs that must work on BOTH projects should click the header nav instead. | mobile viewports | covered incidentally; no dedicated spec |
| **Mobile invariants (#1312)** | Sticky transport stays pinned (`player-controls-sticky`), MediaSession metadata + playbackState, dark-canvas no-white-flash | `mobile-chrome` project | `mobile-invariants.spec.ts` |
| **Trending shows rail (RFC-103)** | Cover-art carousel with cadence sparkline (`trending-shows-rail`, `trending-show-card`) → show page | Home, below Momentum | ⚠️ **none** — see [gaps](#coverage-gaps) |
| **Queue** | Play queue; reorder via `↑`/`↓` chevrons; QueueButton add/remove | `goto('/queue')` (auth) | `auth-queue.spec.ts`, `queue-reorder.spec.ts` |
| **Library** | Saved tab (per-kind **Episodes** / **Insights**), highlights, resurfacing inbox | `goto('/library')` (auth) | `library-saved.spec.ts`, `capture.spec.ts`, `consolidation.spec.ts` |
| **Profile** | User stats; **interests** section → picker; resurfacing settings | `goto('/profile')` (auth) | ⚠️ **none dedicated** |
| **Login** | Dev sign-in — user list + custom subject | `goto('/login')`, auth-guard redirect | `auth-queue.spec.ts` + every authed spec (via `signInIsolated`) |
| **PWA / offline** | Service-worker registration, manifest + icons, `__buildInfo`; offline behaviour of Library/Queue (audio is **not** SW-cached; per-user API is **not** cached) | `goto('/')` then offline | `pwa.spec.ts`, `offline.spec.ts` — **the update toast itself is NOT covered**, see [gaps](#coverage-gaps) |
| **Capture / consolidation** | Mark-moment capture → highlights; consolidation suggestions (derived interests) | Player mark-moment; Library | `capture.spec.ts`, `consolidation.spec.ts`, `full-listen.spec.ts` |
| **Discovery ranking** | Personalized `/api/app/discover` responds to followed-interest levers (PRD-043 #1098) | API-level (`PUT /api/app/interests`) | `recommendation.spec.ts` |

## Coverage gaps

Surfaces that render in the app but have **no owning Playwright spec** (as of this writing). Flagged
so the gap is visible, not silently "covered":

| Surface | Selectors that exist | Note |
| ------- | -------------------- | ---- |
| **Storylines rail** | `home-storylines`, `storyline-chip`, `storyline-follow` | New (option B). Unit-tested (`Storylines.test.ts`); **no e2e**. |
| **EntityCard Follow-storyline** | `ec-follow-storyline` | New (option A). Unit-tested (`EntityCardBody.test.ts`); **no e2e**. |
| **Interests picker (UI)** | `interests-topics`, `interests-storylines` | Unit-tested (`InterestsPicker.test.ts`). e2e drives the **API** only, never the modal. |
| **Trending topics** | `home-trending`, `trend-spark*` | Unit-tested; **no e2e**. |
| **Momentum rail (RFC-103)** | `momentum-rail-{kind}`, `momentum-chip`, `momentum-follow`; `GET /api/app/trending` (server pins `APP_TRENDING_NOW=2026-07-20`) | Unit-tested (`MomentumRail.test.ts`) **+ e2e** (`trending.spec.ts`). Operator global view is on the gi-kg-viewer Dashboard (`TrendingGlobal.vue` → `GET /api/corpus/trending`). |
| **Catalog (Browse)** | — | No dedicated spec. |
| **Profile** | `stats.*` (roles) | No dedicated spec; picker entry point unexercised e2e. |
| **Insight density** | `player-insight-density`, `player-density-band`, `player-density-tick` | No dedicated spec for the player band/ticks. Note `episode-density` **is** asserted visible by `consolidation.spec.ts:35`. Segments are `density-{early,mid,late}` — `density-peak` is a separate caption element, **not** a fourth segment (`EpisodeDensity.vue:17,89`). |
| **PWA update toast** | `pwa-update-*` | **No e2e at all.** `pwa.spec.ts` covers manifest / icons / SW-registration / `__buildInfo` only — it never references a `pwa-update-*` selector. |
| **EntityCard theme members** | `ec-theme-members` | Unit-tested (`EntityCardBody.test.ts`); **no e2e**. |
| **Trending shows rail** | `trending-shows-rail`, `trending-show-card` | Unit-tested; **no e2e**. |
| **Podcast signals band** | `podcast-signals`, `ps-distinctive-heading`, `ps-distinctive-topic`, `ps-topics-heading`, `ps-theme`, `ps-topic`, `ps-trending`, `ps-person` | Unit-tested (`PodcastSignalsBand.test.ts`); **no e2e**. `ps-bubbles` retired — the momentum bubble cloud was removed because its `velocity` is corpus-wide, not show-scoped, so it sized topics by a number that did not answer the band's own question. Topics now split: `ps-distinctive-topic` carries the ones with `lift` above the corpus base rate (what sets the show apart), `ps-topic` the remainder — so a show's signature topic can no longer lose an alphabetical tiebreak to wallpaper every show covers. |
| **Topic conversation arc** | `topic-conversation-arc`, `tca-bars`, `tca-bar-*` | Unit-tested; **no e2e**. |
| **Show activity chart** | `show-activity`, `show-activity-bar-*` | Unit-tested; **no e2e**. |

## Stable selectors and hooks (contract)

Prefer updating this section when Playwright assertions (or the components) change. Views mostly rely
on **roles / accessible names / RouterLinks**; reusable widgets carry `data-testid`.

### Home ([HomeView](../src/views/HomeView.vue))

| Element | Hook |
| ------- | ---- |
| Search input | `#home-search` (label `home.askKicker`) |
| Trending section | `data-testid="home-trending"` |
| Storylines section | `data-testid="home-storylines"` |
| Interests card CTA | button `interests.cardCta` → opens `InterestsPicker` |

### Trending ([TrendingTopics](../src/components/TrendingTopics.vue) + children)

| Element | Hook |
| ------- | ---- |
| Sparklines | `data-testid="trend-sparks"`, `trend-spark-row`, `trend-spark-follow`, `trend-spark-expand` (collapsed to top 5 + "Show N more") |

All views colour topics by **storyline** (theme cluster) — same-cluster topics share a hue, unclustered use a neutral hue; the Sparklines view groups by storyline (hottest cluster first).

### Storylines ([Storylines](../src/components/Storylines.vue))

| Element | Hook |
| ------- | ---- |
| Chip container | `data-testid="storyline-chip"` |
| Open (chip body) | first `button` in the chip → emits `open` with `anchor_topic_id` (opens the topic card) |
| Follow toggle | `data-testid="storyline-follow"` (`aria-pressed`; follows the `thc:` id) |

### EntityCard ([EntityCardBody](../src/components/EntityCardBody.vue) + [EntitySignals](../src/components/EntitySignals.vue) / [TopicPerspectives](../src/components/TopicPerspectives.vue))

| Element | Hook |
| ------- | ---- |
| Follow (this entity) | header `button` text `Follow` / `Following` (`aria-pressed`; token = the entity id) |
| Corpus scope | `role="tablist"` named **"Card scope"**, with `role="tab"` **"All"** / **"My corpus"** (`ec.scopeAll` / `ec.scopeMine`) — the visible label is "My corpus", not "Mine" |
| Theme members | `data-testid="ec-theme-members"` |
| **Follow storyline** | `data-testid="ec-follow-storyline"` (`aria-pressed`; follows the `thc:` cluster) |
| Perspectives | `data-testid="topic-perspectives"`, per-take `topic-perspective` |
| Signals | `data-testid="entity-signals"`, rows `es-grounding` / `es-coappears` / `es-consensus` / `es-consensus-row` / `es-momentum` (similar + discussed-alongside topics render once on the card itself — `ec-theme-members` + the cluster-members chips — not here) |

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
