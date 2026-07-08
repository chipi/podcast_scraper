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

**Key distinction from the operator viewer.** The operator specs are mostly **route-mocked**
(`page.route(**/api/**)`). The player specs run against the **real API** over the **committed
validation corpus** (`tests/fixtures/app-validation-corpus/v3`), **no mocks** — the Playwright
`webServer` boots a real backend on `:8011` and the built app on `:4174`. So a player spec exercises
the actual server surface (search, discover ranking, capture, consolidation), and fixtures live in
the corpus, not in per-spec route handlers.

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
(catalog), **Library**, **Profile** when signed in; **Sign in** / **Sign up** links when signed out.

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
| `/:pathMatch(.*)*` | — | → `home` | — | Catch-all redirect |

`meta.requiresAuth` routes redirect a signed-out visitor to `login` with `?redirect=<fullPath>`
([router/index.ts](../src/router/index.ts)).

## Surfaces and owning specs

| Surface | Intent (short) | Typical entry | Spec files |
| ------- | -------------- | ------------- | ---------- |
| **App shell / nav** | Header brand → home; `<nav>` NavIconLink **Browse** / **Library** / **Profile**; **Sign in** / **Sign up** when signed out | Every page | `smoke.spec.ts` (+ implicit in all) |
| **Home** | Adaptive hero (**Continue** when signed-in with in-progress history, else **Ask your library**); search bar (`#home-search`); dismissible **set-your-interests** card → picker; **What's new** (featured `01` + ranked rows `02–06`); **Trending topics**; **Storylines**; **Recommended**; **Your shows** | `goto('/')` | `home-search.spec.ts`, `smoke.spec.ts`, `full-listen.spec.ts` (entry) |
| **Trending topics** | Corpus "heating up" (`temporal_velocity`) — views **Pills / Sparklines / Over time / Momentum** (`trend-view-*`); chips open the topic card + one-tap follow | Home, below What's new | ⚠️ **none** — see [gaps](#coverage-gaps) |
| **Storylines** | Theme clusters (topics discussed together) as a browsable rail; chip opens the anchor topic card, `＋`/`✓` follows the `thc:` cluster | Home, below Trending | ⚠️ **none** — see [gaps](#coverage-gaps) |
| **Momentum rail (RFC-103)** | Read-time "Trending now" (`GET /api/app/trending`, EWMA momentum anchored to `APP_TRENDING_NOW`) — generic per-kind chips: label + weekly sparkline + `↑` velocity + follow (interest-token kinds). `momentum-rail-{kind}`, `momentum-chip`, `momentum-follow`. Wired for `kind=topic` (opens topic card) | Home, below Storylines | `trending.spec.ts` |
| **EntityCard (person/topic)** | Overlay (from Search/Home) or inline (from Insights) card: **Follow**, **Your corpus** scope (all/mine), cluster identity (**Theme** + **Similar**), theme members, **Follow storyline**, **Perspectives**, **Signals**, related people/topics; re-entrant back stack | Trending/Storyline chip, Search entity hit, KnowledgePanel | `perspectives.spec.ts` (Perspectives), `entity-signals.spec.ts` (Signals) |
| **Interests picker** | Modal: **Topics** (semantic `tc:`) + **Storylines** (`thc:`) sections; Save replaces only the offered subset (preserves `topic:`/`person:` follows) | Home interests card **or** Profile → **Choose interests** | ⚠️ **UI: none** — `recommendation.spec.ts` drives `/api/app/interests` directly |
| **Catalog (Browse)** | Episode catalog / browse-all | `goto('/catalog')` (nav **Browse**, Home **Browse all →**) | ⚠️ **none dedicated** |
| **Search** | Corpus semantic search; passage hits + **KnowledgePanel** (entity chips → card); entity-in-search resolution | `goto('/search?q=…')`, Home search submit | `home-search.spec.ts`, `consolidation.spec.ts` (`?q=index`) |
| **Player (episode)** | Transcript (paragraph-grouped), playback, **capture** (mark moment), summary region, transcript↔audio sync controls, insight **density** strip | `goto('/episode/:slug')`, via Podcast/Library/Queue/Home rows | `transcript.spec.ts`, `transcript-paragraphs.spec.ts`, `full-listen.spec.ts`, `capture.spec.ts`, `entity-signals.spec.ts` |
| **Podcast (show)** | Show page → episode list | `goto('/podcast/:feedId')` (e.g. `p05`) | reached by `auth-queue`, `capture`, `consolidation`, `perspectives`, `entity-signals`, `transcript*` |
| **Queue** | Play queue; reorder via `↑`/`↓` chevrons; QueueButton add/remove | `goto('/queue')` (auth) | `auth-queue.spec.ts`, `queue-reorder.spec.ts` |
| **Library** | Saved tab (per-kind **Episodes** / **Insights**), highlights, resurfacing inbox | `goto('/library')` (auth) | `library-saved.spec.ts`, `capture.spec.ts`, `consolidation.spec.ts` |
| **Profile** | User stats; **interests** section → picker; resurfacing settings | `goto('/profile')` (auth) | ⚠️ **none dedicated** |
| **Login** | Dev sign-in — user list + custom subject | `goto('/login')`, auth-guard redirect | `auth-queue.spec.ts` + every authed spec (via `signInIsolated`) |
| **PWA / offline** | Update toast + service-worker + offline behaviour of Library/Queue | `goto('/')` then offline | `pwa.spec.ts`, `offline.spec.ts` |
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
| **Trending topics** | `home-trending`, `trend-chip`, `trend-chip-follow`, `trend-view-*`, `trend-stream*`, `trend-momentum*`, `trend-spark*` | Unit-tested; **no e2e**. |
| **Momentum rail (RFC-103)** | `momentum-rail-{kind}`, `momentum-chip`, `momentum-follow`; `GET /api/app/trending` (server pins `APP_TRENDING_NOW=2026-07-20`) | Unit-tested (`MomentumRail.test.ts`) **+ e2e** (`trending.spec.ts`). Operator global view is on the gi-kg-viewer Dashboard (`TrendingGlobal.vue` → `GET /api/corpus/trending`). |
| **Catalog (Browse)** | — | No dedicated spec. |
| **Profile** | `stats.*` (roles) | No dedicated spec; picker entry point unexercised e2e. |
| **Insight density** | `episode-density`, `player-insight-density`, `player-density-band`, `player-density-tick`, `density-{early,mid,late,peak}` | No dedicated spec. |

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
| View tabs | `data-testid="trend-view-{chips\|sparks\|stream\|momentum}"` (`role="tab"`) |
| Pills chip / follow | `data-testid="trend-chip"` / `trend-chip-follow` (`aria-pressed`) |
| Sparklines | `data-testid="trend-sparks"`, `trend-spark-row`, `trend-spark-follow` |
| Over-time stream | `data-testid="trend-stream"`, `trend-stream-band`, `trend-stream-legend` |
| Momentum map | `data-testid="trend-momentum"`, `trend-momentum-point` |

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
| Corpus scope | `role="tab"` **All** / **Mine** (`ec.scopeAll` / `ec.scopeMine`) |
| Theme members | `data-testid="ec-theme-members"` |
| **Follow storyline** | `data-testid="ec-follow-storyline"` (`aria-pressed`; follows the `thc:` cluster) |
| Perspectives | `data-testid="topic-perspectives"`, per-take `topic-perspective` |
| Signals | `data-testid="entity-signals"`, rows `es-grounding` / `es-coappears` / `es-disagreements` / `es-disagreement-row` / `es-momentum` / `es-similar` / `es-alongside` |

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
| Insight density | `data-testid="episode-density"` / `player-insight-density`; bands `player-density-band`, ticks `player-density-tick`, segments `density-{early,mid,late,peak}` |
| Capture | `aria-label` `capture.markMoment` → `capture.marked` |
| Sync controls | `aria-label` `player.syncEarlier` / `player.syncLater` / `player.syncReset` |
| Summary region | `role="region"` `player.summaryRegion` |

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
