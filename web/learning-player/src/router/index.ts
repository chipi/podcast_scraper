/**
 * Routes for the consumer Learning Player. LOGIN-FIRST (RFC-120 #2009): the guard denies by
 * default — only routes marked `meta.public` (the `/welcome` lure landing + `/login`) are
 * reachable logged-out; everything else redirects to the landing with a `?redirect` back.
 * (The per-route `meta.requiresAuth` flags are legacy no-ops now that deny-is-default.)
 */

import {
  createRouter,
  createWebHistory,
  type RouteLocationNormalized,
  type RouteRecordRaw,
} from 'vue-router'
import { i18n } from '../i18n'
import { track } from '../services/analytics'
import { noteNavigation } from '../services/provenance'
import { emailLinkOf, firstSighting } from '../services/inboundLink'
import { useAuthStore } from '../stores/auth'
import {
  holdScroll,
  offsetWithin,
  waitForSettledElement,
  waitUntilScrollable,
} from '../utils/scrollRestore'
import { SHEET_HISTORY_KEYS } from '../composables/useModalSheet'
import { anchorFromLastClick, restoreToAnchor, trackClicks, type ClickAnchor } from '../utils/backAnchor'

// Which control each page was left by — Back puts it back where it was (utils/backAnchor).
const backAnchors = new Map<string, ClickAnchor>()
trackClicks()

/** Clearance above a section an anchor lands on. */
const ANCHOR_GAP = 8

/**
 * Same page, and the only query keys that changed are sheet keys — a sheet opened over it or closed
 * off it. Neither moves the page: the sheet's own opener is put back by `useModalSheet`.
 */
function sheetOnlyChange(to: RouteLocationNormalized, from: RouteLocationNormalized): boolean {
  if (to.path !== from.path) return false
  const keys = new Set([...Object.keys(to.query), ...Object.keys(from.query)])
  const changed = [...keys].filter((k) => String(to.query[k] ?? '') !== String(from.query[k] ?? ''))
  return changed.length > 0 && changed.every((k) => (SHEET_HISTORY_KEYS as readonly string[]).includes(k))
}
// `getAuthToken` / `isNative` are no longer imported here: the native-token check moved into the
// shared `auth.hasSession` getter, which the masthead reads too, so the guard and the header cannot
// disagree about who is signed in (2026-09-16).
import { safeInternalPath } from '../utils/redirect'

declare module 'vue-router' {
  interface RouteMeta {
    /** Gate the route behind a signed-in session (per-user features; reads stay open). */
    requiresAuth?: boolean
    /**
     * Reachable while logged OUT (RFC-120 login-first). Only the lure landing + the login/signup
     * entry are public; every other route requires a free account. Default (absent) = login-first.
     */
    public?: boolean
  }
}

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'home',
    component: () => import('../views/HomeView.vue'),
  },
  {
    path: '/catalog',
    name: 'catalog',
    component: () => import('../views/CatalogView.vue'),
  },
  {
    path: '/search',
    name: 'search',
    component: () => import('../views/SearchView.vue'),
  },
  {
    path: '/podcast/:feedId',
    name: 'podcast',
    component: () => import('../views/PodcastView.vue'),
    props: true,
  },
  {
    path: '/episode/:slug',
    name: 'player',
    component: () => import('../views/PlayerView.vue'),
    props: true,
  },
  {
    path: '/queue',
    name: 'queue',
    component: () => import('../views/QueueView.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/library',
    name: 'library',
    component: () => import('../views/LibraryView.vue'),
    meta: { requiresAuth: true },
  },
  {
    path: '/profile',
    name: 'profile',
    component: () => import('../views/ProfileView.vue'),
    meta: { requiresAuth: true },
  },
  {
    // RFC-120: the logged-out lure landing. Public; the guard sends unauthenticated
    // visitors here (with ?redirect) instead of straight to login.
    path: '/welcome',
    name: 'landing',
    component: () => import('../views/LandingView.vue'),
    meta: { public: true },
  },
  {
    path: '/login',
    name: 'login',
    component: () => import('../views/LoginView.vue'),
    meta: { public: true },
  },
  {
    /*
     * "On this device" — downloads, offline, signed out (operator 2026-09-23).
     *
     * PUBLIC by necessity, not by preference: the whole point is that it is reachable when there is
     * no network to sign in over, and every non-public route bounces to the landing. The view
     * itself is the real gate — it redirects unless BOTH offline and signed out — so `public` here
     * buys reachability, not openness.
     */
    path: '/offline',
    name: 'offline-downloads',
    component: () => import('../views/OfflineDownloadsView.vue'),
    meta: { public: true },
  },
  // #1261-6: subject deep-link + browse routes — full-page equivalents of the
  // modal EntityCard for topic / person ids, plus trending-backed index pages.
  {
    path: '/topic/:id',
    name: 'topic',
    component: () => import('../views/TopicView.vue'),
    props: true,
  },
  {
    path: '/person/:id',
    name: 'person',
    component: () => import('../views/PersonView.vue'),
    props: true,
  },
  {
    // Theme page. `:id` is the THEME's own id (`tc:…`), not an anchor topic — a theme has a real
    // id and a real endpoint, so a theme link stays valid when its biggest member changes. It is
    // NOT served by /topic/:id: a theme is a grouping, never a node on an episode, so the topic
    // card matched nothing and the page rendered empty.
    path: '/theme/:id',
    name: 'theme',
    component: () => import('../views/ThemeView.vue'),
    props: true,
  },
  {
    // Storyline page (F4.5). `:id` is the storyline's ANCHOR TOPIC id — the theme cluster is
    // derived from that topic's card (there is no dedicated storyline endpoint).
    path: '/storyline/:id',
    name: 'storyline',
    component: () => import('../views/StorylineView.vue'),
    props: true,
  },
  {
    path: '/browse',
    name: 'browse',
    component: () => import('../views/BrowseView.vue'),
  },
  {
    // #2273. PUBLIC on purpose: Google Play's listing needs a web address that explains deletion
    // to someone who is not signed in. Signed in, the same page performs it.
    path: '/account/delete',
    name: 'account-delete',
    component: () => import('../views/DeleteAccountView.vue'),
    meta: { public: true },
  },
  {
    path: '/settings',
    name: 'settings',
    component: () => import('../views/SettingsView.vue'),
  },
  {
    // About/legal pages (3rd-party / privacy / terms). PUBLIC: a privacy policy must be readable
    // before signing up, and the store listings link to it (#2210).
    path: '/about/:page',
    name: 'about-page',
    component: () => import('../views/AboutPageView.vue'),
    props: true,
    meta: { public: true },
  },
  // The address store listings give for the privacy policy (#2210).
  { path: '/privacy', redirect: { name: 'about-page', params: { page: 'privacy' } } },
  { path: '/terms', redirect: { name: 'about-page', params: { page: 'terms' } } },
  /**
   * Browse is ONE surface with tabs — these paths are aliases into it (#2004 follow-up).
   *
   * Each used to render its view standalone, without the hub's tab strip and with its own heading
   * and a "‹ Back to Home". So the same content had two presentations depending on how you arrived,
   * and the standalone one lost the tabs entirely: from `/browse/shows` there was no way back to
   * Episodes and no indication Browse was where you were.
   *
   * Nothing in the app linked here — only `BottomNav`'s owned-routes list (which keeps the Browse
   * tab lit) and tests — so they were reachable by URL alone. Redirecting keeps those URLs working
   * while making the hub the only Browse there is.
   */
  { path: '/browse/shows', name: 'browse-shows', redirect: { name: 'browse', query: { tab: 'shows' } } },
  { path: '/browse/topics', name: 'browse-topics', redirect: { name: 'browse', query: { tab: 'topics' } } },
  { path: '/browse/people', name: 'browse-people', redirect: { name: 'browse', query: { tab: 'people' } } },
  { path: '/:pathMatch(.*)*', redirect: { name: 'home' } },
]

export const router = createRouter({
  // BASE_URL is vite's runtime-injected base (matches vite.config.ts APP_BASE).
  // Under a subpath deploy this ensures router URLs, history entries, and
  // <router-link> hrefs all include the base — no dead-links when deployed
  // under /app/ or a preview /pr-N/ prefix.
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
  // Top on every navigation, EXCEPT when the link names an anchor — Home's "See all →" lands on
  // Discover's trends section, which sits below the fold, so scrolling to the top would drop the
  // reader above the very thing they asked for (operator 2026-09-17). `behavior: 'smooth'` makes
  // the jump legible as a move rather than a page swap.
  //
  // BACK returns to where you were, not the top (operator 2026-10-04): opening a person from a
  // section halfway down an episode and coming back must land on that section. The browser hands
  // us the position (`saved`); detail pages re-fetch on return, so wait for the page to be tall
  // enough first — see utils/scrollRestore.
  //
  // A SHEET opening is not a new page: it only adds its `?card=` / `?theme=` … key, and the page
  // underneath must stay exactly where it is, so closing the whole stack lands where it started.
  //
  // An anchor (a note's Open → `#notes`) usually names a section that renders only after the page's
  // own fetch, and then gets pushed down by the sections above it as they stream in. Wait for it to
  // exist AND stop moving, or the reader lands at the top or above it.
  //
  // Both then HOLD the position for a few seconds (utils/scrollRestore `holdScroll`): content that
  // arrives later still moves the page, and a one-shot restore loses to it under load.
  scrollBehavior: async (to, from, saved) => {
    if (sheetOnlyChange(to, from)) return false
    if (saved) {
      // The control the reader left by, put back where it sat — robust to the page having grown.
      const anchor = backAnchors.get(to.fullPath)
      const top = anchor ? await restoreToAnchor(anchor) : null
      if (top != null) return { top }
      // Not identifiable: the browser's offset. No hold — it would fight the browser's own scroll
      // anchoring, which is what keeps the content in place as the page grows.
      await waitUntilScrollable(null, saved.top, 5000)
      return saved
    }
    if (to.hash) {
      // Already on the page (Home's "See all" onto Discover's trends): nothing will move under it,
      // so the smooth scroll the operator asked for (2026-09-17) stays.
      if (document.querySelector(to.hash)) return { el: to.hash, behavior: 'smooth', top: ANCHOR_GAP }
      // Rendered later (a note's Open onto `#notes`): land INSTANTLY and hold. A smooth scroll here
      // animates toward where the section WAS; content arriving above then moves the section, and
      // the animation carries the reader away from it (measured 2026-10-04: scroll anchoring put the
      // page at ~2210, the animation dragged it back to 1053, the notes ended off screen).
      const anchor = await waitForSettledElement(to.hash)
      if (anchor) {
        holdScroll(null, () => offsetWithin(null, anchor) - ANCHOR_GAP)
        return { top: offsetWithin(null, anchor) - ANCHOR_GAP }
      }
    }
    return { top: 0 }
  },
})

// Login-first guard (RFC-120): a free account is required for everything except the two
// `public` routes (the lure landing + login/signup). `ensureLoaded()` resolves the session
// once BEFORE deciding, so a signed-in visitor never flashes the landing on cold start / refresh
// (native rehydrates its Bearer here too). An unauthenticated visitor is sent to the landing with
// a `redirect` back to the intended path so a shared deep link survives signup.
router.beforeEach((to, from) => {
  // A SHEET opening keeps the page: its anchor is still the one the page was left by.
  if (sheetOnlyChange(to, from)) return
  const anchor = anchorFromLastClick()
  if (anchor) backAnchors.set(from.fullPath, anchor)
  else backAnchors.delete(from.fullPath)
})

router.beforeEach(async (to) => {
  const auth = useAuthStore()
  await auth.ensureLoaded()
  // A native device with a stored bearer token HAS a session even when the cold-start `getMe`
  // couldn't confirm it — a transport failure, not a 401 (2026-09-15). Treat that as signed-in for
  // ROUTING, so a returning user is never stranded on the public landing while the header already
  // shows them logged in (the reported desync). A token that is genuinely dead surfaces as a 401 on
  // the first authed call and the interceptor (onUnauthorized → markSignedOut) then routes to
  // signed-out cleanly — which it now actually does for this case too (see main.ts).
  //
  // `auth.hasSession` is the SHARED definition: the masthead reads the same getter, so the guard
  // can no longer admit a user that the header then renders as signed-out.
  //
  // NB: reads are NOT open — `/me`, `/episodes`, `/podcasts` all require auth server-side — so an
  // admitted-but-unresolved session sees empty surfaces, which is exactly why the two must agree.
  const signedIn = auth.hasSession
  if (to.meta.public) {
    // Don't strand a signed-in user on the landing/login — bounce to their destination.
    //
    // `isAuthenticated`, NOT `signedIn` (2026-09-24). `hasSession` is true on a bearer token ALONE,
    // which is right for ADMITTING someone to authed routes — a transport failure must not strand a
    // returning user on the landing. It is wrong for BLOCKING. With a token the server no longer
    // honours and no resolved user, the app believed it was signed in, showed nothing (every authed
    // call 401s) and redirected away from `/login` — the one screen that could have fixed it. A
    // locked-out user with no way back in.
    //
    // Found on device: the dev picker "never rendered" across a dozen runs because `/login` kept
    // bouncing to Home, while the masthead showed Search, Queue and "Your profile" — auth-gated
    // controls on a session that could not name its own user.
    //
    // The asymmetry is the point: a token may ADMIT you, but only a known identity may BLOCK you
    // from signing in. If the app cannot say who you are, you must always be able to say it.
    if (auth.isAuthenticated && (to.name === 'landing' || to.name === 'login')) {
      return safeInternalPath(to.query.redirect) ?? { name: 'home' }
    }
    return true
  }
  reportEmailClick(to, signedIn)
  if (!signedIn) {
    return { name: 'landing', query: { redirect: to.fullPath } }
  }
  return true
})

const EMAIL_TARGETS = new Set(['player', 'podcast', 'topic', 'person', 'storyline', 'theme'])

/**
 * `email_link_opened` (operator 2026-10-05, services/inboundLink): a click on a link in one of our
 * emails. Here, in the gate, because only the gate knows BOTH where the link points and whether the
 * person is signed in — a signed-out click is reported before it is sent to sign in.
 */
function reportEmailClick(to: RouteLocationNormalized, signedIn: boolean): void {
  const link = emailLinkOf(to.query)
  if (!link || !firstSighting(to.fullPath, signedIn)) return
  const name = typeof to.name === 'string' ? to.name : ''
  track('email_link_opened', {
    ...link,
    target: EMAIL_TARGETS.has(name) ? (name as 'player') : 'other',
    signed_in: signedIn,
  })
}

/**
 * The browser tab title, per route (operator 2026-09-18).
 *
 * Every route rendered the same string — and that string was "Learning Player", the internal
 * project name, not the product. So every open tab looked identical, every bookmark was named
 * after a repo directory, and history was unusable.
 *
 * `<page> · Close Listening`: the page first, because a tab strip truncates from the right and the
 * distinguishing word has to survive. An unmapped route falls back to the brand alone rather than
 * rendering "undefined · Close Listening".
 *
 * `afterEach`, not a guard: a title is a consequence of having navigated, and setting it in
 * `beforeEach` would rename the tab for a navigation that a guard then redirects away from.
 */
router.afterEach((to) => {
  const brand = i18n.global.t('app.title')
  const key = typeof to.name === 'string' ? `pageTitles.${to.name}` : ''
  const page = key && i18n.global.te(key) ? i18n.global.t(key) : ''
  document.title = page ? `${page} · ${brand}` : brand
})

/**
 * `screen_view` for every route change (#2267).
 *
 * Umami already auto-tracks SPA page views, and those are kept — this is additive and exists for a
 * different reason. Route PATHS carry slugs and ids (`/episode/:slug`, `/person/:id`), so the
 * automatic page views are thousands of distinct URLs: useful for "what was visited", useless for
 * "what KIND of screen". Reporting the route NAME lets a report group by `player` / `topic` /
 * `person`, which is what makes the spec's "Journeys starting at Home" readable at all.
 *
 * `afterEach` for the same reason the title is set there: a view is a consequence of having
 * navigated, and `beforeEach` would report a screen the guard then redirects away from.
 *
 * The route name only — never the path, which would put the search term back into analytics that
 * `data-exclude-search` exists to keep out.
 */
router.afterEach((to, from) => {
  // A query- or hash-only change is not a new screen. On prod 2026-10-04 one open episode logged
  // seven `screen_view`s without its path ever changing. Umami's own page views already ignore the
  // query (`data-exclude-search`), so this keeps the two counts comparable.
  const sameScreen = from.matched.length > 0 && to.path === from.path
  if (typeof to.name === 'string' && !sameScreen) track('screen_view', { screen: to.name })
  // Remember where this navigation came FROM, for events that cannot be emitted until later —
  // `episode_open` needs the episode's feed to know whether the show is followed, and by then the
  // previous route is gone. See services/provenance.ts.
  noteNavigation(from)
})
