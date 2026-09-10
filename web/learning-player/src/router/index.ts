/**
 * Routes for the consumer Learning Player. LOGIN-FIRST (RFC-120 #2009): the guard denies by
 * default — only routes marked `meta.public` (the `/welcome` lure landing + `/login`) are
 * reachable logged-out; everything else redirects to the landing with a `?redirect` back.
 * (The per-route `meta.requiresAuth` flags are legacy no-ops now that deny-is-default.)
 */

import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'
import { useAuthStore } from '../stores/auth'
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
    path: '/settings',
    name: 'settings',
    component: () => import('../views/SettingsView.vue'),
  },
  {
    // Placeholder About/legal pages (3rd-party / privacy / terms) — empty content for now.
    path: '/about/:page',
    name: 'about-page',
    component: () => import('../views/AboutPageView.vue'),
    props: true,
  },
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
  scrollBehavior: () => ({ top: 0 }),
})

// Login-first guard (RFC-120): a free account is required for everything except the two
// `public` routes (the lure landing + login/signup). `ensureLoaded()` resolves the session
// once BEFORE deciding, so a signed-in visitor never flashes the landing on cold start / refresh
// (native rehydrates its Bearer here too). An unauthenticated visitor is sent to the landing with
// a `redirect` back to the intended path so a shared deep link survives signup.
router.beforeEach(async (to) => {
  const auth = useAuthStore()
  await auth.ensureLoaded()
  if (to.meta.public) {
    // Don't strand a signed-in user on the landing/login — bounce to their destination.
    if (auth.isAuthenticated && (to.name === 'landing' || to.name === 'login')) {
      return safeInternalPath(to.query.redirect) ?? { name: 'home' }
    }
    return true
  }
  if (!auth.isAuthenticated) {
    return { name: 'landing', query: { redirect: to.fullPath } }
  }
  return true
})
