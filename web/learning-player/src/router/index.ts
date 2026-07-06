/**
 * Routes for the consumer Learning Player (RFC-099 §1). Player-first MVP: Catalog → Player,
 * with a Login entry. Reads are open, so routes are not auth-gated here; per-user features
 * gate on the auth store. Discovery/Capture/Corpus routes arrive in later tasks.
 */

import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'
import { useAuthStore } from '../stores/auth'

declare module 'vue-router' {
  interface RouteMeta {
    /** Gate the route behind a signed-in session (per-user features; reads stay open). */
    requiresAuth?: boolean
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
    path: '/login',
    name: 'login',
    component: () => import('../views/LoginView.vue'),
  },
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

// Auth guard for per-user routes. Reads are open, so only routes that opt in via
// `meta.requiresAuth` are gated; an unauthenticated visitor is sent to login with a
// `redirect` back to the intended path. No route opts in yet (queue/library land in C6).
router.beforeEach(async (to) => {
  if (!to.meta.requiresAuth) return true
  const auth = useAuthStore()
  await auth.ensureLoaded()
  if (!auth.isAuthenticated) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }
  return true
})
