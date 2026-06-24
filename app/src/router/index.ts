/**
 * Routes for the consumer Learning Player (RFC-099 §1). Player-first MVP: Catalog → Player,
 * with a Login entry. Reads are open, so routes are not auth-gated here; per-user features
 * gate on the auth store. Discovery/Capture/Corpus routes arrive in later tasks.
 */

import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'catalog',
    component: () => import('../views/CatalogView.vue'),
  },
  {
    path: '/episode/:slug',
    name: 'player',
    component: () => import('../views/PlayerView.vue'),
    props: true,
  },
  {
    path: '/login',
    name: 'login',
    component: () => import('../views/LoginView.vue'),
  },
  { path: '/:pathMatch(.*)*', redirect: { name: 'catalog' } },
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior: () => ({ top: 0 }),
})
