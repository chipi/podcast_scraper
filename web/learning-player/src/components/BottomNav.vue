<script setup lang="ts">
/**
 * Mobile bottom tab bar (#1594).
 *
 * All navigation lived in a top header, so on a phone every move was a top-of-screen reach — on an
 * app whose primary Playwright project is a Pixel 7. The plumbing was always phone-first (`dvh`,
 * safe areas, sticky transport, touch-first rails); the layout idiom was not.
 *
 * Four destinations, chosen so each answers a different question: **Home** (what should I listen to),
 * **Search** (find a specific moment — the differentiator, previously reachable from one page only),
 * **Library** (my saved things), **Profile** (me). Browse folds into Home and Search rather than
 * taking a fifth slot; it is a corpus index, not a daily destination.
 *
 * Mobile only — `sm:hidden`. Desktop keeps the header nav, where a top reach costs nothing and the
 * horizontal space is free.
 *
 * ## Safe areas and the mini-player
 *
 * `pb-[env(safe-area-inset-bottom)]` clears the iOS home indicator. When the global mini-player
 * (#1587) lands it sits directly ABOVE this bar, so both must be accounted for in the page's bottom
 * padding — `mobile-invariants.test.ts` pins that the two never overlap the sticky transport.
 *
 * ## The background is OPAQUE, deliberately
 *
 * This shipped as `bg-canvas/95 backdrop-blur`, which looked better and quietly broke WCAG: axe
 * composites text against whatever is actually behind the bar, so with a tinted storyline chip
 * scrolled underneath, both the active label (`text-accent`, 4.28:1) and the inactive labels
 * (`text-muted`, 4.28:1) fell under the 4.5:1 AA floor. Contrast became a function of scroll
 * position, which is why it surfaced as an intermittent e2e failure rather than a steady one.
 *
 * A fixed bar over arbitrary content cannot be translucent and also guarantee contrast. Opacity is
 * the fix at cause; `spec-conformance.test.ts` pins it so the frosted look cannot come back.
 */
import { RouterLink, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useAuthStore } from '../stores/auth'

const { t } = useI18n()
const route = useRoute()
const auth = useAuthStore()

/**
 * Library and Profile require auth. They stay VISIBLE signed-out and route to sign-in (#1590):
 * hiding them would hide the capability from exactly the visitors deciding whether to sign up.
 */
const TABS = [
  { name: 'home', label: 'nav.home' },
  { name: 'browse', label: 'nav.browse' },
  { name: 'search', label: 'nav.search' },
  { name: 'library', label: 'library.title' },
  { name: 'profile', label: 'profile.title' },
] as const

function target(name: string): { name: string; query?: Record<string, string> } {
  const needsAuth = name === 'library' || name === 'profile'
  if (needsAuth && !auth.isAuthenticated) {
    return { name: 'login', query: { redirect: route.fullPath } }
  }
  return { name }
}

/**
 * Which routes each tab OWNS — not just the route it links to.
 *
 * Highlighting only on an exact name match meant no tab was active on `player`, `podcast` or
 * `catalog` — the three routes users spend most of their time on. The bar went blank exactly when
 * it was most needed for orientation, which is the opposite of what a tab bar is for.
 *
 * Browse now has its own tab (#14): the hub plus the corpus indexes (catalogue, topic/people browse)
 * and show pages belong to it, since it is the destination that gathers them. Search owns only the
 * search route again.
 *
 * The player owns NOTHING, deliberately. You can reach an episode from any tab, so lighting one up
 * would assert a path the user may not have taken — and a wrong "you are here" is worse than none.
 */
const OWNED_ROUTES: Record<string, readonly string[]> = {
  home: ['home'],
  browse: ['browse', 'catalog', 'podcast', 'browse-topics', 'browse-people'],
  search: ['search'],
  library: ['library'],
  profile: ['profile'],
}

/** Highlight by the routes the tab OWNS, not the resolved target, so a gated tab still reads active. */
const isActive = (name: string): boolean =>
  OWNED_ROUTES[name]?.includes(String(route.name)) ?? false
</script>

<template>
  <nav
    class="fixed inset-x-0 bottom-0 z-40 border-t border-border bg-canvas sm:hidden"
    :aria-label="t('nav.primary')"
    data-testid="bottom-nav"
  >
    <ul class="mx-auto flex max-w-lg items-stretch justify-around pb-[env(safe-area-inset-bottom)]">
      <li v-for="tab in TABS" :key="tab.name" class="flex-1">
        <RouterLink
          :to="target(tab.name)"
          :data-testid="`bottom-nav-${tab.name}`"
          :aria-current="isActive(tab.name) ? 'page' : undefined"
          class="flex min-h-[3rem] flex-col items-center justify-center gap-0.5 py-2 text-[0.65rem] font-bold no-underline transition-colors"
          :class="isActive(tab.name) ? 'text-accent' : 'text-muted'"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5" aria-hidden="true">
            <template v-if="tab.name === 'home'">
              <path d="M3 10.5 12 3l9 7.5" /><path d="M5 9.5V21h14V9.5" />
            </template>
            <template v-else-if="tab.name === 'browse'">
              <circle cx="12" cy="12" r="10" />
              <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
            </template>
            <template v-else-if="tab.name === 'search'">
              <circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" />
            </template>
            <template v-else-if="tab.name === 'library'">
              <path d="m16 6 4 14" /><path d="M12 6v14" /><path d="M8 8v12" /><path d="M4 4v16" />
            </template>
            <template v-else>
              <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
            </template>
          </svg>
          {{ t(tab.label) }}
        </RouterLink>
      </li>
    </ul>
  </nav>
</template>
