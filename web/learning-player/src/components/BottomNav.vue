<script setup lang="ts">
/**
 * Mobile bottom tab bar (#1594).
 *
 * All navigation lived in a top header, so on a phone every move was a top-of-screen reach — on an
 * app whose primary Playwright project is a Pixel 7. The plumbing was always phone-first (`dvh`,
 * safe areas, sticky transport, touch-first rails); the layout idiom was not.
 *
 * THREE destinations, each answering a different question: **Home** (what should I listen to),
 * **Discovery** (what is out there), **Library** (my saved things).
 *
 * It shipped with four. Profile left for the masthead avatar (2026-09-09) and Search left for
 * Discovery (2026-09-20) — see the notes on `TABS` and `OWNED_ROUTES` below for why each moved and
 * what replaced it. Both are still one tap away; neither needs a slot in the scarcest strip on the
 * screen.
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
import { computed } from 'vue'
import { useAuthStore } from '../stores/auth'
import { useResurfacingStore } from '../stores/resurfacing'
import { ownsRoute } from '../utils/navOwnership'

const { t } = useI18n()
const route = useRoute()
const auth = useAuthStore()
const resurfacing = useResurfacingStore()

/**
 * The Library tab's due-count badge (#1592).
 *
 * The badge prop already existed on `NavIconLink`, but that component is `sm:` and up — desktop
 * only — so wiring it there alone would have lit up on the surface phones never see. This is the
 * primary platform, so the count has to render here too, from the same store, or the two navs can
 * disagree about how much is waiting.
 */
const dueCount = computed(() => resurfacing.dueCount)

/** The count belongs in the LABEL: a bare number beside an icon is a number with no noun. */
function tabLabel(name: string): string {
  const base = t(TABS.find((tb) => tb.name === name)!.label)
  return name === 'library' && dueCount.value ? `${base} (${dueCount.value})` : base
}

/**
 * Library requires auth. It stays VISIBLE signed-out and routes to sign-in (#1590): hiding it would
 * hide the capability from exactly the visitors deciding whether to sign up.
 *
 * Profile is NO LONGER a tab (operator 2026-09-09) — it moved to the masthead avatar, so the same
 * destination isn't reachable from two navs at once.
 *
 * Search is NO LONGER a tab either (operator 2026-09-20) — it is part of Discovery. It keeps three
 * entry points, which is MORE than the two it had as a tab: the masthead magnifier (visible at every
 * width, so reachable from any screen — the #1588 requirement), Discovery's own search box, and
 * Home's "Ask" box. Removing a nav entry for search without the masthead icon would re-open #1588,
 * which existed precisely because search had one entry point and was unreachable elsewhere.
 */
const TABS = [
  { name: 'home', label: 'nav.home' },
  { name: 'browse', label: 'nav.browse' },
  { name: 'library', label: 'library.title' },
] as const

function target(name: string): { name: string; query?: Record<string, string> } {
  const needsAuth = name === 'library'
  if (needsAuth && !auth.isAuthenticated) {
    return { name: 'login', query: { redirect: route.fullPath } }
  }
  // A badge is a promise about what you will find. Library opens on Saved, so tapping a "5" put you
  // on a screen with no 5 anywhere on it and no hint that Revisit — one of four sub-tabs — was what
  // the number meant. Land on the counted tab while there is something to count; once it clears,
  // Library opens where it always did.
  if (name === 'library' && dueCount.value) return { name, query: { tab: 'revisit' } }
  return { name }
}

/* Route ownership is shared with the desktop masthead — see utils/navOwnership.ts. */

/** Highlight by the routes the tab OWNS, not the resolved target, so a gated tab still reads active. */
const isActive = (name: string): boolean => ownsRoute(name, route.name as string | undefined)
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
          :aria-label="tabLabel(tab.name)"
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
            <template v-else-if="tab.name === 'library'">
              <path d="m16 6 4 14" /><path d="M12 6v14" /><path d="M8 8v12" /><path d="M4 4v16" />
            </template>
            <template v-else>
              <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
            </template>
          </svg>
          <span class="relative">
            {{ t(tab.label) }}
            <!-- aria-hidden: the count is already in the link's aria-label, said once and with a
                 noun attached. -->
            <span
              v-if="tab.name === 'library' && dueCount"
              aria-hidden="true"
              data-testid="bottom-nav-badge"
              class="absolute -right-2.5 -top-2 flex h-3.5 min-w-[0.875rem] items-center justify-center rounded-full bg-accent px-1 text-[9px] font-bold text-accent-foreground"
            >{{ dueCount }}</span>
          </span>
        </RouterLink>
      </li>
    </ul>
  </nav>
</template>
