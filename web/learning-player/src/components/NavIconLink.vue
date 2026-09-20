<script setup lang="ts">
/**
 * Header icon link (UXS-014) — one canonical icon-nav affordance: a round hit-area, an optional
 * count badge, and a hover/focus **tooltip** with the label (icons alone are ambiguous). The icon
 * is passed as the default slot (an inline `currentColor` SVG so it inherits theme colours).
 */
import { computed } from 'vue'
import { RouterLink, useRoute } from 'vue-router'
import type { RouteLocationRaw } from 'vue-router'
import { ownsRoute } from '../utils/navOwnership'

const props = defineProps<{ to: RouteLocationRaw; label: string; badge?: number
  /**
   * Which nav destination this icon represents, for highlighting (`utils/navOwnership.ts`).
   *
   * Without it this fell back to `RouterLink`'s exact-active, which answers a DIFFERENT question:
   * "is this the current URL" rather than "does this destination own the current screen". On
   * `/search` that lit Search here while the phone bar lit Discovery — the same URL answered two
   * ways depending on window width. Omit it and the link simply never reads active.
   */
  owns?: string
  /**
   * Which edge the hover tooltip hangs from.
   *
   * `center` is right for every item with room on both sides. The LAST item in a right-aligned rail
   * has none: its label is `whitespace-nowrap` and centred with `-translate-x-1/2`, so half of it
   * sits past the item's right edge — and the profile item is labelled with the USER'S NAME, which
   * can be arbitrarily long. That pushed the page 26px wider than the viewport on every desktop
   * surface, because the nav is global chrome. Anchoring the last one to its right edge sends the
   * overflow inward, where there is space.
   */
  tooltipAlign?: 'center' | 'end'
}>()

/**
 * The badge is decoration to the a11y tree, and its meaning goes in the LABEL instead (#1592).
 *
 * A bare "3" beside an icon is a number with no noun. Screen readers announced only "Library", so
 * the one thing the badge exists to say — that something is waiting — was the one thing not
 * conveyed. `aria-hidden` on the pill plus a counted label says it once, properly.
 */
const ariaLabel = computed(() =>
  props.badge ? `${props.label} (${props.badge})` : props.label,
)

const route = useRoute()
const active = computed(() =>
  props.owns ? ownsRoute(props.owns, route.name as string | undefined) : false,
)
</script>

<template>
  <RouterLink
    :to="to"
    :aria-label="ariaLabel"
    :aria-current="active ? 'page' : undefined"
    class="group relative inline-flex h-9 w-9 items-center justify-center rounded-full transition-colors hover:bg-overlay hover:text-canvas-foreground focus-visible:text-canvas-foreground"
    :class="active ? 'text-accent' : 'text-muted'"
  >
    <slot />
    <span
      v-if="badge"
      aria-hidden="true"
      data-testid="nav-badge"
      class="absolute -right-0.5 -top-0.5 flex h-4 min-w-[1rem] items-center justify-center rounded-full bg-overlay px-1 text-[10px] font-bold text-canvas-foreground"
    >{{ badge }}</span>
    <span
      :class="[
        'pointer-events-none absolute top-full z-50 mt-1.5 whitespace-nowrap',
        tooltipAlign === 'end' ? 'right-0' : 'left-1/2 -translate-x-1/2',
        'rounded-md bg-elevated px-2 py-1 text-xs font-medium text-canvas-foreground opacity-0 shadow-xl transition-opacity duration-150 group-hover:opacity-100 group-focus-visible:opacity-100',
      ]"
      role="tooltip"
    >{{ label }}</span>
  </RouterLink>
</template>
