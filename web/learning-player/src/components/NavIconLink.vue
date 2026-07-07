<script setup lang="ts">
/**
 * Header icon link (UXS-014) — one canonical icon-nav affordance: a round hit-area, an optional
 * count badge, and a hover/focus **tooltip** with the label (icons alone are ambiguous). The icon
 * is passed as the default slot (an inline `currentColor` SVG so it inherits theme colours).
 */
import { RouterLink } from 'vue-router'
import type { RouteLocationRaw } from 'vue-router'

defineProps<{ to: RouteLocationRaw; label: string; badge?: number }>()
</script>

<template>
  <RouterLink
    :to="to"
    :aria-label="label"
    class="group relative inline-flex h-9 w-9 items-center justify-center rounded-full text-muted transition-colors hover:bg-overlay hover:text-canvas-foreground focus-visible:text-canvas-foreground"
  >
    <slot />
    <span
      v-if="badge"
      class="absolute -right-0.5 -top-0.5 flex h-4 min-w-[1rem] items-center justify-center rounded-full bg-accent px-1 text-[10px] font-bold text-accent-foreground"
    >{{ badge }}</span>
    <span
      class="pointer-events-none absolute left-1/2 top-full z-50 mt-1.5 -translate-x-1/2 whitespace-nowrap rounded-md bg-elevated px-2 py-1 text-xs font-medium text-canvas-foreground opacity-0 shadow-xl transition-opacity duration-150 group-hover:opacity-100 group-focus-visible:opacity-100"
      role="tooltip"
    >{{ label }}</span>
  </RouterLink>
</template>
