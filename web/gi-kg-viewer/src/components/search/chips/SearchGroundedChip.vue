<script setup lang="ts">
/**
 * Grounded-only chip (Search v3 §S1 — Explore merge). Simple toggle: passes
 * ``grounded_only=true`` to /api/search when active.
 */
import { computed } from 'vue'
import { useSearchStore } from '../../../stores/search'

const search = useSearchStore()

const isActive = computed(() => Boolean(search.filters.groundedOnly))

const chipLabel = computed(() => (isActive.value ? 'Grounded ✓' : 'Grounded'))

function toggle(): void {
  search.filters.groundedOnly = !search.filters.groundedOnly
}
</script>

<template>
  <button
    type="button"
    class="inline-flex h-6 items-center rounded border px-2 text-[11px] leading-none transition-colors hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
    :class="
      isActive
        ? 'border-border font-medium text-surface-foreground'
        : 'border-border/70 text-muted'
    "
    data-testid="search-chip-grounded-only"
    :aria-pressed="isActive"
    aria-label="Grounded insights only"
    @click="toggle"
  >
    {{ chipLabel }}
  </button>
</template>
