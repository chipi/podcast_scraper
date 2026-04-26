<script setup lang="ts">
/**
 * More chip (#671) — opens the slimmed Advanced explore dialog
 * (Grounded only, Strict schema, Limit, Sort, Min confidence). Active
 * count reflects values that deviate from the API default.
 */
import { computed } from 'vue'
import { useExploreStore } from '../../../stores/explore'

const emit = defineEmits<{ open: [] }>()

const ex = useExploreStore()

const activeCount = computed(() => {
  const f = ex.filters
  let n = 0
  if (f.groundedOnly) n += 1
  if (f.strict) n += 1
  const lim = Number(f.limit)
  if (Number.isFinite(lim) && lim !== 50) n += 1
  if (f.sortBy !== 'confidence') n += 1
  if (f.minConfidence.trim()) n += 1
  return n
})

const isActive = computed(() => activeCount.value > 0)

const chipLabel = computed(() => {
  if (!isActive.value) return 'More ▾'
  return `More: ${activeCount.value} ▾`
})
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
    data-testid="explore-chip-more"
    aria-haspopup="dialog"
    aria-label="More explore filters"
    @click="emit('open')"
  >
    {{ chipLabel }}
  </button>
</template>
