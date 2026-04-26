<script setup lang="ts">
/**
 * More chip (#671) — opens the slimmed Advanced search dialog (Feed
 * substring, Speaker, Embedding model, Grounded only, Merge KG surfaces).
 * Active when any of those values deviate from the default.
 */
import { computed } from 'vue'
import { useSearchStore } from '../../../stores/search'

const emit = defineEmits<{ open: [] }>()

const search = useSearchStore()

const isActive = computed(() => {
  const f = search.filters
  return Boolean(
    f.groundedOnly
      || f.feed.trim()
      || f.speaker.trim()
      || f.embeddingModel.trim()
      || !f.dedupeKgSurfaces,
  )
})

const activeCount = computed(() => {
  const f = search.filters
  let n = 0
  if (f.groundedOnly) n += 1
  if (f.feed.trim()) n += 1
  if (f.speaker.trim()) n += 1
  if (f.embeddingModel.trim()) n += 1
  if (!f.dedupeKgSurfaces) n += 1
  return n
})

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
    data-testid="search-chip-more"
    aria-haspopup="dialog"
    aria-label="More search filters"
    @click="emit('open')"
  >
    {{ chipLabel }}
  </button>
</template>
