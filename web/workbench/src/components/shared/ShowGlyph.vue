<script setup lang="ts">
import { computed } from 'vue'
import { graphNodeTypeStyles } from '../../utils/colors'

/**
 * Small square glyph for a show/podcast in a list (Across shows, …). A rounded
 * SQUARE with the show initial in the shared Podcast colour — deliberately
 * distinct from the round person avatars ({@link PersonInitialAvatar}) so shows
 * and voices read apart at a glance. (Cover art would need feed-image wiring.)
 */
const props = defineProps<{ name: string; sizeClass?: string }>()

const initial = computed((): string => {
  const raw = (props.name ?? '').trim()
  const ch = raw.replace(/^[^\p{L}\p{N}]+/u, '')[0]
  return ch ? ch.toUpperCase() : '#'
})

const chrome = graphNodeTypeStyles.Podcast
</script>

<template>
  <span
    class="inline-flex shrink-0 items-center justify-center rounded font-semibold leading-none"
    :class="sizeClass ?? 'h-4 w-4 text-[9px]'"
    :style="{ backgroundColor: chrome.background, color: chrome.labelColor }"
    data-testid="show-glyph"
    aria-hidden="true"
    >{{ initial }}</span
  >
</template>
