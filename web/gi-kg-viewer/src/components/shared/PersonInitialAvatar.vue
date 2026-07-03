<script setup lang="ts">
import { computed } from 'vue'
import { graphNodeTypeStyles } from '../../utils/colors'

/**
 * Small circular letter avatar for a person shown in a list (co-speakers,
 * key voices, …). We have no person images, so the initial + the shared
 * person colour (matches the graph legend / node-detail header) gives each
 * name a consistent visual anchor. Defined once, reused app-wide.
 */
const props = defineProps<{ name: string; sizeClass?: string }>()

const initial = computed((): string => {
  const raw = (props.name ?? '').trim()
  // skip leading punctuation/emoji so "· Jay" → "J", "SPEAKER_03" → "S"
  const ch = raw.replace(/^[^\p{L}\p{N}]+/u, '')[0]
  return ch ? ch.toUpperCase() : '?'
})

const chrome = graphNodeTypeStyles.Entity_person
</script>

<template>
  <span
    class="inline-flex shrink-0 items-center justify-center rounded-full font-semibold leading-none"
    :class="sizeClass ?? 'h-4 w-4 text-[9px]'"
    :style="{ backgroundColor: chrome.background, color: chrome.labelColor }"
    data-testid="person-initial-avatar"
    aria-hidden="true"
    >{{ initial }}</span
  >
</template>
