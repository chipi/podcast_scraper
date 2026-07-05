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
  // Up to two initials from the first two words ("Katie Martin" → "KM",
  // "jay powell" → "JP"); single-word / speaker labels keep one ("Katie" → "K").
  const words = raw
    .split(/[\s.]+/)
    .map((w) => w.replace(/^[^\p{L}\p{N}]+/u, ''))
    .filter(Boolean)
  if (words.length === 0) return '?'
  const a = words[0][0] ?? ''
  const b = words.length > 1 ? (words[1][0] ?? '') : ''
  return (a + b).toUpperCase() || '?'
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
