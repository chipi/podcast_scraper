<script setup lang="ts">
/**
 * Synced transcript (RFC-099 §2 / UXS-011). The active segment gets the accent left-rule +
 * weight-600 treatment and is announced via an ARIA live region. Autoscroll keeps it in view
 * but backs off while the user is manually scrolling (re-enabled after idle) and respects
 * prefers-reduced-motion. Tapping a segment emits `seek` with its start time.
 */
import { nextTick, ref, watch } from 'vue'
import type { Segment } from '../services/types'
import { formatTime } from '../player/transcriptSync'

const props = defineProps<{ segments: Segment[]; activeIndex: number }>()
const emit = defineEmits<{ (e: 'seek', start: number): void }>()

const items = ref<HTMLElement[]>([])
let userScrolling = false
let idleTimer: ReturnType<typeof setTimeout> | undefined

const reduceMotion =
  typeof window !== 'undefined' &&
  typeof window.matchMedia === 'function' &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches

function onUserScroll(): void {
  userScrolling = true
  if (idleTimer) clearTimeout(idleTimer)
  idleTimer = setTimeout(() => {
    userScrolling = false
  }, 5000)
}

watch(
  () => props.activeIndex,
  async (idx) => {
    if (idx < 0 || userScrolling) return
    await nextTick()
    items.value[idx]?.scrollIntoView?.({
      behavior: reduceMotion ? 'auto' : 'smooth',
      block: 'center',
    })
  },
)

function speakerLabel(s: string | null): string | null {
  if (!s) return null
  return s.startsWith('person:') ? s.slice('person:'.length).replace(/-/g, ' ') : s
}
</script>

<template>
  <div class="overflow-y-auto" @scroll.passive="onUserScroll">
    <p aria-live="polite" class="sr-only">
      {{ activeIndex >= 0 ? segments[activeIndex]?.text : '' }}
    </p>
    <button
      v-for="(seg, i) in segments"
      :key="seg.id"
      :ref="(el) => { if (el) items[i] = el as HTMLElement }"
      type="button"
      class="block w-full text-left py-2 transition-colors"
      :class="
        i === activeIndex
          ? 'border-l-2 border-accent bg-overlay pl-3 -ml-3 rounded'
          : 'border-l-2 border-transparent pl-3 -ml-3'
      "
      @click="emit('seek', seg.start)"
    >
      <span
        v-if="speakerLabel(seg.speaker)"
        class="lp-kicker block mb-0.5"
      >{{ speakerLabel(seg.speaker) }}</span>
      <span class="flex gap-3">
        <span class="font-mono text-xs text-muted shrink-0 pt-0.5 tabular-nums">
          {{ formatTime(seg.start) }}
        </span>
        <span
          class="text-sm leading-relaxed"
          :class="i === activeIndex ? 'text-surface-foreground font-semibold' : 'text-muted'"
        >{{ seg.text }}</span>
      </span>
    </button>
  </div>
</template>
