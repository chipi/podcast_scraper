<script setup lang="ts">
/**
 * Synced transcript (RFC-099 §2 / UXS-011). The active segment gets the accent left-rule +
 * weight-600 treatment and is announced via an ARIA live region. Autoscroll keeps it in view
 * but backs off while the user is manually scrolling (re-enabled after idle) and respects
 * prefers-reduced-motion. Tapping a segment emits `seek` with its start time.
 */
import { computed, nextTick, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import type { Segment } from '../services/types'
import type { GroundedSpan } from '../player/insights'
import { quoteHighlight } from '../player/insights'
import { formatTime } from '../player/transcriptSync'

type Split = { pre: string; match: string; post: string }

const props = withDefaults(
  defineProps<{
    segments: Segment[]
    activeIndex: number
    /** segmentIndex → grounded insight whose quote lands here (highlight + tap-to-reveal). */
    grounded?: Record<number, GroundedSpan>
  }>(),
  { grounded: () => ({}) },
)
const emit = defineEmits<{
  (e: 'seek', start: number): void
  (e: 'insight', insightId: string): void
}>()
const { t } = useI18n()

function onSegmentClick(i: number, seg: Segment): void {
  emit('seek', seg.start)
  const g = props.grounded[i]
  if (g) emit('insight', g.insightId)
}

// Char-level highlight split per grounded segment (3.6); null → whole-segment underline fallback.
const highlights = computed<Record<number, Split | null>>(() => {
  const out: Record<number, Split | null> = {}
  for (const key of Object.keys(props.grounded)) {
    const i = Number(key)
    out[i] = quoteHighlight(props.segments[i]?.text ?? '', props.grounded[i].quote)
  }
  return out
})

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
          : grounded[i]
            ? 'border-l-2 border-grounded pl-3 -ml-3'
            : 'border-l-2 border-transparent pl-3 -ml-3'
      "
      :aria-label="grounded[i] ? t('player.groundedSegment') : undefined"
      @click="onSegmentClick(i, seg)"
    >
      <span
        v-if="speakerLabel(seg.speaker)"
        class="lp-kicker block mb-0.5"
      >{{ speakerLabel(seg.speaker) }}</span>
      <span class="flex gap-3">
        <span class="shrink-0 pt-0.5 font-mono text-xs tabular-nums" :class="grounded[i] ? 'text-grounded' : 'text-muted'">
          {{ grounded[i] ? '●' : formatTime(seg.start) }}
        </span>
        <span
          class="text-sm leading-relaxed"
          :class="i === activeIndex ? 'text-surface-foreground font-semibold' : 'text-muted'"
        >
          <!-- Char-level: underline only the matched quote phrase within the segment (3.6). -->
          <template v-if="grounded[i] && highlights[i]">{{ highlights[i]!.pre }}<span class="text-grounded underline decoration-grounded decoration-2 underline-offset-2">{{ highlights[i]!.match }}</span>{{ highlights[i]!.post }}</template>
          <!-- Fallback: whole-segment underline (grounded but quote not locatable in this segment). -->
          <span
            v-else
            :class="grounded[i] ? 'underline decoration-grounded decoration-2 underline-offset-2' : ''"
          >{{ seg.text }}</span>
        </span>
      </span>
    </button>
  </div>
</template>
