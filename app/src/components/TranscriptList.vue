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
import { speakerLabel } from '../utils/format'

type Split = { pre: string; match: string; post: string }

const props = withDefaults(
  defineProps<{
    segments: Segment[]
    activeIndex: number
    /** segmentIndex → grounded insight whose quote lands here (highlight + tap-to-reveal). */
    grounded?: Record<number, GroundedSpan>
    /** Show the per-line "save highlight" affordance (auth-gated; off → transcript unchanged). */
    canCapture?: boolean
    /** Segment ids already captured as a span (drives the saved/toggle state). */
    savedSegmentIds?: Set<string>
  }>(),
  { grounded: () => ({}), canCapture: false, savedSegmentIds: () => new Set<string>() },
)
const emit = defineEmits<{
  (e: 'seek', start: number): void
  (e: 'insight', insightId: string): void
  (e: 'capture', segment: Segment): void
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

// Show the speaker name only at the START of a run — i.e. when this segment's speaker differs from
// the previous one. Avoids repeating the same name on every short segment of one continuous turn.
function showSpeaker(i: number): boolean {
  const s = props.segments[i]?.speaker ?? null
  if (!s) return false
  return i === 0 || (props.segments[i - 1]?.speaker ?? null) !== s
}
</script>

<template>
  <div class="overflow-y-auto" @scroll.passive="onUserScroll">
    <p aria-live="polite" class="sr-only">
      {{ activeIndex >= 0 ? segments[activeIndex]?.text : '' }}
    </p>
    <div
      v-for="(seg, i) in segments"
      :key="seg.id"
      class="group relative"
    >
      <button
        :ref="(el) => { if (el) items[i] = el as HTMLElement }"
        type="button"
        data-testid="seg"
        class="block w-full text-left py-2 transition-colors"
        :class="[
          i === activeIndex
            ? 'border-l-2 border-accent bg-overlay pl-3 -ml-3 rounded'
            : grounded[i]
              ? 'border-l-2 border-grounded pl-3 -ml-3'
              : 'border-l-2 border-transparent pl-3 -ml-3',
          canCapture ? 'pr-8' : '',
        ]"
        :aria-label="grounded[i] ? t('player.groundedSegment') : undefined"
        @click="onSegmentClick(i, seg)"
      >
        <span
          v-if="showSpeaker(i)"
          class="lp-speaker block mb-0.5"
        >{{ speakerLabel(seg.speaker) }}</span>
        <span class="flex gap-3">
          <span class="shrink-0 pt-0.5 font-mono text-xs tabular-nums" :class="grounded[i] ? 'text-grounded' : 'text-muted'">
            <span v-if="grounded[i]" aria-hidden="true" class="mr-0.5">●</span>{{ formatTime(seg.start) }}
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
      <!-- Save this line as a highlight (auth-gated). Quiet until row hover/focus; filled when saved. -->
      <button
        v-if="canCapture"
        type="button"
        class="absolute right-0 top-1.5 rounded-full p-1 opacity-0 transition focus-visible:opacity-100 group-hover:opacity-100"
        :class="savedSegmentIds.has(seg.id) ? 'text-accent opacity-100' : 'text-muted hover:text-accent'"
        :aria-pressed="savedSegmentIds.has(seg.id)"
        :aria-label="savedSegmentIds.has(seg.id) ? t('capture.savedLine') : t('capture.saveLine')"
        :title="savedSegmentIds.has(seg.id) ? t('capture.savedLine') : t('capture.saveLine')"
        @click="emit('capture', seg)"
      >
        <svg viewBox="0 0 24 24" :fill="savedSegmentIds.has(seg.id) ? 'currentColor' : 'none'" stroke="currentColor" stroke-width="2" class="h-4 w-4" aria-hidden="true">
          <path d="M6 3h12a1 1 0 0 1 1 1v17l-7-4-7 4V4a1 1 0 0 1 1-1z" />
        </svg>
      </button>
    </div>
  </div>
</template>
