<script setup lang="ts">
/**
 * Synced transcript (RFC-099 §2 / UXS-011). Segments are grouped into **paragraphs** — a speaker
 * turn (broken every ~TIMESTAMP_GAP_S in a long run) — and the words flow as continuous prose rather
 * than one row per fragment. Each paragraph carries a single leading speaker + timestamp; inside it,
 * the active segment is highlighted and tap-to-seek still works per segment. Grounded segments
 * underline their supporting quote and reveal the insight on tap. Capture (auth-gated) saves a
 * selected phrase, or the whole paragraph when nothing is selected.
 */
import { computed, nextTick, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import type { Segment } from '../services/types'
import type { GroundedSpan } from '../player/insights'
import { quoteHighlight } from '../player/insights'
import { formatTime } from '../player/transcriptSync'
import { selectionSubRange, spanFromParagraph, type ParagraphSpan } from '../player/transcriptCapture'
import { speakerLabel } from '../utils/format'

type Split = { pre: string; match: string; post: string }
interface Paragraph {
  key: string
  speaker: string | null
  showSpeaker: boolean
  start: number
  hasGrounded: boolean
  indices: number[]
}

const props = withDefaults(
  defineProps<{
    segments: Segment[]
    activeIndex: number
    /** segmentIndex → grounded insight whose quote lands here (highlight + tap-to-reveal). */
    grounded?: Record<number, GroundedSpan>
    /** Show the per-paragraph "save highlight" affordance (auth-gated; off → transcript unchanged). */
    canCapture?: boolean
    /** Segment ids already captured (drives the saved state of the paragraph's save control). */
    savedSegmentIds?: Set<string>
  }>(),
  { grounded: () => ({}), canCapture: false, savedSegmentIds: () => new Set<string>() },
)
const emit = defineEmits<{
  (e: 'seek', start: number): void
  (e: 'insight', insightId: string): void
  (e: 'capture', span: ParagraphSpan): void
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

// Show the speaker name only at the START of a run — i.e. when this segment's speaker differs from
// the previous one. Avoids repeating the same name within one continuous turn.
function showSpeaker(i: number): boolean {
  const s = props.segments[i]?.speaker ?? null
  if (!s) return false
  return i === 0 || (props.segments[i - 1]?.speaker ?? null) !== s
}

// A paragraph break starts at each speaker turn, then inside a long run only at a *sentence
// boundary* once ~TIMESTAMP_GAP_S has passed — so a monologue / undiarized transcript still gets
// readable, anchored paragraphs without ever splitting in the middle of a sentence.
const TIMESTAMP_GAP_S = 25
const endsSentence = (text: string): boolean => /[.!?]["')\]]?\s*$/.test(text ?? '')
const breakAt = computed<Set<number>>(() => {
  const set = new Set<number>()
  let lastShown = -Infinity
  let prevSpeaker: string | null | undefined
  props.segments.forEach((seg, i) => {
    const speaker = seg.speaker ?? null
    const turnStart = i === 0 || speaker !== prevSpeaker
    const longEnough = seg.start - lastShown >= TIMESTAMP_GAP_S
    const afterSentence = i > 0 && endsSentence(props.segments[i - 1]?.text ?? '')
    if (turnStart || (longEnough && afterSentence)) {
      set.add(i)
      lastShown = seg.start
    }
    prevSpeaker = speaker
  })
  return set
})

const paragraphs = computed<Paragraph[]>(() => {
  const out: Paragraph[] = []
  props.segments.forEach((seg, i) => {
    if (out.length === 0 || breakAt.value.has(i)) {
      out.push({
        key: seg.id,
        speaker: seg.speaker ?? null,
        showSpeaker: showSpeaker(i),
        start: seg.start,
        hasGrounded: false,
        indices: [],
      })
    }
    const p = out[out.length - 1]
    p.indices.push(i)
    if (props.grounded[i]) p.hasGrounded = true
  })
  return out
})

function paraSaved(p: Paragraph): boolean {
  return p.indices.some((i) => props.savedSegmentIds.has(props.segments[i].id))
}

// Per-paragraph text elements, so a "save" can read the live selection inside the paragraph and
// capture the exact phrase (FR1.2) — or the whole paragraph when nothing is selected.
const paraEls = ref<HTMLElement[]>([])
function onCaptureParagraph(pi: number, p: Paragraph): void {
  const el = paraEls.value[pi]
  const sub = el ? selectionSubRange(el) : null
  emit('capture', spanFromParagraph(p.indices.map((i) => props.segments[i]), sub))
}

// Per-segment elements for autoscroll-to-active.
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
</script>

<template>
  <div class="overflow-y-auto" @scroll.passive="onUserScroll">
    <p aria-live="polite" class="sr-only">
      {{ activeIndex >= 0 ? segments[activeIndex]?.text : '' }}
    </p>
    <div v-for="(para, pi) in paragraphs" :key="para.key" class="group relative py-2">
      <span v-if="para.showSpeaker" class="lp-speaker mb-0.5 block">{{ speakerLabel(para.speaker) }}</span>
      <div class="flex items-start gap-3" :class="canCapture ? 'pr-8' : ''">
        <!-- One timestamp per paragraph (seeks to its start). -->
        <button
          type="button"
          class="shrink-0 pt-0.5 font-mono text-xs tabular-nums"
          :class="para.hasGrounded ? 'text-grounded' : 'text-muted'"
          :aria-label="t('player.jumpToTime', { time: formatTime(para.start) })"
          @click="emit('seek', para.start)"
        ><span v-if="para.hasGrounded" aria-hidden="true" class="mr-0.5">●</span>{{ formatTime(para.start) }}</button>

        <!-- Flowing paragraph: segments are inline, the active one highlighted, each tap-to-seek. -->
        <p
          :ref="(el) => { if (el) paraEls[pi] = el as HTMLElement }"
          class="min-w-0 flex-1 select-text text-sm leading-relaxed text-muted"
        ><template v-for="i in para.indices" :key="segments[i].id"><span
            :ref="(el) => { if (el) items[i] = el as HTMLElement }"
            data-testid="seg"
            class="cursor-pointer rounded transition-colors"
            :class="[
              i === activeIndex ? 'bg-overlay font-semibold text-surface-foreground' : '',
              grounded[i] ? 'text-grounded' : '',
            ]"
            :aria-label="grounded[i] ? t('player.groundedSegment') : undefined"
            @click="onSegmentClick(i, segments[i])"
          ><template v-if="grounded[i] && highlights[i]">{{ highlights[i]!.pre }}<span class="underline decoration-grounded decoration-2 underline-offset-2">{{ highlights[i]!.match }}</span>{{ highlights[i]!.post }}</template><span
              v-else
              :class="grounded[i] ? 'underline decoration-grounded decoration-2 underline-offset-2' : ''"
            >{{ segments[i].text }}</span></span>{{ ' ' }}</template></p>

        <!-- Save the paragraph (or the selected phrase). Quiet until hover/focus; filled when saved. -->
        <button
          v-if="canCapture"
          type="button"
          class="absolute right-0 top-1 rounded-full p-1 opacity-0 transition focus-visible:opacity-100 group-hover:opacity-100"
          :class="paraSaved(para) ? 'text-accent opacity-100' : 'text-muted hover:text-accent'"
          :aria-pressed="paraSaved(para)"
          :aria-label="paraSaved(para) ? t('capture.savedLine') : t('capture.saveLine')"
          :title="paraSaved(para) ? t('capture.savedLine') : t('capture.saveLine')"
          @click="onCaptureParagraph(pi, para)"
        >
          <svg viewBox="0 0 24 24" :fill="paraSaved(para) ? 'currentColor' : 'none'" stroke="currentColor" stroke-width="2" class="h-4 w-4" aria-hidden="true">
            <path d="M6 3h12a1 1 0 0 1 1 1v17l-7-4-7 4V4a1 1 0 0 1 1-1z" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>
