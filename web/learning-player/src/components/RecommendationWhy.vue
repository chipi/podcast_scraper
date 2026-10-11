<script setup lang="ts">
/**
 * Why a recommendation is here (ADR-163) — the one rendering every Discover rail uses.
 *
 * Extractive only: the reason line is written HERE from the server's parts (kind, subject, the
 * listener's capture), and the quote is the transcript's own words with where they are said. When
 * nothing grounds the recommendation (`quote: null`) no quote is shown — never a stand-in.
 * "Play from" opens the player playing at the quote (ADR-163 §8: no separate clip player). The
 * serendipity share (§7) is labelled.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { Reason } from '../services/types'
import { formatTime } from '../player/transcriptSync'

const props = defineProps<{ why: Reason }>()
/** "Play from" was tapped — the rail reports it (useRailTelemetry) before the player opens. */
const emit = defineEmits<{ (e: 'play'): void }>()
const { t } = useI18n()

const line = computed(() => {
  const w = props.why
  const subject = w.subject?.label ?? ''
  switch (w.kind) {
    case 'rising_topic':
      return t('why.rising_topic', { subject })
    case 'most_engaged':
      return t('why.most_engaged')
    case 'captured':
      if (w.capture?.kind === 'highlight' && w.capture.text) return t('why.captured_highlight', { capture: w.capture.text })
      if (w.capture?.kind === 'note' && w.capture.text) return t('why.captured_note', { capture: w.capture.text })
      if (w.capture?.kind === 'save' && subject) return t('why.captured_save', { subject })
      return t('why.captured_plain')
    case 'thread':
      return t('why.thread', { subject })
    case 'person':
      return t('why.person', { subject })
    case 'another_view':
      return t('why.another_view', { subject })
  }
  return ''
})

const playTo = computed(() => {
  const q = props.why.quote
  if (!q || q.start_ms == null) return null
  return {
    name: 'player',
    params: { slug: q.episode_slug },
    query: { t: String(Math.floor(q.start_ms / 1000)), play: '1' },
  }
})
const time = computed(() => {
  const ms = props.why.quote?.start_ms
  return ms == null ? '' : formatTime(ms / 1000)
})
</script>

<template>
  <div class="flex flex-col gap-1.5" data-testid="why">
    <p class="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs font-bold text-muted">
      <span data-testid="why-reason">{{ line }}</span>
    </p>
    <figure v-if="why.quote" class="m-0" data-testid="why-quote">
      <blockquote class="m-0 text-sm leading-snug text-canvas-foreground">“{{ why.quote.text }}”</blockquote>
      <figcaption class="mt-1 flex flex-wrap items-center gap-x-3 text-xs text-muted">
        <span v-if="why.quote.speaker" data-testid="why-speaker">{{ why.quote.speaker }}</span>
        <RouterLink
          v-if="playTo"
          :to="playTo"
          class="lp-tap relative z-30 font-bold text-accent"
          :aria-label="t('why.playFromLabel', { quote: why.quote.text, time })"
          data-testid="why-play-from"
          @click="emit('play')"
        >▶ {{ t('why.playFrom', { time }) }}</RouterLink>
      </figcaption>
    </figure>
  </div>
</template>
