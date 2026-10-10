<script setup lang="ts">
/**
 * Every moment of the reel, below the transport while Moments plays (operator 2026-10-10: the
 * artwork shows the current moment; the list of all of them sits under the transport, where the
 * transcript is otherwise). Played ones ticked, the current one marked, any one a tap away.
 */
import { useI18n } from 'vue-i18n'
import type { ReelMoment } from '../stores/player'
import { formatTime } from '../player/transcriptSync'
import { minutesValue, reelSeconds } from '../services/moments'

const props = defineProps<{ moments: ReelMoment[]; index: number; done: boolean }>()
const emit = defineEmits<{ (e: 'go', index: number): void }>()
const { t } = useI18n()

function state(i: number): 'done' | 'now' | 'todo' {
  if (props.done || i < props.index) return 'done'
  return i === props.index ? 'now' : 'todo'
}
</script>

<template>
  <section class="mt-6" data-testid="moments-index" :aria-label="t('moments.index')">
    <h2 class="font-display text-xl font-extrabold text-accent">{{ t('moments.title') }}</h2>
    <p class="text-sm text-muted" data-testid="moments-episode">
      {{ t('moments.subtitle', { count: moments.length, length: t('moments.minutes', { n: minutesValue(reelSeconds(moments)) }) }) }}
    </p>
    <ol class="mt-2 border-t border-border">
      <li v-for="(m, i) in moments" :key="m.insightId">
        <button
          type="button"
          class="grid w-full grid-cols-[1rem_2.75rem_minmax(0,1fr)] items-baseline gap-2 border-b border-border py-2.5 text-left text-sm"
          :class="state(i) === 'now' && !done ? 'text-canvas-foreground' : 'text-muted'"
          :aria-current="state(i) === 'now' && !done ? 'true' : undefined"
          data-testid="moments-index-item"
          @click="emit('go', i)"
        >
          <span class="font-mono text-xs" :class="state(i) === 'now' && !done ? 'text-accent' : ''">
            {{ state(i) === 'done' ? '✓' : state(i) === 'now' ? '▶' : i + 1 }}
          </span>
          <span class="font-mono text-xs tabular-nums">{{ formatTime(m.startMs / 1000) }}</span>
          <span class="line-clamp-2">{{ m.text }}</span>
        </button>
      </li>
    </ol>
  </section>
</template>
