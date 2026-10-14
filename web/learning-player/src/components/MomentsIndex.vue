<script setup lang="ts">
/**
 * Every moment of the reel, below the transport while Moments plays (operator 2026-10-10: the
 * artwork shows the current moment; the list of all of them sits under the transport, where the
 * transcript is otherwise). The current one marked, any one a tap away, and each one heard to its
 * end ticked — heard ever, now or on an earlier visit (operator 2026-10-14), not merely passed: a
 * moment skipped keeps its number.
 */
import { useI18n } from 'vue-i18n'
import type { ReelMoment } from '../stores/player'
import { formatTime } from '../player/transcriptSync'
import { minutesValue, reelSeconds } from '../services/moments'

const props = withDefaults(
  defineProps<{ moments: ReelMoment[]; index: number; done: boolean; heard?: ReadonlySet<string> }>(),
  { heard: () => new Set<string>() },
)
const emit = defineEmits<{ (e: 'go', index: number): void }>()
const { t } = useI18n()

function state(i: number): 'heard' | 'now' | 'todo' {
  if (i === props.index && !props.done) return 'now'
  return props.heard.has(props.moments[i]!.insightId) ? 'heard' : 'todo'
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
          :class="state(i) === 'now' ? 'text-canvas-foreground' : 'text-muted'"
          :aria-current="state(i) === 'now' ? 'true' : undefined"
          data-testid="moments-index-item"
          @click="emit('go', i)"
        >
          <span
            class="font-mono text-xs"
            :class="state(i) === 'now' ? 'text-accent' : state(i) === 'heard' ? 'text-learn' : ''"
            :data-state="state(i)"
            data-testid="moments-index-mark"
          >
            {{ state(i) === 'heard' ? '✓' : state(i) === 'now' ? '▶' : i + 1 }}
          </span>
          <span class="font-mono text-xs tabular-nums">{{ formatTime(m.startMs / 1000) }}</span>
          <span class="line-clamp-2">{{ m.text }}</span>
        </button>
      </li>
    </ol>
  </section>
</template>
