<script setup lang="ts">
/**
 * Leave the clips for the episode itself (operator 2026-10-15): one block, the same on an episode's
 * Moments, a short listen and a learning day's end card. "Keep listening" said neither where nor
 * from when, so the two ways in are named: on from the clip (its time in the label), or the whole
 * episode from 0:00.
 */
import { useI18n } from 'vue-i18n'
import { formatTime } from '../player/transcriptSync'

defineProps<{
  /** Where "Play on" picks up, in seconds. */
  time: number
  /** Test ids: the block, "Play on" and "Play from the start". */
  testid: string
  keepTestid: string
  startTestid: string
  /** Centred under a centred transport (the short listen's card). */
  centered?: boolean
}>()
const emit = defineEmits<{ (e: 'play-on'): void; (e: 'from-start'): void }>()
const { t } = useI18n()
</script>

<template>
  <div :class="centered ? 'text-center' : ''" :data-testid="testid">
    <p class="text-xs text-muted">{{ t('listenChoice.heading') }}</p>
    <div class="mt-0.5 flex flex-wrap gap-x-4" :class="centered ? 'justify-center' : ''">
      <button type="button" class="lp-choice text-accent" :data-testid="keepTestid" @click="emit('play-on')">
        {{ t('listenChoice.playOn', { time: formatTime(time) }) }}
      </button>
      <button type="button" class="lp-choice text-canvas-foreground" :data-testid="startTestid" @click="emit('from-start')">
        {{ t('listenChoice.fromStart') }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.lp-choice {
  display: inline-flex;
  min-height: 44px;
  align-items: center;
  font-size: 0.875rem;
  font-weight: 700;
}
.lp-choice:focus-visible {
  outline: 2px solid var(--lp-accent);
  outline-offset: 2px;
}
</style>
