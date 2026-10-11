<script setup lang="ts">
/**
 * "▶ 15 min on this" — the way into a topic's, theme's or storyline's reel from its own card or
 * page (operator 2026-10-11): fifteen minutes of clips about it across episodes (`TopicReelView`).
 * It sits on the subject page's own "listen" row, beside "Learn this in 7 days", and the two look
 * different on purpose (operator 2026-10-11): this one PLAYS now — a filled accent pill with ▶ —
 * the other starts a week-long path — an outlined pill with a calendar mark.
 *
 * It appears only once the reel is known to have at least two clips, so it never opens an empty
 * reel; a subject nobody speaks to has no door.
 */
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { getTopicReel } from '../services/api'

const props = defineProps<{ subjectId: string; label: string }>()
const { t } = useI18n()
const MINUTES = 15
const MIN_CLIPS = 2
const ready = ref(false)

let latest = 0
async function check(): Promise<void> {
  const run = ++latest
  ready.value = false
  const reel = await getTopicReel(props.subjectId, MINUTES).catch(() => null)
  if (run === latest) ready.value = (reel?.clips.length ?? 0) >= MIN_CLIPS
}
watch(() => props.subjectId, check, { immediate: true })
</script>

<template>
  <RouterLink
    v-if="ready"
    :to="{ name: 'topic-reel', params: { id: subjectId }, query: { minutes: String(MINUTES) } }"
    class="lp-tap inline-flex h-9 shrink-0 items-center gap-1.5 rounded-full bg-accent px-4 text-sm font-bold text-accent-foreground no-underline transition hover:opacity-90"
    :aria-label="t('reel.doorLabel', { minutes: MINUTES, topic: label })"
    data-testid="reel-door"
  >
    <span aria-hidden="true">▶ {{ t('reel.door', { minutes: MINUTES }) }}</span>
  </RouterLink>
</template>
