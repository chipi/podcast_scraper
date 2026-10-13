<script setup lang="ts">
/**
 * Discover's "Short listens" row (ADR-163 §8 amendment; goal-shaped listening): a few topics, themes
 * and storylines, each a way into up to fifteen minutes of clips about it across episodes
 * (`TopicReelView`). Under Mine, the listener's own; under Everyone, the rising ones — and only
 * those whose reel has at least two clips (the server checks), so a chip never opens an empty reel.
 * Each chip wears its kind's colours (`kindPill`), as topics, themes and storylines do on episode
 * notes. Hidden when there are none.
 */
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { getReelTopics, type ReelTopic } from '../services/api'
import { kindPill } from '../utils/interests'
import KindWord from './KindWord.vue'
import { reelMinutes } from '../utils/reelLength'
import { track } from '../services/analytics'

const props = defineProps<{ scope: 'mine' | 'corpus' }>()
const { t } = useI18n()
const topics = ref<ReelTopic[]>([])
const MINUTES = 15

let latest = 0
async function load(): Promise<void> {
  const run = ++latest
  const got = await getReelTopics(props.scope).catch(() => [] as ReelTopic[])
  if (run === latest) topics.value = got
}
watch(() => props.scope, load, { immediate: true })
</script>

<template>
  <section v-if="topics.length" class="mt-7" data-testid="reel-topics">
    <!-- Each pill says what it actually runs (operator 2026-10-11: "real length everywhere"); the
         heading promises no number. -->
    <h2 class="lp-section mb-3">{{ t('reel.row') }}</h2>
    <ul class="flex flex-wrap gap-2">
      <li v-for="topic in topics" :key="topic.subject_id">
        <RouterLink
          :to="{ name: 'topic-reel', params: { id: topic.subject_id }, query: { minutes: String(MINUTES) } }"
          class="lp-tap inline-flex items-center rounded-full px-3 py-1.5 text-xs font-semibold transition hover:opacity-80"
          :class="kindPill(topic.subject_kind)"
          :data-kind="topic.subject_kind"
          :aria-label="t('reel.chipLabel', { minutes: reelMinutes(topic.total_seconds), topic: topic.subject_label }, reelMinutes(topic.total_seconds))"
          data-testid="reel-topic-chip"
          @click="track('reel_open', { source: 'discover_row', kind: topic.subject_kind })"
        ><span aria-hidden="true">▶ <KindWord :kind="topic.subject_kind" />{{ topic.subject_label }}</span></RouterLink>
      </li>
    </ul>
  </section>
</template>
