<script setup lang="ts">
/**
 * Discover's "Minutes on" row (ADR-163 §8 amendment; goal-shaped listening): a few topics, themes
 * and storylines, each a way into fifteen minutes of clips about it across episodes
 * (`TopicReelView`). Under Mine, the listener's own; under Everyone, the rising ones — and only
 * those whose reel has at least two clips (the server checks), so a chip never opens an empty reel.
 * Each chip wears its kind's colours (`kindPill`), as topics, themes and storylines do on episode
 * notes. Hidden when there are none.
 */
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { getReelTopics, type ReelTopic } from '../services/api'
import { kindPill } from '../utils/interests'

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
    <h2 class="lp-section mb-3">{{ t('reel.row') }}</h2>
    <ul class="flex flex-wrap gap-2">
      <li v-for="topic in topics" :key="topic.subject_id">
        <RouterLink
          :to="{ name: 'topic-reel', params: { id: topic.subject_id }, query: { minutes: String(MINUTES) } }"
          class="lp-tap inline-flex min-h-[44px] items-center rounded-full px-4 text-sm font-semibold transition hover:opacity-80"
          :class="kindPill(topic.subject_kind)"
          :data-kind="topic.subject_kind"
          data-testid="reel-topic-chip"
        >▶ {{ t('reel.chip', { minutes: MINUTES, topic: topic.subject_label }) }}</RouterLink>
      </li>
    </ul>
  </section>
</template>
