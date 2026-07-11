<script setup lang="ts">
/**
 * "What this show's about" — show-level signals on the consumer show page (PodcastView):
 * dominant themes, the topics it covers, what's trending here, and who's on it. Reads
 * GET /api/app/podcasts/{feedId}/signals — a listener projection of the operator feed-signals
 * (the operator-only grounding/QA score is dropped server-side). Chips emit `open` so the page
 * opens the shared entity card. Best-effort: the whole band hides when the show has no signals.
 *
 * People shows one section (key people) — on a real show "recurring guests" is nearly the same
 * set (hosts dominate both), so we don't render it twice.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { getPodcastSignals } from '../services/api'
import type { PodcastSignals } from '../services/types'
import type { RisingTopic } from './trending'
import TrendingMomentum from './TrendingMomentum.vue'

const props = defineProps<{ feedId: string }>()
const emit = defineEmits<{ (e: 'open', payload: { kind: 'topic' | 'person'; id: string }): void }>()
const { t } = useI18n()

const signals = ref<PodcastSignals | null>(null)
watch(
  () => props.feedId,
  () => {
    signals.value = null
    void getPodcastSignals(props.feedId, 10)
      .then((s) => {
        signals.value = s
      })
      .catch(() => {
        signals.value = null
      })
  },
  { immediate: true },
)

const themes = computed(() => signals.value?.dominant_themes ?? [])
const topics = computed(() => signals.value?.top_topics ?? [])
const trending = computed(() => signals.value?.trending_topics ?? [])
const people = computed(() => signals.value?.key_people ?? [])

// The show's topics as a momentum bubble cloud — size = velocity (how hot the topic is;
// falls back to 1× when the corpus has no velocity for it), reusing the Home component.
const bubbleTopics = computed<RisingTopic[]>(() =>
  topics.value.map((t) => ({
    id: t.topic_id,
    label: t.label,
    v: t.velocity ?? 1,
    total: t.episode_count,
    series: [],
  })),
)
const hasAny = computed(
  () =>
    themes.value.length > 0 ||
    topics.value.length > 0 ||
    trending.value.length > 0 ||
    people.value.length > 0,
)
</script>

<template>
  <section
    v-if="hasAny"
    class="mb-6 rounded-2xl border border-border bg-surface p-4"
    data-testid="podcast-signals"
  >
    <h2 class="lp-section mb-3">{{ t('podcast.about') }}</h2>

    <!-- A momentum bubble of the show's topics (the "little chart") — bigger = hotter. -->
    <div v-if="bubbleTopics.length >= 3" class="mb-3" data-testid="ps-bubbles">
      <TrendingMomentum
        :topics="bubbleTopics"
        @open="emit('open', { kind: 'topic', id: $event })"
      />
    </div>

    <div v-if="themes.length" class="mb-3">
      <h3 class="lp-kicker mb-1.5">{{ t('podcast.sigThemes') }}</h3>
      <div class="flex flex-wrap gap-1.5">
        <button
          v-for="th in themes"
          :key="th.theme_id"
          type="button"
          data-testid="ps-theme"
          class="lp-theme-chip rounded-full px-2.5 py-1 text-xs font-semibold text-surface-foreground transition disabled:opacity-60"
          :disabled="!th.anchor_topic_id"
          @click="th.anchor_topic_id && emit('open', { kind: 'topic', id: th.anchor_topic_id })"
        >
          {{ th.label }} <span class="opacity-70">· {{ th.topic_count }}</span>
        </button>
      </div>
    </div>

    <div v-if="topics.length" class="mb-3">
      <h3 class="lp-kicker mb-1.5">{{ t('podcast.sigTopics') }}</h3>
      <div class="flex flex-wrap gap-1.5">
        <button
          v-for="tp in topics"
          :key="tp.topic_id"
          type="button"
          data-testid="ps-topic"
          class="rounded-full bg-overlay px-2.5 py-1 text-xs text-topic transition hover:bg-elevated"
          @click="emit('open', { kind: 'topic', id: tp.topic_id })"
        >
          {{ tp.label }}
        </button>
      </div>
    </div>

    <div v-if="trending.length" class="mb-3">
      <h3 class="lp-kicker mb-1.5">{{ t('podcast.sigTrending') }}</h3>
      <div class="flex flex-wrap gap-1.5">
        <button
          v-for="tr in trending"
          :key="tr.topic_id"
          type="button"
          data-testid="ps-trending"
          class="inline-flex items-center gap-1 rounded-full bg-emerald-500/20 px-2.5 py-1 text-xs font-semibold text-emerald-300 transition hover:bg-emerald-500/30"
          @click="emit('open', { kind: 'topic', id: tr.topic_id })"
        >
          {{ tr.label }} <span class="opacity-80">↑ {{ tr.velocity }}×</span>
        </button>
      </div>
    </div>

    <div v-if="people.length">
      <h3 class="lp-kicker mb-1.5">{{ t('podcast.sigPeople') }}</h3>
      <div class="flex flex-wrap gap-1.5">
        <button
          v-for="p in people"
          :key="p.person_id"
          type="button"
          data-testid="ps-person"
          class="rounded-full bg-overlay px-2.5 py-1 text-xs text-person transition hover:bg-elevated"
          @click="emit('open', { kind: 'person', id: p.person_id })"
        >
          {{ p.name }} <span class="text-muted">· {{ p.episode_count }}</span>
        </button>
      </div>
    </div>
  </section>
</template>
