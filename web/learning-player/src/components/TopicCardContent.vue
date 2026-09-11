<script setup lang="ts">
/**
 * Topic card BODY — the topic-specific sections of the entity card, in operator-reviewed order:
 * the rising-momentum badge LEADS, then similar topics, the storyline link (opens the storyline
 * overlay on top), strongest shows, search, episodes, conversation arc, perspectives, top voices,
 * notes. The shell ({@link EntityCardBody}) owns the back-stack, header and load; this renders the
 * loaded `TopicCard`. Graph navigation emits `open`; `close` dismisses the whole card.
 */
import { computed, ref } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink, useRouter } from "vue-router"
import type { Entity, EpisodeSummary, TopicCard } from "../services/types"
import { useTrendingIndex } from "../composables/useTrendingIndex"
import ProfileAvatar from "./ProfileAvatar.vue"
import NoteComposer from "./NoteComposer.vue"
import EpisodeRow from "./EpisodeRow.vue"
import StorylineCard from "./StorylineCard.vue"
import TrendMomentum from "./TrendMomentum.vue"
import TopicPerspectives from "./TopicPerspectives.vue"
import TopicConversationArc from "./TopicConversationArc.vue"

const props = defineProps<{ topic: TopicCard }>()
const emit = defineEmits<{
  (e: "open", payload: { kind: "person" | "topic"; id: string }): void
  (e: "close"): void
}>()
const { t } = useI18n()
const router = useRouter()

const label = computed(() => props.topic.label ?? "")
const episodes = computed<EpisodeSummary[]>(() => props.topic.episodes ?? [])
const episodeCount = computed(() => props.topic.episode_count ?? 0)
const siblings = computed(() => props.topic.sibling_topics ?? [])
// Theme cluster (co-occurrence "discussed together") — the STORYLINE this topic is part of.
const themeClusterLabel = computed(() => props.topic.theme_cluster_label ?? null)
const themeClusterSize = computed(() => props.topic.theme_cluster_size ?? 0)
// The people who drive this topic — related_people is server-ranked by co-occurrence, so the top
// few ARE the key voices. Prominent avatar chips.
const topVoices = computed<Entity[]>(() => (props.topic.related_people ?? []).slice(0, 8))
// Storyline overlay ("open on top" — StorylineCard), keyed by this topic's id.
const storylineOpen = ref(false)

// Topic momentum (BT.4) — the same "↑ Rising · N× vs avg" badge the storyline sheet shows, leading
// the card. /trending?kind=topic is keyed by topic id; only when genuinely rising (≥1.5×) so the
// hardcoded "Rising" copy stays honest — a steady/cooling topic shows no badge.
const trendingTopics = useTrendingIndex("topic")
const topicMomentum = computed(() => {
  const row = trendingTopics.value[props.topic.id]
  return row && row.v >= 1.5 ? row : null
})

// Strongest shows (TD.6): which shows cover this topic most, from the discussed episodes grouped by
// feed. Only worth showing when the topic spans MORE THAN ONE show.
const topShows = computed(() => {
  const byFeed = new Map<string, { feed_id: string; title: string; count: number }>()
  for (const e of episodes.value) {
    if (!e.feed_id) continue
    const cur = byFeed.get(e.feed_id)
    if (cur) cur.count++
    else
      byFeed.set(e.feed_id, { feed_id: e.feed_id, title: e.podcast_title ?? e.feed_id, count: 1 })
  }
  return [...byFeed.values()].sort((a, b) => b.count - a.count).slice(0, 5)
})

function searchLibrary(): void {
  const term = label.value.trim()
  emit("close")
  if (term) void router.push({ name: "search", query: { q: term } })
}
</script>

<template>
  <!-- Momentum LEADS the card (operator review): gated to genuinely rising topics (≥1.5×). -->
  <TrendMomentum
    v-if="topicMomentum"
    variant="badge"
    :velocity="topicMomentum.v"
    :series="topicMomentum.series"
    class="mb-4 block"
    data-testid="ec-topic-momentum"
  />

  <!-- Semantically SIMILAR topics: the one you're on (ringed) + siblings. Distinct from the
       storyline below, which is co-occurrence (#1603). Chips drill in place via the back stack. -->
  <section v-if="siblings.length" class="mb-4">
    <h3 class="lp-section mb-2">
      {{ t("ec.clusterMembers", siblings.length + 1, { named: { count: siblings.length + 1 } }) }}
    </h3>
    <div class="flex flex-wrap gap-1.5">
      <span
        class="rounded-full bg-overlay px-2.5 py-1 text-xs font-semibold text-topic ring-1 ring-topic"
      >
        {{ label }}
      </span>
      <button
        v-for="s in siblings"
        :key="s.id"
        type="button"
        data-testid="ec-similar-topic"
        class="rounded-full bg-overlay px-2.5 py-1 text-xs text-topic transition hover:bg-elevated"
        @click="emit('open', { kind: 'topic', id: s.id })"
      >
        {{ s.label }}
      </button>
    </div>
  </section>

  <!-- Part of a storyline: ONE link that opens the whole storyline ON TOP (StorylineCard overlay).
       A topic with no cluster says so, quietly. -->
  <section v-if="themeClusterLabel" class="mb-4" data-testid="ec-storyline">
    <h3 class="lp-section mb-2">{{ t("ec.storylineHeading") }}</h3>
    <button
      type="button"
      data-testid="ec-storyline-link"
      class="flex w-full items-center gap-2 rounded-xl border border-border bg-overlay px-3 py-2.5 text-left transition hover:bg-elevated"
      @click="storylineOpen = true"
    >
      <span class="min-w-0 flex-1">
        <span class="block text-sm font-bold text-theme">{{ themeClusterLabel }}</span>
        <span v-if="themeClusterSize" class="lp-kicker">{{
          t("ec.clusterSize", themeClusterSize, { named: { count: themeClusterSize } })
        }}</span>
      </span>
      <span class="shrink-0 text-muted" aria-hidden="true">›</span>
    </button>
  </section>
  <p v-else class="mb-4 text-xs text-muted" data-testid="ec-single-topic">
    {{ t("ec.singleTopic") }}
  </p>

  <!-- The storyline, opened ON TOP (teleported sheet) rather than navigating away. -->
  <StorylineCard v-if="storylineOpen" :id="topic.id" @close="storylineOpen = false" />

  <!-- Strongest shows on this topic — only when it spans more than one show. -->
  <section v-if="topShows.length > 1" class="mb-4" data-testid="ec-top-shows">
    <h3 class="lp-section mb-2">{{ t("ec.topShows") }}</h3>
    <ul class="flex flex-col">
      <li v-for="s in topShows" :key="s.feed_id">
        <RouterLink
          :to="{ name: 'podcast', params: { feedId: s.feed_id } }"
          class="flex items-center justify-between gap-3 border-b border-border py-2 no-underline text-canvas-foreground hover:bg-overlay"
        >
          <span class="min-w-0 truncate text-sm font-semibold">{{ s.title }}</span>
          <span class="shrink-0 text-xs text-muted">{{
            t("ec.topShowCount", s.count, { named: { count: s.count } })
          }}</span>
        </RouterLink>
      </li>
    </ul>
  </section>

  <!-- Search transcripts — between the strongest shows and the episode list (operator review). -->
  <button
    type="button"
    class="mb-4 block w-fit max-w-full rounded-full border border-border px-4 py-2 text-left text-sm font-bold text-canvas-foreground transition hover:bg-overlay"
    data-testid="ec-search-library"
    @click="searchLibrary"
  >
    {{ t("ec.searchLibrary", { term: label }) }}
  </button>

  <!-- Episodes (newest-first, STATED not offered as a control — #2004 item 11). -->
  <section v-if="episodes.length" class="mb-4">
    <h3 class="lp-section mb-2 flex flex-wrap items-baseline gap-x-2">
      <span>{{ t("ec.topicEpisodes", episodeCount, { named: { count: episodeCount } }) }}</span>
      <span class="lp-kicker" data-testid="episodes-order">{{ t("ec.newestFirst") }}</span>
    </h3>
    <ul class="flex flex-col">
      <li v-for="e in episodes" :key="e.slug">
        <EpisodeRow :episode="e" @navigate="emit('close')" />
      </li>
    </ul>
  </section>

  <!-- Multi-perspective synthesis (#1146): each guest's take on this topic; hides when none. -->
  <TopicConversationArc :id="topic.id" />
  <TopicPerspectives :id="topic.id" @open="(p) => emit('open', p)" />

  <!-- Top voices (wave-G): the people who drive THIS topic, as prominent avatar chips. -->
  <section v-if="topVoices.length" class="mb-4" data-testid="ec-top-voices">
    <h3 class="lp-section mb-2">{{ t("ec.topVoices") }}</h3>
    <div class="flex flex-wrap gap-3">
      <button
        v-for="p in topVoices"
        :key="p.id"
        type="button"
        class="flex w-16 flex-col items-center gap-1"
        :aria-label="p.name"
        data-testid="ec-top-voice"
        @click="emit('open', { kind: 'person', id: p.id })"
      >
        <ProfileAvatar :name="p.name" :src="p.image_url" :size="44" />
        <span class="line-clamp-2 text-center text-xs font-medium text-canvas-foreground">
          {{ p.name }}
        </span>
      </button>
    </div>
  </section>

  <!-- Notes on this topic (TD.7). -->
  <NoteComposer target="topic" :target-id="topic.id" />
</template>
