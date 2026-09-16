<script setup lang="ts">
/**
 * Topic card BODY — the topic-specific sections of the entity card, in operator-reviewed order:
 * the rising-momentum badge LEADS, then similar topics, the storyline link (opens the storyline
 * overlay on top), strongest shows, TOP VOICES, search, episodes, conversation arc, perspectives,
 * notes. The shell ({@link EntityCardBody}) owns the back-stack, header and load; this renders the
 * loaded `TopicCard`. Graph navigation emits `open`; `close` dismisses the whole card.
 *
 * Top voices moved up to sit directly beneath strongest shows (operator 2026-09-16). It answers the
 * same question as the section above it — who and what carries this topic — so splitting the two
 * with the transcript search, the episode list and two analysis panels buried the people at the
 * very bottom, where a reader had already decided whether the topic was worth their time.
 */
import { computed, defineAsyncComponent, ref } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink, useRouter } from "vue-router"
import type { Entity, EpisodeSummary, TopicCard } from "../services/types"
import { useTrendingIndex } from "../composables/useTrendingIndex"
import ProfileAvatar from "./ProfileAvatar.vue"
import NoteComposer from "./NoteComposer.vue"
import EpisodeRow from "./EpisodeRow.vue"
import StorylineCard from "./StorylineCard.vue"
import TrendMomentum from "./TrendMomentum.vue"
import Sparkline from "./Sparkline.vue"
import TopicPerspectives from "./TopicPerspectives.vue"
import TopicConversationArc from "./TopicConversationArc.vue"

// ASYNC on purpose: EntityCard renders EntityCardBody, which renders THIS component, so a static
// import would be a cycle. Deferring the resolve to first open breaks it and costs nothing — the
// chunk is already loaded by the time a topic card is on screen.
const EntityCard = defineAsyncComponent(() => import("./EntityCard.vue"))

const props = withDefaults(
  defineProps<{
    topic: TopicCard
    /**
     * May this card open a person / storyline as a sheet ON TOP?
     *
     * App-wide rule: a sheet may layer over another SHEET or over a PAGE, never over an inline
     * PANEL. Inside the Knowledge Panel the panel IS the layer, so a modal on top would put two
     * dismissables on screen with two different Back meanings (replace-in-panel, UXS-014). There
     * everything replaces in place instead, topics included, so the gesture stays predictable.
     *
     * The shell passes its `dismissAtRoot`, which is already true exactly when this card is the
     * whole destination and false when it is a drill-down inside a host (operator 2026-09-16).
     */
    canLayer?: boolean
  }>(),
  { canLayer: true }
)
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
function openStoryline(): void {
  // There is no storyline equivalent of the shell's back stack, so inside a panel the honest
  // fallback is the standalone page rather than a modal stacked over the panel.
  if (props.canLayer) storylineOpen.value = true
  else void router.push({ name: "storyline", params: { id: props.topic.id } })
}

/**
 * A person opened from THIS topic layers ON TOP, exactly as the storyline does (operator
 * 2026-09-16) — it no longer replaces the topic via the shell's back stack. Walking topic → person
 * is a widening of what you are reading, not a departure from it, and the stacked sheet keeps the
 * topic's kicker + title on screen so the relationship stays visible.
 *
 * TOPIC chips still drill in place through the back stack: topic → topic is the same KIND of thing,
 * so replacing is right there and stacking would pile up identical-looking sheets.
 */
const personOpen = ref<string | null>(null)
function openPerson(id: string): void {
  if (props.canLayer) personOpen.value = id
  else emit("open", { kind: "person", id }) // panel: replace in place via the shell's back stack
}

// Topic momentum (BT.4) — the same "↑ Rising · N× vs avg" badge the storyline sheet shows, leading
// the card. /trending?kind=topic is keyed by topic id; only when genuinely rising (≥1.5×) so the
// hardcoded "Rising" copy stays honest — a steady/cooling topic shows no badge.
const trendingTopics = useTrendingIndex("topic")
const topicMomentum = computed(() => {
  const row = trendingTopics.value[props.topic.id]
  return row && row.v >= 1.5 ? row : null
})

// Universal "discussed over time" activity sparkline (operator 2026-09-14): EVERY topic shows the
// same chart, from any entry point — not just trending ones. Derived client-side from the topic's
// own episodes (monthly counts across their span), so it needs NO backend series and makes no
// "rising" claim; the trending pill above is the only rising signal, and only when earned.
const activitySeries = computed<number[]>(() => {
  const months = episodes.value
    .map((e) => e.publish_date)
    .filter((d): d is string => !!d)
    .map((d) => {
      const t = Date.parse(d)
      return Number.isNaN(t) ? null : (() => {
        const dt = new Date(t)
        return dt.getUTCFullYear() * 12 + dt.getUTCMonth()
      })()
    })
    .filter((m): m is number => m !== null)
  if (months.length < 2) return []
  const min = Math.min(...months)
  const max = Math.max(...months)
  const span = max - min + 1
  if (span < 2) return [] // all in one month — a flat single bar is not a trend
  // Guard runaway spans (a topic with a decade of episodes) — cap the buckets, oldest folded in.
  const CAP = 36
  const buckets = new Array(Math.min(span, CAP)).fill(0)
  for (const m of months) {
    const i = Math.max(0, buckets.length - 1 - (max - m))
    buckets[i]++
  }
  return buckets
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
  <!-- Momentum LEADS the card (operator review). Two honest, separate signals:
       1. The "↑ Rising · N×" pill — ONLY when genuinely trending (≥1.5×), so the copy stays true.
       2. A universal "discussed over time" activity line — on EVERY topic, from any entry point
          (operator 2026-09-14), derived from the topic's own episodes. The pill renders `hide-spark`
          so it does not draw a second, redundant chart above this one. -->
  <div v-if="topicMomentum || activitySeries.length" class="mb-4">
    <TrendMomentum
      v-if="topicMomentum"
      variant="badge"
      hide-spark
      :velocity="topicMomentum.v"
      class="mb-2"
      data-testid="ec-topic-momentum"
    />
    <figure v-if="activitySeries.length > 1" data-testid="ec-topic-activity">
      <Sparkline :values="activitySeries" class="h-8 w-full text-topic" />
      <figcaption class="lp-kicker mt-1">{{ t("ec.discussedOverTime") }}</figcaption>
    </figure>
  </div>

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
      @click="openStoryline"
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
  <StorylineCard v-if="storylineOpen" :id="topic.id" stacked @close="storylineOpen = false" />

  <!-- A person, layered over this topic. `history-key` MUST differ from the parent sheet's `card`
       or the two fight over one history entry (see EntityCard). -->
  <EntityCard
    v-if="personOpen"
    kind="person"
    :id="personOpen"
    history-key="card2"
    stacked
    @close="personOpen = null"
  />

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

  <!-- Top voices (wave-G): the people who drive THIS topic, as prominent avatar chips. -->
  <section v-if="topVoices.length" class="mb-4" data-testid="ec-top-voices">
    <h3 class="lp-section mb-2">{{ t("ec.topVoices") }}</h3>
    <!-- A 4-column grid that fills the row width (operator 2026-09-15): the old flex-wrap left a
         dead gap on the right of each row; the grid spreads the avatars evenly and lets a partial
         last row sit left with empty space below rather than an uneven ragged edge. -->
    <div class="grid grid-cols-4 gap-3">
      <button
        v-for="p in topVoices"
        :key="p.id"
        type="button"
        class="flex flex-col items-center gap-1"
        :aria-label="p.name"
        data-testid="ec-top-voice"
        @click="openPerson(p.id)"
      >
        <ProfileAvatar :name="p.name" :src="p.image_url" :size="44" />
        <span class="line-clamp-2 text-center text-xs font-medium text-canvas-foreground">
          {{ p.name }}
        </span>
      </button>
    </div>
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
        <EpisodeRow :episode="e" />
      </li>
    </ul>
  </section>

  <!-- Multi-perspective synthesis (#1146): each guest's take on this topic; hides when none. -->
  <TopicConversationArc :id="topic.id" />
  <TopicPerspectives
    :id="topic.id"
    @open="(p) => (p.kind === 'person' ? openPerson(p.id) : emit('open', p))"
  />

  <!-- Notes on this topic (TD.7). -->
  <NoteComposer target="topic" :target-id="topic.id" />
</template>
