<script setup lang="ts">
/**
 * Topic card BODY — the topic-specific sections of the entity card, in operator-reviewed order:
 * the rising-momentum badge and the activity sparkline LEAD, then the conversation arc (the two
 * charts read as a pair), similar topics, the storyline link (opens the storyline overlay on top),
 * strongest shows, TOP VOICES, perspectives (the same people the voices name), search, episodes,
 * notes. The shell ({@link EntityCardBody}) owns the back-stack, header and load; this renders the
 * loaded `TopicCard`. Graph navigation emits `open`; `close` dismisses the whole card.
 *
 * Top voices moved up to sit directly beneath strongest shows (operator 2026-09-16). It answers the
 * same question as the section above it — who and what carries this topic — so splitting the two
 * with the transcript search, the episode list and two analysis panels buried the people at the
 * very bottom, where a reader had already decided whether the topic was worth their time.
 */
import CollapsibleSection from "./CollapsibleSection.vue"
import { computed, defineAsyncComponent, ref } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink, useRouter } from "vue-router"
import type { Entity, EpisodeSummary, TopicCard } from "../services/types"
import { useTrendingIndex } from "../composables/useTrendingIndex"
import { resolveMediaUrl } from "../services/tier"
import TopVoices from "./TopVoices.vue"
import NoteComposer from "./NoteComposer.vue"
import EntityEpisodeList from "./EntityEpisodeList.vue"
import StorylineCard from "./StorylineCard.vue"
import TrendMomentum from "./TrendMomentum.vue"
import Sparkline from "./Sparkline.vue"
import ThemeCard from "./ThemeCard.vue"
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
    /** This card's own stack depth; anything it opens sits one level deeper. */
    depth?: number
    /**
     * The card is the whole page at desktop width (the standalone /topic route): pairs that answer
     * one question sit side by side, half the width each, from `lg` (operator 2026-10-05). In a
     * sheet or the Knowledge Panel the card is narrow, so it stays one column.
     */
    wide?: boolean
  }>(),
  { canLayer: true, depth: 0, wide: false }
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
// Topic cluster ("means the same thing") — the THEME this topic belongs to. The payload has carried
// `cluster_id` / `cluster_label` / `cluster_size` since the topic card existed, and nothing ever
// rendered them: the theme appeared only as loose "similar topics" chips, which show its MEMBERS
// without ever naming the thing they are members OF. The storyline has had a named link all along,
// so a topic announced one of its two groupings and stayed silent about the other.
const themeId = computed(() => props.topic.cluster_id ?? null)
const themeLabel = computed(() => props.topic.cluster_label ?? null)
// Member count INCLUDING this topic, which is what "N topics" in the link means. `sibling_topics`
// deliberately excludes the topic you are on, so it is one short of the cluster.
const themeSize = computed(() => props.topic.cluster_size ?? 0)
// Theme cluster (co-occurrence "discussed together") — the STORYLINE this topic is part of.
const storylineLabel = computed(() => props.topic.storyline_label ?? null)
const storylineSize = computed(() => props.topic.storyline_size ?? 0)
// The people who drive this topic — related_people is server-ranked by co-occurrence, so the top
// few ARE the key voices. Prominent avatar chips.
const topVoices = computed<Entity[]>(() => (props.topic.related_people ?? []).slice(0, 8))
// Theme overlay (ThemeCard), keyed by the theme's OWN `tc:` id — a theme has a real endpoint and
// does not need reconstructing from a member the way a storyline does from its anchor.
//
// A sheet, not a route, for the reason spelled out under `openStoryline` below: inside the
// Knowledge Panel a `router.push` changes the page underneath the top-layer dialog and the tap
// reads as dead. Both groupings now open the same way.
const themeOpen = ref(false)
function openTheme(): void {
  themeOpen.value = true
}

// Storyline overlay ("open on top" — StorylineCard), keyed by this topic's id.
const storylineOpen = ref(false)
function openStoryline(): void {
  // ALWAYS open the storyline here. This control is "Part of a storyline" on the topic card, and
  // the one thing it must do is show that storyline.
  //
  // It used to route to the standalone page when `canLayer` was false. Inside the Knowledge Panel
  // that made the tap look completely dead: the panel opens with `showModal()`, so it sits in the
  // browser's top layer over everything, and `router.push` changed the page UNDERNEATH it. Nothing
  // moved, nothing new appeared, and the storyline was only discoverable by closing the panel —
  // reported as "opening storyline from this field on topic is not working" (operator 2026-09-16).
  //
  // Stacking it is also what was asked for: the topic stays visible by its title and the storyline
  // sits one card lower, to whatever depth the chain reaches.
  storylineOpen.value = true
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
  const byFeed = new Map<string, { feed_id: string; title: string; count: number; art: string | null }>()
  for (const e of episodes.value) {
    if (!e.feed_id) continue
    const cur = byFeed.get(e.feed_id)
    if (cur) {
      cur.count++
      // Episodes vary in whether they carry the feed image; take the first one that does.
      cur.art ??= resolveMediaUrl(e.feed_artwork_url || e.feed_image_url)
    } else {
      byFeed.set(e.feed_id, {
        feed_id: e.feed_id,
        title: e.podcast_title ?? e.feed_id,
        count: 1,
        // The FEED image, never the episode's own — this row is the show, not an episode of it.
        // `feed_artwork_url` is our stored copy; `feed_image_url` is the feed-hosted original and
        // only a fallback, since it points off-origin and is frequently unreachable.
        art: resolveMediaUrl(e.feed_artwork_url || e.feed_image_url),
      })
    }
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
  <div :class="props.wide ? 'lg:grid lg:grid-cols-2 lg:items-start lg:gap-6' : ''" data-testid="ec-topic-time-pair">
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

  <!-- The conversation arc sits directly under the activity sparkline (operator 2026-09-19). It was
       at the foot of the page, below the episode list, which put the two time-series charts about
       this topic at opposite ends of a long scroll. They answer the same question at different
       resolutions — how much, and how it changed — so they read as a pair or not at all. -->
  <TopicConversationArc :id="topic.id" :known-weeks="topic.conversation_arc_weeks" />
  </div>

  <!-- Semantically SIMILAR topics. Distinct from the storyline below, which is co-occurrence
       (#1603). Chips drill in place via the back stack.

       The topic you are ON is not in this list (operator 2026-09-19). It used to lead it as a
       ringed chip, on the reasoning that a cluster is best shown whole, with your position in it
       marked. On the page that reasoning does not survive: the heading says "N similar topics" and
       the first thing under it is the topic whose page you are reading, which is not similar to
       itself. The count now counts what is actually listed. -->
  <section v-if="siblings.length" class="mb-4" data-testid="ec-similar-topics">
    <CollapsibleSection :title="t('ec.clusterMembers', siblings.length, { named: { count: siblings.length } })" section-key="topic-similar" :level="3">
      <div class="flex flex-wrap gap-1.5">
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
    </CollapsibleSection>
  </section>

  <!-- Part of a theme: the grouping this topic MEANS the same thing as. Sits directly above the
       storyline link so a reader meets the two groupings as a pair and can see they are different
       claims — "means the same thing" against "keeps coming up together" — rather than meeting one
       of them and inferring the other from a chip list. -->
  <div :class="props.wide ? 'lg:grid lg:grid-cols-2 lg:items-start lg:gap-6' : ''" data-testid="ec-topic-group-pair">
  <section v-if="themeLabel && themeId" class="mb-4" data-testid="ec-theme">
    <CollapsibleSection :title="t('ec.themeHeading')" section-key="topic-theme" :level="3">
      <button
        type="button"
        data-testid="ec-theme-link"
        class="flex w-full items-center gap-2 rounded-xl border border-border bg-overlay px-3 py-2.5 text-left transition hover:bg-elevated"
        @click="openTheme"
      >
        <span class="min-w-0 flex-1">
          <span class="block text-sm font-bold text-theme">{{ themeLabel }}</span>
          <span v-if="themeSize" class="lp-kicker">{{
            t("ec.clusterSize", themeSize, { named: { count: themeSize } })
          }}</span>
        </span>
        <span class="shrink-0 text-muted" aria-hidden="true">›</span>
      </button>
    </CollapsibleSection>
  </section>

  <!-- Part of a storyline: ONE link that opens the whole storyline ON TOP (StorylineCard overlay).
       A topic with no cluster says so, quietly. -->
  <section v-if="storylineLabel" class="mb-4" data-testid="ec-storyline">
    <CollapsibleSection :title="t('ec.storylineHeading')" section-key="topic-storyline" :level="3">
      <button
        type="button"
        data-testid="ec-storyline-link"
        class="flex w-full items-center gap-2 rounded-xl border border-border bg-overlay px-3 py-2.5 text-left transition hover:bg-elevated"
        @click="openStoryline"
      >
        <span class="min-w-0 flex-1">
          <span class="block text-sm font-bold text-storyline">{{ storylineLabel }}</span>
          <span v-if="storylineSize" class="lp-kicker">{{
            t("ec.clusterSize", storylineSize, { named: { count: storylineSize } })
          }}</span>
        </span>
        <span class="shrink-0 text-muted" aria-hidden="true">›</span>
      </button>
    </CollapsibleSection>
  </section>
  <p v-else class="mb-4 text-xs text-muted" data-testid="ec-single-topic">
    {{ t("ec.singleTopic") }}
  </p>
  </div>

  <!-- The theme, opened ON TOP (teleported sheet) rather than navigating away. -->
  <ThemeCard
    v-if="themeOpen && themeId"
    :id="themeId"
    :depth="depth + 1"
    @close="themeOpen = false"
  />

  <!-- The storyline, opened ON TOP (teleported sheet) rather than navigating away. -->
  <StorylineCard
    v-if="storylineOpen"
    :id="topic.id"
    :depth="depth + 1"
    @close="storylineOpen = false"
  />

  <!-- A person, layered over this topic. `history-key` MUST differ from the parent sheet's `card`
       or the two fight over one history entry (see EntityCard). -->
  <EntityCard
    v-if="personOpen"
    kind="person"
    :id="personOpen"
    history-key="card2"
    :depth="depth + 1"
    @close="personOpen = null"
  />

  <!-- Strongest shows on this topic — only when it spans more than one show. -->
  <section v-if="topShows.length > 1" class="mb-4" data-testid="ec-top-shows">
    <CollapsibleSection :title="t('ec.topShows')" section-key="topic-top-shows" :level="3">
      <!-- Artwork, then the name, then the tally — the compact row `EpisodeRow` uses, at its 40px
           thumbnail (operator 2026-09-19). It was a bare line of text with a number on the right,
           which is the one way a show does NOT get recognised: cover art is how you know a podcast at
           a glance, and every other list of shows in the app shows it. Deliberately NOT the full
           `ShowRow` (128px artwork + description) — this sits inside a card as a short aside, not as
           the page's subject. -->
      <ul class="flex flex-col">
        <li v-for="s in topShows" :key="s.feed_id">
          <RouterLink
            :to="{ name: 'podcast', params: { feedId: s.feed_id } }"
            class="flex items-center gap-2.5 border-b border-border py-2 no-underline text-canvas-foreground hover:bg-overlay"
          >
            <img
              v-if="s.art"
              :src="s.art"
              alt=""
              loading="lazy"
              class="h-10 w-10 shrink-0 rounded-md bg-elevated object-cover"
            />
            <div v-else class="h-10 w-10 shrink-0 rounded-md bg-elevated" aria-hidden="true" />
            <span class="min-w-0 flex-1 truncate text-sm font-semibold">{{ s.title }}</span>
            <span class="shrink-0 text-xs text-muted">{{
              t("ec.topShowCount", s.count, { named: { count: s.count } })
            }}</span>
          </RouterLink>
        </li>
      </ul>
    </CollapsibleSection>
  </section>

  <!-- Top voices (wave-G): the people who drive THIS topic. Shared with the storyline page. -->
  <TopVoices class="mb-4" :people="topVoices" @open="(id) => openPerson(id)" />

  <!-- Search transcripts — between the strongest shows and the episode list (operator review). -->
  <button
    type="button"
    class="mb-4 block w-fit max-w-full rounded-full border border-border px-4 py-2 text-left text-sm font-bold text-canvas-foreground transition hover:bg-overlay"
    data-testid="ec-search-library"
    @click="searchLibrary"
  >
    {{ t("ec.searchLibrary", { term: label }) }}
  </button>

  <!-- Multi-perspective synthesis (#1146): each guest's take on this topic; hides when none.
       Directly ABOVE the episode list (operator 2026-10-01), the same place it sits on the theme
       and storyline pages — a reader meets what people SAID about the subject before being handed
       everything that mentions it. It used to sit under Top voices (operator 2026-09-19, because
       it names those same people); the across-all-three consistency won.
       The order asked for, in the operator's words: "between topics, list of topics, and the list
       of episodes ... on all three surfaces". -->
  <!-- Multi-perspective synthesis (#1146): each guest's take on this topic; hides when none.
       Directly under Top voices (operator 2026-09-19) — it names the same people and says what they
       actually argued, so it belongs beside the faces rather than at the foot of the page. -->
  <TopicPerspectives
    :id="topic.id"
    @open="(p) => (p.kind === 'person' ? openPerson(p.id) : emit('open', p))"
  />

  <!-- Episodes (newest-first, STATED not offered as a control — #2004 item 11). -->
  <section v-if="episodes.length" class="mb-4">
    <CollapsibleSection section-key="topic-episodes" :level="3">
      <template #title>
        <span>{{ t("ec.topicEpisodes", episodeCount, { named: { count: episodeCount } }) }}</span>
        <span class="lp-kicker" data-testid="episodes-order">{{ t("ec.newestFirst") }}</span>
      </template>
      <EntityEpisodeList :episodes="episodes" />
    </CollapsibleSection>
  </section>

  <!-- Notes on this topic (TD.7). -->
  <NoteComposer target="topic" :target-id="topic.id" />
</template>
