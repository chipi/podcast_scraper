<script setup lang="ts">
/**
 * Trending topics (Plan B #4 — temporal_velocity on Home). Topics "heating up"
 * across the corpus: last month running >= 1.5x their 6-month average, with a
 * floor on total mentions to cut sample noise. Reads the shared, memoized
 * /api/app/corpus/enrichment; hides when nothing is rising. Chips open the
 * topic entity card (whose Signals show the same momentum).
 *
 * ## One view, not four (#1589)
 *
 * This shipped with a four-way view switcher — Pills / Sparklines / Over time / Momentum — whose
 * own comment said it existed "so the operator can flip between to decide what to keep". That
 * decision was never made, so an internal A/B lab reached users: four ways to read the same data,
 * a control nobody outside the team could interpret, and three components' worth of maintenance.
 *
 * Sparklines won. It is the only view that shows BOTH the current level and the shape that earned
 * the "trending" label, which is the question the section exists to answer; the stream and momentum
 * views answered analyst questions on a consumer surface, and pills dropped the trend entirely.
 *
 * If a second view is ever wanted, add it deliberately with a reason — not as an unresolved
 * experiment.
 */
import { computed, ref } from 'vue'
import { useSectionState } from '../composables/useSectionState'
import SectionStatus from './SectionStatus.vue'
import { storeToRefs } from 'pinia'
import { useI18n } from 'vue-i18n'
import { getCorpusEnrichment } from '../services/api'
import { useAuthStore } from '../stores/auth'
import { useInterestsStore } from '../stores/interests'
import { THEME_NEUTRAL, THEME_PALETTE, type RisingTopic, type TopicTheme } from './trending'
import TrendingSparkChips from './TrendingSparkChips.vue'

const emit = defineEmits<{ (e: 'open', id: string): void }>()
const { t } = useI18n()

// #12 — follow a trending topic straight into the profile interests (same store the entity-card
// follows use). Only when signed in; the store persists + reconciles with the server.
const auth = useAuthStore()
const interests = useInterestsStore()
const { ids: followedIds } = storeToRefs(interests)
const canFollow = computed(() => auth.isAuthenticated)
if (auth.isAuthenticated) void interests.ensureLoaded().catch(() => {})
function onFollow(id: string): void {
  void interests.toggle(id)
}

const RISING = 1.5
const MIN_TOTAL = 3
const MAX = 12

const months = ref<string[]>([])
const topics = ref<RisingTopic[]>([])
// Topic ids that belong to a co-occurrence theme cluster ("storyline") — used to mark them
// the standard way (teal theme chrome), same as the topic card + storyline chips.
const themeMemberIds = ref<Set<string>>(new Set())

// Per-topic theme colour + label: topics in the same co-occurrence theme ("storyline") share a
// distinct hue so related trending topics read as a group; unclustered topics fall back to neutral.
const topicTheme = ref<Record<string, TopicTheme>>({})

const section = useSectionState<null>(null)
function load(): Promise<void> {
  return section.load(async () => {
    applyEnrichment(await getCorpusEnrichment())
    return null
  })
}
void load()

function applyEnrichment(s: Awaited<ReturnType<typeof getCorpusEnrichment>>): void {
  {
    const tv = s.temporal_velocity
    const rows = tv?.topics ?? []
    const clusters = s.topic_theme_clusters?.clusters ?? []
    themeMemberIds.value = new Set(
      clusters.flatMap((c) => (c.members ?? []).map((m) => m.topic_id)),
    )
    // topic → { hue, theme label, group } so the sparkline view can colour + group by storyline.
    const themeMap: Record<string, TopicTheme> = {}
    clusters.forEach((c, i) => {
      const color = THEME_PALETTE[i % THEME_PALETTE.length]
      const label = c.canonical_label?.trim() || null
      for (const m of c.members ?? []) themeMap[m.topic_id] = { color, label, group: i }
    })
    topicTheme.value = themeMap
    // Month axis: the envelope's window_months, else the union of keys seen.
    const axis =
      tv?.window_months && tv.window_months.length
        ? [...tv.window_months]
        : [...new Set(rows.flatMap((r) => Object.keys(r.monthly_counts ?? {})))].sort()
    months.value = axis
    // Did we actually get a velocity reading? "Nothing is rising" is only sayable when we HAVE the
    // measurements and none of them clear the bar. With no enricher there is nothing to conclude,
    // and claiming quiet would be inventing a result.
    hasVelocityData.value = rows.length > 0
    topics.value = rows
      .filter((x) => (x.velocity_last_over_6mo ?? 0) >= RISING && (x.total ?? 0) >= MIN_TOTAL)
      .sort((a, b) => (b.velocity_last_over_6mo ?? 0) - (a.velocity_last_over_6mo ?? 0))
      .slice(0, MAX)
      .map((x) => ({
        id: x.topic_id,
        label: x.topic_label?.trim() || x.topic_id.replace(/^topic:/, '').replace(/[-_]+/g, ' '),
        v: Math.round((x.velocity_last_over_6mo ?? 0) * 10) / 10,
        total: x.total ?? 0,
        series: axis.map((m) => x.monthly_counts?.[m] ?? 0),
      }))
  }
}

const hasAny = computed(() => topics.value.length > 0)
/** True once the velocity envelope has been read and carried at least one topic. */
const hasVelocityData = ref(false)
/** Show the rail when it has something to say — rising topics, a measured quiet, or a failure. */
const showsSection = computed(
  () => hasAny.value || hasVelocityData.value || !section.isReady.value,
)
</script>

<template>
  <!--
    Renders even when nothing clears the bar — a deliberate exception to #1591's "hide when the
    SYSTEM is empty".

    This rail and the momentum rail below it are two independent measures of "what's hot", and they
    disagree: on the validation corpus `systems thinking` is 0.86x here (last month vs its 6-month
    average) and 1.78x on the momentum rail (EWMA anchored to today). Because this one hides when
    its 1.5x gate finds nothing, that disagreement was invisible — the rail has never appeared, so
    nobody could compare them.

    Keeping it on screen with an honest "nothing clears the bar" line makes the metric observable
    instead of silent, which is the whole point while both are being evaluated against real data
    (#1595-followup). It says the system found nothing; it does not pretend to be broken or loading.
  -->
  <section v-if="showsSection" class="mt-7" data-testid="home-trending">
    <h2 class="lp-section">{{ t('home.trending') }}</h2>
    <SectionStatus :phase="section.phase.value" :rows="2" @retry="load" />
    <p
      v-if="section.isReady.value && !hasAny && hasVelocityData"
      class="mt-1 text-sm text-muted"
      data-testid="home-trending-quiet"
    >
      {{ t('home.trendingQuiet') }}
    </p>
    <template v-if="hasAny">
    <p class="mb-2 text-sm text-muted">{{ t('home.trendingHint') }}</p>

    <TrendingSparkChips
      :topics="topics"
      :topic-theme="topicTheme"
      :neutral-color="THEME_NEUTRAL"
      :followed-ids="followedIds"
      :can-follow="canFollow"
      @open="emit('open', $event)"
      @follow="onFollow"
    />
    </template>
  </section>
</template>
