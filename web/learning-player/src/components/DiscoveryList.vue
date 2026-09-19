<script setup lang="ts">
/**
 * ONE discovery list for Home (operator 2026-09-14) — topics, storylines, or people, ranked by
 * Rising (velocity) or Trending (volume), over a scope + window. Replaces the three bespoke rails
 * (`MomentumRail` + `TrendingTopics` + `Storylines`): every kind renders the SAME row — label ·
 * optional subtitle · colour-pulsed sparkline · trailing metric (follows the active sort) · follow —
 * so the tabs are identical by construction rather than by hand-matched CSS that drifts.
 *
 * All three kinds come from one endpoint (`GET /api/app/trending`), which carries BOTH `velocity`
 * and `volume` per entity — so Rising vs Trending is a client-side re-sort, not a second fetch.
 * Storylines are the one exception: the trending row lacks their `size` (for the "N topics" subtitle)
 * and `anchor_topic_id` (what the overlay opens on), so we merge in `getStorylines` by id.
 *
 * Controls (kind tabs, sort/scope toggles, window) live in the parent (HomeView); this is a
 * controlled presentational list. It emits `open` with the kind + the id to open.
 */
import { computed, ref, watch } from "vue"
import { storeToRefs } from "pinia"
import { useI18n } from "vue-i18n"
import { getStorylines, getTrending, type TrendWindow } from "../services/api"
import type { Storyline } from "../services/types"
import { useAuthStore } from "../stores/auth"
import { useInterestsStore } from "../stores/interests"
import { useSectionState } from "../composables/useSectionState"
import SectionStatus from "./SectionStatus.vue"
import { RouterLink } from "vue-router"
import ProfileAvatar from "./ProfileAvatar.vue"
import Sparkline from "./Sparkline.vue"
import TrendWindowTabs from "./TrendWindowTabs.vue"
import { trendArrow, trendColor } from "./trending"

type Kind = "topic" | "storyline" | "person"
type Sort = "rising" | "trending"
interface Row {
  id: string // entity id (topic:/thc:/person:) — the follow token
  openId: string // what `open` targets: topic/person id, or a storyline's anchor topic
  label: string
  count?: number // storylines: how many topics the cluster holds (shown as a "(N)" after the label)
  image?: string | null // people: avatar photo (falls back to initials)
  velocity: number
  volume: number
  series: number[]
}

const props = withDefaults(
  defineProps<{
    kind: Kind
    sort: Sort
    scope?: "corpus" | "mine"
    limit?: number
    // Rows shown before the inline "show more" (operator 2026-09-14: Home caps at 5, Discover at 10).
    collapsed?: number
    // Suppress the inline "show more" entirely — Discover surfaces a "See all →" in the section
    // header instead (in DiscoveryExplorer), so a bottom affordance would be redundant.
    hideMore?: boolean
  }>(),
  { scope: "corpus", limit: 20, collapsed: 5, hideMore: false }
)
const emit = defineEmits<{ (e: "open", payload: { kind: Kind; id: string }): void }>()

const { t } = useI18n()
// The trend window lives here (not the parent) so it can share the row with the metric hint. #2030.
const window = ref<TrendWindow>("3m")
const auth = useAuthStore()
const interests = useInterestsStore()
const { ids: followedIds } = storeToRefs(interests)
if (auth.isAuthenticated) void interests.ensureLoaded().catch(() => {})

// Only interest tokens are followable (topic: / tc: / thc: / person:).
const _FOLLOWABLE = /^(topic:|tc:|thc:|person:)/
const canFollow = (id: string): boolean => auth.isAuthenticated && _FOLLOWABLE.test(id)
const isFollowed = (id: string): boolean => followedIds.value.includes(id)
const onFollow = (id: string): void => void interests.toggle(id)

const section = useSectionState<Row[]>([])
async function fetchRows(): Promise<Row[]> {
  const trending = await getTrending(props.kind, props.scope, props.limit, window.value)
  if (props.kind !== "storyline") {
    return trending.map((e) => ({
      id: e.entity_id,
      openId: e.entity_id,
      label: e.label,
      image: e.image_url,
      velocity: e.velocity,
      volume: e.volume,
      series: e.series,
    }))
  }
  // Storyline: join the trending momentum with the storyline list for size + anchor topic.
  const stories = await getStorylines(props.limit).catch(() => [] as Storyline[])
  const byId = new Map(stories.map((s) => [s.id, s]))
  return trending.map((e) => {
    const s = byId.get(e.entity_id)
    return {
      id: e.entity_id,
      openId: s?.anchor_topic_id ?? e.entity_id,
      label: e.label,
      count: s?.size,
      velocity: e.velocity,
      volume: e.volume,
      series: e.series,
    }
  })
}
function load(): Promise<void> {
  return section.load(fetchRows)
}
void load()
// Re-fetch on kind/scope/window; sort is client-side (no refetch).
watch(() => [props.kind, props.scope, window.value] as const, load)

const rows = computed<Row[]>(() =>
  [...section.data.value].sort((a, b) =>
    props.sort === "rising" ? b.velocity - a.velocity : b.volume - a.volume
  )
)
const hasAny = computed(() => rows.value.length > 0)

const expanded = ref(false)
const visible = computed(() =>
  expanded.value ? rows.value : rows.value.slice(0, props.collapsed)
)

const vFmt = (v: number): number => Math.round(v * 10) / 10
function rowLabel(r: Row): string {
  const metric =
    props.sort === "rising"
      ? `${vFmt(r.velocity)}× momentum`
      : t("home.discoveryVolume", { count: r.volume })
  const named = r.count != null ? `${r.label} (${r.count})` : r.label
  return `${named} — ${metric}`
}
</script>

<template>
  <section :data-testid="`discovery-list-${kind}`">
    <!-- Window + metric hint share ONE row (operator 2026-09-14): the 1M/3M/6M/1Y tabs on the left,
         the on-screen explanation of the trailing metric on the right (truncates on a phone; full
         text in the title). The hint is required so the ×/count is decoded on touch, where title
         attrs don't exist (#1595); its copy names the active sort's window so Rising vs Trending
         read differently (#1668). -->
    <div class="mb-2 flex items-start gap-6">
      <TrendWindowTabs v-model="window" class="shrink-0" />
      <p
        v-if="hasAny"
        class="line-clamp-2 min-w-0 flex-1 text-xs leading-tight text-muted"
        :title="`${sort === 'rising' ? '↑2×' : '2×'} = ${sort === 'rising' ? t('home.momentumHint') : t('home.trendingHint')}`"
        data-testid="discovery-hint"
      >
        <!-- The metric example is BOLD (not quoted) so the ×/count reads at a glance and is decoded
             on screen, not only via title (#1595). -->
        <b class="font-semibold text-canvas-foreground">{{ sort === "rising" ? "↑2×" : "2×" }}</b>
        = {{ sort === "rising" ? t("home.momentumHint") : t("home.trendingHint") }}
      </p>
    </div>
    <SectionStatus :phase="section.phase.value" :rows="4" @retry="load" />
    <ul v-if="hasAny" class="flex flex-col">
      <li
        v-for="r in visible"
        :key="r.id"
        class="flex items-center gap-1 rounded-lg transition hover:bg-overlay"
        data-testid="discovery-row"
      >
        <button
          type="button"
          class="flex min-h-10 min-w-0 flex-1 items-center gap-2.5 rounded-lg px-2 py-1.5 text-left"
          :aria-label="rowLabel(r)"
          @click="emit('open', { kind, id: r.openId })"
        >
          <!-- People carry their photo (falls back to initials); topics/storylines don't — which is
               why the row above pins `min-h-10`.
               The avatar is 28px and the tallest thing in a topic/storyline row is the 20px
               sparkline, so People rows were 40px and the other two 32px. Switching tabs then moved
               everything below the section by 8px per row, and the page jumped under the reader
               (operator 2026-09-19). The height is now the SAME whatever the tab renders, so the
               kind can change without the layout moving. -->
          <ProfileAvatar
            v-if="kind === 'person'"
            :name="r.label"
            :src="r.image"
            :size="28"
            class="shrink-0"
          />
          <!-- Label truncates; a storyline's topic count rides at the END as "(N)" after the
               ellipsis (operator 2026-09-14) rather than a second subtitle line. -->
          <span class="flex min-w-0 flex-1 items-baseline gap-1">
            <span class="min-w-0 truncate text-sm">{{ r.label }}</span>
            <span v-if="r.count != null" class="shrink-0 text-xs text-muted">({{ r.count }})</span>
          </span>
          <!-- The pulse hue always encodes direction (rising green / cooling red / steady amber). -->
          <Sparkline
            :values="r.series"
            :width="56"
            :height="20"
            :stroke-width="1.4"
            class="shrink-0"
            :style="{ color: trendColor(r.velocity) }"
          />
          <!-- Trailing metric follows the active sort, so the ordering always reads as intentional:
               velocity ×N under Rising, the volume count under Trending. -->
          <span
            class="w-14 shrink-0 text-right text-xs font-semibold tabular-nums"
            :style="sort === 'rising' ? { color: trendColor(r.velocity) } : undefined"
            :class="{ 'text-muted': sort === 'trending' }"
          >
            <template v-if="sort === 'rising'">{{ trendArrow(r.velocity) }} {{ vFmt(r.velocity) }}×</template>
            <template v-else>{{ r.volume }}</template>
          </span>
        </button>
        <button
          v-if="canFollow(r.id)"
          type="button"
          class="shrink-0 rounded-full px-2 py-1 text-base leading-none transition"
          :class="isFollowed(r.id) ? 'text-accent' : 'text-muted hover:text-accent'"
          data-testid="discovery-follow"
          :aria-pressed="isFollowed(r.id)"
          :aria-label="isFollowed(r.id) ? t('ec.following') : t('ec.follow')"
          :title="isFollowed(r.id) ? t('ec.following') : t('ec.follow')"
          @click="onFollow(r.id)"
        >
          {{ isFollowed(r.id) ? "✓" : "+" }}
        </button>
      </li>
    </ul>
    <!-- "Show N more" NAVIGATES to the full trends page (operator 2026-09-16) rather than growing
         the rail in place. Home is a summary surface: expanding inline pushed everything below the
         rail down the page and still left the reader on a capped list, somewhere they had not
         chosen to be. The trends page is the surface built for the full list, and carrying `tab`
         lands them on the kind they were already reading rather than resetting to Topics.
         Discover sets `hideMore` and offers its own "See all →" in the section header.

         Shown on EVERY tab, not only when rows are hidden (operator 2026-09-16): a tab that happens
         to fit its rows is still a summary of a bigger list, and a link that appears and disappears
         by tab reads as a bug rather than as a rule. "See all" also says what it does now — the old
         "Show N more" promised an inline expansion this no longer performs.

         Lands on DISCOVER's trends section with this kind selected, not on the standalone /trends
         page (operator 2026-09-17). `?trends=` and not `?tab=`: the latter drives Discover's own
         Episodes/Shows tabs, so it would both miss the kind and reset the page. -->
    <RouterLink
      v-if="!hideMore && hasAny"
      :to="{ name: 'browse', query: { trends: kind }, hash: '#trends' }"
      class="mt-2 inline-block text-sm font-bold text-accent no-underline"
      data-testid="discovery-expand"
    >
      {{ t("home.seeAllTrends") }} →
    </RouterLink>
  </section>
</template>
