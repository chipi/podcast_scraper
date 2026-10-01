<script setup lang="ts">
/**
 * Theme page — a THEME is a set of topics that MEAN the same thing (cosine similarity).
 *
 * Modelled on `StorylineView`, deliberately: the two are the same kind of object — a grouping over
 * topics, not an entity — so they should read the same and differ only where the idea differs. The
 * one difference that matters is the member heading: a storyline's members are "discussed
 * together" (co-occurrence), a theme's "mean the same thing" (similarity). That sentence is the
 * whole distinction, and it is the page's job to make it.
 *
 * ## Why this is not the topic page
 *
 * A theme is never a node on an episode, so `build_topic_card` — which matches a topic node by id —
 * found nothing for a `tc:` id. Routed to `/topic/:id`, a theme rendered an EMPTY page with a
 * "TOPIC" eyebrow. This view talks to `/api/app/themes/{id}`, which exists to serve a grouping.
 *
 * ## Why the route carries the real `tc:` id
 *
 * `/storyline/:id` takes its ANCHOR TOPIC's id, because there is no storyline endpoint and the
 * anchor's card carries the cluster. A theme has a real id and a real endpoint, so it uses them: a
 * theme link stays valid even when its biggest member changes, which an anchor-topic link does not.
 */
import { computed, ref, watch } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink, useRouter } from "vue-router"

import EntityEpisodeList from "../components/EntityEpisodeList.vue"
import MemberTrendBadge from "../components/MemberTrendBadge.vue"
import AddToCollectionButton from "../components/AddToCollectionButton.vue"
import FavoriteButton from "../components/FavoriteButton.vue"
import FollowButton from "../components/FollowButton.vue"
import NoteComposer from "../components/NoteComposer.vue"
import ShareMenu from "../components/ShareMenu.vue"
import TopVoices from "../components/TopVoices.vue"
import TrendMomentum from "../components/TrendMomentum.vue"
import { accentForKind, type EntityCardModel } from "../composables/entityShareCard"
import { useTrendingIndex } from "../composables/useTrendingIndex"
import { getThemeCard } from "../services/api"
import type { ClusterMember, Entity, EpisodeSummary } from "../services/types"
import { useAuthStore } from "../stores/auth"
import { useInterestsStore } from "../stores/interests"

type Member = { id: string; label: string; episodeCount: number; firstSeen: string | null; lastSeen: string | null; trend: ClusterMember['trend'] }

const props = defineProps<{ id: string }>()

const { t } = useI18n()
const router = useRouter()
const auth = useAuthStore()
const interests = useInterestsStore()

watch(
  () => auth.isAuthenticated,
  (authed) => {
    if (authed) void interests.ensureLoaded()
  },
  { immediate: true },
)

const loading = ref(true)
const failed = ref(false)
const label = ref("")
const topics = ref<Member[]>([])
const people = ref<Entity[]>([])
const episodes = ref<EpisodeSummary[]>([])

async function load(themeId: string): Promise<void> {
  loading.value = true
  failed.value = false
  try {
    const card = await getThemeCard(themeId)
    label.value = card.label
    // No `anchor` and no pair line, unlike the storyline. A theme groups topics that MEAN the same
    // thing — symmetric, no centre — so flagging one member or claiming a co-occurrence would both
    // assert something the grouping does not say. The count still earns its place: it shows how
    // much of the theme each member actually carries.
    topics.value = (card.member_topics ?? []).map((tp) => ({
      id: tp.id,
      label: tp.label,
      episodeCount: tp.episode_count,
      firstSeen: tp.first_seen ?? null,
      lastSeen: tp.last_seen ?? null,
      trend: tp.trend,
    }))
    people.value = card.related_people ?? []
    episodes.value = card.episodes ?? []
  } catch {
    failed.value = true
  } finally {
    loading.value = false
  }
}
watch(
  () => props.id,
  (id) => void load(id),
  { immediate: true },
)

// Momentum, the same band the storyline page carries. `/trending?kind=theme` already exists and
// the momentum engine already aggregates a weekly series per theme cluster
// (`app_momentum._add_cluster_series(..., "theme")`) — it was simply never read by a theme surface.
// Keyed by the route param, because a theme's `tc:` id IS its trending id.
//
// This is one of the two diagrams a storyline has, and it belongs on BOTH: "is this getting more
// attention lately" is a question about any grouping, unlike the anchor and the co-occurrence pair,
// which only a storyline can answer. Best-effort — no badge when the theme is outside the top set.
const trendingThemes = useTrendingIndex("theme")
const momentum = computed(() => trendingThemes.value[props.id] ?? null)

// The route param IS the follow token — a theme is followed by its `tc:` id, the same id the
// profile stores. No lookup needed, unlike the storyline page which has to learn its cluster id
// from the anchor topic's card before it can offer a follow.
const following = computed(() => interests.has(props.id))
function toggleFollow(): void {
  void interests.toggle(props.id)
}

const shareModel = computed<EntityCardModel>(() => {
  const parts = [`${topics.value.length} ${topics.value.length === 1 ? "topic" : "topics"}`]
  if (episodes.value.length) {
    parts.push(`${episodes.value.length} ${episodes.value.length === 1 ? "episode" : "episodes"}`)
  }
  return {
    kicker: t("share.kickerTheme"),
    title: label.value || props.id,
    stats: parts.join(" · "),
    // Themes share the storyline accent: both are groupings, and giving them one hue against the
    // topic's is the colour saying "this is a set, not a thing".
    accent: accentForKind("storyline"),
    url: typeof window !== "undefined" ? `${window.location.origin}/theme/${props.id}` : null,
  }
})

function goBack(): void {
  if (window.history.length > 1) router.back()
  else void router.push({ name: "browse", query: { tab: "topics" } })
}
</script>

<template>
  <section class="mx-auto max-w-3xl px-4 pb-8 pt-4" data-testid="theme-view">
    <button type="button" class="lp-nav" :aria-label="t('nav.back')" @click="goBack">
      <span aria-hidden="true" class="text-base leading-none">‹</span>
      <span>{{ t("nav.back") }}</span>
    </button>

    <!-- Header order matches the storyline and topic pages (UXS-014): kicker, title on its own
         row, then actions on theirs. -->
    <div class="mt-3">
      <span class="lp-kicker min-w-0 text-accent">{{ t("home.themes") }}</span>
      <h1 class="mt-2 line-clamp-2 font-display text-2xl font-extrabold tracking-tight">
        {{ label || "…" }}
      </h1>
      <!-- The same action set as the topic and storyline pages: Save, Share, Follow. These were
           absent while `FavoriteKind` / `NoteTarget` did not admit "theme" — the buttons would have
           offered an action the API rejects. Both contracts now carry it end to end. -->
      <div class="mt-3 flex flex-wrap items-center gap-2">
        <FavoriteButton :item="{ kind: 'theme', ref: id, label: label || id }" />
        <AddToCollectionButton :item="{ kind: 'theme', ref: id }" variant="pill" />
        <ShareMenu :model="shareModel" />
        <FollowButton
          v-if="auth.isAuthenticated"
          variant="theme"
          :following="following"
          :label="label || undefined"
          @toggle="toggleFollow"
        />
      </div>
    </div>

    <!-- Its own full-width row under the actions, matching the storyline page: the pill and the
         sparkline sit on one bottom-aligned line instead of the sparkline wrapping under the pill. -->
    <TrendMomentum
      v-if="momentum"
      variant="badge"
      :velocity="momentum.v"
      :series="momentum.series"
      class="mt-3 block"
    />

    <p v-if="loading" class="mt-4 text-sm text-muted">{{ t("home.themeSheetLoading") }}</p>
    <p v-else-if="failed || !topics.length" class="mt-4 text-sm text-muted">
      {{ t("home.themeSheetEmpty") }}
    </p>

    <template v-else>
      <!-- The members, ranked by how much of the corpus each carries, so the first row is the one a
           reader is most likely to recognise. The heading is the product distinction: these mean
           the same thing, as against a storyline's "discussed together". -->
      <section class="mt-6">
        <h2 class="lp-section mb-2">{{ t("home.themeTopicsHeading") }}</h2>
        <ol class="flex flex-col">
          <li v-for="(tp, i) in topics" :key="tp.id">
            <RouterLink
              :to="{ name: 'topic', params: { id: tp.id } }"
              class="flex items-center gap-3 border-b border-border py-2 text-canvas-foreground no-underline hover:bg-overlay"
            >
              <span class="w-5 shrink-0 text-center text-xs font-bold tabular-nums text-muted">{{
                i + 1
              }}</span>
              <span class="min-w-0 flex-1 truncate text-sm font-semibold text-topic">{{
                tp.label
              }}</span>
              <MemberTrendBadge
                :trend="tp.trend"
                :first-seen="tp.firstSeen"
                :last-seen="tp.lastSeen"
              />
              <span class="shrink-0 text-xs tabular-nums text-muted" data-testid="member-episodes">{{
                t("home.memberEpisodes", tp.episodeCount, { named: { n: tp.episodeCount } })
              }}</span>
              <span class="shrink-0 text-muted" aria-hidden="true">›</span>
            </RouterLink>
          </li>
        </ol>
      </section>

      <!-- The MERGED list: every episode discussing any member, de-duplicated. This is the page's
           reason to exist — a reader on one member's topic page sees only that member's episodes,
           and a similarity grouping exists precisely because that misses the rest. -->
      <section v-if="episodes.length" class="mt-6">
        <h2 class="lp-section mb-2 flex flex-wrap items-baseline gap-x-2">
          <span>{{
            t("ec.topicEpisodes", episodes.length, { named: { count: episodes.length } })
          }}</span>
          <span class="lp-kicker" data-testid="episodes-order">{{ t("ec.newestFirst") }}</span>
        </h2>
        <EntityEpisodeList :episodes="episodes" />
      </section>

      <!-- Counted across the whole union, so these are the voices that recur across the theme
           rather than inside one member. -->
      <TopVoices
        class="mt-6"
        :people="people"
        :heading-level="2"
        :route-for="(pid) => ({ name: 'person', params: { id: pid } })"
      />

      <!-- Notes, like the storyline and topic pages. Keyed by the theme's own id. -->
      <NoteComposer target="theme" :target-id="id" />
    </template>
  </section>
</template>
