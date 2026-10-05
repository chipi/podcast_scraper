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
import CollapsibleSection from "../components/CollapsibleSection.vue"
import { computed, defineAsyncComponent, ref, watch } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink, useRouter } from "vue-router"

import BackIcon from "../components/BackIcon.vue"
import CloseIcon from "../components/CloseIcon.vue"
import EntityEpisodeList from "../components/EntityEpisodeList.vue"
import MemberTrendBadge from "../components/MemberTrendBadge.vue"
import AddToCollectionButton from "../components/AddToCollectionButton.vue"
import FavoriteButton from "../components/FavoriteButton.vue"
import FollowButton from "../components/FollowButton.vue"
import NoteComposer from "../components/NoteComposer.vue"
import ShowAllToggle from "../components/ShowAllToggle.vue"
import { useCappedSections } from "../composables/useCappedSections"
import TopicPerspectives from "../components/TopicPerspectives.vue"
import ShareMenu from "../components/ShareMenu.vue"
import TopVoices from "../components/TopVoices.vue"
import TrendMomentum from "../components/TrendMomentum.vue"
import { useTrendingIndex } from "../composables/useTrendingIndex"
import { getThemeCard } from "../services/api"
import type { ClusterMember, Entity, EpisodeSummary } from "../services/types"
import { useAuthStore } from "../stores/auth"
import { useInterestsStore } from "../stores/interests"

type Member = { id: string; label: string; episodeCount: number; firstSeen: string | null; lastSeen: string | null; trend: ClusterMember['trend'] }

// `embedded` — rendered INSIDE the theme overlay sheet (ThemeCard) rather than as a page. Same
// contract as StorylineView: the two are the same kind of object and must behave the same when a
// topic card opens one on top.
const props = withDefaults(
  defineProps<{ id: string; embedded?: boolean; depth?: number }>(),
  { embedded: false, depth: 0 }
)
const emit = defineEmits<{ (e: "close"): void }>()

/**
 * Inside the SHEET a member topic or person opens as a sheet ON TOP; on the PAGE the links stay
 * links. Identical to StorylineView — the same gesture has to give the same result whichever
 * grouping you opened.
 */
const entityOpen = ref<{ kind: "topic" | "person"; id: string } | null>(null)
function openEntity(kind: "topic" | "person", id: string, e?: MouseEvent): void {
  if (!props.embedded) return // page: let the RouterLink navigate
  e?.preventDefault()
  entityOpen.value = { kind, id }
}
function openPerspective(p: { kind: "person" | "topic"; id: string }): void {
  if (props.embedded) {
    entityOpen.value = { kind: p.kind, id: p.id }
    return
  }
  void router.push({ name: p.kind, params: { id: p.id } })
}

// ASYNC: EntityCard → EntityCardBody → TopicCardContent → ThemeCard → this file is a cycle, so the
// resolve is deferred to first open. Same reason StorylineView defers it.
const EntityCard = defineAsyncComponent(() => import("../components/EntityCard.vue"))

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
// Member topics page five at a time (operator 2026-10-05): the server sends the whole grouping.
const memberCaps = useCappedSections(5, 5)
const pagedTopics = computed(() => memberCaps.visible("members", topics.value))
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


function goBack(): void {
  if (window.history.length > 1) router.back()
  else void router.push({ name: "browse", query: { tab: "topics" } })
}
</script>

<template>
  <section
    :class="embedded ? '' : 'lp-page pb-8 pt-4'"
    data-testid="theme-view"
  >
    <!-- Back on its own row. Suppressed when embedded — the sheet closes with its own ✕. -->
    <button
      v-if="!embedded"
      type="button"
      class="lp-nav"
      :aria-label="t('nav.back')"
      @click="goBack"
    >
      <BackIcon />
      <span>{{ t("nav.back") }}</span>
    </button>

    <!-- Header order matches the storyline and topic pages (UXS-014): kicker, title on its own
         row, then actions on theirs. -->
    <div :class="embedded ? '' : 'mt-3'">
      <div class="flex items-start justify-between gap-3">
        <span class="lp-kicker min-w-0 text-theme">{{ t("ec.theme") }}</span>
        <!-- Close ✕ — embedded only; standalone uses the Back row above. -->
        <button
          v-if="embedded"
          type="button"
          class="lp-nav shrink-0"
          :aria-label="t('ec.close')"
          data-testid="theme-card-close"
          @click="emit('close')"
        >
          <CloseIcon />
        </button>
      </div>
      <h1
        class="mt-2 line-clamp-2 font-display font-extrabold tracking-tight"
        :class="embedded ? 'text-xl' : 'text-2xl'"
      >
        {{ label || "…" }}
      </h1>
      <!-- The same action set as the topic and storyline pages: Save, Share, Follow. These were
           absent while `FavoriteKind` / `NoteTarget` did not admit "theme" — the buttons would have
           offered an action the API rejects. Both contracts now carry it end to end. -->
      <div class="mt-3 flex flex-wrap items-center gap-2">
        <FavoriteButton :item="{ kind: 'theme', ref: id, label: label || id }" />
        <AddToCollectionButton :item="{ kind: 'theme', ref: id }" variant="pill" />
        <!-- Shares as a THEME — its own server card and link (operator 2026-10-05); it used to share
             as a topic. Analytics has no theme bucket, so `target-kind` stays topic. -->
        <ShareMenu kind="theme" :id="id" :title="label || id" target-kind="topic" />
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
      <!-- ORDER (operator 2026-10-05): members + top voices -> what they SAID -> episodes.
           Top voices moved up beside the member list — side by side, half the width each, on the
           desktop page; stacked on a phone and in the sheet. The episode list closes the page at
           full width. (2026-10-01 had members -> said -> episodes -> voices: the quotes moved up
           from the foot, where the long episode list hid them; that part stands.) -->
      <div :class="embedded ? '' : 'lg:grid lg:grid-cols-2 lg:items-start lg:gap-6'" data-testid="theme-opening-pair">
      <section class="mt-6" data-testid="theme-topics">
        <CollapsibleSection :title="t('home.themeTopicsHeading')" section-key="theme-topics" :level="2">
          <ol class="flex flex-col">
            <li v-for="(tp, i) in pagedTopics" :key="tp.id">
              <RouterLink
                :to="{ name: 'topic', params: { id: tp.id } }"
                class="flex items-center gap-3 border-b border-border py-2 text-canvas-foreground no-underline hover:bg-overlay"
                @click="openEntity('topic', tp.id, $event)"
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
          <ShowAllToggle
            v-if="memberCaps.overflows(topics.length, false, 'members')"
            :expanded="memberCaps.remaining('members', topics.length) === 0"
            :count="topics.length"
            :remaining="memberCaps.remaining('members', topics.length)"
            data-testid="members-more"
            @toggle="memberCaps.toggle('members', topics.length)"
          />
        </CollapsibleSection>
      </section>

      <!-- Counted across the whole union, so these are the voices that recur across the theme
           rather than inside one member. -->
      <TopVoices
        class="mt-6"
        :people="people"
        :heading-level="2"
        :route-for="(pid) => ({ name: 'person', params: { id: pid } })"
        @open="(pid, e) => openEntity('person', pid, e)"
      />
      </div>

      <!-- What is SAID across the grouping — its members' insights, grouped by speaker, each with
           a jump-to-moment link. Until this, nothing on either grouping page was a sentence anybody
           actually uttered: the pages listed member topics and episodes and left the reader to
           infer what the grouping sounded like. Renders nothing when no member has a
           speaker-attributable insight, which is the honest outcome for a grouping whose members
           are abstract labels nobody says aloud. -->
      <TopicPerspectives
        class="mt-6"
        :id="id"
        kind="theme"
        @open="openPerspective"
      />

      <!-- The MERGED list: every episode discussing any member, de-duplicated. This is the page's
           reason to exist — a reader on one member's topic page sees only that member's episodes,
           and a similarity grouping exists precisely because that misses the rest. -->
      <section v-if="episodes.length" class="mt-6" data-testid="theme-episodes">
        <CollapsibleSection section-key="theme-episodes" :level="2">
          <template #title>
            <span>{{
              t("ec.topicEpisodes", episodes.length, { named: { count: episodes.length } })
            }}</span>
            <span class="lp-kicker" data-testid="episodes-order">{{ t("ec.newestFirst") }}</span>
          </template>
          <EntityEpisodeList :episodes="episodes" />
        </CollapsibleSection>
      </section>


      <!-- Notes, like the storyline and topic pages. Keyed by the theme's own id. -->
      <NoteComposer target="theme" :target-id="id" />
    </template>

    <!-- A topic or person opened from inside this sheet, layered over it. `history-key` MUST differ
         from the parent sheet's or the two fight over one history entry (see EntityCard). -->
    <EntityCard
      v-if="entityOpen"
      :kind="entityOpen.kind"
      :id="entityOpen.id"
      history-key="card2"
      :depth="depth + 1"
      @close="entityOpen = null"
    />
  </section>
</template>
