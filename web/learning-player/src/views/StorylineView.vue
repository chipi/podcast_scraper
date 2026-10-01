<script setup lang="ts">
/**
 * Storyline page (F4.5) — a storyline is a THEME CLUSTER (topics discussed together). It used to
 * open as a half-screen sheet; it is now a full page with the topic-page look: back at top, title +
 * actions on one row, the member topics, top episodes and the people involved, and notes.
 *
 * There is no dedicated storyline endpoint — the anchor topic's card IS the storyline (its
 * `storyline_*` + `storyline_sibling_topics` + `related_people` + `episodes`), so the route param is
 * the anchor topic id and everything derives from `getTopicCard`.
 */
import { computed, ref, watch, defineAsyncComponent } from "vue"
import CloseIcon from "../components/CloseIcon.vue"
import { useI18n } from "vue-i18n"
import { RouterLink, useRouter } from "vue-router"
import { getStorylineCard } from "../services/api"
import { useTrendingIndex } from "../composables/useTrendingIndex"
import { useAuthStore } from "../stores/auth"
import { useInterestsStore } from "../stores/interests"
import EntityEpisodeList from "../components/EntityEpisodeList.vue"
import MemberTrendBadge from "../components/MemberTrendBadge.vue"
import NoteComposer from "../components/NoteComposer.vue"
import TopicPerspectives from "../components/TopicPerspectives.vue"
import TopVoices from "../components/TopVoices.vue"
// ASYNC: EntityCard → EntityCardBody → TopicCardContent → StorylineCard → this file is a cycle, so
// the resolve is deferred to first open. Same reason TopicCardContent defers EntityCard.
const EntityCard = defineAsyncComponent(() => import("../components/EntityCard.vue"))
import AddToCollectionButton from "../components/AddToCollectionButton.vue"
import FavoriteButton from "../components/FavoriteButton.vue"
import FollowButton from "../components/FollowButton.vue"
import TrendMomentum from "../components/TrendMomentum.vue"
import ShareMenu from "../components/ShareMenu.vue"
import { accentForKind, type EntityCardModel } from "../composables/entityShareCard"
import type { ClusterMember, ClusterPair, Entity, EpisodeSummary } from "../services/types"

type Member = { id: string; label: string; episodeCount: number; anchor: boolean; firstSeen: string | null; lastSeen: string | null; trend: ClusterMember['trend'] }

// `embedded` — rendered INSIDE the storyline overlay sheet (StorylineCard) rather than as a
// standalone route. Drops the back button + page padding/width; the sheet supplies its own chrome.
const props = withDefaults(
  // `depth` — the hosting sheet's stack level, so anything opened from here lands one deeper. A
  // standalone page is 0: nothing is underneath it.
  defineProps<{ id: string; embedded?: boolean; depth?: number }>(),
  { embedded: false, depth: 0 }
)

/**
 * Inside the SHEET, a member topic or person opens as a sheet ON TOP — the same gesture giving the
 * same result as everywhere else (operator 2026-09-16). These used to be RouterLinks to the full
 * page, so tapping a topic here closed the storyline and dropped you on a page, while the identical
 * tap on a topic card opened a layer.
 *
 * On the PAGE the links stay links: a page is where navigation belongs, and there is no sheet to
 * layer over.
 */
const entityOpen = ref<{ kind: "topic" | "person"; id: string } | null>(null)
function openEntity(kind: "topic" | "person", id: string, e?: MouseEvent): void {
  if (!props.embedded) return // page: let the RouterLink navigate
  e?.preventDefault()
  entityOpen.value = { kind, id }
}
/**
 * Perspectives emit a payload rather than wrapping a RouterLink, so this view has to route them
 * itself — and it must honour `embedded`, where every other open in this file resolves into the
 * overlay instead of navigating. `openEntity` alone would not do: it no-ops on the standalone page
 * because there a RouterLink normally handles it, and a perspective has no RouterLink to fall back
 * on. Getting this wrong makes the speaker names inert in the sheet, the same affordance lie the
 * dimmed theme pills were.
 */
function openPerspective(p: { kind: "person" | "topic"; id: string }): void {
  if (props.embedded) {
    entityOpen.value = { kind: p.kind, id: p.id }
    return
  }
  void router.push({ name: p.kind, params: { id: p.id } })
}

// When embedded in the overlay sheet the ✕ lives in THIS header's action row (unified with the
// topic/person card), so the close intent has to reach StorylineCard. Standalone ignores it.
const emit = defineEmits<{ (e: "close"): void }>()
const { t } = useI18n()
const router = useRouter()
const auth = useAuthStore()
const interests = useInterestsStore()
watch(
  () => auth.isAuthenticated,
  (authed) => {
    if (authed) void interests.ensureLoaded()
  },
  { immediate: true }
)

const loading = ref(true)
const failed = ref(false)
const label = ref("")
const topics = ref<Member[]>([])
const people = ref<Entity[]>([])
const episodes = ref<EpisodeSummary[]>([])
const storylineId = ref<string | null>(null)
/** The most co-occurring pair — the storyline's evidence, in one line. */
const pair = ref<ClusterPair | null>(null)

async function load(anchorTopicId: string): Promise<void> {
  loading.value = true
  failed.value = false
  try {
    // The STORYLINE endpoint, not the anchor's topic card.
    //
    // This page used to derive everything from `getTopicCard(anchor)`: its members from
    // `storyline_sibling_topics`, and its episodes from `card.episodes` — which are the ANCHOR's
    // episodes, not the storyline's. So it said "Discussed in 30 episodes" for a storyline that
    // spans 40, showing one member's corpus under the storyline's name. `/storylines/:id` returns
    // the de-duplicated UNION across every member, which is what the heading always claimed.
    //
    // The route still passes an anchor TOPIC id (there was no endpoint when it was built); the
    // endpoint accepts either that or the `thc:` id.
    const card = await getStorylineCard(anchorTopicId)
    label.value = card.label
    storylineId.value = card.id
    topics.value = card.member_topics.map((tp) => ({
      id: tp.id,
      label: tp.label,
      episodeCount: tp.episode_count,
      anchor: tp.anchor,
      firstSeen: tp.first_seen ?? null,
      lastSeen: tp.last_seen ?? null,
      trend: tp.trend,
    }))
    pair.value = card.strongest_pair ?? null
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
  { immediate: true }
)

// Storyline momentum (BT.4): /trending?kind=storyline keys the same thc: id as the theme cluster,
// so match the loaded storyline by its storylineId. Same badge idiom as the topic card
// (TrendMomentum badge variant). Best-effort — no badge when this storyline isn't in the top set.
const trendingStorylines = useTrendingIndex("storyline")
const storylineMomentum = computed(() =>
  storylineId.value ? trendingStorylines.value[storylineId.value] ?? null : null
)

const following = computed(() => !!storylineId.value && interests.has(storylineId.value))
function toggleFollow(): void {
  if (storylineId.value) void interests.toggle(storylineId.value)
}

// #2036 — the shareable card for this storyline: the cluster label + how many topics/episodes it
// spans + a canonical link. Violet accent (accentForKind("storyline")) — distinct from topic cyan.
const shareModel = computed<EntityCardModel>(() => {
  const parts = [`${topics.value.length} ${topics.value.length === 1 ? "topic" : "topics"}`]
  if (episodes.value.length) {
    parts.push(`${episodes.value.length} ${episodes.value.length === 1 ? "episode" : "episodes"}`)
  }
  return {
    kicker: t("share.kickerStoryline"),
    title: label.value || props.id,
    stats: parts.join(" · "),
    accent: accentForKind("storyline"),
    url: typeof window !== "undefined" ? `${window.location.origin}/storyline/${props.id}` : null,
  }
})

function goBack(): void {
  if (window.history.length > 1) router.back()
  else void router.push({ name: "browse", query: { tab: "topics" } })
}
</script>

<template>
  <section :class="embedded ? '' : 'mx-auto max-w-3xl px-4 pb-8 pt-4'" data-testid="storyline-view">
    <!-- Back on its own row, then kicker → title (UXS-014 header order). Suppressed when embedded
         in the overlay sheet — the sheet carries its own ✕ close. -->
    <button
      v-if="!embedded"
      type="button"
      class="lp-nav"
      :aria-label="t('nav.back')"
      @click="goBack"
    >
      <span aria-hidden="true" class="text-base leading-none">‹</span>
      <span>{{ t("nav.back") }}</span>
    </button>

    <!-- Header unified with the topic/person card (EntityCardBody): kicker (+ the close ✕ when
         embedded) on the top row, then the TITLE on its own full-width row, then the actions on
         their OWN row after the title (operator: the kicker+actions row was too cramped). -->
    <div :class="embedded ? '' : 'mt-3'">
      <div class="flex items-start justify-between gap-3">
        <span class="lp-kicker min-w-0 text-storyline">{{ t("home.storylines") }}</span>
        <!-- Close ✕ — embedded only; standalone uses the Back row above. -->
        <button
          v-if="embedded"
          type="button"
          class="lp-nav shrink-0"
          :aria-label="t('ec.close')"
          data-testid="storyline-card-close"
          @click="emit('close')"
        >
          <CloseIcon />
        </button>
      </div>
      <h1
        class="mt-2 font-display font-extrabold tracking-tight line-clamp-2"
        :class="embedded ? 'text-xl' : 'text-2xl'"
      >
        {{ label || "…" }}
      </h1>
      <!-- Actions on their OWN aligned row, AFTER the title (operator). -->
      <div class="mt-3 flex flex-wrap items-center gap-2">
        <!-- Save (heart) is a per-kind favorite — a storyline lands in Library › Saved like any
             other kind (F2.2). Distinct from Follow, which subscribes to the theme cluster. -->
        <FavoriteButton :item="{ kind: 'storyline', ref: id, label: label || id }" />
        <!-- Share (card / link / text) — #2036. -->
        <AddToCollectionButton :item="{ kind: 'storyline', ref: id }" variant="pill" />
        <ShareMenu :model="shareModel" />
        <FollowButton
          v-if="auth.isAuthenticated && storylineId"
          variant="storyline"
          :following="following"
          :label="label || undefined"
          @toggle="toggleFollow"
        />
      </div>
    </div>

    <!-- Momentum on its OWN full-width row, not squeezed into the title column beside the actions —
         so the pill + sparkline sit side-by-side on one bottom-aligned line even in the narrow
         overlay sheet, instead of the sparkline wrapping under the pill. -->
    <TrendMomentum
      v-if="storylineMomentum"
      variant="badge"
      :velocity="storylineMomentum.v"
      :series="storylineMomentum.series"
      class="mt-3 block"
    />

    <p v-if="loading" class="mt-4 text-sm text-muted">{{ t("home.storylineSheetLoading") }}</p>
    <p v-else-if="failed || !topics.length" class="mt-4 text-sm text-muted">
      {{ t("home.storylineSheetEmpty") }}
    </p>

    <template v-else>
      <!-- Member topics, an ordered list (SL.1). -->
      <section class="mt-6">
        <h2 class="lp-section mb-1">{{ t("home.storylineTopicsHeading") }}</h2>
        <!-- WHY these topics are one storyline, stated as a fact rather than asserted by the
             heading. One line, under the heading, so it reads as the section's subtitle. -->
        <p v-if="pair" class="mb-2 text-xs text-muted" data-testid="storyline-pair">
          {{
            t("home.storylinePair", {
              a: pair.a_label,
              b: pair.b_label,
              n: pair.shared_episode_count,
            })
          }}
        </p>
        <ol class="flex flex-col">
          <li v-for="(tp, i) in topics" :key="tp.id">
            <RouterLink
              :to="{ name: 'topic', params: { id: tp.id } }"
              class="flex items-center gap-3 border-b border-border py-2 no-underline text-canvas-foreground hover:bg-overlay"
              @click="openEntity('topic', tp.id, $event)"
            >
              <span class="w-5 shrink-0 text-center text-xs font-bold tabular-nums text-muted">{{
                i + 1
              }}</span>
              <span class="min-w-0 flex-1 truncate text-sm font-semibold text-topic">{{
                tp.label
              }}</span>
              <!-- The ANCHOR as a mark, not a sentence: the row is already carrying a rank, a
                   label and a count, and "Anchors this storyline" spelled out would wrap the row
                   on a phone. The word lives in the accessible name instead. -->
              <svg
                v-if="tp.anchor"
                data-testid="storyline-anchor"
                class="h-3.5 w-3.5 shrink-0 text-accent"
                viewBox="0 0 16 16"
                fill="none"
                stroke="currentColor"
                stroke-width="1.6"
                stroke-linecap="round"
                stroke-linejoin="round"
                role="img"
                :aria-label="t('home.storylineAnchor')"
              >
                <circle cx="8" cy="3" r="1.6" />
                <path d="M8 4.6V14" />
                <path d="M4.5 7.5h7" />
                <path d="M2.5 10.5a5.5 5.5 0 0 0 11 0" />
              </svg>
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

      <!-- In the OVERLAY the episodes + people below just re-present the topic card sitting beneath
           it, so the sheet stays a compact preview (members + momentum + follow) and links out to
           the full storyline page for the rest. The standalone page has nothing beneath it, so it
           shows everything. This link doubles as the overlay's "open in page" escape hatch. -->

      <!-- Top episodes for the storyline (SL.2). Standalone page only — see the note above. -->
      <section v-if="episodes.length" class="mt-6">
        <!-- Says "newest first" like the topic, person and org lists do (operator 2026-09-19).
             This was the one of the four that never did, which is the drift the shared
             `EntityEpisodeList` exists to stop repeating. -->
        <h2 class="lp-section mb-2 flex flex-wrap items-baseline gap-x-2">
          <span>{{ t("ec.topicEpisodes", episodes.length, { named: { count: episodes.length } }) }}</span>
          <span class="lp-kicker" data-testid="episodes-order">{{ t("ec.newestFirst") }}</span>
        </h2>
        <EntityEpisodeList :episodes="episodes" />
      </section>

      <!-- Top voices (SL.2) — the SAME grid the topic card shows (operator 2026-09-30): it was
           "Related people" as plain chips here, the same people from the same `related_people`
           drawn a second way. Standalone page only (redundant with the topic card in overlay). -->
      <TopVoices
        class="mt-6"
        :people="people"
        :heading-level="2"
        :route-for="(id) => ({ name: 'person', params: { id } })"
        @open="(id, e) => openEntity('person', id, e)"
      />

      <!-- What is SAID across the storyline — its members' insights, grouped by speaker, each with
           a jump-to-moment link. Until this, nothing on the page was a sentence anybody actually
           uttered: it listed member topics and episodes and left the reader to infer what the
           storyline sounded like. Scoped to the UNION of the member topics, which is what the
           storyline IS — a single member's perspectives would be the anchor topic's page again.
           Renders nothing when no member has a speaker-attributable insight. -->
      <TopicPerspectives
        class="mt-6"
        :id="id"
        kind="storyline"
        @open="openPerspective"
      />

      <!-- Notes on this storyline (SL.3). Shown in the sheet too (operator 2026-09-16): the sheet is
           no longer a preview of the page, it IS the page's content, so withholding notes here was
           the last thing making the two differ. -->
      <NoteComposer target="storyline" :target-id="id" />
    </template>

    <!-- A topic or person opened FROM this sheet, layered on top. Its own history key, or it would
         fight the parent sheet's entry (see EntityCard). -->
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
