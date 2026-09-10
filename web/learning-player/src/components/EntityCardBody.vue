<script setup lang="ts">
/**
 * Entity card body (PRD-043 FR2/FR3; UXS-014 interaction patterns) — the shared person/topic card
 * content, rendered two ways:
 *   • `inline`  — replaces a panel's content with a ‹ Back (Insights → entity); no new layer.
 *   • `overlay` — wrapped in EntityCard's modal (Search → entity, a page-level surface).
 * KG-grounded from the dedicated `/api/app/persons|topics/{id}` endpoints; the library search is one
 * explicit action inside. Re-entrant via an internal back stack (walk the graph, step back).
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import { getPersonCard, getTopicCard } from '../services/api'
import type {
  Entity,
  EpisodeSummary,
  PersonCard,
  PersonShow,
  Topic,
  TopicCard,
} from '../services/types'
import AddToCollectionButton from './AddToCollectionButton.vue'
import FavoriteButton from './FavoriteButton.vue'
import NoteComposer from './NoteComposer.vue'
import Tabs from './Tabs.vue'
import type { TabSpec } from './tabs'
import EntitySignals from './EntitySignals.vue'
import TopicPerspectives from './TopicPerspectives.vue'
import TopicConversationArc from './TopicConversationArc.vue'
import { useAuthStore } from '../stores/auth'
import { useInterestsStore } from '../stores/interests'
import { useFavoritesStore } from '../stores/favorites'
import { episodeArtwork } from '../utils/episode'

type Target = { kind: 'person' | 'topic'; id: string }

const props = withDefaults(
  defineProps<{
    kind: 'person' | 'topic'
    id: string
    variant?: 'inline' | 'overlay'
    /**
     * What the control means when there is nothing left on the card's own back stack.
     *
     * `back` — the card DRILLED DOWN inside something you were already reading, and dismissing it
     * returns you to that thing. The Knowledge Panel works this way: tapping a person chip replaces
     * the panel body, and the panel is still what you are in.
     *
     * `close` — the card is the whole destination. An overlay sheet you opened, or the standalone
     * topic / person route, where nothing contains it. A back arrow here is a back arrow whose only
     * job is to close, which is what it looked like on the full-page route.
     */
    rootControl?: 'back' | 'close'
  }>(),
  { variant: 'overlay', rootControl: undefined },
)
const emit = defineEmits<{ (e: 'close'): void }>()

const { t } = useI18n()
const router = useRouter()
const auth = useAuthStore()
const interests = useInterestsStore()
const favorites = useFavoritesStore()
// Load follow + favourite state once we know the user is signed in — auth may resolve after mount.
watch(
  () => auth.isAuthenticated,
  (authed) => {
    if (authed) {
      void interests.ensureLoaded()
      void favorites.ensureLoaded()
    }
  },
  { immediate: true },
)

// Follow this person/topic → its id is the interest token (person:… / topic:…), which feeds
// personalized discovery. Following shapes "Recommended for you" on Home.
const following = computed(() => interests.has(current.value.id))
function toggleFollow(): void {
  void interests.toggle(current.value.id)
}

// Back stack — the bottom is the entity opened on; the top is what's shown.
const stack = ref<Target[]>([{ kind: props.kind, id: props.id }])
const current = computed<Target>(() => stack.value[stack.value.length - 1])
const atRoot = computed(() => stack.value.length === 1)
// An overlay is a thing you opened; an inline card is, by default, a drill-down inside its host.
// A host that is itself the destination (the standalone routes) says so with `rootControl`.
const dismissAtRoot = computed(
  () => atRoot.value && (props.rootControl ?? (props.variant === 'overlay' ? 'close' : 'back')) === 'close',
)

const person = ref<PersonCard | null>(null)
const topic = ref<TopicCard | null>(null)
const loading = ref(false)
const failed = ref(false)

// "Your corpus" lens (P3 #1125): 'mine' restricts the card to the episodes the user has heard
// ("you also heard them in …"). Auth-gated; a global card otherwise.
const corpusScope = ref<'all' | 'mine'>('all')

async function load(target: Target): Promise<void> {
  loading.value = true
  failed.value = false
  person.value = null
  topic.value = null
  const scope = corpusScope.value === 'mine' ? 'mine' : undefined
  try {
    if (target.kind === 'person') person.value = await getPersonCard(target.id, scope)
    else topic.value = await getTopicCard(target.id, scope)
  } catch {
    failed.value = true
  } finally {
    loading.value = false
  }
}

const scopeTabs = computed<TabSpec<'all' | 'mine'>[]>(() => [
  { key: 'all', label: t('ec.scopeAll') },
  { key: 'mine', label: t('ec.scopeMine') },
])

function setCorpusScope(s: 'all' | 'mine'): void {
  if (corpusScope.value === s) return
  corpusScope.value = s
  void load(current.value)
}

// Re-open on a brand-new target (parent opened a different chip) — reset the stack.
watch(
  () => [props.kind, props.id] as const,
  ([kind, id]) => {
    stack.value = [{ kind, id }]
  },
)
watch(current, (target) => void load(target), { immediate: true })

function open(kind: 'person' | 'topic', id: string): void {
  stack.value = [...stack.value, { kind, id }]
}
// Left control: pop the stack if deeper, else dismiss the whole card (back to panel / close modal).
function onBack(): void {
  if (stack.value.length > 1) stack.value = stack.value.slice(0, -1)
  else emit('close')
}

const label = computed(() => person.value?.label ?? topic.value?.label ?? '')

// Speaker role badge (host / guest / mentioned) — mirrors the operator viewer's person role
// badge, KG-grounded from the person node's aggregate role. Empty for topics / unknown role.
const ROLE_LABEL_KEYS: Record<string, string> = {
  host: 'ec.roleHost',
  guest: 'ec.roleGuest',
  mentioned: 'ec.roleMentioned',
}
const personRole = computed(() =>
  current.value.kind === 'person' ? (person.value?.role ?? '').toLowerCase() : '',
)
const personRoleLabel = computed(() => {
  const key = ROLE_LABEL_KEYS[personRole.value]
  return key ? t(key) : ''
})
const episodes = computed<EpisodeSummary[]>(
  () => person.value?.episodes ?? topic.value?.episodes ?? [],
)

// Per-show role (#3 follow-up): a person hosts some shows and guests on others. Surface the
// shows they HOST up top ("Host of"), and drop those shows' back-catalogue from the episode
// list below — a daily-show host shouldn't list 500 own episodes; show other-show appearances.
const hostShows = computed<PersonShow[]>(() =>
  (person.value?.shows ?? []).filter((s) => (s.role ?? '').toLowerCase() === 'host'),
)
const hostFeedIds = computed(() => new Set(hostShows.value.map((s) => s.feed_id)))
const shownEpisodes = computed<EpisodeSummary[]>(() =>
  hostShows.value.length ? episodes.value.filter((e) => !hostFeedIds.value.has(e.feed_id)) : episodes.value,
)
const relatedPeople = computed<Entity[]>(
  () => person.value?.related_people ?? topic.value?.related_people ?? [],
)
const relatedTopics = computed<Topic[]>(() => person.value?.related_topics ?? [])
const siblings = computed<Topic[]>(() => topic.value?.sibling_topics ?? [])
const episodeCount = computed(() => person.value?.episode_count ?? topic.value?.episode_count ?? 0)
const themeLabel = computed(() => topic.value?.cluster_label ?? null)
const clusterSize = computed(() => topic.value?.cluster_size ?? 0)
// Theme cluster (co-occurrence "discussed together") — distinct from the semantic cluster above.
const themeClusterLabel = computed(() => topic.value?.theme_cluster_label ?? null)
const themeClusterSize = computed(() => topic.value?.theme_cluster_size ?? 0)
const themeSiblings = computed<Topic[]>(() => topic.value?.theme_sibling_topics ?? [])
// Follow the whole storyline (the theme cluster, `thc:…`) as one interest token — distinct from
// following just this topic (the header button). Feeds the same personalized discovery ranking.
const themeClusterId = computed(() => topic.value?.theme_cluster_id ?? null)
const followingStoryline = computed(() => {
  const id = themeClusterId.value
  return id != null && interests.has(id)
})
function toggleStoryline(): void {
  const id = themeClusterId.value
  if (id) void interests.toggle(id)
}
const isTopic = computed(() => current.value.kind === 'topic')

const epArt = episodeArtwork

function searchLibrary(): void {
  const term = label.value.trim()
  emit('close')
  if (term) void router.push({ name: 'search', query: { q: term } })
}
</script>

<template>
  <div class="flex min-h-0 flex-1 flex-col bg-surface">
    <!-- Header mirrors the episode-detail masthead (UXS-014): back-nav on its own row, then the
         kicker, then the title — never back crammed beside the kicker/name. -->
    <header class="border-b border-border px-4 py-3">
      <!-- Two different jobs, so two different marks. Inside the card you can drill from a topic
           into a sibling topic or a person, and "‹ Back" pops that stack — it means "up one level,
           still here". At the root the mark depends on what CONTAINS the card — see `rootControl`.
           It was gated on `variant === 'overlay'` alone, so the full-page topic route showed
           "‹ Back" at its root: a back arrow whose only job was to close the page. -->
      <button
        type="button"
        class="lp-nav"
        :aria-label="dismissAtRoot ? t('ec.close') : t('ec.back')"
        @click="onBack"
      >
        <span aria-hidden="true" class="text-base leading-none">{{ dismissAtRoot ? '✕' : '‹' }}</span>
        <span>{{ dismissAtRoot ? t('ec.close') : t('ec.back') }}</span>
      </button>
      <span class="mt-3 flex items-center gap-2">
        <span class="lp-kicker">{{ current.kind === 'person' ? t('ec.person') : t('ec.topic') }}</span>
        <!-- Host / guest / mentioned — the person's aggregate speaker role (mirrors the operator
             viewer). Host gets the ringed emphasis idiom used for the "current" chip elsewhere. -->
        <span
          v-if="personRoleLabel"
          data-testid="ec-person-role"
          :data-role="personRole"
          class="rounded-full bg-overlay px-2 py-0.5 text-[0.65rem] font-bold uppercase tracking-wide text-person"
          :class="personRole === 'host' ? 'ring-1 ring-person' : ''"
        >{{ personRoleLabel }}</span>
      </span>
      <!-- Title + primary actions on ONE row (UXS-014 detail template): the name reads on the left,
           Follow and the other actions sit at the right edge of the same row, not stacked beneath. -->
      <div class="mt-1 flex items-start justify-between gap-3">
        <span class="min-w-0 flex-1 truncate font-display text-xl font-extrabold">{{ label || '…' }}</span>
        <div v-if="label" class="flex shrink-0 items-center gap-2">
          <button
            v-if="auth.isAuthenticated"
            type="button"
            class="inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-bold transition"
            :class="following ? 'bg-accent text-accent-foreground' : 'bg-overlay text-canvas-foreground hover:bg-elevated'"
            :aria-pressed="following"
            :title="t('ec.followHint')"
            @click="toggleFollow"
          >
            <span aria-hidden="true">{{ following ? '✓' : '+' }}</span>
            {{ following ? t('ec.following') : t('ec.follow') }}
          </button>
          <!-- Save (heart) — the ONE save affordance; distinct from Follow (F2.2). -->
          <FavoriteButton :item="{ kind: current.kind, ref: current.id, label }" />
          <!-- Pin this topic/person into a collection (RFC-119) — self-gates when signed out.
               Pill on this roomy detail header (CO.1). -->
          <AddToCollectionButton :item="{ kind: current.kind, ref: current.id }" variant="pill" />
        </div>
      </div>
      <!-- #1261-9: escape hatch from the modal to the standalone page. Only
           in overlay mode — inline is already the standalone page or an
           embedded panel where a link would go nowhere useful. -->
      <RouterLink
        v-if="variant === 'overlay' && label"
        :to="{ name: current.kind === 'topic' ? 'topic' : 'person', params: { id: current.id } }"
        class="mt-2 ml-2 inline-flex items-center gap-1 rounded-full bg-overlay px-3 py-1 text-xs font-bold text-canvas-foreground transition hover:bg-elevated"
        data-testid="ec-open-in-page"
        @click="emit('close')"
      >
        {{ t('ec.openInPage') }} ›
      </RouterLink>
      <!-- "Your corpus" lens (P3 #1125): all episodes, or just the ones you've heard.
           Gated on auth ALONE, deliberately — NOT on `label`. The switcher is chrome that belongs
           to the open card, not content derived from the payload, and keying it on `label` made it
           destroy itself: `load()` nulls person/topic before awaiting, so `label` goes empty on
           every scope change and the tablist unmounted the instant it was clicked. It came back
           only if the new payload happened to carry a label — and scoping to "My corpus" on an
           entity you have not heard is *honest-empty by design*, so it did not. The control you
           needed to get back to "All" was the one that disappeared, leaving the card a dead end
           until you closed and reopened it. -->
      <!--
        A radiogroup, not a tablist (#1594 item 7): the scope re-queries the one card body below
        rather than switching between panels.
      -->
      <Tabs
        v-if="auth.isAuthenticated"
        :model-value="corpusScope"
        :tabs="scopeTabs"
        :label="t('ec.scopeLabel')"
        id-prefix="ec-scope"
        variant="pill"
        pattern="radio"
        class="mt-2 text-xs"
        @update:model-value="setCorpusScope"
      />
    </header>

    <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
      <p v-if="loading" class="text-sm text-muted">{{ t('ec.loading') }}</p>
      <p v-else-if="failed || (!person && !topic)" class="text-sm text-muted">{{ t('ec.notFound') }}</p>

      <template v-else>
        <!-- Cluster identity: theme (co-occurrence "Theme") + semantic ("Similar"), or standalone.
             The Theme line carries a "Follow storyline" toggle (follows the whole thc: cluster). -->
        <div v-if="themeClusterLabel" class="mb-1 flex flex-wrap items-center gap-x-2 gap-y-1">
          <!-- The storyline this topic belongs to is a SECTION HEADING, not a caption: it names
               what you are looking at. It was `text-xs`, the smallest type in the app, which read
               as a footnote under the title. The count stays small beside it — that is metadata
               about the heading, and the instrument voice is where measured values live. -->
          <p class="lp-section text-theme">
            {{ t('kp.theme', { cluster: themeClusterLabel })
            }}<span v-if="themeClusterSize" class="lp-kicker ml-1"> · {{ t('ec.clusterSize', themeClusterSize, { named: { count: themeClusterSize } }) }}</span>
          </p>
          <button
            v-if="auth.isAuthenticated && themeClusterId"
            type="button"
            data-testid="ec-follow-storyline"
            class="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[0.7rem] font-bold transition"
            :class="followingStoryline ? 'bg-accent text-accent-foreground' : 'bg-overlay text-canvas-foreground hover:bg-elevated'"
            :aria-pressed="followingStoryline"
            :title="t('ec.followStorylineHint')"
            @click="toggleStoryline"
          >
            <span aria-hidden="true">{{ followingStoryline ? '✓' : '+' }}</span>
            {{ followingStoryline ? t('ec.followingStoryline') : t('ec.followStoryline') }}
          </button>
        </div>
        <p v-if="themeLabel" class="mb-3 text-xs text-topic">
          {{ t('kp.similar', { cluster: themeLabel })
          }}<span v-if="clusterSize"> · {{ t('ec.clusterSize', clusterSize, { named: { count: clusterSize } }) }}</span>
        </p>
        <p v-if="isTopic && !themeLabel && !themeClusterLabel" class="mb-3 text-xs text-muted">
          {{ t('ec.singleTopic') }}
        </p>

        <!-- Enrichment signals (Plan B) — momentum first, up top (operator feedback): momentum /
             similar / discussed-alongside (topic); grounding / co-appears / consensus (person).
             Hides itself when empty.

             SYNTHESIS BEFORE SEARCH (#1595). A full-width accent "Search every episode for X"
             button used to sit above this, so the card's most prominent control sent you AWAY to a
             list of matches — on the one surface whose entire purpose is the synthesis below it
             (perspectives, consensus, conversation arc, who talks about this). Search is still one
             tap away, demoted to a secondary control after the signals. -->
        <EntitySignals
          :kind="current.kind"
          :id="current.id"
          @open="(p) => open(p.kind, p.id)"
        />

        <button
          type="button"
          class="mb-4 w-full rounded-full border border-border px-4 py-2 text-sm font-bold text-canvas-foreground transition hover:bg-overlay"
          @click="searchLibrary"
        >
          {{ t('ec.searchLibrary', { term: label }) }}
        </button>

        <!-- Every semantically SIMILAR topic: the one you're on (ringed) + siblings, with a count.
             Distinct from the storyline section below, which is co-occurrence (#1603). -->
        <section v-if="siblings.length" class="mb-4">
          <h3 class="lp-section mb-2">
            {{ t('ec.clusterMembers', siblings.length + 1, { named: { count: siblings.length + 1 } }) }}
          </h3>
          <div class="flex flex-wrap gap-1.5">
            <span class="rounded-full bg-overlay px-2.5 py-1 text-xs font-semibold text-topic ring-1 ring-topic">
              {{ label }}
            </span>
            <button
              v-for="s in siblings"
              :key="s.id"
              type="button"
              class="rounded-full bg-overlay px-2.5 py-1 text-xs text-topic transition hover:bg-elevated"
              @click="open('topic', s.id)"
            >{{ s.label }}</button>
          </div>
        </section>

        <!-- Theme-cluster members (co-occurrence): topics discussed together with this one. -->
        <section v-if="themeSiblings.length" class="mb-4" data-testid="ec-theme-members">
          <h3 class="lp-section mb-2">
            {{ t('ec.themeMembers', themeSiblings.length + 1, { named: { count: themeSiblings.length + 1 } }) }}
          </h3>
          <div class="flex flex-wrap gap-1.5">
            <span class="lp-theme-chip rounded-full px-2.5 py-1 text-xs font-semibold text-surface-foreground">
              {{ label }}
            </span>
            <button
              v-for="s in themeSiblings"
              :key="s.id"
              type="button"
              class="lp-theme-chip rounded-full px-2.5 py-1 text-xs text-surface-foreground transition"
              @click="open('topic', s.id)"
            >{{ s.label }}</button>
          </div>
        </section>

        <!-- Shows this person hosts (their own shows) — kept distinct from guest appearances
             below. A host can be a guest elsewhere, so this is per-show, not a global role. -->
        <section v-if="hostShows.length" class="mb-4" data-testid="ec-host-shows">
          <h3 class="lp-section mb-2">{{ t('ec.hostOf') }}</h3>
          <div class="flex flex-col">
            <RouterLink
              v-for="s in hostShows"
              :key="s.feed_id"
              :to="{ name: 'podcast', params: { feedId: s.feed_id } }"
              class="flex items-center gap-3 border-b border-border py-2 no-underline text-canvas-foreground hover:bg-overlay"
              @click="emit('close')"
            >
              <span class="min-w-0 flex-1 truncate text-sm font-semibold">{{ s.title }}</span>
              <span class="lp-kicker shrink-0">{{
                t('ec.showEpisodeCount', s.episode_count, { named: { count: s.episode_count } })
              }}</span>
            </RouterLink>
          </div>
        </section>

        <section v-if="shownEpisodes.length" class="mb-4">
          <!--
            The order is STATED rather than offered as a control (#2004 item 11).

            The list was already newest-first — `_sorted_episode_cards` in
            `server/app_relational_view.py:138` sorts on `publish_date` descending and the client
            only filters — but nothing said so, which leaves a reader unable to tell a deliberate
            order from an arbitrary one. Marko asked for the fact, not a sort control: a control
            invites a decision where there is nothing to decide.

            The qualifier is a kicker, so it reads as an annotation on the heading rather than part
            of the count. It works for the person headings too, which sort through the same function.
          -->
          <h3 class="lp-section mb-2 flex flex-wrap items-baseline gap-x-2">
            <span>{{
              current.kind !== 'person'
                ? t('ec.topicEpisodes', episodeCount, { named: { count: episodeCount } })
                : hostShows.length
                  ? t('ec.personOtherEpisodes', shownEpisodes.length, { named: { count: shownEpisodes.length } })
                  : t('ec.personEpisodes', episodeCount, { named: { count: episodeCount } })
            }}</span>
            <span class="lp-kicker" data-testid="episodes-order">{{ t('ec.newestFirst') }}</span>
          </h3>
          <ul class="flex flex-col">
            <li v-for="e in shownEpisodes" :key="e.slug">
              <RouterLink
                :to="{ name: 'player', params: { slug: e.slug } }"
                class="flex items-start gap-3 border-b border-border py-2 no-underline text-canvas-foreground hover:bg-overlay"
                @click="emit('close')"
              >
                <img
                  v-if="epArt(e)"
                  :src="epArt(e)!"
                  alt=""
                  loading="lazy"
                  class="h-10 w-10 shrink-0 rounded-md bg-elevated object-cover"
                />
                <div v-else class="h-10 w-10 shrink-0 rounded-md bg-elevated" />
                <span class="min-w-0 flex-1">
                  <span class="block text-sm font-semibold">{{ e.title }}</span>
                  <span v-if="e.podcast_title" class="lp-kicker block">{{ e.podcast_title }}</span>
                </span>
              </RouterLink>
            </li>
          </ul>
        </section>

        <!-- Multi-perspective synthesis (#1146): each guest's take on this topic. Topic-only;
             hides itself when the topic has no speaker-attributable insight. The arc is a
             corpus-wide aggregate (no per-user cut), so it clears under "My corpus" like the
             rest of the card (operator feedback). -->
        <TopicConversationArc v-if="isTopic" :id="current.id" :scope="corpusScope" />

        <TopicPerspectives
          v-if="isTopic"
          :id="current.id"
          :scope="corpusScope"
          @open="(p) => open(p.kind, p.id)"
        />

        <section v-if="relatedPeople.length" class="mb-4">
          <h3 class="lp-section mb-2">{{ t('ec.relatedPeople') }}</h3>
          <div class="flex flex-wrap gap-1.5">
            <button
              v-for="p in relatedPeople"
              :key="p.id"
              type="button"
              class="rounded-full bg-overlay px-2.5 py-1 text-xs text-person transition hover:bg-elevated"
              @click="open('person', p.id)"
            >{{ p.name }}</button>
          </div>
        </section>

        <section v-if="relatedTopics.length">
          <h3 class="lp-section mb-2">{{ t('ec.relatedTopics') }}</h3>
          <div class="flex flex-wrap gap-1.5">
            <button
              v-for="tp in relatedTopics"
              :key="tp.id"
              type="button"
              class="rounded-full bg-overlay px-2.5 py-1 text-xs text-topic transition hover:bg-elevated"
              @click="open('topic', tp.id)"
            >{{ tp.label }}</button>
          </div>
        </section>

        <!-- Notes on this topic/person (TD.7 / PD.4). -->
        <NoteComposer :target="current.kind" :target-id="current.id" />
      </template>
    </div>
  </div>
</template>
