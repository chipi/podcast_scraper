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
import type { Entity, EpisodeSummary, PersonCard, Topic, TopicCard } from '../services/types'
import EntitySignals from './EntitySignals.vue'
import TopicPerspectives from './TopicPerspectives.vue'
import TopicConversationArc from './TopicConversationArc.vue'
import { useAuthStore } from '../stores/auth'
import { useInterestsStore } from '../stores/interests'
import { episodeArtwork } from '../utils/episode'

type Target = { kind: 'person' | 'topic'; id: string }

const props = withDefaults(
  defineProps<{ kind: 'person' | 'topic'; id: string; variant?: 'inline' | 'overlay' }>(),
  { variant: 'overlay' },
)
const emit = defineEmits<{ (e: 'close'): void }>()

const { t } = useI18n()
const router = useRouter()
const auth = useAuthStore()
const interests = useInterestsStore()
// Load follow-state once we know the user is signed in — auth may resolve after this mounts.
watch(
  () => auth.isAuthenticated,
  (authed) => {
    if (authed) void interests.ensureLoaded()
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
      <button
        type="button"
        class="lp-nav"
        :aria-label="atRoot && variant === 'overlay' ? t('ec.close') : t('ec.back')"
        @click="onBack"
      >
        <span aria-hidden="true" class="text-base leading-none">{{ atRoot && variant === 'overlay' ? '✕' : '‹' }}</span>
        <span>{{ atRoot && variant === 'overlay' ? t('ec.close') : t('ec.back') }}</span>
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
      <span class="block truncate font-display text-xl font-extrabold">{{ label || '…' }}</span>
      <button
        v-if="auth.isAuthenticated && label"
        type="button"
        class="mt-2 inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-bold transition"
        :class="following ? 'bg-accent text-accent-foreground' : 'bg-overlay text-canvas-foreground hover:bg-elevated'"
        :aria-pressed="following"
        :title="t('ec.followHint')"
        @click="toggleFollow"
      >
        <span aria-hidden="true">{{ following ? '✓' : '+' }}</span>
        {{ following ? t('ec.following') : t('ec.follow') }}
      </button>
      <!-- "Your corpus" lens (P3 #1125): all episodes, or just the ones you've heard. -->
      <div
        v-if="auth.isAuthenticated && label"
        role="tablist"
        :aria-label="t('ec.scopeLabel')"
        class="mt-2 inline-flex gap-1 rounded-full border border-border p-0.5 text-xs"
      >
        <button
          v-for="opt in (['all', 'mine'] as const)"
          :key="opt"
          type="button"
          role="tab"
          :aria-selected="corpusScope === opt"
          class="rounded-full px-2.5 py-0.5 font-semibold transition"
          :class="corpusScope === opt ? 'bg-accent text-accent-foreground' : 'text-muted hover:text-canvas-foreground'"
          @click="setCorpusScope(opt)"
        >
          {{ opt === 'all' ? t('ec.scopeAll') : t('ec.scopeMine') }}
        </button>
      </div>
    </header>

    <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
      <p v-if="loading" class="text-sm text-muted">{{ t('ec.loading') }}</p>
      <p v-else-if="failed || (!person && !topic)" class="text-sm text-muted">{{ t('ec.notFound') }}</p>

      <template v-else>
        <!-- Cluster identity: theme (co-occurrence "Theme") + semantic ("Similar"), or standalone.
             The Theme line carries a "Follow storyline" toggle (follows the whole thc: cluster). -->
        <div v-if="themeClusterLabel" class="mb-1 flex flex-wrap items-center gap-x-2 gap-y-1">
          <p class="text-xs text-theme">
            {{ t('kp.theme', { cluster: themeClusterLabel })
            }}<span v-if="themeClusterSize"> · {{ t('ec.clusterSize', themeClusterSize, { named: { count: themeClusterSize } }) }}</span>
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

        <button
          type="button"
          class="mb-4 w-full rounded-full bg-accent px-4 py-2 text-sm font-bold text-accent-foreground"
          @click="searchLibrary"
        >
          {{ t('ec.searchLibrary', { term: label }) }}
        </button>

        <!-- Enrichment signals (Plan B) — momentum first, up top (operator feedback): momentum /
             similar / discussed-alongside (topic); grounding / co-appears / consensus (person).
             Hides itself when empty. -->
        <EntitySignals
          :kind="current.kind"
          :id="current.id"
          @open="(p) => open(p.kind, p.id)"
        />

        <!-- All topics in this cluster: the one you're on (ringed) + every sibling, with a count. -->
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

        <section v-if="episodes.length" class="mb-4">
          <h3 class="lp-section mb-2">
            {{
              current.kind === 'person'
                ? t('ec.personEpisodes', episodeCount, { named: { count: episodeCount } })
                : t('ec.topicEpisodes', episodeCount, { named: { count: episodeCount } })
            }}
          </h3>
          <ul class="flex flex-col">
            <li v-for="e in episodes" :key="e.slug">
              <RouterLink
                :to="{ name: 'player', params: { slug: e.slug } }"
                class="flex items-center gap-3 border-b border-border py-2 no-underline text-canvas-foreground hover:bg-overlay"
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
                  <span class="block truncate text-sm font-semibold">{{ e.title }}</span>
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
      </template>
    </div>
  </div>
</template>
