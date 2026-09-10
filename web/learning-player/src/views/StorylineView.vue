<script setup lang="ts">
/**
 * Storyline page (F4.5) — a storyline is a THEME CLUSTER (topics discussed together). It used to
 * open as a half-screen sheet; it is now a full page with the topic-page look: back at top, title +
 * actions on one row, the member topics, top episodes and the people involved, and notes.
 *
 * There is no dedicated storyline endpoint — the anchor topic's card IS the storyline (its
 * `theme_cluster_*` + `theme_sibling_topics` + `related_people` + `episodes`), so the route param is
 * the anchor topic id and everything derives from `getTopicCard`.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import { getTopicCard } from '../services/api'
import { useAuthStore } from '../stores/auth'
import { useInterestsStore } from '../stores/interests'
import { episodeArtwork } from '../utils/episode'
import NoteComposer from '../components/NoteComposer.vue'
import FavoriteButton from '../components/FavoriteButton.vue'
import type { Entity, EpisodeSummary } from '../services/types'

type Member = { id: string; label: string }

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
const label = ref('')
const topics = ref<Member[]>([])
const people = ref<Entity[]>([])
const episodes = ref<EpisodeSummary[]>([])
const themeClusterId = ref<string | null>(null)

const epArt = episodeArtwork

async function load(anchorTopicId: string): Promise<void> {
  loading.value = true
  failed.value = false
  try {
    const card = await getTopicCard(anchorTopicId)
    label.value = card.theme_cluster_label ?? card.label
    themeClusterId.value = card.theme_cluster_id ?? null
    // Anchor + its theme siblings = the storyline's topics; de-dupe (the API may include the anchor).
    const members: Member[] = [
      { id: card.id, label: card.label },
      ...(card.theme_sibling_topics ?? []).map((tp) => ({ id: tp.id, label: tp.label })),
    ]
    const seen = new Set<string>()
    topics.value = members.filter((tp) => tp.id && !seen.has(tp.id) && seen.add(tp.id))
    people.value = card.related_people ?? []
    episodes.value = card.episodes ?? []
  } catch {
    failed.value = true
  } finally {
    loading.value = false
  }
}
watch(() => props.id, (id) => void load(id), { immediate: true })

const following = computed(() => !!themeClusterId.value && interests.has(themeClusterId.value))
function toggleFollow(): void {
  if (themeClusterId.value) void interests.toggle(themeClusterId.value)
}

function goBack(): void {
  if (window.history.length > 1) router.back()
  else void router.push({ name: 'browse', query: { tab: 'topics' } })
}
</script>

<template>
  <section class="mx-auto max-w-3xl px-4 pb-8 pt-4" data-testid="storyline-view">
    <!-- Back on its own row, then kicker → title (UXS-014 header order). -->
    <button type="button" class="lp-nav" :aria-label="t('nav.back')" @click="goBack">
      <span aria-hidden="true" class="text-base leading-none">‹</span>
      <span>{{ t('nav.back') }}</span>
    </button>

    <div class="mt-3 flex items-start justify-between gap-3">
      <div class="min-w-0">
        <span class="lp-kicker text-theme">{{ t('home.storylines') }}</span>
        <h1 class="mt-1 font-display text-2xl font-extrabold tracking-tight">{{ label || '…' }}</h1>
      </div>
      <div class="flex shrink-0 items-center gap-2">
        <!-- Save (heart) is a per-kind favorite — a storyline lands in Library › Saved like any
             other kind (F2.2). Distinct from Follow, which subscribes to the theme cluster. -->
        <FavoriteButton :item="{ kind: 'storyline', ref: id, label: label || id }" />
        <button
          v-if="auth.isAuthenticated && themeClusterId"
          type="button"
          class="inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-bold transition"
          :class="following ? 'bg-accent text-accent-foreground' : 'bg-overlay text-canvas-foreground hover:bg-elevated'"
          :aria-pressed="following"
          data-testid="storyline-follow"
          @click="toggleFollow"
        >
          <span aria-hidden="true">{{ following ? '✓' : '+' }}</span>
          {{ following ? t('ec.followingStoryline') : t('ec.followStoryline') }}
        </button>
      </div>
    </div>

    <p v-if="loading" class="mt-4 text-sm text-muted">{{ t('home.storylineSheetLoading') }}</p>
    <p v-else-if="failed || !topics.length" class="mt-4 text-sm text-muted">
      {{ t('home.storylineSheetEmpty') }}
    </p>

    <template v-else>
      <!-- Member topics, an ordered list (SL.1). -->
      <section class="mt-6">
        <h2 class="lp-section mb-2">{{ t('home.storylineTopicsHeading') }}</h2>
        <ol class="flex flex-col">
          <li v-for="(tp, i) in topics" :key="tp.id">
            <RouterLink
              :to="{ name: 'topic', params: { id: tp.id } }"
              class="flex items-center gap-3 border-b border-border py-2 no-underline text-canvas-foreground hover:bg-overlay"
            >
              <span class="w-5 shrink-0 text-center text-xs font-bold tabular-nums text-muted">{{ i + 1 }}</span>
              <span class="min-w-0 flex-1 truncate text-sm font-semibold text-topic">{{ tp.label }}</span>
              <span class="shrink-0 text-muted" aria-hidden="true">›</span>
            </RouterLink>
          </li>
        </ol>
      </section>

      <!-- Top episodes for the storyline (SL.2). -->
      <section v-if="episodes.length" class="mt-6">
        <h2 class="lp-section mb-2">{{ t('ec.topicEpisodes', episodes.length, { named: { count: episodes.length } }) }}</h2>
        <ul class="flex flex-col">
          <li v-for="e in episodes" :key="e.slug">
            <RouterLink
              :to="{ name: 'player', params: { slug: e.slug } }"
              class="flex items-start gap-3 border-b border-border py-2 no-underline text-canvas-foreground hover:bg-overlay"
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

      <!-- People involved (SL.2). -->
      <section v-if="people.length" class="mt-6">
        <h2 class="lp-section mb-2">{{ t('ec.relatedPeople') }}</h2>
        <div class="flex flex-wrap gap-1.5">
          <RouterLink
            v-for="p in people"
            :key="p.id"
            :to="{ name: 'person', params: { id: p.id } }"
            class="rounded-full bg-overlay px-2.5 py-1 text-xs text-person no-underline transition hover:bg-elevated"
          >{{ p.name }}</RouterLink>
        </div>
      </section>

      <!-- Notes on this storyline (SL.3). -->
      <NoteComposer target="storyline" :target-id="id" />
    </template>
  </section>
</template>
