<script setup lang="ts">
/**
 * Knowledge Panel (PRD-039 FR4 / RFC-099 §5) — the learning surface beside the player.
 * Sections (each independently hidden when its artifact is absent):
 *   Ask (extractive grounded search, no LLM) · Summary · Topics · People · Insights.
 * Every timestamp + Ask result emits `seek` for jump-to-moment. Tapping a person filters
 * the insight list. The "surfacing now" insight (driven by playback) is highlighted.
 *
 * "More like this" surfaces semantic peer episodes (vector similarity) at the foot of the
 * panel — the consolidation loop: finish here, keep learning next.
 */
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { getRelated, searchEpisode } from '../services/api'
import type {
  EpisodeDetail,
  EpisodeSummary,
  Entity,
  FavoriteAdd,
  Insight,
  SearchHit,
  Topic,
} from '../services/types'
import { formatTime } from '../player/transcriptSync'
import { hitStartSeconds, insightStartSeconds } from '../player/insights'
import { speakerLabel } from '../utils/format'
import { episodeArtwork } from '../utils/episode'
import { useAuthStore } from '../stores/auth'
import { useQueueStore } from '../stores/queue'
import { useCaptureStore } from '../stores/capture'
import EntityCardBody from './EntityCardBody.vue'
import FavoriteButton from './FavoriteButton.vue'

function favInsight(ins: Insight): FavoriteAdd {
  const secs = insightStartSeconds(ins)
  return {
    kind: 'insight',
    ref: `${props.slug}#${ins.id}`,
    label: ins.text,
    sublabel: props.episode.title,
    slug: props.slug,
    start_ms: secs != null ? Math.round(secs * 1000) : undefined,
  }
}

const props = withDefaults(
  defineProps<{
    episode: EpisodeDetail
    insights: Insight[]
    topics: Topic[]
    persons: Entity[]
    slug: string
    activeInsightId: string | null
    /** An insight tapped from the transcript — scroll it into view + highlight it. */
    focusInsightId?: string | null
  }>(),
  { focusInsightId: null },
)
const emit = defineEmits<{ (e: 'seek', seconds: number): void; (e: 'close'): void }>()

const { t } = useI18n()

const summary = computed(() => props.episode.summary_text || props.episode.summary_title || null)
const hasAnything = computed(
  () =>
    Boolean(summary.value) ||
    props.topics.length > 0 ||
    props.persons.length > 0 ||
    props.insights.length > 0,
)

// --- Topics + People as one compact, expandable row; topics cluster-first (RFC-102) ---
type Tag = {
  key: string
  label: string
  kind: 'topic' | 'person'
  dominant: boolean
  themeMember: boolean
}

// Tapping a chip opens its entity card (PRD-043; library search now lives inside the card).
const cardTarget = ref<{ kind: 'person' | 'topic'; id: string } | null>(null)
function openCard(tag: Tag): void {
  cardTarget.value = { kind: tag.kind, id: tag.key }
}

// How many of THIS episode's topics fall in each corpus cluster (intra-episode dominance).
const topicClusterCounts = computed<Record<string, number>>(() => {
  const c: Record<string, number> = {}
  for (const t of props.topics) if (t.cluster_id) c[t.cluster_id] = (c[t.cluster_id] ?? 0) + 1
  return c
})
// The dominant cluster = the one with the most of this episode's topics (≥2), tie → larger corpus
// cluster; null when no topic is clustered or none reaches 2 (then it's a flat list).
const dominantClusterId = computed<string | null>(() => {
  const counts = topicClusterCounts.value
  let best: string | null = null
  let bestCount = 1
  let bestSize = -1
  for (const t of props.topics) {
    if (!t.cluster_id) continue
    const n = counts[t.cluster_id] ?? 0
    if (n > bestCount || (n === bestCount && t.cluster_size > bestSize)) {
      best = t.cluster_id
      bestCount = n
      bestSize = t.cluster_size
    }
  }
  return best
})
const dominantClusterLabel = computed(
  () => props.topics.find((t) => t.cluster_id === dominantClusterId.value)?.cluster_label ?? null,
)

// Theme clusters (co-occurrence "discussed together") — parallel to the semantic dominant above.
// Marked on the pills (theme ring) + a "Theme ·" lead-in. No-op when topics carry no theme_cluster_id.
const themeClusterCounts = computed<Record<string, number>>(() => {
  const c: Record<string, number> = {}
  for (const t of props.topics)
    if (t.theme_cluster_id) c[t.theme_cluster_id] = (c[t.theme_cluster_id] ?? 0) + 1
  return c
})
const themeDominantId = computed<string | null>(() => {
  const counts = themeClusterCounts.value
  let best: string | null = null
  let bestCount = 1
  let bestSize = -1
  for (const t of props.topics) {
    if (!t.theme_cluster_id) continue
    const n = counts[t.theme_cluster_id] ?? 0
    if (n > bestCount || (n === bestCount && (t.theme_cluster_size ?? 0) > bestSize)) {
      best = t.theme_cluster_id
      bestCount = n
      bestSize = t.theme_cluster_size ?? 0
    }
  }
  return best
})
const themeDominantLabel = computed(
  () =>
    props.topics.find((t) => t.theme_cluster_id === themeDominantId.value)?.theme_cluster_label ??
    null,
)
const allTags = computed<Tag[]>(() => {
  const counts = topicClusterCounts.value
  const dom = dominantClusterId.value
  // Rank: dominant cluster first, then other clustered (larger intra-episode groups earlier),
  // then singletons. Stable sort keeps original order within a rank.
  const rank = (t: { cluster_id: string | null }): number =>
    t.cluster_id === dom && dom ? 0 : t.cluster_id ? 100 - (counts[t.cluster_id] ?? 0) : 1000
  const topics = [...props.topics].sort((a, b) => rank(a) - rank(b))
  return [
    ...topics.map((tp) => ({
      key: tp.id,
      label: tp.label,
      kind: 'topic' as const,
      dominant: Boolean(dom) && tp.cluster_id === dom,
      themeMember: Boolean(tp.theme_cluster_id),
    })),
    ...props.persons.map((p) => ({
      key: p.id,
      label: p.name,
      kind: 'person' as const,
      dominant: false,
      themeMember: false,
    })),
  ]
})
const TAG_COLLAPSED = 6
const tagsExpanded = ref(false)
const visibleTags = computed(() =>
  tagsExpanded.value ? allTags.value : allTags.value.slice(0, TAG_COLLAPSED),
)
const hiddenTagCount = computed(() => Math.max(0, allTags.value.length - TAG_COLLAPSED))

// A grounded insight is one with a timestamped supporting quote (sourced in the audio); the
// rest are ungrounded claims — that's why only some show a quote + play button.
function isGrounded(ins: Insight): boolean {
  return insightStartSeconds(ins) != null
}

const showAll = ref(false)
const visibleInsights = computed(() =>
  showAll.value ? props.insights : props.insights.slice(0, 5),
)

// Scroll a transcript-tapped insight into view (and reveal it past the 5-item fold).
const insightEls = ref<Record<string, HTMLElement>>({})
watch(
  () => props.focusInsightId,
  async (id) => {
    if (!id) return
    showAll.value = true
    await nextTick()
    // rAF so the panel (and on mobile, its open transition) has laid out before we centre —
    // scrollIntoView walks every scroll ancestor, bringing the claim into the viewport too.
    requestAnimationFrame(() => {
      insightEls.value[id]?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    })
  },
)

// --- ask (extractive grounded search) ---
const q = ref('')
const results = ref<SearchHit[]>([])
const searching = ref(false)
const askError = ref(false)
async function runSearch(): Promise<void> {
  const query = q.value.trim()
  if (!query) {
    results.value = []
    return
  }
  searching.value = true
  askError.value = false
  try {
    const resp = await searchEpisode(props.slug, query)
    results.value = resp.results
    askError.value = Boolean(resp.error)
  } catch {
    askError.value = true
  } finally {
    searching.value = false
  }
}

// --- capture (P2, PRD-040): save a grounded insight to the personal highlights corpus ---
const capture = useCaptureStore()
const savedInsightIds = computed(() => capture.savedInsightIds)
function captureInsight(ins: Insight): void {
  const secs = insightStartSeconds(ins)
  void capture.captureInsight(props.slug, {
    id: ins.id,
    text: ins.text,
    start_ms: secs != null ? Math.round(secs * 1000) : null,
  })
}

// --- related ("more like this") ---
const auth = useAuthStore()
const queue = useQueueStore()
const epArt = episodeArtwork

// Queue a peer episode to play right after the current one (RFC-099 §4 "Play next").
function playNext(slug: string): void {
  void queue.playNext(slug, props.slug)
}
const related = ref<EpisodeSummary[]>([])
async function loadRelated(slug: string): Promise<void> {
  try {
    related.value = (await getRelated(slug)).items.filter((e) => e.slug !== slug)
  } catch {
    related.value = []
  }
}
onMounted(() => {
  loadRelated(props.slug)
  if (auth.isAuthenticated) void capture.ensureLoaded()
})
watch(() => props.slug, (s) => loadRelated(s))
watch(
  () => auth.isAuthenticated,
  (yes) => {
    if (yes) void capture.ensureLoaded()
  },
)
</script>

<template>
  <aside class="flex h-full flex-col bg-surface" :aria-label="t('kp.title')">
    <!-- Mobile bottom-sheet grab handle (signals the player sits behind; hidden on desktop rail). -->
    <div class="flex shrink-0 justify-center pt-2 lg:hidden" aria-hidden="true">
      <span class="h-1.5 w-10 rounded-full bg-border"></span>
    </div>
    <!-- Replace-in-panel (UXS-014): a tapped chip swaps the panel content to the entity card with a
         ‹ Back — no overlay, no second backdrop. -->
    <EntityCardBody
      v-if="cardTarget"
      variant="inline"
      :kind="cardTarget.kind"
      :id="cardTarget.id"
      @close="cardTarget = null"
    />
    <template v-else>
    <header class="flex items-center justify-between border-b border-border px-4 py-3">
      <span class="font-display text-lg font-bold">{{ t('kp.title') }}</span>
      <button type="button" class="text-muted" :aria-label="t('kp.close')" @click="emit('close')">✕</button>
    </header>

    <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
      <!-- Ask -->
      <form class="mb-5" @submit.prevent="runSearch">
        <label class="sr-only" for="kp-ask">{{ t('kp.ask') }}</label>
        <div class="flex gap-2">
          <input
            id="kp-ask"
            v-model="q"
            type="search"
            :placeholder="t('kp.askPlaceholder')"
            class="min-w-0 flex-1 rounded-full border border-border bg-canvas px-4 py-2 text-sm"
          />
          <button type="submit" class="rounded-full bg-accent px-4 py-2 text-sm font-bold text-accent-foreground">
            {{ t('kp.ask') }}
          </button>
        </div>
        <p v-if="searching" class="mt-2 text-sm text-muted">{{ t('kp.searching') }}</p>
        <p v-else-if="askError" class="mt-2 text-sm text-danger">{{ t('kp.searchError') }}</p>
        <ul v-else-if="results.length" class="mt-3 flex flex-col gap-2">
          <li v-for="hit in results" :key="hit.doc_id" class="rounded-xl border border-border p-3">
            <p class="text-sm text-surface-foreground">{{ hit.text }}</p>
            <button
              v-if="hitStartSeconds(hit) != null"
              type="button"
              class="mt-1 font-mono text-xs text-accent"
              @click="emit('seek', hitStartSeconds(hit) as number)"
            >
              ▶ {{ formatTime(hitStartSeconds(hit) as number) }}
            </button>
          </li>
        </ul>
        <p v-else-if="q.trim() && !searching" class="mt-2 text-sm text-muted">{{ t('kp.noResults') }}</p>
      </form>

      <p v-if="!hasAnything" class="text-sm text-muted">{{ t('kp.empty') }}</p>

      <!-- Summary -->
      <section v-if="summary" class="mb-5">
        <h3 class="lp-section mb-1">{{ t('kp.summary') }}</h3>
        <p class="text-sm leading-relaxed text-surface-foreground">{{ summary }}</p>
      </section>

      <!-- Topics & People — one compact, expandable row; topics cluster-first (RFC-102) -->
      <section v-if="allTags.length" class="mb-5">
        <div class="mb-2 flex items-baseline justify-between gap-2">
          <h3 class="lp-section">{{ t('kp.tags') }}</h3>
          <span
            v-if="themeDominantLabel || dominantClusterLabel"
            class="flex min-w-0 flex-col items-end text-xs leading-tight"
          >
            <span v-if="themeDominantLabel" class="truncate text-theme">
              {{ t('kp.theme', { cluster: themeDominantLabel }) }}
            </span>
            <span v-if="dominantClusterLabel" class="truncate text-topic">
              {{ t('kp.similar', { cluster: dominantClusterLabel }) }}
            </span>
          </span>
        </div>
        <div class="flex flex-wrap gap-1.5">
          <button
            v-for="tag in visibleTags"
            :key="tag.key"
            type="button"
            class="rounded-full bg-overlay px-2.5 py-1 text-xs transition hover:bg-elevated"
            :class="[
              tag.kind === 'topic' ? 'text-topic' : 'text-person',
              tag.themeMember
                ? 'ring-1 ring-theme'
                : tag.dominant
                  ? 'ring-1 ring-topic'
                  : '',
            ]"
            :aria-label="t('kp.openEntity', { term: tag.label })"
            @click="openCard(tag)"
          >
            {{ tag.label }}
          </button>
          <button
            v-if="!tagsExpanded && hiddenTagCount > 0"
            type="button"
            class="rounded-full px-2 py-1 text-xs font-bold text-accent"
            @click="tagsExpanded = true"
          >
            +{{ hiddenTagCount }} …
          </button>
        </div>
      </section>

      <!-- Insights -->
      <section v-if="insights.length">
        <div class="mb-2 flex items-center justify-between">
          <h3 class="lp-section">{{ t('kp.insights') }} · {{ insights.length }}</h3>
        </div>
        <ul class="flex flex-col gap-3">
          <li
            v-for="ins in visibleInsights"
            :key="ins.id"
            :ref="(el) => { if (el) insightEls[ins.id] = el as HTMLElement }"
            class="rounded-xl border p-3 transition-colors"
            :class="
              ins.id === activeInsightId || ins.id === focusInsightId
                ? 'border-accent bg-overlay'
                : 'border-border'
            "
          >
            <div class="flex items-center justify-between gap-2">
              <span class="flex items-center gap-1.5">
                <span
                  v-if="isGrounded(ins)"
                  class="text-grounded"
                  :title="t('kp.groundedHint')"
                  aria-hidden="true"
                >●</span>
                <span v-if="ins.insight_type" class="lp-kicker">{{ ins.insight_type }}</span>
              </span>
              <span class="flex items-center gap-2">
                <button
                  v-if="insightStartSeconds(ins) != null"
                  type="button"
                  class="font-mono text-xs text-accent"
                  @click="emit('seek', insightStartSeconds(ins) as number)"
                >
                  ▶ {{ formatTime(insightStartSeconds(ins) as number) }}
                </button>
                <!-- Save this insight to the personal highlights corpus (P2; auth-gated). -->
                <button
                  v-if="auth.isAuthenticated"
                  type="button"
                  class="rounded-full p-0.5 transition"
                  :class="savedInsightIds.has(ins.id) ? 'text-accent' : 'text-muted hover:text-accent'"
                  :aria-pressed="savedInsightIds.has(ins.id)"
                  :aria-label="savedInsightIds.has(ins.id) ? t('capture.savedInsight') : t('capture.saveInsight')"
                  :title="savedInsightIds.has(ins.id) ? t('capture.savedInsight') : t('capture.saveInsight')"
                  @click="captureInsight(ins)"
                >
                  <svg viewBox="0 0 24 24" :fill="savedInsightIds.has(ins.id) ? 'currentColor' : 'none'" stroke="currentColor" stroke-width="2" class="h-4 w-4" aria-hidden="true">
                    <path d="M6 3h12a1 1 0 0 1 1 1v17l-7-4-7 4V4a1 1 0 0 1 1-1z" />
                  </svg>
                </button>
                <FavoriteButton :item="favInsight(ins)" />
              </span>
            </div>
            <p class="mt-1 text-sm font-semibold text-surface-foreground">{{ ins.text }}</p>
            <blockquote v-if="ins.quotes[0]" class="mt-2 border-l-2 border-border pl-3 text-sm text-muted">
              “{{ ins.quotes[0].text }}”
              <span v-if="speakerLabel(ins.quotes[0].speaker)" class="lp-speaker block">
                — {{ speakerLabel(ins.quotes[0].speaker) }}
              </span>
            </blockquote>
          </li>
        </ul>
        <button
          v-if="!showAll && insights.length > 5"
          type="button"
          class="mt-3 text-sm font-bold text-accent"
          @click="showAll = true"
        >
          {{ t('kp.showAll') }}
        </button>
      </section>

      <!-- More like this (semantic peers; hidden when the index has no neighbours). -->
      <section v-if="related.length" class="mt-5">
        <h3 class="lp-section mb-2">{{ t('kp.related') }}</h3>
        <ul class="flex flex-col">
          <li v-for="r in related" :key="r.slug" class="flex items-center gap-1 border-b border-border">
            <RouterLink
              :to="{ name: 'player', params: { slug: r.slug } }"
              class="flex min-w-0 flex-1 items-center gap-3 py-2 no-underline text-canvas-foreground hover:bg-overlay"
            >
              <img
                v-if="epArt(r)"
                :src="epArt(r)!"
                alt=""
                loading="lazy"
                class="h-10 w-10 shrink-0 rounded-md bg-elevated object-cover"
              />
              <div v-else class="h-10 w-10 shrink-0 rounded-md bg-elevated" />
              <span class="min-w-0 flex-1">
                <span class="block truncate text-sm font-semibold">{{ r.title }}</span>
                <span v-if="r.podcast_title" class="lp-kicker block">{{ r.podcast_title }}</span>
              </span>
            </RouterLink>
            <!-- Play next: queue this peer right after the current episode (RFC-099 §4). -->
            <button
              v-if="auth.isAuthenticated"
              type="button"
              class="shrink-0 rounded-full p-1.5 transition hover:bg-overlay hover:text-accent"
              :class="queue.has(r.slug) ? 'text-accent' : 'text-muted'"
              :aria-label="t('queue.playNext')"
              :title="t('queue.playNext')"
              @click="playNext(r.slug)"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" class="h-4 w-4" aria-hidden="true">
                <path d="M5 5l9 7-9 7V5z" /><rect x="16" y="5" width="2.4" height="14" rx="1" />
              </svg>
            </button>
          </li>
        </ul>
      </section>
    </div>
    </template>
  </aside>
</template>
