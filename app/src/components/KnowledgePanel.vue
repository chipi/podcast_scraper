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
import { RouterLink, useRouter } from 'vue-router'
import { getRelated, searchEpisode } from '../services/api'
import type { EpisodeDetail, EpisodeSummary, Entity, Insight, SearchHit, Topic } from '../services/types'
import { formatTime } from '../player/transcriptSync'
import { hitStartSeconds, insightStartSeconds } from '../player/insights'

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
const router = useRouter()

const summary = computed(() => props.episode.summary_text || props.episode.summary_title || null)
const hasAnything = computed(
  () =>
    Boolean(summary.value) ||
    props.topics.length > 0 ||
    props.persons.length > 0 ||
    props.insights.length > 0,
)

// A topic/person chip explores that term across the whole library (clear, consistent action).
function exploreSearch(term: string): void {
  const q = term.replace(/^person:/, '').replace(/-/g, ' ').trim()
  if (q) void router.push({ name: 'search', query: { q } })
}

// --- Topics + People as one compact, expandable row (height-optimised) ---
type Tag = { key: string; label: string; kind: 'topic' | 'person' }
const allTags = computed<Tag[]>(() => [
  ...props.topics.map((tp) => ({ key: tp.id, label: tp.label, kind: 'topic' as const })),
  ...props.persons.map((p) => ({ key: p.id, label: p.name, kind: 'person' as const })),
])
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

function speakerLabel(s: string | null): string | null {
  if (!s) return null
  return s.startsWith('person:') ? s.slice('person:'.length).replace(/-/g, ' ') : s
}

// --- related ("more like this") ---
const epArt = (e: EpisodeSummary) => e.artwork_url || e.episode_image_url || e.feed_image_url
const related = ref<EpisodeSummary[]>([])
async function loadRelated(slug: string): Promise<void> {
  try {
    related.value = (await getRelated(slug)).items.filter((e) => e.slug !== slug)
  } catch {
    related.value = []
  }
}
onMounted(() => loadRelated(props.slug))
watch(() => props.slug, (s) => loadRelated(s))
</script>

<template>
  <aside class="flex h-full flex-col bg-surface" :aria-label="t('kp.title')">
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
        <h3 class="lp-kicker mb-1">{{ t('kp.summary') }}</h3>
        <p class="text-sm leading-relaxed text-surface-foreground">{{ summary }}</p>
      </section>

      <!-- Topics & People — one compact, expandable row; tap a chip to explore the library -->
      <section v-if="allTags.length" class="mb-5">
        <h3 class="lp-kicker mb-2">{{ t('kp.tags') }}</h3>
        <div class="flex flex-wrap gap-1.5">
          <button
            v-for="tag in visibleTags"
            :key="tag.key"
            type="button"
            class="rounded-full bg-overlay px-2.5 py-1 text-xs transition hover:bg-elevated"
            :class="tag.kind === 'topic' ? 'text-topic' : 'text-person'"
            :aria-label="t('kp.exploreTerm', { term: tag.label })"
            @click="exploreSearch(tag.label)"
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
          <h3 class="lp-kicker">{{ t('kp.insights') }} · {{ insights.length }}</h3>
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
              <button
                v-if="insightStartSeconds(ins) != null"
                type="button"
                class="font-mono text-xs text-accent"
                @click="emit('seek', insightStartSeconds(ins) as number)"
              >
                ▶ {{ formatTime(insightStartSeconds(ins) as number) }}
              </button>
            </div>
            <p class="mt-1 text-sm font-semibold text-surface-foreground">{{ ins.text }}</p>
            <blockquote v-if="ins.quotes[0]" class="mt-2 border-l-2 border-border pl-3 text-sm text-muted">
              “{{ ins.quotes[0].text }}”
              <span v-if="speakerLabel(ins.quotes[0].speaker)" class="block text-xs text-person">
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
        <h3 class="lp-kicker mb-2">{{ t('kp.related') }}</h3>
        <ul class="flex flex-col">
          <li v-for="r in related" :key="r.slug">
            <RouterLink
              :to="{ name: 'player', params: { slug: r.slug } }"
              class="flex items-center gap-3 border-b border-border py-2 no-underline text-canvas-foreground hover:bg-overlay"
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
                <span v-if="r.podcast_title" class="lp-kicker block truncate">{{ r.podcast_title }}</span>
              </span>
            </RouterLink>
          </li>
        </ul>
      </section>
    </div>
  </aside>
</template>
