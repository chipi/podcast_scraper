<script setup lang="ts">
/**
 * Knowledge Panel (PRD-039 FR4 / RFC-099 §5) — the learning surface beside the player.
 * Sections (each independently hidden when its artifact is absent):
 *   Ask (extractive grounded search, no LLM) · Summary · Topics · People · Insights.
 * Every timestamp + Ask result emits `seek` for jump-to-moment. Tapping a person filters
 * the insight list. The "surfacing now" insight (driven by playback) is highlighted.
 *
 * A "Related / more-like-this" section is a planned addition (logged for the gap-seek) —
 * the layout leaves room for it above the footer.
 */
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { searchEpisode } from '../services/api'
import type { EpisodeDetail, Entity, Insight, SearchHit, Topic } from '../services/types'
import { formatTime } from '../player/transcriptSync'
import { hitStartSeconds, insightStartSeconds } from '../player/insights'

const props = defineProps<{
  episode: EpisodeDetail
  insights: Insight[]
  topics: Topic[]
  persons: Entity[]
  slug: string
  activeInsightId: string | null
}>()
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

// --- person filter ---
const selectedPerson = ref<string | null>(null)
function norm(s: string): string {
  return s
    .replace(/^person:/, '')
    .replace(/-/g, ' ')
    .trim()
    .toLowerCase()
}
function togglePerson(name: string): void {
  selectedPerson.value = selectedPerson.value === name ? null : name
}
const showAll = ref(false)
const filteredInsights = computed(() => {
  if (!selectedPerson.value) return props.insights
  const want = norm(selectedPerson.value)
  return props.insights.filter((ins) => ins.quotes.some((qt) => qt.speaker && norm(qt.speaker) === want))
})
const visibleInsights = computed(() =>
  showAll.value ? filteredInsights.value : filteredInsights.value.slice(0, 5),
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
</script>

<template>
  <aside class="flex h-full flex-col bg-surface" aria-label="Knowledge">
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

      <!-- Topics -->
      <section v-if="topics.length" class="mb-5">
        <h3 class="lp-kicker mb-2">{{ t('kp.topics') }}</h3>
        <div class="flex flex-wrap gap-2">
          <span v-for="topic in topics" :key="topic.id" class="rounded-full bg-overlay px-3 py-1 text-xs text-topic">
            {{ topic.label }}
          </span>
        </div>
      </section>

      <!-- People (filter insights) -->
      <section v-if="persons.length" class="mb-5">
        <h3 class="lp-kicker mb-2">{{ t('kp.people') }}</h3>
        <div class="flex flex-wrap gap-2">
          <button
            v-for="p in persons"
            :key="p.id"
            type="button"
            class="rounded-full px-3 py-1 text-xs"
            :class="selectedPerson === p.name ? 'bg-accent text-accent-foreground' : 'bg-overlay text-person'"
            :aria-pressed="selectedPerson === p.name"
            @click="togglePerson(p.name)"
          >
            {{ p.name }}
          </button>
        </div>
      </section>

      <!-- Insights -->
      <section v-if="insights.length">
        <div class="mb-2 flex items-center justify-between">
          <h3 class="lp-kicker">{{ t('kp.insights') }} · {{ insights.length }}</h3>
          <button
            v-if="selectedPerson"
            type="button"
            class="text-xs text-accent"
            @click="selectedPerson = null"
          >
            {{ t('kp.clearFilter') }}
          </button>
        </div>
        <ul class="flex flex-col gap-3">
          <li
            v-for="ins in visibleInsights"
            :key="ins.id"
            class="rounded-xl border p-3"
            :class="ins.id === activeInsightId ? 'border-accent bg-overlay' : 'border-border'"
          >
            <div class="flex items-center justify-between gap-2">
              <span v-if="ins.insight_type" class="lp-kicker">{{ ins.insight_type }}</span>
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
          v-if="!showAll && filteredInsights.length > 5"
          type="button"
          class="mt-3 text-sm font-bold text-accent"
          @click="showAll = true"
        >
          {{ t('kp.showAll') }}
        </button>
      </section>
    </div>
  </aside>
</template>
