<script setup lang="ts">
/**
 * Corpus-wide grounded search (PRD-042 FR5 / RFC-099 §Home). Searches the whole library via
 * GET /api/app/search (extractive, no request-time LLM) and renders grounded passages; each
 * jumps to the source episode at the moment (player `?t=` deep-link). Graceful empty/no-index.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { searchCorpus } from '../services/api'
import type { SearchHit } from '../services/types'
import { hitStartSeconds } from '../player/insights'
import { formatTime } from '../player/transcriptSync'

const { t } = useI18n()
const route = useRoute()
const router = useRouter()

const query = ref(String(route.query.q ?? ''))
const results = ref<SearchHit[]>([])
const searching = ref(false)
const error = ref(false)
const ran = ref(false)

const hitSlug = (h: SearchHit) => (h.metadata as { episode_slug?: string }).episode_slug
const hitEpisode = (h: SearchHit) =>
  (h.metadata as { episode_title?: string; podcast_title?: string }).episode_title
const hitShow = (h: SearchHit) => (h.metadata as { podcast_title?: string }).podcast_title

async function run(q: string): Promise<void> {
  const term = q.trim()
  if (!term) {
    results.value = []
    ran.value = false
    return
  }
  searching.value = true
  error.value = false
  try {
    const resp = await searchCorpus(term)
    results.value = resp.results
    error.value = Boolean(resp.error)
  } catch {
    error.value = true
  } finally {
    searching.value = false
    ran.value = true
  }
}

function submit(): void {
  void router.replace({ name: 'search', query: { q: query.value.trim() } })
}

function jump(h: SearchHit): void {
  const slug = hitSlug(h)
  if (!slug) return
  const s = hitStartSeconds(h)
  void router.push({
    name: 'player',
    params: { slug },
    query: s != null ? { t: String(Math.floor(s)) } : {},
  })
}

watch(() => route.query.q, (q) => run(String(q ?? '')), { immediate: true })

const showEmpty = computed(() => ran.value && !searching.value && !error.value && results.value.length === 0)
</script>

<template>
  <section>
    <h1 class="mb-4 font-display text-3xl font-extrabold tracking-tight">{{ t('search.title') }}</h1>

    <form class="flex gap-2" @submit.prevent="submit">
      <label class="sr-only" for="search-q">{{ t('search.title') }}</label>
      <input
        id="search-q"
        v-model="query"
        type="search"
        :placeholder="t('search.placeholder')"
        class="min-w-0 flex-1 rounded-full border border-border bg-surface px-4 py-3 text-sm"
      />
      <button type="submit" class="rounded-full bg-accent px-5 py-3 font-bold text-accent-foreground">
        {{ t('search.title') }}
      </button>
    </form>

    <p v-if="searching" class="mt-4 text-muted">{{ t('search.searching') }}</p>
    <p v-else-if="error" class="mt-4 text-muted">{{ t('search.error') }}</p>
    <p v-else-if="showEmpty" class="mt-4 text-muted">{{ t('search.noResults') }}</p>

    <ul v-else class="mt-4 flex flex-col">
      <li v-for="(h, i) in results" :key="h.doc_id + i" class="border-b border-border py-4">
        <p class="text-sm leading-relaxed text-surface-foreground">{{ h.text }}</p>
        <div class="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
          <span v-if="hitEpisode(h)" class="text-muted">
            {{ t('search.in') }} <span class="text-canvas-foreground">{{ hitEpisode(h) }}</span>
            <template v-if="hitShow(h)"> · {{ hitShow(h) }}</template>
          </span>
          <button
            v-if="hitSlug(h)"
            type="button"
            class="font-mono text-accent"
            :aria-label="t('search.jumpTo', { time: formatTime(hitStartSeconds(h) ?? 0), episode: hitEpisode(h) ?? '' })"
            @click="jump(h)"
          >
            ▶ {{ formatTime(hitStartSeconds(h) ?? 0) }}
          </button>
        </div>
      </li>
    </ul>
  </section>
</template>
