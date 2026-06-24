<script setup lang="ts">
/**
 * Corpus-wide grounded search (PRD-042 FR5 / RFC-099 §Home). Searches the whole library via
 * GET /api/app/search (extractive, no request-time LLM). Rather than a flat wall of mixed
 * passages, results are **grouped by episode** (ranked by their best hit) and each passage is
 * labelled by kind (Insight / Transcript / Topic). A "Play from …" jump appears only when the
 * passage carries a real timestamp — otherwise we open the episode rather than fake a 0:00.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { searchCorpus } from '../services/api'
import type { SearchHit } from '../services/types'
import { hitStartSeconds } from '../player/insights'
import { formatTime } from '../player/transcriptSync'
import { formatPublishDate } from '../utils/format'

const { t, locale } = useI18n()
const route = useRoute()
const router = useRouter()

const query = ref(String(route.query.q ?? ''))
const results = ref<SearchHit[]>([])
const searching = ref(false)
const error = ref(false)
const ran = ref(false)

type Kind = 'insight' | 'transcript' | 'topic' | 'passage'
interface EpisodeGroup {
  slug: string | null
  title: string
  show: string | null
  date: string | null
  art: string | null
  hits: SearchHit[]
}

const md = (h: SearchHit) => h.metadata as Record<string, unknown>
const hitSlug = (h: SearchHit) => (md(h).episode_slug as string | undefined) ?? null
const hitEpisode = (h: SearchHit) => (md(h).episode_title as string | undefined) ?? null
const hitShow = (h: SearchHit) => (md(h).podcast_title as string | undefined) ?? null
const hitDate = (h: SearchHit) => (md(h).publish_date as string | undefined) ?? null
const hitArt = (h: SearchHit) => (md(h).episode_artwork as string | undefined) ?? null

function hitKind(h: SearchHit): Kind {
  const dt = md(h).doc_type
  if (dt === 'insight') return 'insight'
  if (dt === 'transcript') return 'transcript'
  if (dt === 'kg_topic') return 'topic'
  return 'passage'
}

// Group passages under their source episode, preserving rank order (results arrive best-first,
// so an episode's rank is its first appearance).
const groups = computed<EpisodeGroup[]>(() => {
  const byKey = new Map<string, EpisodeGroup>()
  const order: string[] = []
  for (const h of results.value) {
    const slug = hitSlug(h)
    const key = slug ?? `doc:${h.doc_id}`
    let g = byKey.get(key)
    if (!g) {
      g = {
        slug,
        title: hitEpisode(h) ?? t('player.notFound'),
        show: hitShow(h),
        date: hitDate(h),
        art: hitArt(h),
        hits: [],
      }
      byKey.set(key, g)
      order.push(key)
    }
    g.hits.push(h)
  }
  return order.map((k) => byKey.get(k)!)
})

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

function openEpisode(slug: string | null, hit?: SearchHit): void {
  if (!slug) return
  const s = hit ? hitStartSeconds(hit) : null
  void router.push({
    name: 'player',
    params: { slug },
    query: s != null ? { t: String(Math.floor(s)) } : {},
  })
}

watch(() => route.query.q, (q) => run(String(q ?? '')), { immediate: true })

const showEmpty = computed(
  () => ran.value && !searching.value && !error.value && results.value.length === 0,
)
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

    <template v-else-if="results.length">
      <p class="mt-4 text-xs font-semibold uppercase tracking-wider text-muted">
        {{ t('search.summary', { passages: results.length, episodes: groups.length }) }}
      </p>

      <ul class="mt-3 flex flex-col gap-3">
        <li
          v-for="g in groups"
          :key="g.slug ?? g.title"
          class="overflow-hidden rounded-xl border border-border bg-surface"
        >
          <!-- Episode header: opens the player -->
          <button
            type="button"
            class="flex w-full items-center gap-3 px-4 pt-4 text-left"
            @click="openEpisode(g.slug)"
          >
            <img
              v-if="g.art"
              :src="g.art"
              alt=""
              loading="lazy"
              class="h-12 w-12 shrink-0 rounded-md bg-elevated object-cover"
            />
            <span class="min-w-0 flex-1">
              <span class="block font-display text-base font-bold leading-snug text-canvas-foreground">
                {{ g.title }}
              </span>
              <span v-if="g.show || g.date" class="lp-kicker mt-0.5 block truncate">
                {{ g.show }}<template v-if="g.show && g.date"> · </template>{{ g.date ? formatPublishDate(g.date, locale) : '' }}
              </span>
            </span>
            <span class="shrink-0 text-xs font-semibold text-muted">{{ t('search.matchCount', g.hits.length) }}</span>
          </button>

          <!-- Matching passages -->
          <ul class="mt-3 flex flex-col">
            <li
              v-for="(h, i) in g.hits"
              :key="h.doc_id + i"
              class="border-t border-border px-4 py-3"
            >
              <div class="flex items-center gap-2">
                <span
                  class="rounded bg-overlay px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider"
                  :class="{
                    'text-grounded': hitKind(h) === 'insight',
                    'text-canvas-foreground': hitKind(h) === 'transcript' || hitKind(h) === 'passage',
                    'text-topic': hitKind(h) === 'topic',
                  }"
                >
                  {{ t(`search.kind.${hitKind(h)}`) }}
                </span>
                <button
                  v-if="hitStartSeconds(h) != null && g.slug"
                  type="button"
                  class="ml-auto font-mono text-xs font-bold text-accent"
                  :aria-label="t('search.jumpTo', { time: formatTime(hitStartSeconds(h) ?? 0), episode: g.title })"
                  @click="openEpisode(g.slug, h)"
                >
                  ▶ {{ t('search.playHere', { time: formatTime(hitStartSeconds(h) ?? 0) }) }}
                </button>
              </div>
              <p
                class="mt-1.5 line-clamp-2 text-sm leading-relaxed"
                :class="hitKind(h) === 'topic' ? 'italic text-muted' : 'text-surface-foreground'"
              >
                {{ h.text }}
              </p>
            </li>
          </ul>
        </li>
      </ul>
    </template>
  </section>
</template>
