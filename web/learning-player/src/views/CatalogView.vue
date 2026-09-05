<script setup lang="ts">
/**
 * Catalog — global all-episodes view (PRD-038 FR1). Newest-first, paginated with a "load more"
 * control (20/page). A shared ListToolbar (UXS-014) filters/sorts/searches the list; engaging any
 * control auto-loads the remaining pages so the controls cover the whole catalog, not just what's
 * been paged in.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
defineOptions({ name: 'CatalogView' }) // stable name for <keep-alive :include> (App.vue)
import EpisodeCard from '../components/EpisodeCard.vue'
import ListToolbar from '../components/ListToolbar.vue'
import { getPodcasts, listEpisodes } from '../services/api'
import type { EpisodeSummary } from '../services/types'

// `embedded` — rendered as the Episodes tab panel inside the Browse hub, which supplies the page
// heading; drop our own so it isn't shown twice.
withDefaults(defineProps<{ embedded?: boolean }>(), { embedded: false })

const PAGE_SIZE = 20
const { t } = useI18n()
const episodes = ref<EpisodeSummary[]>([])
const page = ref(0)
const hasMore = ref(false)
const loading = ref(false)
const error = ref(false)

const search = ref('')
const sort = ref('newest')
const filter = ref('all')
const show = ref('')
const shows = ref<{ id: string; label: string }[]>([])
const controlsActive = computed(
  () =>
    search.value.trim() !== '' ||
    sort.value !== 'newest' ||
    filter.value !== 'all' ||
    show.value !== '',
)

async function loadMore(): Promise<void> {
  loading.value = true
  error.value = false
  try {
    const next = page.value + 1
    const res = await listEpisodes({ page: next, pageSize: PAGE_SIZE })
    episodes.value.push(...res.items)
    page.value = next
    hasMore.value = res.has_more
  } catch {
    error.value = true
  } finally {
    loading.value = false
  }
}

// Filtering/sorting only makes sense over the whole list — pull the rest in when a control is used.
async function loadAll(): Promise<void> {
  while (hasMore.value && !error.value) await loadMore()
}
watch(controlsActive, (active) => {
  if (active && hasMore.value) void loadAll()
})

const visible = computed<EpisodeSummary[]>(() => {
  let list = episodes.value
  const q = search.value.trim().toLowerCase()
  if (q) {
    list = list.filter(
      (e) =>
        e.title.toLowerCase().includes(q) ||
        (e.podcast_title ?? '').toLowerCase().includes(q),
    )
  }
  if (filter.value === 'insights') list = list.filter((e) => e.has_gi)
  if (show.value) list = list.filter((e) => e.feed_id === show.value)
  const byDate = (e: EpisodeSummary) => e.publish_date ?? ''
  const sorted = [...list]
  if (sort.value === 'newest') sorted.sort((a, b) => byDate(b).localeCompare(byDate(a)))
  else if (sort.value === 'oldest') sorted.sort((a, b) => byDate(a).localeCompare(byDate(b)))
  else if (sort.value === 'title') sorted.sort((a, b) => a.title.localeCompare(b.title))
  return sorted
})

/**
 * The visible list, cut into time bands — or one unlabelled band when time is not the order.
 *
 * Bands are relative to now rather than to calendar boundaries, so "this week" means the last
 * seven days rather than "since Monday": a Monday-morning reader should not watch the section
 * they were reading yesterday empty itself out.
 *
 * `oldest` sort still groups, but the bands arrive in the order the sort dictates — the labels
 * describe the content either way, which is the whole point of preferring them to an every-Nth
 * divider.
 */
const grouped = computed<Array<{ key: string; label: string; items: EpisodeSummary[] }>>(() => {
  const timeOrdered = sort.value === 'newest' || sort.value === 'oldest'
  if (!timeOrdered || search.value.trim()) {
    return [{ key: 'all', label: '', items: visible.value }]
  }
  const now = Date.now()
  const DAY = 86_400_000
  const band = (e: EpisodeSummary): string => {
    const t = e.publish_date ? Date.parse(e.publish_date) : NaN
    if (Number.isNaN(t)) return 'undated'
    const age = (now - t) / DAY
    if (age < 7) return 'week'
    if (age < 31) return 'month'
    if (age < 366) return 'year'
    return 'older'
  }
  const labels: Record<string, string> = {
    week: t('catalog.groupWeek'),
    month: t('catalog.groupMonth'),
    year: t('catalog.groupYear'),
    older: t('catalog.groupOlder'),
    undated: t('catalog.groupUndated'),
  }
  const out: Array<{ key: string; label: string; items: EpisodeSummary[] }> = []
  for (const ep of visible.value) {
    const k = band(ep)
    const last = out[out.length - 1]
    if (last && last.key === k) last.items.push(ep)
    else out.push({ key: k, label: labels[k] ?? '', items: [ep] })
  }
  return out
})

const countLabel = computed(() =>
  controlsActive.value ? t('list.count', { shown: visible.value.length, total: episodes.value.length }) : '',
)

onMounted(async () => {
  await loadMore()
  shows.value = (await getPodcasts().catch(() => []))
    .filter((p) => p.feed_id)
    .map((p) => ({ id: p.feed_id, label: p.title ?? p.feed_id }))
})
</script>

<template>
  <section>
    <h1 v-if="!embedded" class="mb-5 font-display text-3xl font-extrabold tracking-tight">
      {{ t('catalog.heading') }}
    </h1>

    <p v-if="loading && episodes.length === 0" class="text-muted">{{ t('catalog.loading') }}</p>
    <p v-else-if="error && episodes.length === 0" class="text-danger">{{ t('catalog.loadError') }}</p>
    <p v-else-if="episodes.length === 0" class="text-muted">{{ t('catalog.empty') }}</p>

    <div v-else>
      <ListToolbar
        v-model:search="search"
        v-model:sort="sort"
        v-model:filter="filter"
        v-model:show="show"
        :shows="shows"
        :count="countLabel"
      />

      <p v-if="visible.length === 0" class="text-muted">{{ t('list.noMatches') }}</p>

      <!-- Grouped by WHEN, not chunked by count (#1978).
           The catalogue's compositional problem was measured, not assumed: 29 structurally
           identical rows down a 4,929px page with nothing to break them — "a spreadsheet with
           pictures". The critic's own alternative was "a divider every six rows", which breaks
           monotony while meaning nothing; these headings carry information instead, and they are
           the eyebrow system already used everywhere else rather than a new device.
           Only when the list is actually IN time order. Under `title` sort, or with a search
           term active, a "This week" heading over an alphabetical list would be a lie, so the
           grouping disappears and the flat list returns. -->
      <template v-for="group in grouped" :key="group.key">
        <h2 v-if="group.label" class="lp-kicker mb-2 mt-6 first:mt-0">{{ group.label }}</h2>
        <EpisodeCard v-for="ep in group.items" :key="ep.slug" :episode="ep" />
      </template>

      <div class="mt-6 flex justify-center">
        <button
          v-if="hasMore && !controlsActive"
          type="button"
          :disabled="loading"
          class="rounded-full border border-border px-5 py-2 font-bold disabled:opacity-50"
          @click="loadMore"
        >
          {{ loading ? t('catalog.loading') : t('catalog.loadMore') }}
        </button>
        <p v-else-if="loading && controlsActive" class="text-sm text-muted">{{ t('catalog.loading') }}</p>
      </div>
    </div>
  </section>
</template>
