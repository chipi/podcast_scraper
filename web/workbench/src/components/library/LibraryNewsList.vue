<script setup lang="ts">
/**
 * The Library's News mode (UXS-019 "The Library: News mode"): the shown news items, newest first
 * (`GET /api/corpus/news/items`, paged 50), each as the news result row. A publisher filter narrows
 * it to one publisher; with one chosen, its name and domain head the list (the publisher page).
 */
import { computed, ref, watch } from 'vue'
import { fetchNewsItems } from '../../api/newsApi'
import type { NewsItem } from '../../api/searchApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import { StaleGeneration } from '../../utils/staleGeneration'
import NewsResultRow from '../search/NewsResultRow.vue'

const PAGE = 50
/** The items route's largest page: the publisher sweep reads the whole list in few requests. */
const SWEEP = 500

type Publisher = { id: string; name: string; domain: string }

const shell = useShellStore()
const subject = useSubjectStore()
const items = ref<NewsItem[]>([])
const total = ref(0)
/** The publisher filter lives in the shell store, so another surface can open a publisher page. */
const publisherId = computed({
  get: () => shell.libraryNewsPublisherId,
  set: (v: string) => (shell.libraryNewsPublisherId = v),
})
/** Every publisher of the corpus's shown items, for the select. */
const publishers = ref<Publisher[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const gate = new StaleGeneration()
const sweepGate = new StaleGeneration()

const chosen = computed(() => publishers.value.find((p) => p.id === publisherId.value) ?? null)

function addPublishers(from: NewsItem[]): void {
  const seen = new Map(publishers.value.map((p) => [p.id, p]))
  for (const it of from) if (it.publisher) seen.set(it.publisher.id, it.publisher)
  publishers.value = [...seen.values()].sort((a, b) => a.name.localeCompare(b.name))
}

/**
 * Every publisher in the corpus, from the whole unfiltered list. The pages shown hold only the
 * newest items, so a select built from them alone missed the publishers further back (7 of 17 on
 * the 2026-11-20 test corpus). A failed sweep leaves the publishers the pages gave.
 */
async function loadPublishers(): Promise<void> {
  const root = shell.corpusPath.trim()
  if (!root) return
  const seq = sweepGate.bump()
  try {
    for (let offset = 0; ; offset += SWEEP) {
      const page = await fetchNewsItems(root, { limit: SWEEP, offset })
      if (sweepGate.isStale(seq)) return
      addPublishers(page.items)
      if (!page.items.length || offset + page.items.length >= page.total) return
    }
  } catch (e) {
    if (sweepGate.isCurrent(seq)) console.warn('news publisher list incomplete:', e)
  }
}

async function load(offset = 0): Promise<void> {
  const root = shell.corpusPath.trim()
  if (!root) return
  const seq = gate.bump()
  loading.value = true
  error.value = null
  try {
    const page = await fetchNewsItems(root, {
      publisherId: publisherId.value || undefined,
      limit: PAGE,
      offset,
    })
    if (gate.isStale(seq)) return
    items.value = offset ? [...items.value, ...page.items] : page.items
    total.value = page.total
    // Opened straight onto one publisher (another surface set it), the select holds it before
    // the sweep ends, for the select and the page's heading.
    addPublishers(page.items)
  } catch (e) {
    if (gate.isStale(seq)) return
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (gate.isCurrent(seq)) loading.value = false
  }
}

// Declared first: on a corpus change the old corpus's publishers go before either load adds any.
watch(
  () => shell.corpusPath,
  () => {
    publishers.value = []
    void loadPublishers()
  },
  { immediate: true },
)
watch(() => [shell.corpusPath, publisherId.value], () => void load(0), { immediate: true })

function open(item: NewsItem): void {
  subject.focusArticle(item.item_id, { uiTitle: item.title })
}
</script>

<template>
  <div class="flex min-h-0 flex-1 flex-col gap-2" data-testid="library-news-list">
    <div class="flex flex-wrap items-center gap-2 px-2">
      <label class="text-[10px] font-semibold uppercase tracking-wider text-muted" for="lib-news-publisher">
        Publisher
      </label>
      <select
        id="lib-news-publisher"
        v-model="publisherId"
        class="rounded border border-border bg-elevated px-2 py-1 text-xs text-surface-foreground"
        data-testid="library-news-publisher"
      >
        <option value="">All publishers</option>
        <option v-for="p in publishers" :key="p.id" :value="p.id">{{ p.name }}</option>
      </select>
      <span class="ml-auto text-[10px] text-muted">{{ total }} item{{ total === 1 ? '' : 's' }}</span>
    </div>
    <div v-if="chosen" class="px-2" data-testid="library-news-publisher-head">
      <h3 class="text-sm font-semibold text-surface-foreground">{{ chosen.name }}</h3>
      <p v-if="chosen.domain" class="text-[11px] text-muted">{{ chosen.domain }}</p>
    </div>
    <p v-if="error" class="px-2 text-[11px] text-warning">{{ error }}</p>
    <div class="min-h-0 flex-1 space-y-2 overflow-y-auto px-2 pb-2">
      <NewsResultRow v-for="item in items" :key="item.item_id" :item="item" @open="open" />
      <p v-if="!loading && !items.length && !error" class="text-[11px] text-muted">No news items.</p>
      <button
        v-if="items.length < total"
        type="button"
        class="w-full rounded border border-border px-2 py-1 text-[11px] text-muted hover:bg-overlay"
        data-testid="library-news-more"
        :disabled="loading"
        @click="load(items.length)"
      >
        {{ loading ? 'Loading…' : 'Load more' }}
      </button>
    </div>
  </div>
</template>
