<script setup lang="ts">
/**
 * The Digest's news band (UXS-019 "The Digest: a news band"): the news items published in the
 * Digest's window, newest first, at most ten, as compact rows (title, publisher, time). "See all
 * in Library" opens the Library in News mode. Only when the corpus holds news; nothing when the
 * window has none.
 */
import { ref, watch } from 'vue'
import { fetchNewsItems } from '../../api/newsApi'
import type { NewsItem } from '../../api/searchApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import { formatRelativeShort } from '../../utils/relativeTime'
import { StaleGeneration } from '../../utils/staleGeneration'

const SHOWN = 10

const props = defineProps<{ sinceYmd: string }>()
const emit = defineEmits<{ 'switch-main-tab': [tab: 'library'] }>()

const shell = useShellStore()
const subject = useSubjectStore()
const items = ref<NewsItem[]>([])
const total = ref(0)
const gate = new StaleGeneration()

async function load(): Promise<void> {
  const seq = gate.bump()
  const root = shell.corpusPath.trim()
  items.value = []
  total.value = 0
  if (!root || !shell.healthStatus || !shell.hasNews) return
  try {
    const page = await fetchNewsItems(root, { since: props.sinceYmd || undefined, limit: SHOWN })
    if (gate.isStale(seq)) return
    items.value = page.items
    total.value = page.total
  } catch {
    if (!gate.isStale(seq)) items.value = []
  }
}

watch(
  () => [props.sinceYmd, shell.corpusPath, shell.healthStatus, shell.hasNews],
  () => void load(),
  { immediate: true },
)

function meta(item: NewsItem): string {
  return [item.publisher?.name ?? item.site, formatRelativeShort(item.published_at, Date.now())]
    .filter((s) => !!s)
    .join(' · ')
}

function open(item: NewsItem): void {
  subject.focusArticle(item.item_id, { uiTitle: item.title })
}

function seeAll(): void {
  shell.libraryMode = 'news'
  shell.libraryNewsPublisherId = ''
  emit('switch-main-tab', 'library')
}
</script>

<template>
  <section
    v-if="items.length"
    class="shrink-0 rounded-sm border border-border bg-surface"
    aria-labelledby="digest-news-heading"
    data-testid="digest-news-band"
  >
    <div class="flex items-center justify-between gap-2 border-b border-border p-2">
      <h2 id="digest-news-heading" class="text-sm font-semibold text-surface-foreground">
        News
        <span class="font-normal tabular-nums text-muted">{{ total }}</span>
      </h2>
      <button
        type="button"
        class="text-[11px] text-link hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        data-testid="digest-news-see-all"
        @click="seeAll"
      >
        See all in Library
      </button>
    </div>
    <ul class="divide-y divide-border">
      <li
        v-for="item in items"
        :key="item.item_id"
        class="border-l-[3px] border-l-news px-2 py-1.5"
        data-testid="digest-news-row"
      >
        <button
          type="button"
          class="block w-full min-w-0 break-words text-left text-[12px] font-medium leading-snug text-surface-foreground hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          @click="open(item)"
        >
          {{ item.title }}
        </button>
        <p class="text-[10px] text-muted">{{ meta(item) }}</p>
      </li>
    </ul>
  </section>
</template>
