<script setup lang="ts">
/**
 * "In the news" on a topic, person or organisation rail (UXS-019 "The node rail"): the news items
 * whose KG names the node, newest first (`POST /api/corpus/node-episodes` with `content_types`,
 * field `news_items`). Only when the corpus holds news; renders nothing when no item names the node.
 */
import { computed, ref, watch } from 'vue'
import { fetchNodeEpisodes } from '../../api/corpusLibraryApi'
import type { NewsItem } from '../../api/searchApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import { formatRelativeShort } from '../../utils/relativeTime'
import { StaleGeneration } from '../../utils/staleGeneration'

/** Entries shown before "Show all". */
const SHOWN = 5

const props = defineProps<{ nodeId: string }>()

const shell = useShellStore()
const subject = useSubjectStore()
const items = ref<NewsItem[]>([])
const showAll = ref(false)
const gate = new StaleGeneration()

const bareId = computed(() => props.nodeId.trim().replace(/^(?:g:|k:|kg:)+/, ''))
const visible = computed(() => (showAll.value ? items.value : items.value.slice(0, SHOWN)))

async function load(): Promise<void> {
  const seq = gate.bump()
  const root = shell.corpusPath.trim()
  items.value = []
  showAll.value = false
  if (!root || !bareId.value || !shell.healthStatus || !shell.hasNews) return
  try {
    const res = await fetchNodeEpisodes(root, props.nodeId.trim(), 1, ['news_article'])
    if (gate.isStale(seq)) return
    items.value = res.news_items ?? []
  } catch {
    if (!gate.isStale(seq)) items.value = []
  }
}

watch(
  () => [props.nodeId, shell.corpusPath, shell.healthStatus, shell.hasNews],
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
</script>

<template>
  <section v-if="items.length" class="mb-3" data-testid="node-news-section">
    <h4 class="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted">In the news</h4>
    <ul class="space-y-1">
      <li
        v-for="item in visible"
        :key="item.item_id"
        class="border-l-[3px] border-l-news pl-1.5"
        data-testid="node-news-item"
      >
        <button
          type="button"
          class="block w-full min-w-0 break-words text-left text-[11px] font-medium leading-snug text-surface-foreground hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          @click="open(item)"
        >
          {{ item.title }}
        </button>
        <p class="text-[10px] text-muted">{{ meta(item) }}</p>
      </li>
    </ul>
    <button
      v-if="!showAll && items.length > SHOWN"
      type="button"
      class="mt-1 text-[10px] text-link hover:underline"
      data-testid="node-news-show-all"
      @click="showAll = true"
    >
      Show all ({{ items.length }})
    </button>
  </section>
</template>
