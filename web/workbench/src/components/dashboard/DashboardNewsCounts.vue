<script setup lang="ts">
/**
 * The Dashboard's news counts (UXS-019 "The Dashboard: news counts"): one line under the corpus
 * briefing, "News: N items from P publishers (T teasers, W paywalled)", from
 * `GET /api/corpus/news/summary`. Only when the corpus holds news.
 */
import { computed, ref, watch } from 'vue'
import { fetchNewsSummary, type NewsSummary } from '../../api/newsApi'
import { useShellStore } from '../../stores/shell'
import { StaleGeneration } from '../../utils/staleGeneration'

const shell = useShellStore()
const summary = ref<NewsSummary | null>(null)
const gate = new StaleGeneration()

async function load(): Promise<void> {
  const seq = gate.bump()
  summary.value = null
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus || !shell.hasNews) return
  try {
    const got = await fetchNewsSummary(root)
    if (gate.isCurrent(seq)) summary.value = got
  } catch {
    if (gate.isCurrent(seq)) summary.value = null
  }
}

watch(() => [shell.corpusPath, shell.healthStatus, shell.hasNews], () => void load(), {
  immediate: true,
})

function plural(n: number, one: string, many: string): string {
  return `${n} ${n === 1 ? one : many}`
}

const line = computed(() => {
  const s = summary.value
  if (!s || !s.items) return ''
  const extras = [
    s.teasers ? plural(s.teasers, 'teaser', 'teasers') : '',
    s.paywalled ? `${s.paywalled} paywalled` : '',
  ].filter((x) => !!x)
  return (
    `News: ${plural(s.items, 'item', 'items')} from ${plural(s.publishers, 'publisher', 'publishers')}` +
    (extras.length ? ` (${extras.join(', ')})` : '')
  )
})
</script>

<template>
  <p v-if="line" class="px-1 text-[11px] text-muted" data-testid="dashboard-news-counts">
    <span class="mr-1 inline-block h-2 w-2 rounded-full bg-news align-middle" aria-hidden="true" />
    {{ line }}
  </p>
</template>
