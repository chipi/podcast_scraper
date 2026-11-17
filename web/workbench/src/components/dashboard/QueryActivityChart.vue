<script setup lang="ts">
/**
 * PRD-033 FR6.2 — search-activity chart (#888 follow-up).
 *
 * Daily search volume from the append-only query log (`/api/corpus/query-activity`).
 * Honest scope: this is *search volume over time*, not "query volume by topic" — queries
 * are not topic-tagged, so a single-series daily count is the supported signal. Tufte:
 * single y-axis, one series, no decoration (delegated to VerticalBarChart). Renders only
 * once at least one search has been logged.
 */
import { onMounted, ref, watch } from 'vue'
import { fetchQueryActivity } from '../../api/queryActivityApi'
import { useShellStore } from '../../stores/shell'
import { StaleGeneration } from '../../utils/staleGeneration'
import VerticalBarChart from './VerticalBarChart.vue'

const shell = useShellStore()

const labels = ref<string[]>([])
const values = ref<number[]>([])
const total = ref(0)
const loaded = ref(false)
const gate = new StaleGeneration()

async function load(): Promise<void> {
  const root = shell.corpusPath?.trim()
  if (!root || !shell.healthStatus) {
    labels.value = []
    values.value = []
    total.value = 0
    loaded.value = false
    return
  }
  const seq = gate.bump()
  try {
    const body = await fetchQueryActivity(root, 30)
    if (gate.isStale(seq)) return
    // Compact axis labels: month-day only.
    labels.value = body.buckets.map((b) => b.date.slice(5))
    values.value = body.buckets.map((b) => b.count)
    total.value = body.total
    loaded.value = true
  } catch {
    if (gate.isStale(seq)) return
    labels.value = []
    values.value = []
    total.value = 0
    loaded.value = false
  }
}

onMounted(load)
watch(() => [shell.corpusPath, shell.healthStatus], load)
</script>

<template>
  <section
    v-if="loaded && total > 0"
    class="rounded border border-border bg-surface p-3"
    data-testid="query-activity-chart"
  >
    <VerticalBarChart
      title="Search activity (last 30 days)"
      :labels="labels"
      :values="values"
      y-axis-label="Searches"
      :insight-text="`${total} searches in the last 30 days.`"
      help-text="Daily count of searches run against this corpus, from the local search-activity log. Volume over time, not query-by-topic."
    />
  </section>
</template>
