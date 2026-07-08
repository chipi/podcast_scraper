<script setup lang="ts">
/**
 * Operator global trending view (RFC-103) — the bird's-eye "what's hot corpus-wide" across EVERY
 * momentum kind (topics, clusters, storylines, people, episodes, shows, insights), backed by
 * GET /api/corpus/trending. One row per entity: kind, label, a weekly sparkline, and its velocity
 * (↑ rising). Read-only; hides when the corpus has no trending signal.
 */
import { computed, ref, watch } from 'vue'
import { fetchCorpusTrending, type TrendingEntity } from '../../api/corpusTrendingApi'
import { useShellStore } from '../../stores/shell'

const shell = useShellStore()

// Kinds in display order (only those with entities render).
const KIND_ORDER = ['storyline', 'topic', 'cluster', 'person', 'episode', 'show', 'insight'] as const
const KIND_LABEL: Record<string, string> = {
  storyline: 'Storylines',
  topic: 'Topics',
  cluster: 'Clusters',
  person: 'People',
  episode: 'Episodes',
  show: 'Shows',
  insight: 'Insights',
}

const asOfWeek = ref('')
const kinds = ref<Record<string, TrendingEntity[]>>({})
const loading = ref(false)

async function load(): Promise<void> {
  const root = shell.corpusPath?.trim()
  if (!root || !shell.healthStatus) {
    kinds.value = {}
    return
  }
  loading.value = true
  const r = await fetchCorpusTrending(root).catch(() => null)
  loading.value = false
  if (r?.status === 'ok') {
    asOfWeek.value = r.document.as_of_week
    kinds.value = r.document.kinds ?? {}
  } else {
    kinds.value = {}
  }
}
watch(() => [shell.corpusPath, shell.healthStatus], () => void load(), { immediate: true })

const sections = computed(() =>
  KIND_ORDER.map((k) => ({ kind: k, label: KIND_LABEL[k], items: kinds.value[k] ?? [] })).filter(
    (s) => s.items.length > 0,
  ),
)
const hasAny = computed(() => sections.value.length > 0)

/** A tiny sparkline polyline for a weekly series over a `w`×`h` box. */
function sparkPoints(series: number[], w = 48, h = 14): string {
  if (!series.length) return ''
  const max = Math.max(1, ...series)
  const step = series.length > 1 ? w / (series.length - 1) : 0
  return series.map((v, i) => `${(i * step).toFixed(1)},${(h - (v / max) * h).toFixed(1)}`).join(' ')
}
function velColor(v: number): string {
  if (v >= 1.5) return 'var(--ps-success, #16a34a)'
  if (v < 0.9) return 'var(--ps-muted, #94a3b8)'
  return 'var(--ps-text, currentColor)'
}
</script>

<template>
  <section v-if="hasAny" class="rounded-lg border border-[--ps-border] p-3" data-testid="trending-global">
    <div class="mb-2 flex items-baseline justify-between">
      <h3 class="text-sm font-semibold">Trending (global)</h3>
      <span class="text-[10px] text-[--ps-muted]" data-testid="trending-global-asof">{{ asOfWeek }}</span>
    </div>
    <div class="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      <div v-for="s in sections" :key="s.kind" :data-testid="`trending-global-kind-${s.kind}`">
        <div class="mb-1 text-[11px] font-semibold uppercase tracking-wide text-[--ps-muted]">
          {{ s.label }}
        </div>
        <ul class="space-y-1">
          <li
            v-for="e in s.items"
            :key="e.entity_id"
            class="flex items-center gap-2 text-xs"
            data-testid="trending-global-row"
          >
            <span class="min-w-0 flex-1 truncate" :title="e.label">{{ e.label }}</span>
            <svg
              :viewBox="`0 0 48 14`"
              width="48"
              height="14"
              class="shrink-0"
              aria-hidden="true"
              data-testid="trending-global-sparkline"
            >
              <polyline
                :points="sparkPoints(e.series)"
                fill="none"
                :stroke="velColor(e.velocity)"
                stroke-width="1.5"
              />
            </svg>
            <span
              class="w-10 shrink-0 text-right font-semibold tabular-nums"
              :style="{ color: velColor(e.velocity) }"
              >{{ Math.round(e.velocity * 10) / 10 }}×</span
            >
          </li>
        </ul>
      </div>
    </div>
  </section>
</template>
