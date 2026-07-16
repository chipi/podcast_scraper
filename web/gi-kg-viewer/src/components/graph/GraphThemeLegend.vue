<script setup lang="ts">
/**
 * graph-v3 Tier 5A-1 — floating legend for the theme-cluster region tints.
 *
 * Renders only when the `themeClusterRegions` lens is on AND the
 * `topic_theme_clusters.json` artifact is loaded. One row per cluster
 * with a colour swatch (matching the underlay hex the graph paints for
 * the same `thc:...` id) + canonical_label + member count.
 *
 * Collapsible; state persisted in localStorage.
 */
import { computed, onMounted, ref } from 'vue'
import type { TopicClustersCluster } from '../../api/corpusTopicClustersApi'
import { useArtifactsStore } from '../../stores/artifacts'
import { useGraphLensesStore } from '../../stores/graphLenses'
import { themeRegionColor } from '../../utils/themeRegionPalette'

const artifacts = useArtifactsStore()
const lenses = useGraphLensesStore()

const COLLAPSED_KEY = 'ps_graph_theme_legend_collapsed'
const collapsed = ref(false)

onMounted(() => {
  try {
    collapsed.value =
      typeof localStorage !== 'undefined' &&
      localStorage.getItem(COLLAPSED_KEY) === '1'
  } catch {
    /* ignore */
  }
})

function toggleCollapsed(): void {
  collapsed.value = !collapsed.value
  try {
    if (typeof localStorage === 'undefined') return
    if (collapsed.value) localStorage.setItem(COLLAPSED_KEY, '1')
    else localStorage.removeItem(COLLAPSED_KEY)
  } catch {
    /* ignore */
  }
}

const visible = computed(() => {
  return lenses.themeClusterRegions && artifacts.themeClustersDoc != null
})

interface LegendRow {
  id: string
  label: string
  colour: string
  memberCount: number
}

const rows = computed<LegendRow[]>(() => {
  const doc = artifacts.themeClustersDoc
  const clusters = (doc?.clusters ?? []) as TopicClustersCluster[]
  const out: LegendRow[] = []
  for (const cl of clusters) {
    const id = typeof cl?.graph_compound_parent_id === 'string' ? cl.graph_compound_parent_id.trim() : ''
    if (!id) continue
    const label =
      typeof cl?.canonical_label === 'string' && cl.canonical_label.trim()
        ? cl.canonical_label.trim()
        : id
    const memberCount = typeof cl?.member_count === 'number' ? cl.member_count : 0
    out.push({ id, label, colour: themeRegionColor(id), memberCount })
  }
  return out
})
</script>

<template>
  <div
    v-if="visible"
    class="pointer-events-auto absolute bottom-2 right-2 z-10 max-w-[13rem] rounded border border-border bg-surface/95 text-surface-foreground shadow-md backdrop-blur-[1px]"
    data-testid="graph-theme-legend"
  >
    <div class="flex items-center justify-between border-b border-border/60 px-2 py-1">
      <span class="text-[10px] uppercase tracking-wide text-muted">Theme regions</span>
      <button
        type="button"
        class="rounded px-1 text-[10px] leading-none text-muted hover:bg-overlay hover:text-surface-foreground"
        data-testid="graph-theme-legend-toggle"
        :aria-label="collapsed ? 'Expand theme legend' : 'Collapse theme legend'"
        @click="toggleCollapsed"
      >
        {{ collapsed ? '▸' : '▾' }}
      </button>
    </div>
    <ul v-if="!collapsed" class="max-h-[14rem] overflow-y-auto px-2 py-1">
      <li
        v-for="r in rows"
        :key="r.id"
        class="flex items-center gap-1.5 py-0.5 text-[11px]"
        :data-testid="`graph-theme-legend-row-${r.id}`"
      >
        <span
          class="inline-block h-3 w-3 shrink-0 rounded-sm border border-border/40"
          :style="{ backgroundColor: r.colour }"
          aria-hidden="true"
        />
        <span class="min-w-0 flex-1 truncate" :title="r.label">{{ r.label }}</span>
        <span class="shrink-0 text-[9px] font-mono text-muted">{{ r.memberCount }}</span>
      </li>
      <li v-if="rows.length === 0" class="py-1 text-[10px] text-muted">
        No clusters in this corpus.
      </li>
    </ul>
  </div>
</template>
