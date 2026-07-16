<script setup lang="ts">
/**
 * graph-v3 Q — Lenses popover in the graph bottom bar.
 *
 * Renders one checkbox per RFC-080 lens (currently four:
 * `aggregatedEdges`, `nodeSizeByDegree`, `themeClusterRegions`,
 * `bridgeRing`). Toggles write to `useGraphLensesStore` which persists to
 * localStorage; GraphCanvas watchers re-apply the class overlays without
 * a full re-layout so the graph responds live.
 *
 * Enricher-gated: the `themeClusterRegions` row is hidden entirely when
 * the `topic_theme_clusters.json` artifact is not available for the
 * current corpus (`artifacts.themeClustersDoc === null`). Operator
 * direction (graph-v3 V): hide, don't disable — no dead controls.
 */
import { computed, ref } from 'vue'
import { useGraphLensesStore } from '../../../stores/graphLenses'
import { useArtifactsStore } from '../../../stores/artifacts'
import { useFilterChipPopover } from '../../../composables/useFilterChipPopover'

const lenses = useGraphLensesStore()
const artifacts = useArtifactsStore()
const anchorRef = ref<HTMLButtonElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)
const { open, toggle } = useFilterChipPopover(anchorRef, panelRef)

/** Theme-cluster lens is enricher-gated. */
const themeClustersAvailable = computed(() => artifacts.themeClustersDoc != null)

interface LensRow {
  key: 'aggregatedEdges' | 'nodeSizeByDegree' | 'themeClusterRegions' | 'bridgeRing'
  label: string
  description: string
  testid: string
  available: boolean
}

const rows = computed<LensRow[]>(() => {
  const all: LensRow[] = [
    {
      key: 'nodeSizeByDegree',
      label: 'Size by connectivity',
      description: 'Topic, Episode, Person, Org — width scales with graph degree.',
      testid: 'lens-node-size-by-degree',
      available: true,
    },
    {
      key: 'bridgeRing',
      label: 'Bridge nodes',
      description: 'Rose ring on Topic/Podcast/Person/Org that bridge communities.',
      testid: 'lens-bridge-ring',
      available: true,
    },
    {
      key: 'themeClusterRegions',
      label: 'Theme regions',
      description: 'Soft underlay tint per theme cluster (needs the theme_clusters enricher).',
      testid: 'lens-theme-cluster-regions',
      available: themeClustersAvailable.value,
    },
    {
      key: 'aggregatedEdges',
      label: 'Aggregated edges',
      description: 'Roll per-Insight edges up to Episode↔Topic / Episode↔Person aggregates.',
      testid: 'lens-aggregated-edges',
      available: true,
    },
  ]
  return all.filter((r) => r.available)
})

const activeCount = computed(() => {
  let n = 0
  for (const r of rows.value) if (lenses[r.key]) n += 1
  return n
})

const chipLabel = computed(() => {
  const total = rows.value.length
  if (activeCount.value === 0) return 'Lenses ▾'
  return `Lenses: ${activeCount.value} of ${total} ▾`
})

function setLens(key: LensRow['key'], value: boolean): void {
  if (key === 'aggregatedEdges') lenses.setAggregatedEdges(value)
  else if (key === 'nodeSizeByDegree') lenses.setNodeSizeByDegree(value)
  else if (key === 'themeClusterRegions') lenses.setThemeClusterRegions(value)
  else if (key === 'bridgeRing') lenses.setBridgeRing(value)
}

function reset(): void {
  lenses.resetToDefaults()
}
</script>

<template>
  <div class="relative inline-flex items-center">
    <button
      ref="anchorRef"
      type="button"
      class="inline-flex h-6 items-center rounded border px-2 text-[11px] leading-none transition-colors hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
      :class="
        activeCount > 0
          ? 'border-border font-medium text-surface-foreground'
          : 'border-border/70 text-muted'
      "
      data-testid="graph-chip-lenses"
      :aria-expanded="open"
      aria-haspopup="dialog"
      aria-label="Graph lenses"
      @click="toggle"
    >
      {{ chipLabel }}
    </button>
    <div
      v-show="open"
      ref="panelRef"
      role="dialog"
      aria-label="Graph lenses"
      data-testid="graph-popover-lenses"
      class="absolute left-0 bottom-full z-[40] mb-1 w-72 rounded-sm border border-border bg-elevated p-2 shadow-md"
    >
      <div class="mb-1 flex items-center justify-between">
        <span class="text-[10px] uppercase tracking-wide text-muted">Render overlays</span>
        <button
          type="button"
          class="text-[10px] text-primary underline"
          data-testid="lens-reset"
          @click="reset"
        >
          reset
        </button>
      </div>
      <ul>
        <li
          v-for="r in rows"
          :key="r.key"
          class="border-t border-border/40 py-1.5 first:border-t-0"
        >
          <label class="flex cursor-pointer items-start gap-2 text-[11px]">
            <input
              type="checkbox"
              class="mt-0.5 shrink-0 rounded border-border"
              :checked="lenses[r.key]"
              :data-testid="r.testid"
              @change="setLens(r.key, ($event.target as HTMLInputElement).checked)"
            >
            <span class="min-w-0 flex-1">
              <span class="block font-medium text-surface-foreground">{{ r.label }}</span>
              <span class="mt-0.5 block text-[10px] leading-tight text-muted">{{ r.description }}</span>
            </span>
          </label>
        </li>
      </ul>
    </div>
  </div>
</template>
