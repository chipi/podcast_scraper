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
import { computed, onMounted, ref, watch } from 'vue'
import { useGraphLensesStore } from '../../../stores/graphLenses'
import { useArtifactsStore } from '../../../stores/artifacts'
import { useShellStore } from '../../../stores/shell'
import { useFilterChipPopover } from '../../../composables/useFilterChipPopover'
import { fetchCachedCorpusEnvelope } from '../../../composables/useEnrichmentEnvelopeCache'

const lenses = useGraphLensesStore()
const artifacts = useArtifactsStore()
const shell = useShellStore()
const anchorRef = ref<HTMLButtonElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)
const { open, toggle } = useFilterChipPopover(anchorRef, panelRef)

/** Theme-cluster lens is enricher-gated. */
const themeClustersAvailable = computed(() => artifacts.themeClustersDoc != null)

/* aggregatedEdges lens is data-gated: it renders Episode↔Topic (`ABOUT_AGG`)
   + Episode↔Person (`SPOKE_IN_AGG`) roll-ups on top of the per-Insight
   edges. If the current merged artifact has zero source Insight→Topic
   ABOUT edges AND zero Quote→Person SPOKEN_BY edges then the toggle
   would render nothing at all — hide the row to avoid a dead control
   (RFC-080 V1). FU2 in graph-v3/HARDEN-FOLLOWUPS-2026-07-17.md. */
const aggregatedEdgesAvailable = computed(() => {
  const art = artifacts.fullArtifact
  const edges = art?.data?.edges
  if (!art || !Array.isArray(edges) || edges.length === 0) return false
  for (const e of edges) {
    if (!e) continue
    const raw = e.type
    const t = typeof raw === 'string' ? raw.trim().toLowerCase() : ''
    if (t === 'about' || t === 'spoken_by') return true
  }
  return false
})

/* graph-v3 Tier 5C/5D — enricher-based lens rows are hidden when the
   underlying corpus artifact is absent. Availability is probed on mount
   + on corpus-path change (each fetch is cached, so subsequent lens
   toggles reuse the same result without a re-request). */
const velocityAvailable = ref(false)
const credibilityAvailable = ref(false)
const consensusAvailable = ref(false)
const coguestAvailable = ref(false)

async function probeEnricherAvailability(root: string): Promise<void> {
  const missingRoot = !root.trim()
  if (missingRoot) {
    velocityAvailable.value = false
    credibilityAvailable.value = false
    consensusAvailable.value = false
    coguestAvailable.value = false
    return
  }
  const [velocity, credibility, consensus, coguest] = await Promise.all([
    fetchCachedCorpusEnvelope(root, 'temporal_velocity').catch(() => null),
    fetchCachedCorpusEnvelope(root, 'grounding_rate').catch(() => null),
    fetchCachedCorpusEnvelope(root, 'topic_consensus').catch(() => null),
    fetchCachedCorpusEnvelope(root, 'guest_coappearance').catch(() => null),
  ])
  velocityAvailable.value = velocity != null
  credibilityAvailable.value = credibility != null
  consensusAvailable.value = consensus != null
  coguestAvailable.value = coguest != null
}

onMounted(() => {
  void probeEnricherAvailability(shell.corpusPath.trim())
})

watch(
  () => shell.corpusPath,
  (next) => {
    void probeEnricherAvailability(String(next).trim())
  },
)

interface LensRow {
  key:
    | 'aggregatedEdges'
    | 'nodeSizeByDegree'
    | 'themeClusterRegions'
    | 'bridgeRing'
    | 'velocityHalo'
    | 'personCredibility'
    | 'consensusEdges'
    | 'coGuestEdges'
    | 'personCommunities'
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
      key: 'velocityHalo',
      label: 'Velocity halo',
      description: 'Green / red / amber border on Topics + Persons by 6-month trend.',
      testid: 'lens-velocity-halo',
      available: velocityAvailable.value,
    },
    {
      key: 'personCredibility',
      label: 'Person credibility',
      description: 'Border colour reflects Person grounded-insight rate (green ≥ 0.7).',
      testid: 'lens-person-credibility',
      available: credibilityAvailable.value,
    },
    {
      key: 'consensusEdges',
      label: 'Consensus edges',
      description: 'Green arcs between Persons who corroborate on a topic (topic_consensus).',
      testid: 'lens-consensus-edges',
      available: consensusAvailable.value,
    },
    {
      key: 'coGuestEdges',
      label: 'Co-guest edges',
      description: 'Dotted arcs between Persons sharing ≥ 2 episodes (guest_coappearance).',
      testid: 'lens-coguest-edges',
      available: coguestAvailable.value,
    },
    {
      key: 'personCommunities',
      label: 'Person communities',
      description: 'Soft underlay grouping Persons who repeatedly co-appear (guest_coappearance).',
      testid: 'lens-person-communities',
      available: coguestAvailable.value,
    },
    {
      key: 'aggregatedEdges',
      label: 'Aggregated edges',
      description: 'Roll per-Insight edges up to Episode↔Topic / Episode↔Person aggregates.',
      testid: 'lens-aggregated-edges',
      available: aggregatedEdgesAvailable.value,
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
  else if (key === 'velocityHalo') lenses.setVelocityHalo(value)
  else if (key === 'personCredibility') lenses.setPersonCredibility(value)
  else if (key === 'consensusEdges') lenses.setConsensusEdges(value)
  else if (key === 'coGuestEdges') lenses.setCoGuestEdges(value)
  else if (key === 'personCommunities') lenses.setPersonCommunities(value)
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
