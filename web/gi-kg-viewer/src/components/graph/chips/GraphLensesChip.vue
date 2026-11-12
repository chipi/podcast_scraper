<script setup lang="ts">
/**
 * graph-v3 Q — Lenses popover in the graph bottom bar.
 *
 * Renders one checkbox per RFC-080 lens (`nodeSizeByDegree`, `bridgeRing`,
 * `personCredibility`, `coGuestEdges`, `personCommunities`, `aggregatedEdges`). Toggles write to `useGraphLensesStore` which persists
 * to localStorage; GraphCanvas watchers re-apply the class overlays
 * without a full re-layout so the graph responds live.
 *
 * Enricher-gated rows are hidden entirely when the corresponding corpus
 * artifact is not available (probed via `fetchCachedCorpusEnvelope` on
 * mount + on corpus-path change). Data-gated `aggregatedEdges` is hidden
 * when the display artifact has no source ABOUT/SPOKEN_BY edges to roll
 * up. Operator direction (graph-v3 V): hide, don't disable — no dead
 * controls.
 *
 * Prose reference: ``docs/guides/GRAPH_VISUALIZATION_GUIDE.md`` §Lenses
 * has the full catalog table (default state + gate + visual effect for
 * each row) and §Enricher artifacts documents the underlying data
 * contracts. Update both when adding a new lens.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { graphElementDecorators, graphLenses as registeredLenses } from '../../../extensions'
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

/* The aggregated-edges lens draws what a registered element decorator rolls up (RFC-080 V1);
   the row shows only when one has something to aggregate in the drawn artifact, so it is never
   a dead control. */
const aggregatedEdgesAvailable = computed(() => {
  const art = artifacts.displayArtifact
  if (!art) return false
  return graphElementDecorators.list().some((d) => d.aggregates?.(art) === true)
})

/* graph-v3 Tier 5C/5D — enricher-based lens rows are hidden when the
   underlying corpus artifact is absent. Availability is probed on mount
   + on corpus-path change (each fetch is cached, so subsequent lens
   toggles reuse the same result without a re-request). */
const credibilityAvailable = ref(false)
const coguestAvailable = ref(false)

/** Availability of lenses a build registered (extension point `graphLenses`), by id. */
const registeredAvailable = ref<Record<string, boolean>>({})

async function probeRegisteredAvailability(root: string): Promise<void> {
  const entries = await Promise.all(
    registeredLenses.list().map(async (l) => {
      if (!l.available) return [l.id, true] as const
      if (!root) return [l.id, false] as const
      return [l.id, await Promise.resolve(l.available(root)).catch(() => false)] as const
    }),
  )
  registeredAvailable.value = Object.fromEntries(entries)
}

async function probeEnricherAvailability(root: string): Promise<void> {
  void probeRegisteredAvailability(root.trim())
  const missingRoot = !root.trim()
  if (missingRoot) {
    credibilityAvailable.value = false
    coguestAvailable.value = false
    return
  }
  const [credibility, coguest] = await Promise.all([
    fetchCachedCorpusEnvelope(root, 'grounding_rate').catch(() => null),
    fetchCachedCorpusEnvelope(root, 'guest_coappearance').catch(() => null),
  ])
  // #1927 — availability must mean "this lens can actually draw something", not "the file
  // exists". grounding_rate became per-EPISODE: its envelope is still written, but it no longer
  // carries `persons`, and applyCredibilityBorder returns early without them. Probing for the
  // file alone left the lens toggling ON and doing nothing, permanently. Requiring the array the
  // overlay reads makes the chip honest, and leaves the plumbing intact if a per-episode-driven
  // credibility lens is built later (tracked separately).
  // ``persons`` lives under the envelope's ``data``, which is what applyCredibilityBorder is
  // handed — reading it off the envelope itself is always undefined, i.e. correct-looking
  // for the wrong reason, and would stay false if a per-episode lens ever lands.
  const credibilityPersons = (credibility?.data as { persons?: unknown } | undefined)?.persons
  credibilityAvailable.value = Array.isArray(credibilityPersons) && credibilityPersons.length > 0
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

/* A registered lens's availability can depend on data that loads after mount (its `watchSource`). */
watch(
  () => registeredLenses.list().map((l) => l.watchSource?.()),
  () => {
    void probeRegisteredAvailability(shell.corpusPath.trim())
  },
)

type BuiltInLensKey =
  | 'aggregatedEdges'
  | 'nodeSizeByDegree'
  | 'bridgeRing'
  | 'personCredibility'
  | 'coGuestEdges'
  | 'personCommunities'

interface LensRow {
  key: string
  label: string
  description: string
  testid: string
  available: boolean
  /** Position among the rows; registered lenses sort in by their `order`. */
  order: number
}

const BUILT_IN_ORDER: Record<BuiltInLensKey, number> = {
  nodeSizeByDegree: 10,
  bridgeRing: 20,
  personCredibility: 40,
  coGuestEdges: 60,
  personCommunities: 70,
  aggregatedEdges: 80,
}

function isBuiltIn(key: string): key is BuiltInLensKey {
  return key in BUILT_IN_ORDER
}

function lensOn(key: string): boolean {
  return isBuiltIn(key) ? lenses[key] : lenses.isLensOn(key)
}

const rows = computed<LensRow[]>(() => {
  const all: LensRow[] = [
    {
      key: 'nodeSizeByDegree',
      order: BUILT_IN_ORDER.nodeSizeByDegree,
      label: 'Size by connectivity',
      description: 'Topic, Episode, Person, Org — width scales with graph degree.',
      testid: 'lens-node-size-by-degree',
      available: true,
    },
    {
      key: 'bridgeRing',
      order: BUILT_IN_ORDER.bridgeRing,
      label: 'Bridge nodes',
      description: 'Rose ring on Topic/Podcast/Person/Org that bridge communities.',
      testid: 'lens-bridge-ring',
      available: true,
    },
    {
      key: 'personCredibility',
      order: BUILT_IN_ORDER.personCredibility,
      label: 'Person credibility',
      description: 'Border colour reflects Person grounded-insight rate (green ≥ 0.7).',
      testid: 'lens-person-credibility',
      available: credibilityAvailable.value,
    },
    {
      key: 'coGuestEdges',
      order: BUILT_IN_ORDER.coGuestEdges,
      label: 'Co-guest edges',
      description: 'Dotted arcs between Persons sharing ≥ 2 episodes (guest_coappearance).',
      testid: 'lens-coguest-edges',
      available: coguestAvailable.value,
    },
    {
      key: 'personCommunities',
      order: BUILT_IN_ORDER.personCommunities,
      label: 'Person communities',
      description: 'Soft underlay grouping Persons who repeatedly co-appear (guest_coappearance).',
      testid: 'lens-person-communities',
      available: coguestAvailable.value,
    },
    {
      key: 'aggregatedEdges',
      order: BUILT_IN_ORDER.aggregatedEdges,
      label: 'Aggregated edges',
      description: 'Roll per-Insight edges up to Episode↔Topic / Episode↔Person aggregates.',
      testid: 'lens-aggregated-edges',
      available: aggregatedEdgesAvailable.value,
    },
  ]
  for (const l of registeredLenses.list()) {
    all.push({
      key: l.id,
      label: l.label,
      description: l.description,
      testid: l.testid,
      available: registeredAvailable.value[l.id] === true,
      order: l.order ?? 0,
    })
  }
  return all.filter((r) => r.available).sort((a, b) => a.order - b.order)
})

const activeCount = computed(() => {
  let n = 0
  for (const r of rows.value) if (lensOn(r.key)) n += 1
  return n
})

const chipLabel = computed(() => {
  const total = rows.value.length
  if (activeCount.value === 0) return 'Lenses ▾'
  return `Lenses: ${activeCount.value} of ${total} ▾`
})

function setLens(key: string, value: boolean): void {
  if (!isBuiltIn(key)) lenses.setLens(key, value)
  else if (key === 'aggregatedEdges') lenses.setAggregatedEdges(value)
  else if (key === 'nodeSizeByDegree') lenses.setNodeSizeByDegree(value)
  else if (key === 'bridgeRing') lenses.setBridgeRing(value)
  else if (key === 'personCredibility') lenses.setPersonCredibility(value)
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
              :checked="lensOn(r.key)"
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
