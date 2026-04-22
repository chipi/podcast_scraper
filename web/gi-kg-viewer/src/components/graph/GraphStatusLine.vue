<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { computed, ref, watch } from 'vue'
import { useGraphExpansionStore } from '../../stores/graphExpansion'
import { useGraphExplorerStore } from '../../stores/graphExplorer'
import { useGraphFilterStore } from '../../stores/graphFilters'
import { useArtifactsStore } from '../../stores/artifacts'
import {
  formatGraphNodeCount,
  graphLensActivePreset,
  graphLensSummaryLabel,
} from '../../utils/graphLensLabels'
import { weaklyConnectedComponentCount } from '../../utils/graphWeakComponents'

/** ``summary`` — counts text only (top of graph card). ``controls`` — lens presets + Reset (bottom bar). */
export type GraphStatusLineVariant = 'summary' | 'controls'

const props = withDefaults(
  defineProps<{
    variant: GraphStatusLineVariant
    /** Smaller preset buttons when ``variant === 'controls'`` in the bottom bar. */
    embedded?: boolean
    /**
     * When ``variant === 'summary'``, omit outer border/padding so a parent strip
     * (e.g. stats + Gestures) owns the chrome.
     */
    bare?: boolean
  }>(),
  { embedded: false, bare: false },
)

const emit = defineEmits<{
  'request-reload': []
  'request-graph-full-reset': []
}>()

const graphExplorer = useGraphExplorerStore()
const graphExpansion = useGraphExpansionStore()
const artifacts = useArtifactsStore()
const gf = useGraphFilterStore()

const { sinceYmd, lastAutoLoadWasCapped } = storeToRefs(graphExplorer)
const { expandedBySeed } = storeToRefs(graphExpansion)
const { loading: artifactsLoading } = storeToRefs(artifacts)

const sinceInput = ref('')

const lensLabel = computed(() => graphLensSummaryLabel(sinceYmd.value))

const lensPreset = computed(() => graphLensActivePreset(sinceYmd.value))

const presetBtnClass =
  'rounded border border-border bg-surface px-2 py-0.5 text-[10px] text-surface-foreground hover:bg-overlay/40'

const presetBtnClassEmbedded =
  'rounded border border-border bg-surface px-1.5 py-px text-[9px] text-surface-foreground hover:bg-overlay/40'

/** Episode anchors visible in the graph after toolbar type filters (matches canvas). */
const episodeCount = computed(() => {
  const nodes = gf.filteredArtifact?.data?.nodes
  if (!Array.isArray(nodes)) return 0
  let n = 0
  for (const node of nodes) {
    if (node && String(node.type || '') === 'Episode') n += 1
  }
  return n
})

const nodeCount = computed(() => {
  const nodes = gf.filteredArtifact?.data?.nodes
  return Array.isArray(nodes) ? nodes.length : 0
})

const componentCount = computed(() => weaklyConnectedComponentCount(gf.filteredArtifact))

/** Cross-episode expand (RFC-076) appended artifacts beyond the auto slice — offer one-click restore. */
const showGraphFullReset = computed(() => Object.keys(expandedBySeed.value).length > 0)

function bumpSinceInputFromStore(): void {
  sinceInput.value = sinceYmd.value.trim()
}

bumpSinceInputFromStore()

watch(sinceYmd, () => {
  bumpSinceInputFromStore()
})

function applyPreset(days: 7 | 30 | 90): void {
  graphExplorer.setPresetDays(days)
  bumpSinceInputFromStore()
  emit('request-reload')
}

function applyAll(): void {
  graphExplorer.setAllTime()
  bumpSinceInputFromStore()
  emit('request-reload')
}

function applySinceInput(): void {
  const raw = sinceInput.value.trim()
  if (!/^\d{4}-\d{2}-\d{2}$/.test(raw)) {
    return
  }
  graphExplorer.setSinceYmd(raw)
  emit('request-reload')
}

function onRequestGraphFullReset(): void {
  if (artifactsLoading.value) {
    return
  }
  emit('request-graph-full-reset')
}

const presetClass = computed(() => (props.embedded ? presetBtnClassEmbedded : presetBtnClass))

const summaryRootClass = computed(() =>
  props.bare
    ? 'flex min-h-0 min-w-0 shrink items-center py-0 pr-1 text-[10px] leading-none text-muted'
    : 'border-b border-border bg-canvas px-2 py-px text-[10px] leading-tight text-muted',
)
</script>

<template>
  <!-- Counts only — top row; ``data-testid="graph-status-line"`` stays on this strip for E2E. -->
  <div v-if="props.variant === 'summary'" :class="summaryRootClass" data-testid="graph-status-line">
    <div class="leading-tight">
      <span data-testid="graph-status-lens-label">Showing {{ lensLabel }}</span>
      <span class="text-border"> · </span>
      <span>
        <span data-testid="graph-status-episode-count">{{ episodeCount }}</span>
        episodes<span v-if="lastAutoLoadWasCapped" data-testid="graph-status-capped"> (capped)</span>
      </span>
      <span class="text-border"> · </span>
      <span>
        <span data-testid="graph-status-node-count">{{ formatGraphNodeCount(nodeCount) }}</span>
        nodes
      </span>
      <span class="text-border"> · </span>
      <span>
        <span data-testid="graph-status-component-count">{{ componentCount }}</span>
        components
      </span>
    </div>
  </div>

  <!-- Lens + Reset only — bottom bar centre -->
  <div
    v-else
    class="flex min-h-0 min-w-0 flex-wrap items-center justify-center gap-x-1 gap-y-0.5 px-1 py-0"
    data-testid="graph-status-line-controls"
    role="group"
    aria-label="Graph time lens"
  >
    <div
      class="flex shrink-0 flex-wrap items-center gap-0.5"
      :class="props.embedded ? 'justify-center' : ''"
      data-testid="graph-status-lens-selector"
    >
      <button
        type="button"
        :class="[presetClass, lensPreset === '7' ? 'ring-2 ring-primary' : '']"
        @click="applyPreset(7)"
      >
        7d
      </button>
      <button
        type="button"
        :class="[presetClass, lensPreset === '30' ? 'ring-2 ring-primary' : '']"
        @click="applyPreset(30)"
      >
        30d
      </button>
      <button
        type="button"
        :class="[presetClass, lensPreset === '90' ? 'ring-2 ring-primary' : '']"
        @click="applyPreset(90)"
      >
        90d
      </button>
      <button
        type="button"
        :class="[presetClass, lensPreset === 'all' ? 'ring-2 ring-primary' : '']"
        @click="applyAll()"
      >
        All
      </button>
      <label class="flex items-center gap-0.5 whitespace-nowrap">
        <span :class="props.embedded ? 'text-[8px]' : 'text-[9px]'">Since</span>
        <input
          v-model="sinceInput"
          type="date"
          data-testid="graph-status-since-input"
          class="max-w-[9rem] rounded border border-border bg-surface px-0.5 py-0 text-[9px] text-surface-foreground"
          :class="[
            lensPreset === 'custom' ? 'ring-2 ring-primary' : '',
            props.embedded ? 'max-w-[7.5rem]' : '',
          ]"
          @change="applySinceInput"
        >
      </label>
      <button
        v-if="showGraphFullReset"
        type="button"
        :disabled="artifactsLoading"
        :class="
          props.embedded
            ? 'rounded border border-border bg-surface px-1.5 py-px text-[9px] font-medium text-surface-foreground hover:bg-overlay/40 disabled:cursor-not-allowed disabled:opacity-40'
            : 'rounded border border-border bg-surface px-2 py-0.5 text-[10px] font-medium text-surface-foreground hover:bg-overlay/40 disabled:cursor-not-allowed disabled:opacity-40'
        "
        data-testid="graph-status-reset"
        title="Clear cross-episode expansions, restore the default time slice (15 episodes max), and fit the graph"
        @click="onRequestGraphFullReset"
      >
        Reset
      </button>
    </div>
  </div>
</template>
