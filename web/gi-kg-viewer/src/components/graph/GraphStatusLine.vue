<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { computed, ref, watch } from 'vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useGraphExpansionStore } from '../../stores/graphExpansion'
import { useGraphFilterStore } from '../../stores/graphFilters'
import { useGraphLensStore } from '../../stores/graphLens'
import {
  formatGraphNodeCount,
  graphLensActivePreset,
  graphLensSummaryLabel,
} from '../../utils/graphLensLabels'

const emit = defineEmits<{
  'request-reload': []
}>()

const artifacts = useArtifactsStore()
const graphLens = useGraphLensStore()
const graphExpansion = useGraphExpansionStore()
const gf = useGraphFilterStore()

const { sinceYmd, lastAutoLoadWasCapped } = storeToRefs(graphLens)
const { expandedBySeed } = storeToRefs(graphExpansion)

const sinceInput = ref('')

const lensLabel = computed(() => graphLensSummaryLabel(sinceYmd.value))

const lensPreset = computed(() => graphLensActivePreset(sinceYmd.value))

const presetBtnClass =
  'rounded border border-border bg-surface px-2 py-0.5 text-[10px] text-surface-foreground hover:bg-overlay/40'

const episodeCount = computed(() => {
  const ids = new Set<string>()
  for (const p of artifacts.parsedList) {
    if (p.kind === 'gi' && p.episodeId) {
      ids.add(p.episodeId)
    }
  }
  return ids.size
})

const nodeCount = computed(() => {
  const nodes = gf.fullArtifact?.data?.nodes
  return Array.isArray(nodes) ? nodes.length : 0
})

const showCappedChip = computed(
  () =>
    lastAutoLoadWasCapped.value &&
    Object.keys(expandedBySeed.value).length === 0,
)

function bumpSinceInputFromStore(): void {
  sinceInput.value = sinceYmd.value.trim()
}

bumpSinceInputFromStore()

watch(sinceYmd, () => {
  bumpSinceInputFromStore()
})

function applyPreset(days: 7 | 30 | 90): void {
  graphLens.setPresetDays(days)
  bumpSinceInputFromStore()
  emit('request-reload')
}

function applyAll(): void {
  graphLens.setAllTime()
  bumpSinceInputFromStore()
  emit('request-reload')
}

function applySinceInput(): void {
  const raw = sinceInput.value.trim()
  if (!/^\d{4}-\d{2}-\d{2}$/.test(raw)) {
    return
  }
  graphLens.setSinceYmd(raw)
  emit('request-reload')
}
</script>

<template>
  <div
    class="flex min-h-[24px] flex-wrap items-center justify-between gap-x-3 gap-y-1 border-b border-border bg-canvas px-2 py-0.5 text-[10px] text-muted"
    data-testid="graph-status-line"
  >
    <div class="min-w-0 flex-1 leading-tight">
      <span data-testid="graph-status-lens-label">Showing {{ lensLabel }}</span>
      <span class="text-border"> · </span>
      <span>
        <span data-testid="graph-status-episode-count">{{ episodeCount }}</span>
        episodes
      </span>
      <span v-if="showCappedChip" class="text-border"> (capped)</span>
      <span class="text-border"> · </span>
      <span>
        <span data-testid="graph-status-node-count">{{ formatGraphNodeCount(nodeCount) }}</span>
        nodes
      </span>
    </div>
    <div
      class="flex shrink-0 flex-wrap items-center gap-1"
      data-testid="graph-status-lens-selector"
    >
      <button
        type="button"
        :class="[presetBtnClass, lensPreset === '7' ? 'ring-2 ring-primary' : '']"
        @click="applyPreset(7)"
      >
        7d
      </button>
      <button
        type="button"
        :class="[presetBtnClass, lensPreset === '30' ? 'ring-2 ring-primary' : '']"
        @click="applyPreset(30)"
      >
        30d
      </button>
      <button
        type="button"
        :class="[presetBtnClass, lensPreset === '90' ? 'ring-2 ring-primary' : '']"
        @click="applyPreset(90)"
      >
        90d
      </button>
      <button
        type="button"
        :class="[presetBtnClass, lensPreset === 'all' ? 'ring-2 ring-primary' : '']"
        @click="applyAll()"
      >
        All
      </button>
      <label class="flex items-center gap-0.5 whitespace-nowrap">
        <span class="text-[9px]">Since</span>
        <input
          v-model="sinceInput"
          type="date"
          data-testid="graph-status-since-input"
          class="max-w-[9rem] rounded border border-border bg-surface px-0.5 py-0 text-[9px] text-surface-foreground"
          :class="lensPreset === 'custom' ? 'ring-2 ring-primary' : ''"
          @change="applySinceInput"
        >
      </label>
    </div>
  </div>
</template>
