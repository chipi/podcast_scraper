<script setup lang="ts">
import { computed } from 'vue'
import { useEpisodeRailStore } from '../../stores/episodeRail'
import { useGraphFilterStore } from '../../stores/graphFilters'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { findRawNodeInArtifact } from '../../utils/parsing'
import NodeDetail from './NodeDetail.vue'

const gf = useGraphFilterStore()
const nav = useGraphNavigationStore()
const episodeRail = useEpisodeRailStore()

const emit = defineEmits<{ 'go-graph': [] }>()

const viewArtifact = computed(() => gf.viewWithEgo(nav.graphEgoFocusCyId))

const nodeId = computed(() => episodeRail.graphNodeCyId)

const rawNode = computed(() => {
  const art = viewArtifact.value
  const id = nodeId.value
  if (!art || !id) {
    return null
  }
  return findRawNodeInArtifact(art, id)
})

/** Same type chip as ``NodeDetail``; used as rail heading (Library uses “Episode”). */
const panelHeading = computed(() => {
  const n = rawNode.value
  if (!n) {
    return 'Node'
  }
  const t = n.type
  return typeof t === 'string' && t.trim() ? t.trim() : 'Node'
})

function onClose(): void {
  episodeRail.showTools({ preserveGraphNodeId: true })
}
</script>

<template>
  <div
    class="mx-2 flex min-h-0 min-w-0 flex-1 flex-col"
    role="region"
    :aria-label="`Graph node: ${panelHeading}`"
    data-testid="graph-node-detail-rail"
  >
    <div class="mt-1 flex shrink-0 items-center justify-between gap-2 border-b border-border pb-2">
      <h2 class="text-xs font-semibold text-surface-foreground">
        {{ panelHeading }}
      </h2>
      <button
        type="button"
        class="shrink-0 rounded border border-border px-2 py-1 text-[10px] font-medium text-elevated-foreground hover:bg-overlay"
        @click="onClose"
      >
        Search & Explore
      </button>
    </div>
    <NodeDetail
      v-if="nodeId"
      embed-in-rail
      :view-artifact="viewArtifact"
      :node-id="nodeId"
      @close="onClose"
      @go-graph="emit('go-graph')"
    />
  </div>
</template>
