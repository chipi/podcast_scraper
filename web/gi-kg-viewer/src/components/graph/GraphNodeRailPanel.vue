<script setup lang="ts">
import { computed } from 'vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useGraphFilterStore } from '../../stores/graphFilters'
import { useSubjectStore } from '../../stores/subject'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { findRawNodeInArtifact } from '../../utils/parsing'
import {
  findClusterByCompoundId,
  findTopicClusterContextForGraphNode,
} from '../../utils/topicClustersOverlay'
import NodeDetail from './NodeDetail.vue'

const gf = useGraphFilterStore()
const nav = useGraphNavigationStore()
const subject = useSubjectStore()
const artifacts = useArtifactsStore()

const emit = defineEmits<{
  'go-graph': []
  'prefill-semantic-search': [{ query: string }]
  'open-explore-topic-filter': [{ topic: string }]
  'open-explore-speaker-filter': [{ speaker: string }]
  'open-explore-insight-filters': [{ groundedOnly: boolean; minConfidence: number | null }]
  'open-library-episode': [{ metadata_relative_path: string }]
  'close-subject-rail': []
}>()

/**
 * Prefer the ego slice when the selected node is inside it (matches canvas). If not — e.g.
 * **TopicCluster** compounds are not linked by **edges** to ego center, only ``parent`` in
 * Cytoscape — fall back to the full filtered graph so ``findRawNodeInArtifact`` resolves and
 * ``NodeDetail`` (member list, connections) can render.
 */
const viewArtifact = computed(() => {
  const id = subject.graphNodeCyId?.trim()
  const ego = gf.viewWithEgo(nav.graphEgoFocusCyId)
  const full = gf.filteredArtifact
  if (!id || !full) {
    return ego
  }
  if (ego && findRawNodeInArtifact(ego, id)) {
    return ego
  }
  return full
})

const nodeId = computed(() => subject.graphNodeCyId)

const rawNode = computed(() => {
  const id = nodeId.value
  if (!id) {
    return null
  }
  const slice = viewArtifact.value
  if (slice) {
    const hit = findRawNodeInArtifact(slice, id)
    if (hit) {
      return hit
    }
  }
  const full = gf.fullArtifact
  return full ? findRawNodeInArtifact(full, id) : null
})

/** Same type chip as ``NodeDetail``; used as rail heading (Library uses “Episode”). */
const panelHeading = computed(() => {
  const n = rawNode.value
  const id = nodeId.value?.trim()
  if (!n) {
    return 'Node'
  }
  if (
    id &&
    (findClusterByCompoundId(artifacts.topicClustersDoc, id) ||
      findTopicClusterContextForGraphNode(id, artifacts.topicClustersDoc))
  ) {
    return 'TopicCluster'
  }
  const t = n.type
  return typeof t === 'string' && t.trim() ? t.trim() : 'Node'
})

function onClose(): void {
  emit('close-subject-rail')
}
</script>

<template>
  <div
    class="mx-3 flex min-h-0 min-w-0 flex-1 flex-col"
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
        aria-label="Close graph node detail"
        @click="onClose"
      >
        Close
      </button>
    </div>
    <NodeDetail
      v-if="nodeId"
      embed-in-rail
      :view-artifact="viewArtifact"
      :node-id="nodeId"
      :bridge-document="artifacts.bridgeDocument"
      @close="onClose"
      @go-graph="emit('go-graph')"
      @prefill-semantic-search="emit('prefill-semantic-search', $event)"
      @open-explore-topic-filter="emit('open-explore-topic-filter', $event)"
      @open-explore-speaker-filter="emit('open-explore-speaker-filter', $event)"
      @open-explore-insight-filters="emit('open-explore-insight-filters', $event)"
      @open-library-episode="emit('open-library-episode', $event)"
    />
  </div>
</template>
