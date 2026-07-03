<script setup lang="ts">
import { computed } from 'vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useGraphFilterStore } from '../../stores/graphFilters'
import { useSubjectStore } from '../../stores/subject'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { findRawNodeInArtifactByIdOrPrefixed } from '../../utils/parsing'
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
  if (ego && findRawNodeInArtifactByIdOrPrefixed(ego, id)) {
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
    const hit = findRawNodeInArtifactByIdOrPrefixed(slice, id)
    if (hit) {
      return hit
    }
  }
  const full = gf.fullArtifact
  return full ? findRawNodeInArtifactByIdOrPrefixed(full, id) : null
})

/** Same type chip as ``NodeDetail``; used as rail heading (Library uses “Episode”). */
const panelHeading = computed(() => {
  const id = nodeId.value?.trim()
  /** Compound ids are stable even before ``topicClustersDoc`` sync or graph node resolution. */
  if (id?.startsWith('tc:')) {
    return 'TopicCluster'
  }
  const n = rawNode.value
  if (!n) {
    // Out-of-slice node (from the full relational graph): NodeDetail still
    // renders the person / topic / entity view, so title the rail accordingly
    // instead of a bare "Node".
    if (id) {
      if (id.includes('person:')) return 'Person'
      if (id.includes('org:')) return 'Organization'
      if (id.includes('topic:')) return 'Topic'
      if (id.includes('podcast:')) return 'Podcast'
    }
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
      <div class="flex min-w-0 items-center gap-1.5">
        <button
          v-if="subject.canGoBack"
          type="button"
          class="shrink-0 rounded border border-border px-1.5 py-0.5 text-xs font-medium text-elevated-foreground hover:bg-overlay"
          data-testid="subject-rail-back"
          aria-label="Back to previous node"
          @click="subject.back()"
        >
          ←
        </button>
        <h2 class="min-w-0 truncate text-xs font-semibold text-surface-foreground">
          {{ panelHeading }}
        </h2>
      </div>
      <button
        type="button"
        class="shrink-0 self-center rounded border border-border px-1.5 py-0.5 text-xs font-medium text-elevated-foreground hover:bg-overlay"
        data-testid="subject-rail-close"
        aria-label="Close graph node detail"
        @click="onClose"
      >
        ×
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
