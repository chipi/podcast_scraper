<script setup lang="ts">
import { computed } from 'vue'
import type { ParsedArtifact } from '../../types/artifact'
import { useArtifactsStore } from '../../stores/artifacts'
import { useGraphFilterStore } from '../../stores/graphFilters'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useShellStore } from '../../stores/shell'
import { graphNodeTypeChrome } from '../../utils/colors'
import { truncate } from '../../utils/formatting'
import {
  findRawNodeInArtifact,
  fullPrimaryNodeLabel,
} from '../../utils/parsing'
import {
  logicalEpisodeIdFromGraphNodeId,
  metadataPathFromEpisodeProperties,
  resolveEpisodeMetadataFromLoadedArtifacts,
} from '../../utils/graphEpisodeMetadata'
import { graphNeighborsForNode, type GraphNeighborRow } from '../../utils/graphNeighbors'
import { graphTypeAvatarLetter } from '../../utils/graphTypeAvatar'
import {
  SEARCH_RESULT_GRAPH_BUTTON_CLASS,
  SEARCH_RESULT_LIBRARY_BUTTON_CLASS,
  SEARCH_RESULT_SEMANTIC_PREFILL_BUTTON_CLASS,
} from '../../utils/searchResultActionStyles'
import GraphNeighborhoodMiniMap from './GraphNeighborhoodMiniMap.vue'

const SEMANTIC_PREFILL_MAX_CHARS = 240

const props = withDefaults(
  defineProps<{
    viewArtifact: ParsedArtifact | null
    nodeId: string | null
    /** When set, use merged neighbor rows (e.g. TopicCluster selects all member topics’ edges). */
    aggregatedNeighborRows?: GraphNeighborRow[] | undefined
    /** Minimap center when aggregating (first member topic id). */
    miniMapCenterId?: string | null
    /** Override empty-state copy (aggregate mode). */
    connectionsEmptyHint?: string | null
    /** Minimap: compound + members + 1-hop from members (TopicCluster selection). */
    topicClusterNeighborhood?: { compoundId: string; memberIds: string[] } | null
    /**
     * When true (default), neighbor rows use a capped inner scroll. When false, the list grows
     * with the parent scroll (e.g. graph rail **Neighbourhood** tab).
     */
    denseNeighborList?: boolean
  }>(),
  { denseNeighborList: true },
)

const emit = defineEmits<{
  'go-graph': []
  'open-library-episode': [{ metadata_relative_path: string }]
  'prefill-semantic-search': [{ query: string }]
}>()

const nav = useGraphNavigationStore()
const shell = useShellStore()
const artifacts = useArtifactsStore()
const graphFilters = useGraphFilterStore()

/** Neighbor rows + Library metadata resolve against merged GI/KG, not only canvas-visible types. */
const connectionsArtifact = computed(
  () => graphFilters.fullArtifact ?? props.viewArtifact,
)

const neighbors = computed((): GraphNeighborRow[] => {
  if (props.aggregatedNeighborRows !== undefined) {
    return props.aggregatedNeighborRows
  }
  return graphNeighborsForNode(connectionsArtifact.value, props.nodeId)
})

const miniMapCenterIdResolved = computed((): string | null => {
  const o = props.miniMapCenterId?.trim()
  if (o) {
    return o
  }
  return props.nodeId
})

const connectionsEmptyMessage = computed((): string => {
  const h = props.connectionsEmptyHint?.trim()
  if (h) {
    return h
  }
  return 'No edges in this view (isolated node or filtered out).'
})

const neighborListUlClass = computed((): string =>
  props.denseNeighborList
    ? 'max-h-48 space-y-1.5 overflow-y-auto text-xs leading-snug'
    : 'space-y-1.5 text-xs leading-snug',
)

/** Show rail neighborhood UI when the center node exists in the current merged view. */
const centerInView = computed(() => {
  const a = props.viewArtifact
  const id = props.nodeId
  return Boolean(a && id && findRawNodeInArtifact(a, id))
})

const graphButtonTooltip =
  'Show on graph — focus this node in the loaded merged graph (same as semantic search G).'

const libraryButtonTooltip =
  'Open episode in the subject panel — same as semantic search L when metadata path resolves.'

const semanticPrefillButtonTooltip =
  "Prefill semantic search with this node's primary text (truncated); switch to Search and run Search to query the index."

function neighborAvatarStyle(vt: string): Record<string, string> {
  const c = graphNodeTypeChrome(vt)
  return {
    backgroundColor: c.background,
    border: `2px solid ${c.border}`,
    color: c.labelColor,
  }
}

function isEpisodeNeighborType(nb: GraphNeighborRow): boolean {
  return nb.type.trim().toLowerCase() === 'episode'
}

function libraryMetadataPathForEpisodeNeighbor(nb: GraphNeighborRow): string | null {
  if (!isEpisodeNeighborType(nb)) return null
  const node = findRawNodeInArtifact(connectionsArtifact.value, nb.id)
  const fromEp = metadataPathFromEpisodeProperties(node)
  if (fromEp?.trim()) return fromEp.trim()
  const logical = logicalEpisodeIdFromGraphNodeId(nb.id)
  if (!logical?.trim()) return null
  return resolveEpisodeMetadataFromLoadedArtifacts(
    logical.trim(),
    artifacts.parsedList,
    artifacts.selectedRelPaths,
  )
}

function libraryNeighborEnabled(nb: GraphNeighborRow): boolean {
  return Boolean(
    libraryMetadataPathForEpisodeNeighbor(nb) &&
      shell.healthStatus &&
      shell.corpusLibraryApiAvailable,
  )
}

function semanticPrefillQueryForNeighbor(nb: GraphNeighborRow): string {
  const raw = findRawNodeInArtifact(connectionsArtifact.value, nb.id)
  const text = raw ? fullPrimaryNodeLabel(raw) : nb.label
  return truncate(text.trim(), SEMANTIC_PREFILL_MAX_CHARS)
}

function onGraphNeighbor(nbId: string, ev: MouseEvent): void {
  ev.stopPropagation()
  nav.requestFocusNode(nbId)
  emit('go-graph')
}

function onOpenLibraryNeighbor(nb: GraphNeighborRow, ev: MouseEvent): void {
  ev.stopPropagation()
  const p = libraryMetadataPathForEpisodeNeighbor(nb)?.trim()
  if (!p || !shell.healthStatus || !shell.corpusLibraryApiAvailable) return
  emit('open-library-episode', { metadata_relative_path: p })
}

function onSemanticPrefillNeighbor(nb: GraphNeighborRow, ev: MouseEvent): void {
  ev.stopPropagation()
  const q = semanticPrefillQueryForNeighbor(nb).trim()
  if (!q || !shell.healthStatus) return
  emit('prefill-semantic-search', { query: q })
}

/** Merged TopicCluster rows: which member topic(s) connect to this neighbor. */
function neighborViaLine(nb: GraphNeighborRow): string {
  const ids = nb.viaMemberTopicIds
  if (!ids?.length || !connectionsArtifact.value) {
    return ''
  }
  const labels = ids
    .map((id) => {
      const n = findRawNodeInArtifact(connectionsArtifact.value, id)
      return n ? fullPrimaryNodeLabel(n) : id
    })
    .filter(Boolean)
  if (!labels.length) {
    return ''
  }
  return `Via: ${labels.join(', ')}`
}
</script>

<template>
  <div
    v-if="centerInView"
    class="border-t border-border bg-surface/30 px-2 pb-2 pt-3"
    role="region"
    aria-label="Graph neighborhood and connections"
    data-testid="graph-connections-section"
  >
    <GraphNeighborhoodMiniMap
      :view-artifact="connectionsArtifact"
      :center-id="miniMapCenterIdResolved"
      :topic-cluster-neighborhood="topicClusterNeighborhood"
    />
    <p
      v-if="aggregatedNeighborRows !== undefined"
      class="mb-1 text-[9px] leading-snug text-muted"
    >
      Edges from member topics to other nodes in this view. Via shows which member topic each edge
      comes from.
    </p>
    <h4 class="mb-2 text-[10px] font-semibold uppercase tracking-wide text-muted">
      <template v-if="aggregatedNeighborRows !== undefined">
        Connections to other nodes ({{ neighbors.length }})
      </template>
      <template v-else>
        Connections ({{ neighbors.length }})
      </template>
    </h4>
    <ul
      v-if="neighbors.length"
      :class="neighborListUlClass"
    >
      <li
        v-for="nb in neighbors"
        :key="`${nb.id}-${nb.direction}-${nb.edgeType}`"
        class="flex items-center gap-1.5 rounded px-0.5 py-0.5 hover:bg-overlay/60"
        :data-connection-node-id="nb.id"
      >
        <div
          class="flex size-7 shrink-0 items-center justify-center rounded-xl text-[11px] font-extrabold shadow-sm ring-1 ring-black/10 dark:ring-white/10"
          :style="neighborAvatarStyle(nb.visualType)"
          aria-hidden="true"
        >
          {{ graphTypeAvatarLetter(nb.visualType) }}
        </div>
        <div class="min-w-0 flex-1">
          <span
            class="block truncate font-medium text-surface-foreground"
            :title="nb.label"
          >{{ nb.label }}</span>
          <span class="mt-0.5 block text-[10px] leading-tight text-muted">
            {{ nb.type }}<template v-if="nb.edgeType"> ({{ nb.edgeType }})</template>
          </span>
          <span
            v-if="neighborViaLine(nb)"
            class="mt-0.5 block text-[9px] leading-tight text-muted/95"
          >
            {{ neighborViaLine(nb) }}
          </span>
        </div>
        <div class="flex shrink-0 items-center gap-1">
          <button
            v-if="isEpisodeNeighborType(nb)"
            type="button"
            :class="SEARCH_RESULT_LIBRARY_BUTTON_CLASS"
            :aria-label="libraryButtonTooltip"
            :title="libraryButtonTooltip"
            data-testid="graph-connection-open-library"
            :disabled="!libraryNeighborEnabled(nb)"
            @click="onOpenLibraryNeighbor(nb, $event)"
          >
            L
          </button>
          <button
            type="button"
            :class="SEARCH_RESULT_GRAPH_BUTTON_CLASS"
            :aria-label="graphButtonTooltip"
            :title="graphButtonTooltip"
            data-testid="graph-connection-focus-graph"
            @click="onGraphNeighbor(nb.id, $event)"
          >
            G
          </button>
          <button
            type="button"
            :class="SEARCH_RESULT_SEMANTIC_PREFILL_BUTTON_CLASS"
            :aria-label="semanticPrefillButtonTooltip"
            :title="semanticPrefillButtonTooltip"
            data-testid="graph-connection-prefill-search"
            :disabled="!shell.healthStatus"
            @click="onSemanticPrefillNeighbor(nb, $event)"
          >
            S
          </button>
        </div>
      </li>
    </ul>
    <p
      v-else
      class="text-[10px] text-muted"
    >
      {{ connectionsEmptyMessage }}
    </p>
  </div>
</template>
