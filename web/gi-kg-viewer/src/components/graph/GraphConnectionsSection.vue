<script setup lang="ts">
import { computed } from 'vue'
import type { ParsedArtifact } from '../../types/artifact'
import { graphNodeTypeChrome } from '../../utils/colors'
import { findRawNodeInArtifact } from '../../utils/parsing'
import { graphNeighborsForNode } from '../../utils/graphNeighbors'
import { graphTypeAvatarLetter } from '../../utils/graphTypeAvatar'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { SEARCH_RESULT_GRAPH_BUTTON_CLASS } from '../../utils/searchResultActionStyles'
import GraphNeighborhoodMiniMap from './GraphNeighborhoodMiniMap.vue'

const props = defineProps<{
  viewArtifact: ParsedArtifact | null
  nodeId: string | null
}>()

const emit = defineEmits<{ 'go-graph': [] }>()

const nav = useGraphNavigationStore()

const neighbors = computed(() =>
  graphNeighborsForNode(props.viewArtifact, props.nodeId),
)

/** Show rail neighborhood UI when the center node exists in the current merged view. */
const centerInView = computed(() => {
  const a = props.viewArtifact
  const id = props.nodeId
  return Boolean(a && id && findRawNodeInArtifact(a, id))
})

const graphButtonTooltip =
  'Show on graph — focus this node in the loaded merged graph (same as semantic search G).'

function neighborAvatarStyle(vt: string): Record<string, string> {
  const c = graphNodeTypeChrome(vt)
  return {
    backgroundColor: c.background,
    border: `2px solid ${c.border}`,
    color: c.labelColor,
  }
}

function onGraphNeighbor(nbId: string, ev: MouseEvent): void {
  ev.stopPropagation()
  nav.requestFocusNode(nbId)
  emit('go-graph')
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
      :view-artifact="viewArtifact"
      :center-id="nodeId"
    />
    <h4 class="mb-2 text-[10px] font-semibold uppercase tracking-wide text-muted">
      Connections ({{ neighbors.length }})
    </h4>
    <ul
      v-if="neighbors.length"
      class="max-h-48 space-y-1.5 overflow-y-auto text-xs leading-snug"
    >
      <li
        v-for="nb in neighbors"
        :key="`${nb.id}-${nb.direction}-${nb.edgeType}`"
        class="flex items-center gap-1.5 rounded px-0.5 py-0.5 hover:bg-overlay/60"
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
        </div>
        <button
          type="button"
          :class="SEARCH_RESULT_GRAPH_BUTTON_CLASS"
          :aria-label="graphButtonTooltip"
          :title="graphButtonTooltip"
          @click="onGraphNeighbor(nb.id, $event)"
        >
          G
        </button>
      </li>
    </ul>
    <p
      v-else
      class="text-[10px] text-muted"
    >
      No edges in this view (isolated node or filtered out).
    </p>
  </div>
</template>
