<script setup lang="ts">
import cytoscape, { type Core } from 'cytoscape'
import { storeToRefs } from 'pinia'
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import type { ParsedArtifact } from '../../types/artifact'
import { useThemeStore } from '../../stores/theme'
import { giKgCoseLayoutOptionsCompact } from '../../utils/cyCoseLayoutOptions'
import { prefersReducedMotionQuery, syncGraphLabelTierClasses } from '../../utils/cyGraphLabelTier'
import { buildGiKgCyStylesheet, cytoscapeSideLabelMarginXCallback } from '../../utils/cyGraphStylesheet'
import {
  filterArtifactEgoAroundTopicCluster,
  filterArtifactEgoOneHop,
  findRawNodeInArtifact,
  toCytoElements,
} from '../../utils/parsing'

const props = defineProps<{
  viewArtifact: ParsedArtifact | null
  centerId: string | null
  /** When set, minimap shows compound + members + 1-hop from members (not ego of compound only). */
  topicClusterNeighborhood?: { compoundId: string; memberIds: string[] } | null
}>()

const showMini = computed(() => {
  const a = props.viewArtifact
  if (!a) {
    return false
  }
  const tc = props.topicClusterNeighborhood
  if (tc?.compoundId?.trim() && tc.memberIds.length > 0) {
    const sub = miniArtifact()
    return Boolean(sub?.data?.nodes?.length)
  }
  const id = props.centerId?.trim()
  return Boolean(id && findRawNodeInArtifact(a, id))
})

const host = ref<HTMLDivElement | null>(null)
let cy: Core | null = null
let ro: ResizeObserver | null = null

const themeStore = useThemeStore()
const { choice: themeChoice } = storeToRefs(themeStore)

function destroyCy(): void {
  if (ro) {
    try {
      ro.disconnect()
    } catch {
      /* ignore */
    }
    ro = null
  }
  if (cy) {
    try {
      cy.destroy()
    } catch {
      /* ignore */
    }
    cy = null
  }
}

function miniArtifact(): ParsedArtifact | null {
  const art = props.viewArtifact
  if (!art) {
    return null
  }
  const tc = props.topicClusterNeighborhood
  if (tc?.compoundId?.trim() && tc.memberIds.length > 0) {
    return filterArtifactEgoAroundTopicCluster(art, tc.compoundId.trim(), tc.memberIds)
  }
  const id = props.centerId?.trim()
  if (!id || !findRawNodeInArtifact(art, id)) {
    return null
  }
  return filterArtifactEgoOneHop(art, id)
}

function mountCy(): void {
  destroyCy()
  const el = host.value
  const sub = miniArtifact()
  if (!el || !sub) {
    return
  }
  const elements = toCytoElements(sub)
  if (elements.length === 0) {
    return
  }
  try {
    const core = cytoscape({
      container: el,
      elements,
      style: [
        ...(buildGiKgCyStylesheet({
          compact: true,
          prefersReducedMotion: prefersReducedMotionQuery(),
        }) as Record<string, unknown>[]),
        {
          selector: 'node',
          style: {
            'text-margin-x': cytoscapeSideLabelMarginXCallback(true),
          },
        },
      ] as never,
      userZoomingEnabled: false,
      userPanningEnabled: false,
      boxSelectionEnabled: false,
      autoungrabify: true,
      minZoom: 0.15,
      maxZoom: 4,
      wheelSensitivity: 0,
    })
    cy = core
    const tc = props.topicClusterNeighborhood
    const layoutId =
      tc?.compoundId?.trim() && !core.$id(tc.compoundId.trim()).empty()
        ? tc.compoundId.trim()
        : tc?.memberIds[0]?.trim() || props.centerId!.trim()
    const cid = layoutId
    const layoutOpts: Record<string, unknown> = giKgCoseLayoutOptionsCompact()
    const layout = core.elements().layout(layoutOpts as never)
    layout.one('layoutstop', () => {
      if (cy !== core) return
      try {
        core.fit(core.elements(), 14)
        core.nodes().unselect()
        const cn = core.$id(cid)
        if (!cn.empty()) {
          cn.select()
        }
        syncGraphLabelTierClasses(core)
      } catch {
        /* ignore */
      }
    })
    layout.run()
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(() => {
        if (cy === core) {
          try {
            core.resize()
          } catch {
            /* ignore */
          }
        }
      })
      ro.observe(el)
    }
  } catch {
    cy = null
  }
}

onBeforeUnmount(() => {
  destroyCy()
})

watch(
  () =>
    [
      showMini.value,
      props.viewArtifact,
      props.centerId,
      props.topicClusterNeighborhood?.compoundId,
      props.topicClusterNeighborhood?.memberIds?.slice(),
      themeChoice.value,
    ] as const,
  () => {
    if (!showMini.value) {
      destroyCy()
      return
    }
    void nextTick(() => mountCy())
  },
  { immediate: true },
)

</script>

<template>
  <div v-if="showMini" class="mb-3">
    <p class="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted">
      <template v-if="topicClusterNeighborhood?.compoundId">Cluster neighborhood</template>
      <template v-else>Local neighborhood</template>
    </p>
    <div
      ref="host"
      class="h-36 w-full overflow-hidden rounded border border-border bg-canvas"
      data-testid="graph-neighborhood-mini"
      role="img"
      :aria-label="
        topicClusterNeighborhood?.compoundId
          ? 'Graph around topic cluster compound and member topics'
          : '1-hop graph around selected node'
      "
    />
  </div>
</template>
