<script setup lang="ts">
import cytoscape, { type Core } from 'cytoscape'
import { storeToRefs } from 'pinia'
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import type { ParsedArtifact } from '../../types/artifact'
import { useThemeStore } from '../../stores/theme'
import { buildGiKgCyStylesheet } from '../../utils/cyGraphStylesheet'
import { filterArtifactEgoOneHop, findRawNodeInArtifact, toCytoElements } from '../../utils/parsing'

const props = defineProps<{
  viewArtifact: ParsedArtifact | null
  centerId: string | null
}>()

const showMini = computed(() => {
  const a = props.viewArtifact
  const id = props.centerId?.trim()
  return Boolean(a && id && findRawNodeInArtifact(a, id))
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
  const id = props.centerId?.trim()
  if (!art || !id || !findRawNodeInArtifact(art, id)) {
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
      style: buildGiKgCyStylesheet({ compact: true }) as never,
      userZoomingEnabled: false,
      userPanningEnabled: false,
      boxSelectionEnabled: false,
      autoungrabify: true,
      minZoom: 0.15,
      maxZoom: 4,
      wheelSensitivity: 0,
    })
    cy = core
    const cid = props.centerId!.trim()
    const root = core.$id(cid)
    const layoutOpts: Record<string, unknown> = {
      name: 'breadthfirst',
      directed: true,
      spacingFactor: 1.35,
      padding: 18,
      animate: false,
      fit: false,
      roots: root.empty() ? core.nodes() : root,
    }
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
      Local neighborhood
    </p>
    <div
      ref="host"
      class="h-36 w-full overflow-hidden rounded border border-border bg-canvas"
      data-testid="graph-neighborhood-mini"
      role="img"
      :aria-label="`1-hop graph around selected node`"
    />
  </div>
</template>
