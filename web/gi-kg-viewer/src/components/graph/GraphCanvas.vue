<script setup lang="ts">
import cytoscape, { type Core } from 'cytoscape'
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import NodeDetail from './NodeDetail.vue'
import { useGraphFilterStore } from '../../stores/graphFilters'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { graphNodeFill } from '../../utils/colors'
import { toCytoElements } from '../../utils/parsing'

const gf = useGraphFilterStore()
const nav = useGraphNavigationStore()
const container = ref<HTMLDivElement | null>(null)
const focusNodeId = ref<string | null>(null)
const selectedNodeId = ref<string | null>(null)
/** Detail side panel opens on double-click only; selection stays for future node actions. */
const detailPanelOpen = ref(false)

const hint = computed(() => {
  if (focusNodeId.value) {
    return (
      'Neighborhood view: double-click empty canvas for full graph. ' +
      'Double-click a node for details. Shift+double-click toggles 1-hop focus.'
    )
  }
  return (
    'Click a node to select. Double-click for details. ' +
    'Shift+double-click for 1-hop neighborhood.'
  )
})

function nodeLabelColor(): string {
  if (typeof window !== 'undefined' && window.matchMedia) {
    try {
      if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        return '#e8ecf1'
      }
    } catch {
      /* ignore */
    }
  }
  return '#1a2332'
}

function buildCyStyle() {
  const types = [
    'Episode',
    'Insight',
    'Quote',
    'Speaker',
    'Topic',
    'Entity_person',
    'Entity_organization',
    'Podcast',
  ]
  const style = [
    {
      selector: 'node',
      style: {
        label: 'data(label)',
        'font-size': '9px',
        'text-wrap': 'wrap',
        'text-max-width': '140px',
        'background-color': '#868e96',
        color: nodeLabelColor(),
        width: 18,
        height: 18,
        'border-width': 0,
      },
    },
    {
      selector: 'node:selected',
      style: {
        'border-width': 3,
        'border-color': '#228be6',
        'border-opacity': 1,
      },
    },
    {
      selector: 'edge',
      style: {
        width: 1.5,
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': '#adb5bd',
        'line-color': '#adb5bd',
        label: 'data(label)',
        'font-size': '8px',
        color: '#495057',
      },
    },
  ]
  for (const t of types) {
    style.push({
      selector: `node[type = "${t}"]`,
      style: {
        'background-color': graphNodeFill(t),
      },
    } as (typeof style)[number])
  }
  return style as never
}

let cy: Core | null = null
let resizeObs: ResizeObserver | null = null
let zoomCenterTimer: ReturnType<typeof setTimeout> | null = null
let lastZoomLevel = 1

function destroyCy(): void {
  if (zoomCenterTimer != null) {
    clearTimeout(zoomCenterTimer)
    zoomCenterTimer = null
  }
  if (resizeObs) {
    resizeObs.disconnect()
    resizeObs = null
  }
  if (cy) {
    cy.destroy()
    cy = null
  }
}

function tryApplyPendingFocus(core: Core): void {
  const id = nav.pendingFocusNodeId
  if (!id) return
  const n = core.$id(id)
  if (n.empty()) {
    return
  }
  core.nodes().unselect()
  n.select()
  selectedNodeId.value = id
  detailPanelOpen.value = true
  try {
    core.animate({
      center: { eles: n },
      duration: 260,
    })
  } catch {
    try {
      core.center(n)
    } catch {
      /* ignore */
    }
  }
  nav.clearPendingFocus()
}

function fitAnimated(): void {
  if (!cy) return
  const els = cy.elements()
  if (els.length === 0) return
  try {
    cy.animate({
      fit: { eles: els, padding: 24 },
      duration: 280,
    })
  } catch {
    cy.fit(els, 24)
  }
}

function canvasExportBg(): string {
  try {
    const v = getComputedStyle(document.documentElement)
      .getPropertyValue('--ps-canvas')
      .trim()
    if (v.length) {
      return v
    }
  } catch {
    /* ignore */
  }
  return '#111418'
}

/** Escape / shortcuts: clear ego focus, selection, detail panel, pending nav. */
function clearInteractionState(): void {
  nav.clearPendingFocus()
  const hadEgo = focusNodeId.value !== null
  focusNodeId.value = null
  selectedNodeId.value = null
  detailPanelOpen.value = false
  const c = cy
  if (c) {
    try {
      c.nodes().unselect()
    } catch {
      /* ignore */
    }
  }
  if (hadEgo) {
    redraw()
  }
}

function exportGraphPng(): void {
  const c = cy
  if (!c) return
  const els = c.elements()
  if (els.length === 0) return
  let uri: string
  try {
    uri = c.png({
      output: 'base64uri',
      full: true,
      scale: 2,
      bg: canvasExportBg(),
    })
  } catch {
    return
  }
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')
  const a = document.createElement('a')
  a.href = uri
  a.download = `gi-kg-viewer-${stamp}.png`
  a.rel = 'noopener'
  document.body.appendChild(a)
  a.click()
  a.remove()
}

function redraw(): void {
  destroyCy()
  const el = container.value
  if (!el) return

  const art = gf.viewWithEgo(focusNodeId.value)
  if (!art) {
    el.innerHTML =
      '<p class="p-4 text-sm text-muted">Load artifacts and use “Load selected” to render the graph.</p>'
    return
  }

  const elements = toCytoElements(art)
  const nodeCount = elements.filter((x) => !('source' in x.data)).length
  if (nodeCount === 0) {
    el.innerHTML =
      '<p class="p-4 text-sm text-muted">No nodes in this view (adjust filters).</p>'
    return
  }

  el.innerHTML = ''
  const core = cytoscape({
    container: el,
    elements,
    layout: {
      name: 'cose',
      padding: 24,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      nodeRepulsion: () => 450000,
    } as any,
    style: buildCyStyle(),
    wheelSensitivity: 0.35,
  })
  cy = core

  const sync = (): void => {
    if (cy === core) core.resize()
  }
  if (typeof ResizeObserver !== 'undefined') {
    resizeObs = new ResizeObserver(sync)
    resizeObs.observe(el)
  }
  requestAnimationFrame(sync)
  requestAnimationFrame(() => requestAnimationFrame(sync))

  const sid = selectedNodeId.value
  if (sid) {
    const n = core.$id(sid)
    if (n.empty()) {
      selectedNodeId.value = null
      detailPanelOpen.value = false
    } else {
      core.nodes().unselect()
      n.select()
    }
  }

  core.on('tap', (evt) => {
    const t = evt.target
    if (t === core) {
      core.nodes().unselect()
      selectedNodeId.value = null
      detailPanelOpen.value = false
      return
    }
    if (typeof t.isNode === 'function' && t.isNode()) {
      core.nodes().unselect()
      t.select()
      selectedNodeId.value = t.id()
      detailPanelOpen.value = false
      return
    }
    selectedNodeId.value = null
    detailPanelOpen.value = false
  })

  core.on('dbltap', (evt) => {
    const raw = evt.originalEvent as MouseEvent | TouchEvent | undefined
    const shift =
      raw && 'shiftKey' in raw ? Boolean((raw as MouseEvent).shiftKey) : false
    const t = evt.target
    if (typeof t.isNode === 'function' && t.isNode()) {
      const id = t.id()
      if (shift) {
        focusNodeId.value = focusNodeId.value === id ? null : id
        redraw()
        return
      }
      core.nodes().unselect()
      t.select()
      selectedNodeId.value = id
      detailPanelOpen.value = true
      return
    }
    if (focusNodeId.value !== null) {
      focusNodeId.value = null
      redraw()
    }
  })

  // COSE often finishes with the cluster off to one side; fit once, then track zoom for re-center.
  core.one('layoutstop', () => {
    if (!cy) return
    try {
      cy.fit(cy.elements(), 24)
    } catch {
      /* ignore */
    }
    lastZoomLevel = cy.zoom()

    // Wheel zoom anchors on cursor; after zooming out, pan so content stays centered in the viewport.
    cy.on('zoom', () => {
      if (!cy) return
      const z = cy.zoom()
      const zoomedOut = z < lastZoomLevel - 1e-9
      lastZoomLevel = z
      if (!zoomedOut) return
      if (zoomCenterTimer != null) {
        clearTimeout(zoomCenterTimer)
      }
      zoomCenterTimer = setTimeout(() => {
        zoomCenterTimer = null
        if (!cy) return
        try {
          cy.center(cy.elements())
        } catch {
          /* ignore */
        }
      }, 160)
    })
    tryApplyPendingFocus(core)
  })
}

watch(
  () => gf.filteredArtifact,
  () => {
    focusNodeId.value = null
    selectedNodeId.value = null
    detailPanelOpen.value = false
    redraw()
  },
  { flush: 'post' },
)

watch(
  () => [nav.pendingFocusNodeId, gf.filteredArtifact] as const,
  () => {
    nextTick(() => {
      const c = cy
      if (c) tryApplyPendingFocus(c)
    })
  },
  { flush: 'post' },
)

onMounted(() => {
  redraw()
})

onUnmounted(() => {
  destroyCy()
})

defineExpose({
  fitAnimated,
  redraw,
  clearInteractionState,
  exportGraphPng,
})
</script>

<template>
  <div
    class="flex min-h-[280px] flex-1 flex-col rounded border border-border bg-canvas sm:min-h-[420px]"
  >
    <div class="flex flex-wrap items-center gap-2 border-b border-border px-2 py-1.5">
      <button
        type="button"
        class="rounded bg-primary px-2 py-1 text-xs font-medium text-primary-foreground hover:opacity-90"
        @click="fitAnimated"
      >
        Fit
      </button>
      <button
        type="button"
        class="rounded border border-border px-2 py-1 text-xs hover:bg-overlay"
        @click="redraw"
      >
        Re-layout
      </button>
      <button
        type="button"
        class="rounded border border-border px-2 py-1 text-xs hover:bg-overlay"
        title="Full graph as PNG (2× scale)"
        @click="exportGraphPng"
      >
        Export PNG
      </button>
      <span class="min-w-0 flex-1 text-xs text-muted">{{ hint }}</span>
    </div>
    <div class="flex min-h-[240px] min-w-0 flex-1 sm:min-h-[380px]">
      <div
        ref="container"
        class="graph-canvas min-h-[240px] min-w-0 flex-1 sm:min-h-[380px]"
      />
      <NodeDetail
        v-if="selectedNodeId && detailPanelOpen"
        class="max-w-xs shrink-0"
        :view-artifact="gf.viewWithEgo(focusNodeId)"
        :node-id="selectedNodeId"
        @close="detailPanelOpen = false"
      />
    </div>
  </div>
</template>
