<script setup lang="ts">
import cytoscape, { type Core } from 'cytoscape'
// @ts-expect-error — package has no TypeScript types (CommonJS extension)
import registerNavigator from 'cytoscape-navigator'
import 'cytoscape-navigator/cytoscape.js-navigator.css'
import { storeToRefs } from 'pinia'
import {
  computed,
  nextTick,
  onActivated,
  onMounted,
  onUnmounted,
  reactive,
  ref,
  watch,
} from 'vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useEpisodeRailStore } from '../../stores/episodeRail'
import { useGraphExplorerStore } from '../../stores/graphExplorer'
import { useGraphFilterStore } from '../../stores/graphFilters'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useSearchStore } from '../../stores/search'
import { useShellStore } from '../../stores/shell'
import { useThemeStore } from '../../stores/theme'
import type { RawGraphNode } from '../../types/artifact'
import { graphNodeFill, graphNodeLegendLabel } from '../../utils/colors'
import {
  DEGREE_BUCKET_ORDER,
  degreeBucketFor,
  emptyDegreeCounts,
} from '../../utils/graphDegreeBuckets'
import {
  logicalEpisodeIdFromGraphNodeId,
  metadataPathFromEpisodeProperties,
  resolveEpisodeMetadataFromLoadedArtifacts,
  resolveEpisodeMetadataViaCorpusCatalog,
} from '../../utils/graphEpisodeMetadata'
import { buildGiKgCyStylesheet } from '../../utils/cyGraphStylesheet'
import { findRawNodeInArtifact, toCytoElements } from '../../utils/parsing'
import { graphNodeIdFromSearchHit, resolveCyNodeId } from '../../utils/searchFocus'
import { visualNodeTypeCounts } from '../../utils/visualGroup'

registerNavigator(cytoscape)

const gf = useGraphFilterStore()
const ge = useGraphExplorerStore()
const { preferredLayout, minimapOpen, activeDegreeBucket } = storeToRefs(ge)
const nav = useGraphNavigationStore()
const episodeRail = useEpisodeRailStore()
const artifacts = useArtifactsStore()
const shell = useShellStore()
const searchStore = useSearchStore()
const themeStore = useThemeStore()

/** Cancels stale async episode-metadata resolution when the user double-taps another node. */
let graphEpisodeOpenGen = 0

async function openGraphEpisodeOrNodeRail(
  cyId: string,
  rawNode: RawGraphNode | null,
): Promise<void> {
  const token = ++graphEpisodeOpenGen
  if (rawNode?.type !== 'Episode') {
    if (token !== graphEpisodeOpenGen) return
    episodeRail.openGraphNodePanel(cyId)
    return
  }
  let meta = metadataPathFromEpisodeProperties(rawNode)?.trim() || null
  const eid =
    logicalEpisodeIdFromGraphNodeId(cyId) ||
    (typeof rawNode.properties?.episode_id === 'string'
      ? rawNode.properties.episode_id.trim()
      : null)
  if (!meta && eid) {
    meta = resolveEpisodeMetadataFromLoadedArtifacts(
      eid,
      artifacts.parsedList,
      artifacts.selectedRelPaths,
    )
  }
  if (!meta && eid && shell.corpusPath.trim() && shell.healthStatus) {
    try {
      meta = await resolveEpisodeMetadataViaCorpusCatalog(
        shell.corpusPath.trim(),
        eid,
      )
    } catch {
      meta = null
    }
  }
  if (token !== graphEpisodeOpenGen) return
  if (meta) {
    episodeRail.openEpisodePanel(meta, { graphConnectionsCyId: cyId })
  } else {
    episodeRail.openGraphNodePanel(cyId)
  }
}

const container = ref<HTMLDivElement | null>(null)
const canvasHost = ref<HTMLDivElement | null>(null)
/** True while cytoscape is rebuilding (after redraw); hides default/square pre-layout frame. */
const graphContentHiddenUntilLayout = ref(false)
const minimapHost = ref<HTMLDivElement | null>(null)

const focusNodeId = ref<string | null>(null)
const selectedNodeId = ref<string | null>(null)
const searchHighlightCount = ref(0)

const zoomPercent = ref(100)
const degreeHistogramCounts = ref<Record<string, number>>({})

const boxZoomRect = reactive({
  show: false,
  left: 0,
  top: 0,
  width: 0,
  height: 0,
})

let cy: Core | null = null
let resizeObs: ResizeObserver | null = null
let zoomCenterTimer: ReturnType<typeof setTimeout> | null = null
let lastZoomLevel = 1
let navInstance: { destroy: () => void } | null = null
let zoomPanListenerAttached = false

let boxDragging = false
let boxStartClient = { x: 0, y: 0 }
let boxListenersAttached = false

/** In-flight element layout (e.g. COSE rAF). Must be stopped before starting another or positions revert. */
let activeElesLayout: { stop: () => void } | null = null
/** Bumps when a new layout run starts so stale layoutstop handlers from a stopped layout are ignored. */
let graphLayoutGeneration = 0

/** Captured before relayout or ego exit; consumed in finishLayoutPass to avoid fit() jumping the view. */
type ViewportPreserveSnap = {
  cyId: string
  zoom: number
  rx: number
  ry: number
  /** Relayout: only restore if this node is still selected. Ego exit: restore regardless of selection. */
  requireSelectedMatch: boolean
}

let pendingViewportPreserve: ViewportPreserveSnap | null = null
/** Full-graph viewport before entering 1-hop (shift+dbl-click); applied when returning to full graph. */
let egoPriorFullGraphViewportPreserve: ViewportPreserveSnap | null = null

/** Target rendered (screen) position for the selected node; kept stable across wheel + toolbar zoom. */
let selectedNodeZoomAnchor: { x: number; y: number } | null = null

/** Skip pan-by-anchor in zoom handler while applyViewportPreserveOrFit runs (zoom+pan ordering). */
let suspendSelectedNodeZoomAnchorCorrection = 0

function clearSelectedNodeZoomAnchor(): void {
  selectedNodeZoomAnchor = null
}

function refreshSelectedNodeZoomAnchor(core: Core): void {
  const sid = selectedNodeId.value
  if (!sid || !nodeOkForViewportPreserve(core, sid)) {
    selectedNodeZoomAnchor = null
    return
  }
  try {
    const rp = core.$id(sid).renderedPosition()
    selectedNodeZoomAnchor = { x: rp.x, y: rp.y }
  } catch {
    selectedNodeZoomAnchor = null
  }
}

function nodeOkForViewportPreserve(core: Core, cyId: string): boolean {
  try {
    const n = core.$id(cyId)
    return (
      !n.empty() &&
      n.visible() &&
      String(n.style('display')) !== 'none'
    )
  } catch {
    return false
  }
}

/** Snapshot of selected node in viewport coords (for restore after layout / degree filter). */
function captureSelectedViewportAnchor(core: Core): ViewportPreserveSnap | null {
  const sid = selectedNodeId.value
  if (!sid || !nodeOkForViewportPreserve(core, sid)) {
    return null
  }
  try {
    const rp = core.$id(sid).renderedPosition()
    return {
      cyId: sid,
      zoom: core.zoom(),
      rx: rp.x,
      ry: rp.y,
      requireSelectedMatch: true,
    }
  } catch {
    return null
  }
}

/** Preserve zoom + screen position of a node (used when entering ego from full graph). */
function captureViewportAnchorForCyId(core: Core, cyId: string): ViewportPreserveSnap | null {
  if (!cyId || !nodeOkForViewportPreserve(core, cyId)) {
    return null
  }
  try {
    const rp = core.$id(cyId).renderedPosition()
    return {
      cyId,
      zoom: core.zoom(),
      rx: rp.x,
      ry: rp.y,
      requireSelectedMatch: false,
    }
  } catch {
    return null
  }
}

function applyViewportPreserveOrFit(
  core: Core,
  snap: ViewportPreserveSnap | null,
): void {
  suspendSelectedNodeZoomAnchorCorrection += 1
  try {
    if (
      snap &&
      nodeOkForViewportPreserve(core, snap.cyId) &&
      (!snap.requireSelectedMatch || selectedNodeId.value === snap.cyId)
    ) {
      try {
        core.zoom(snap.zoom)
        const nr = core.$id(snap.cyId).renderedPosition()
        core.panBy({ x: snap.rx - nr.x, y: snap.ry - nr.y })
        return
      } catch {
        /* fall through to fit */
      }
    }
    try {
      core.fit(core.elements(':visible'), 24)
    } catch {
      /* ignore */
    }
  } finally {
    suspendSelectedNodeZoomAnchorCorrection -= 1
    refreshSelectedNodeZoomAnchor(core)
  }
}

function destroyNavigator(): void {
  if (navInstance) {
    try {
      navInstance.destroy()
    } catch {
      /* ignore */
    }
    navInstance = null
  }
}

/** cytoscape-navigator sets img alt to "Graph navigator", which browsers show before src loads. */
function clearMinimapThumbnailAlt(): void {
  const host = minimapHost.value ?? document.getElementById('gi-kg-graph-minimap')
  const img = host?.querySelector('img')
  if (!img) {
    return
  }
  img.alt = ''
  img.setAttribute('aria-hidden', 'true')
}

function setupNavigator(core: Core): void {
  destroyNavigator()
  if (!minimapOpen.value || !minimapHost.value) {
    return
  }
  try {
    // cytoscape-navigator only mounts into a user container when `container` is a
    // non-empty string selector; a DOM element is ignored and a 400×400 fixed
    // panel is appended to document.body instead.
    const api = (core as unknown as { navigator: (o: object) => { destroy: () => void } }).navigator(
      {
        container: '#gi-kg-graph-minimap',
        removeCustomContainer: false,
        viewLiveFramerate: 0,
        // Throttle full-graph PNG for minimap; too low hammers the main thread on pan/zoom.
        rerenderDelay: 300,
      },
    )
    navInstance = api
    clearMinimapThumbnailAlt()
    // Thumbnail updates hook `cy.onRender`. We often attach the navigator after the
    // graph has already drawn (deferred setup after first paint), so no `render` fires
    // until the user interacts — force one draw so the minimap gets an initial PNG.
    try {
      core.forceRender()
    } catch {
      /* ignore */
    }
  } catch {
    navInstance = null
  }
}

function applySearchHighlights(core: Core): void {
  core.nodes().removeClass('search-hit')
  const matched = new Set<string>()
  for (const hit of searchStore.results) {
    const rawId = graphNodeIdFromSearchHit(hit)
    if (!rawId) continue
    const cyId = resolveCyNodeId(core, rawId)
    if (cyId && !matched.has(cyId)) {
      matched.add(cyId)
      core.$id(cyId).addClass('search-hit')
    }
  }
  for (const raw of nav.libraryHighlightSourceIds) {
    const cyId = resolveCyNodeId(core, raw)
    if (cyId && !matched.has(cyId)) {
      matched.add(cyId)
      core.$id(cyId).addClass('search-hit')
    }
  }
  searchHighlightCount.value = matched.size
}

const hint = computed(() => {
  const parts: string[] = []
  if (focusNodeId.value) {
    parts.push(
      'Neighborhood: dbl-click empty canvas for full graph. Shift+dbl-click toggles 1-hop.',
    )
  } else {
    parts.push('Shift+dbl-click node: 1-hop. Shift+drag canvas: box zoom.')
  }
  return parts.join(' ')
})

function buildCyStyle() {
  return buildGiKgCyStylesheet({ includeSearchHit: true }) as never
}

function layoutOptionsFor(name: string): Record<string, unknown> {
  if (name === 'cose') {
    return {
      name: 'cose',
      padding: 36,
      // Avoid cy.fit() on every COSE refresh tick; finishLayoutPass applies fit / viewport preserve once.
      fit: false,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      nodeRepulsion: () => 800000,
      idealEdgeLength: () => 80,
      edgeElasticity: () => 100,
      gravity: 0.15,
      nodeDimensionsIncludeLabels: true,
    } as any
  }
  return { name, padding: 36 }
}

function applyDegreeVisibility(core: Core): void {
  const bucket = activeDegreeBucket.value
  core.batch(() => {
    if (!bucket) {
      core.nodes().style('display', 'element')
      core.edges().style('display', 'element')
      return
    }
    for (const n of core.nodes()) {
      const deg = n.degree(false)
      const b = degreeBucketFor(deg)
      n.style('display', b === bucket ? 'element' : 'none')
    }
    for (const e of core.edges()) {
      const sh = e.source().style('display')
      const th = e.target().style('display')
      const hidden = sh === 'none' || th === 'none'
      e.style('display', hidden ? 'none' : 'element')
    }
  })
}

function recomputeDegreeHistogram(core: Core): void {
  const counts = emptyDegreeCounts()
  core.nodes().forEach((n) => {
    const d = n.degree(false)
    const b = degreeBucketFor(d)
    counts[b] += 1
  })
  degreeHistogramCounts.value = { ...counts }
}

function updateZoomPercentDisplay(core: Core): void {
  try {
    zoomPercent.value = Math.round(core.zoom() * 100)
  } catch {
    zoomPercent.value = 100
  }
}

function attachZoomRecenter(core: Core): void {
  if (zoomPanListenerAttached) {
    return
  }
  zoomPanListenerAttached = true
  lastZoomLevel = core.zoom()
  core.on('zoom', () => {
    if (!cy) return
    const c = cy
    const z = c.zoom()
    const prevZ = lastZoomLevel

    if (suspendSelectedNodeZoomAnchorCorrection > 0) {
      lastZoomLevel = z
      updateZoomPercentDisplay(c)
      return
    }

    const sid = selectedNodeId.value
    if (
      sid &&
      selectedNodeZoomAnchor &&
      nodeOkForViewportPreserve(c, sid)
    ) {
      const ratio = prevZ > 1e-9 ? z / prevZ : 1
      const incremental = ratio >= 0.65 && ratio <= 1.55
      if (incremental) {
        try {
          const rp = c.$id(sid).renderedPosition()
          c.panBy({
            x: selectedNodeZoomAnchor.x - rp.x,
            y: selectedNodeZoomAnchor.y - rp.y,
          })
        } catch {
          /* ignore */
        }
      }
      refreshSelectedNodeZoomAnchor(c)
    } else if (!sid) {
      clearSelectedNodeZoomAnchor()
    }

    const zoomedOut = z < prevZ - 1e-9
    lastZoomLevel = c.zoom()
    updateZoomPercentDisplay(c)
    if (!zoomedOut) return
    if (selectedNodeId.value) {
      return
    }
    if (zoomCenterTimer != null) {
      clearTimeout(zoomCenterTimer)
    }
    zoomCenterTimer = setTimeout(() => {
      zoomCenterTimer = null
      if (!cy) return
      if (selectedNodeId.value) {
        return
      }
      try {
        cy.center(cy.elements(':visible'))
      } catch {
        /* ignore */
      }
    }, 160)
  })

  core.on('pan', () => {
    if (!cy || suspendSelectedNodeZoomAnchorCorrection > 0) return
    if (selectedNodeId.value) {
      refreshSelectedNodeZoomAnchor(cy)
    }
  })
}

/**
 * Minimap uses cytoscape-navigator, which calls `cy.png({ full: true })` on a throttled
 * `onRender` hook — expensive on large graphs. Schedule it only after the main graph is
 * visible so first paint is not blocked behind that export.
 */
function scheduleMinimapSetup(core: Core): void {
  if (!minimapOpen.value) {
    return
  }
  void nextTick(() => {
    if (!minimapOpen.value || !cy || cy !== core) {
      return
    }
    setupNavigator(core)
  })
}

function releaseGraphPaintAfterLayout(core: Core): void {
  if (!graphContentHiddenUntilLayout.value) {
    scheduleMinimapSetup(core)
    return
  }
  void nextTick(() => {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        graphContentHiddenUntilLayout.value = false
        scheduleMinimapSetup(core)
      })
    })
  })
}

function finishLayoutPass(core: Core): void {
  if (!cy) return
  const snap = pendingViewportPreserve
  pendingViewportPreserve = null

  recomputeDegreeHistogram(cy)
  applyDegreeVisibility(cy)
  applyViewportPreserveOrFit(cy, snap)

  lastZoomLevel = cy.zoom()
  updateZoomPercentDisplay(cy)
  attachZoomRecenter(core)
  tryApplyPendingFocus(core)
  applySearchHighlights(core)
  releaseGraphPaintAfterLayout(core)
}

function destroyCy(): void {
  // Do not clear pendingViewportPreserve here — it must survive destroy+redraw until the new cy's layoutstop
  // (ego exit to full graph, Re-layout). egoPriorFullGraphViewportPreserve must survive enter-ego redraw too.
  clearSelectedNodeZoomAnchor()
  suspendSelectedNodeZoomAnchorCorrection = 0
  zoomPanListenerAttached = false
  teardownBoxZoomListeners()
  destroyNavigator()
  if (zoomCenterTimer != null) {
    clearTimeout(zoomCenterTimer)
    zoomCenterTimer = null
  }
  if (resizeObs) {
    resizeObs.disconnect()
    resizeObs = null
  }
  if (activeElesLayout) {
    try {
      activeElesLayout.stop()
    } catch {
      /* ignore */
    }
    activeElesLayout = null
  }
  graphLayoutGeneration += 1
  if (cy) {
    cy.destroy()
    cy = null
  }
}

function tryApplyPendingFocus(core: Core): void {
  const rawId = nav.pendingFocusNodeId
  if (!rawId) return
  const cyId = resolveCyNodeId(core, rawId)
  if (!cyId) return
  const n = core.$id(cyId)
  core.nodes().unselect()
  n.select()
  selectedNodeId.value = cyId
  const viewArt = gf.viewWithEgo(nav.graphEgoFocusCyId)
  const rawNode = findRawNodeInArtifact(viewArt, cyId)
  void openGraphEpisodeOrNodeRail(cyId, rawNode)
  suspendSelectedNodeZoomAnchorCorrection += 1
  try {
    const targetZoom = Math.max(core.zoom(), 1.6)
    core.animate({
      center: { eles: n },
      zoom: targetZoom,
      duration: 320,
      complete: () => {
        suspendSelectedNodeZoomAnchorCorrection -= 1
        refreshSelectedNodeZoomAnchor(core)
        lastZoomLevel = core.zoom()
        updateZoomPercentDisplay(core)
      },
    })
  } catch {
    suspendSelectedNodeZoomAnchorCorrection -= 1
    try {
      core.center(n)
    } catch {
      /* ignore */
    }
    refreshSelectedNodeZoomAnchor(core)
    lastZoomLevel = core.zoom()
    updateZoomPercentDisplay(core)
  }
  nav.clearPendingFocus()
}

function fitAnimated(): void {
  const c = cy
  if (!c) return
  const els = c.elements(':visible')
  if (els.length === 0) return
  suspendSelectedNodeZoomAnchorCorrection += 1
  try {
    c.animate({
      fit: { eles: els, padding: 24 },
      duration: 280,
      complete: () => {
        suspendSelectedNodeZoomAnchorCorrection -= 1
        refreshSelectedNodeZoomAnchor(c)
        lastZoomLevel = c.zoom()
        updateZoomPercentDisplay(c)
      },
    })
  } catch {
    suspendSelectedNodeZoomAnchorCorrection -= 1
    try {
      c.fit(els, 24)
    } catch {
      /* ignore */
    }
    refreshSelectedNodeZoomAnchor(c)
    lastZoomLevel = c.zoom()
    updateZoomPercentDisplay(c)
  }
}

function zoomIn(): void {
  const c = cy
  if (!c) return
  try {
    const z = Math.min(c.zoom() * 1.2, c.maxZoom())
    c.zoom(z)
  } catch {
    /* ignore */
  }
}

function zoomOut(): void {
  const c = cy
  if (!c) return
  try {
    const z = Math.max(c.zoom() / 1.2, c.minZoom())
    c.zoom(z)
  } catch {
    /* ignore */
  }
}

function zoomReset100(): void {
  const c = cy
  if (!c) return
  try {
    c.zoom(1)
  } catch {
    /* ignore */
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

function clearInteractionState(): void {
  nav.clearPendingFocus()
  nav.clearLibraryEpisodeHighlights()
  clearSelectedNodeZoomAnchor()
  const hadEgo = focusNodeId.value !== null
  if (hadEgo) {
    pendingViewportPreserve = egoPriorFullGraphViewportPreserve
    egoPriorFullGraphViewportPreserve = null
  }
  focusNodeId.value = null
  selectedNodeId.value = null
  episodeRail.showTools()
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
  const els = c.elements(':visible')
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

/** Model-space bbox for the shift-drag rectangle (matches Cytoscape container client coords). */
function modelBoundingBoxFromClientRect(
  core: Core,
  cx1: number,
  cy1: number,
  cx2: number,
  cy2: number,
): { x1: number; y1: number; x2: number; y2: number } | null {
  const r = (core as unknown as {
    renderer: () => {
      projectIntoViewport: (clientX: number, clientY: number) => [number, number]
      invalidateContainerClientCoordsCache?: () => void
    }
  }).renderer()
  try {
    r.invalidateContainerClientCoordsCache?.()
  } catch {
    /* ignore */
  }
  const minX = Math.min(cx1, cx2)
  const maxX = Math.max(cx1, cx2)
  const minY = Math.min(cy1, cy2)
  const maxY = Math.max(cy1, cy2)
  const corners: Array<[number, number]> = [
    [minX, minY],
    [maxX, minY],
    [maxX, maxY],
    [minX, maxY],
  ]
  let x1 = Infinity
  let y1 = Infinity
  let x2 = -Infinity
  let y2 = -Infinity
  for (const [cx, cy] of corners) {
    let mx: number
    let my: number
    try {
      ;[mx, my] = r.projectIntoViewport(cx, cy)
    } catch {
      return null
    }
    if (!Number.isFinite(mx) || !Number.isFinite(my)) {
      return null
    }
    x1 = Math.min(x1, mx)
    y1 = Math.min(y1, my)
    x2 = Math.max(x2, mx)
    y2 = Math.max(y2, my)
  }
  if (!(x2 > x1 && y2 > y1)) {
    return null
  }
  return { x1, y1, x2, y2 }
}

function onWindowMouseMove(ev: MouseEvent): void {
  if (!boxDragging || !canvasHost.value) return
  const br = canvasHost.value.getBoundingClientRect()
  const x1 = boxStartClient.x - br.left
  const y1 = boxStartClient.y - br.top
  const x2 = ev.clientX - br.left
  const y2 = ev.clientY - br.top
  boxZoomRect.left = Math.min(x1, x2)
  boxZoomRect.top = Math.min(y1, y2)
  boxZoomRect.width = Math.abs(x2 - x1)
  boxZoomRect.height = Math.abs(y2 - y1)
  boxZoomRect.show = boxZoomRect.width > 4 && boxZoomRect.height > 4
}

function onWindowMouseUp(ev: MouseEvent): void {
  if (!boxDragging) return
  boxDragging = false
  const c = cy
  const host = canvasHost.value
  boxZoomRect.show = false
  if (c && host) {
    const br = host.getBoundingClientRect()
    const rx1 = boxStartClient.x - br.left
    const ry1 = boxStartClient.y - br.top
    const rx2 = ev.clientX - br.left
    const ry2 = ev.clientY - br.top
    if (Math.abs(rx2 - rx1) > 6 && Math.abs(ry2 - ry1) > 6) {
      const modelBb = modelBoundingBoxFromClientRect(
        c,
        boxStartClient.x,
        boxStartClient.y,
        ev.clientX,
        ev.clientY,
      )
      if (modelBb) {
        suspendSelectedNodeZoomAnchorCorrection += 1
        try {
          // fit() uses getFitViewport(), which returns nothing while panning is disabled — we
          // turn panning off during shift-drag, so re-enable before fit.
          try {
            c.panningEnabled(true)
            c.zoomingEnabled(true)
          } catch {
            /* ignore */
          }
          // Cytoscape accepts a plain { x1,y1,x2,y2 } bbox in model space (undocumented in types).
          ;(c as unknown as { fit: (target: unknown, padding?: number) => void }).fit(modelBb, 32)
        } catch {
          /* ignore */
        } finally {
          suspendSelectedNodeZoomAnchorCorrection -= 1
          refreshSelectedNodeZoomAnchor(c)
          lastZoomLevel = c.zoom()
          updateZoomPercentDisplay(c)
        }
      }
    }
  }
  try {
    c?.panningEnabled(true)
    c?.zoomingEnabled(true)
  } catch {
    /* ignore */
  }
}

function teardownBoxZoomListeners(): void {
  if (!boxListenersAttached) return
  window.removeEventListener('mousemove', onWindowMouseMove)
  window.removeEventListener('mouseup', onWindowMouseUp)
  boxListenersAttached = false
  boxDragging = false
  boxZoomRect.show = false
}

function setupBoxZoomListeners(): void {
  if (boxListenersAttached) return
  window.addEventListener('mousemove', onWindowMouseMove)
  window.addEventListener('mouseup', onWindowMouseUp)
  boxListenersAttached = true
}

function runRelayout(): void {
  const c = cy
  if (!c) return
  const eles = c.elements(':visible')
  if (eles.length === 0) return

  graphLayoutGeneration += 1
  const gen = graphLayoutGeneration
  if (activeElesLayout) {
    try {
      activeElesLayout.stop()
    } catch {
      /* ignore */
    }
    activeElesLayout = null
  }

  pendingViewportPreserve = captureSelectedViewportAnchor(c)
  const name = preferredLayout.value
  const opts = layoutOptionsFor(name)
  let lo: { stop: () => void; one: (ev: string, fn: () => void) => void; run: () => void }
  try {
    lo = eles.layout({
      ...opts,
      name,
    } as never) as typeof lo
  } catch {
    return
  }
  activeElesLayout = lo
  lo.one('layoutstop', () => {
    if (gen !== graphLayoutGeneration) {
      return
    }
    if (activeElesLayout === lo) {
      activeElesLayout = null
    }
    if (!cy) {
      return
    }
    finishLayoutPass(cy)
  })
  try {
    lo.run()
  } catch {
    activeElesLayout = null
  }
}

function redraw(): void {
  graphContentHiddenUntilLayout.value = true
  destroyCy()
  const el = container.value
  if (!el) {
    graphContentHiddenUntilLayout.value = false
    return
  }

  const art = gf.viewWithEgo(focusNodeId.value)
  if (!art) {
    el.innerHTML =
      '<p class="p-4 text-sm text-muted">Load artifacts and use “Load selected” to render the graph.</p>'
    degreeHistogramCounts.value = {}
    graphContentHiddenUntilLayout.value = false
    return
  }

  const elements = toCytoElements(art)
  const nodeCount = elements.filter((x) => !('source' in x.data)).length
  if (nodeCount === 0) {
    el.innerHTML =
      '<p class="p-4 text-sm text-muted">No nodes in this view (adjust filters).</p>'
    degreeHistogramCounts.value = {}
    graphContentHiddenUntilLayout.value = false
    return
  }

  el.innerHTML = ''
  const layoutName = preferredLayout.value
  const layoutOpts = layoutOptionsFor(layoutName)
  const core = cytoscape({
    container: el,
    elements,
    style: buildCyStyle(),
    wheelSensitivity: 0.35,
  })
  cy = core

  graphLayoutGeneration += 1
  const layoutGen = graphLayoutGeneration
  let initialLo: { stop: () => void; one: (ev: string, fn: () => void) => void; run: () => void }
  try {
    initialLo = core.elements().layout({
      ...layoutOpts,
      name: layoutName,
    } as never) as typeof initialLo
  } catch {
    cy = null
    core.destroy()
    graphContentHiddenUntilLayout.value = false
    return
  }
  activeElesLayout = initialLo
  initialLo.one('layoutstop', () => {
    if (layoutGen !== graphLayoutGeneration) {
      return
    }
    if (activeElesLayout === initialLo) {
      activeElesLayout = null
    }
    if (!cy || cy !== core) {
      return
    }
    finishLayoutPass(core)
  })
  try {
    initialLo.run()
  } catch {
    activeElesLayout = null
    cy = null
    core.destroy()
    graphContentHiddenUntilLayout.value = false
    return
  }

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
      episodeRail.showTools()
      clearSelectedNodeZoomAnchor()
    } else {
      core.nodes().unselect()
      n.select()
    }
  }

  setupBoxZoomListeners()

  core.on('tap', (evt) => {
    const t = evt.target
    if (t === core) {
      core.nodes().unselect()
      selectedNodeId.value = null
      episodeRail.showTools()
      clearSelectedNodeZoomAnchor()
      return
    }
    if (typeof t.isNode === 'function' && t.isNode()) {
      core.nodes().unselect()
      t.select()
      selectedNodeId.value = t.id()
      refreshSelectedNodeZoomAnchor(core)
      return
    }
    selectedNodeId.value = null
    episodeRail.showTools()
    clearSelectedNodeZoomAnchor()
  })

  core.on('dbltap', (evt) => {
    const raw = evt.originalEvent as MouseEvent | TouchEvent | undefined
    const shift =
      raw && 'shiftKey' in raw ? Boolean((raw as MouseEvent).shiftKey) : false
    const t = evt.target
    if (typeof t.isNode === 'function' && t.isNode()) {
      const id = t.id()
      if (shift) {
        const c = cy
        const cur = focusNodeId.value
        if (cur === id) {
          pendingViewportPreserve = egoPriorFullGraphViewportPreserve
          egoPriorFullGraphViewportPreserve = null
          focusNodeId.value = null
        } else {
          if (cur === null && c) {
            const snap = captureViewportAnchorForCyId(c, id)
            if (snap) {
              egoPriorFullGraphViewportPreserve = snap
            }
          }
          focusNodeId.value = id
        }
        redraw()
        return
      }
      core.nodes().unselect()
      t.select()
      selectedNodeId.value = id
      const viewArt = gf.viewWithEgo(nav.graphEgoFocusCyId)
      const rawNode = findRawNodeInArtifact(viewArt, id)
      void openGraphEpisodeOrNodeRail(id, rawNode)
      refreshSelectedNodeZoomAnchor(core)
      return
    }
    if (focusNodeId.value !== null) {
      pendingViewportPreserve = egoPriorFullGraphViewportPreserve
      egoPriorFullGraphViewportPreserve = null
      focusNodeId.value = null
      redraw()
    }
  })

  core.on('mousedown', (evt) => {
    if (evt.target !== core) return
    const o = evt.originalEvent as MouseEvent | undefined
    if (!o?.shiftKey) return
    o.preventDefault()
    boxDragging = true
    boxStartClient = { x: o.clientX, y: o.clientY }
    try {
      core.panningEnabled(false)
    } catch {
      /* ignore */
    }
  })
}

const typeHistogramCounts = computed(() => {
  const art = gf.viewWithEgo(focusNodeId.value)
  const nodes = art?.data.nodes
  if (!nodes) return {} as Record<string, number>
  return visualNodeTypeCounts(nodes as RawGraphNode[])
})

const typeFilterKeys = computed(() => {
  const st = gf.state
  if (!st) return [] as string[]
  return Object.keys(st.allowedTypes).sort()
})

const edgeTypeKeys = computed(() => {
  const aet = gf.state?.allowedEdgeTypes
  if (!aet) return [] as string[]
  return Object.keys(aet).sort()
})

const graphKind = computed(() => gf.fullArtifact?.kind)
const showLayerToggles = computed(() => graphKind.value === 'both')
const showGroundedFilter = computed(
  () => graphKind.value === 'gi' || graphKind.value === 'both',
)
const showSourcesChromeRow = computed(
  () =>
    showLayerToggles.value ||
    showGroundedFilter.value ||
    gf.filtersAreActive,
)

function onLayoutSelectChange(): void {
  runRelayout()
}

watch(
  () => [searchStore.results, nav.libraryHighlightSourceIds] as const,
  () => {
    const c = cy
    if (c) applySearchHighlights(c)
  },
  { flush: 'post', deep: true },
)

watch(
  focusNodeId,
  (v) => {
    nav.setGraphEgoFocusCyId(v)
  },
  { immediate: true },
)

watch(
  () => gf.filteredArtifact,
  () => {
    ge.resetForNewArtifact()
    focusNodeId.value = null
    selectedNodeId.value = null
    episodeRail.showTools()
    nav.clearLibraryEpisodeHighlights()
    pendingViewportPreserve = null
    egoPriorFullGraphViewportPreserve = null
    redraw()
  },
  { flush: 'post' },
)

watch(
  activeDegreeBucket,
  () => {
    const c = cy
    if (!c) return
    const snap = captureSelectedViewportAnchor(c)
    applyDegreeVisibility(c)
    applyViewportPreserveOrFit(c, snap)
    lastZoomLevel = c.zoom()
    updateZoomPercentDisplay(c)
  },
)

watch(
  minimapOpen,
  (open) => {
    const c = cy
    if (!c) return
    if (open) {
      void nextTick(() => setupNavigator(c))
    } else {
      destroyNavigator()
    }
  },
)

watch(
  () => themeStore.choice,
  () => {
    pendingViewportPreserve = null
    egoPriorFullGraphViewportPreserve = null
    nextTick(() => redraw())
  },
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

onActivated(() => {
  const c = cy
  if (!c) {
    return
  }
  void nextTick(() => {
    try {
      c.resize()
    } catch {
      /* ignore */
    }
    requestAnimationFrame(() => {
      try {
        c.resize()
      } catch {
        /* ignore */
      }
    })
  })
})

onUnmounted(() => {
  destroyCy()
  pendingViewportPreserve = null
  egoPriorFullGraphViewportPreserve = null
  graphContentHiddenUntilLayout.value = false
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
      <span
        v-if="searchHighlightCount > 0"
        class="rounded-full bg-yellow-500/20 px-2 py-0.5 text-[10px] font-medium text-yellow-600"
      >
        {{ searchHighlightCount }}
        {{ searchHighlightCount === 1 ? 'highlight' : 'highlights' }}
      </span>
      <span class="min-w-0 flex-1 text-xs text-muted">{{ hint }}</span>
    </div>

    <div
      v-if="gf.state"
      class="flex flex-col gap-2 border-b border-border px-2 py-2 text-surface-foreground"
    >
      <div
        v-if="showSourcesChromeRow"
        class="flex flex-wrap items-center gap-x-4 gap-y-1"
      >
        <span class="text-[10px] font-semibold uppercase tracking-wide text-muted">Sources</span>
        <template v-if="showLayerToggles">
          <label class="flex cursor-pointer items-center gap-1 text-[10px]">
            <input
              type="checkbox"
              class="rounded border-border"
              :checked="gf.state!.showGiLayer"
              @change="gf.setShowGiLayer(!gf.state!.showGiLayer)"
            >
            <span>GI</span>
          </label>
          <label class="flex cursor-pointer items-center gap-1 text-[10px]">
            <input
              type="checkbox"
              class="rounded border-border"
              :checked="gf.state!.showKgLayer"
              @change="gf.setShowKgLayer(!gf.state!.showKgLayer)"
            >
            <span>KG</span>
          </label>
        </template>
        <label
          v-if="showGroundedFilter"
          class="flex cursor-pointer items-center gap-1 text-[10px]"
        >
          <input
            type="checkbox"
            class="rounded border-border"
            :checked="gf.state!.hideUngroundedInsights"
            @change="gf.setHideUngrounded(!gf.state!.hideUngroundedInsights)"
          >
          <span>Hide ungrounded</span>
        </label>
        <span
          v-if="gf.filtersAreActive"
          class="text-[10px] font-medium text-warning"
        >
          filters active
        </span>
      </div>
      <div class="flex flex-wrap items-center gap-2">
        <label class="flex cursor-pointer items-center gap-1 text-xs">
          <input
            v-model="minimapOpen"
            type="checkbox"
            class="rounded border-border"
          >
          <span class="text-[10px] text-muted">Minimap</span>
        </label>
      </div>
      <div v-if="edgeTypeKeys.length" class="flex flex-wrap items-start gap-x-3 gap-y-1">
        <span class="text-[10px] font-semibold uppercase tracking-wide text-muted">Edges</span>
        <button
          type="button"
          class="text-[10px] text-primary underline"
          @click="gf.selectAllEdgeTypes()"
        >
          all
        </button>
        <label
          v-for="et in edgeTypeKeys"
          :key="et"
          class="flex cursor-pointer items-center gap-1 text-[10px]"
        >
          <input
            type="checkbox"
            class="rounded border-border"
            :checked="gf.state!.allowedEdgeTypes[et]"
            @change="gf.toggleAllowedEdgeType(et)"
          >
          <span class="max-w-[10rem] truncate" :title="et">{{ et }}</span>
        </label>
      </div>
      <div class="flex flex-wrap items-start gap-x-3 gap-y-1">
        <span class="text-[10px] font-semibold uppercase tracking-wide text-muted">Types</span>
        <button
          type="button"
          class="text-[10px] text-primary underline"
          @click="gf.selectAllTypes()"
        >
          all
        </button>
        <button
          type="button"
          class="text-[10px] text-primary underline"
          @click="gf.deselectAllTypes()"
        >
          none
        </button>
        <label
          v-for="t in typeFilterKeys"
          :key="t"
          class="flex cursor-pointer items-center gap-1 text-[10px]"
        >
          <input
            type="checkbox"
            class="rounded border-border"
            :checked="gf.state!.allowedTypes[t]"
            @change="gf.toggleAllowedType(t)"
          >
          <span
            class="h-2.5 w-2.5 shrink-0 rounded-sm ring-1 ring-black/15 dark:ring-white/20"
            :style="{ backgroundColor: graphNodeFill(String(t)) }"
            :title="`${String(t)} (node fill)`"
            aria-hidden="true"
          />
          <span>{{ graphNodeLegendLabel(t) }}</span>
          <span class="font-medium text-muted">({{ typeHistogramCounts[t] ?? 0 }})</span>
        </label>
      </div>
    </div>

    <div class="flex min-h-[240px] min-w-0 flex-1 sm:min-h-[380px]">
      <div
        ref="canvasHost"
        class="relative isolate min-h-[240px] min-w-0 flex-1 overflow-hidden transition-opacity duration-150 sm:min-h-[380px]"
        :class="
          graphContentHiddenUntilLayout
            ? 'pointer-events-none opacity-0'
            : 'opacity-100'
        "
        :aria-busy="graphContentHiddenUntilLayout ? 'true' : 'false'"
      >
        <div
          ref="container"
          class="graph-canvas absolute inset-0 min-h-[240px] sm:min-h-[380px]"
        />
        <div
          v-show="boxZoomRect.show"
          class="pointer-events-none absolute z-[5] box-border border-2 border-dashed border-primary bg-primary/10"
          :style="{
            left: `${boxZoomRect.left}px`,
            top: `${boxZoomRect.top}px`,
            width: `${boxZoomRect.width}px`,
            height: `${boxZoomRect.height}px`,
          }"
        />
        <div
          v-if="gf.state"
          class="graph-layout-controls pointer-events-auto absolute right-2 top-2 z-[22] flex w-[6.75rem] max-w-[min(6.75rem,calc(100%-1rem))] flex-col gap-0.5 rounded border border-border bg-surface/95 p-1 shadow-md backdrop-blur-sm"
          role="region"
          aria-label="Graph layout, re-layout, and degree filter"
        >
          <button
            type="button"
            class="w-full rounded border border-border px-1 py-px text-[10px] font-medium leading-tight hover:bg-overlay"
            @click="runRelayout"
          >
            Re-layout
          </button>
          <div class="flex flex-col gap-0">
            <span class="text-[9px] font-semibold uppercase leading-none tracking-wide text-muted">Layout</span>
            <select
              v-model="preferredLayout"
              class="mt-0.5 w-full rounded border border-border bg-elevated py-0.5 pl-0.5 pr-0 text-[10px] leading-tight text-surface-foreground"
              aria-label="Graph layout algorithm"
              @change="onLayoutSelectChange"
            >
              <option value="cose">
                COSE
              </option>
              <option value="breadthfirst">
                Breadthfirst
              </option>
              <option value="circle">
                Circle
              </option>
              <option value="grid">
                Grid
              </option>
            </select>
          </div>
          <div class="flex flex-col gap-0.5 border-t border-border/80 pt-0.5">
            <span class="text-[9px] font-semibold uppercase leading-none tracking-wide text-muted">Degree</span>
            <div class="grid grid-cols-2 gap-0.5">
              <button
                v-for="bid in DEGREE_BUCKET_ORDER"
                :key="bid"
                type="button"
                class="rounded border px-0.5 py-px text-[10px] leading-tight hover:bg-overlay"
                :class="
                  activeDegreeBucket === bid
                    ? 'border-primary bg-primary/15 font-medium'
                    : 'border-border'
                "
                :aria-pressed="activeDegreeBucket === bid"
                @click="ge.toggleDegreeBucket(bid)"
              >
                {{ bid }}
                <span class="text-muted">({{ degreeHistogramCounts[bid] ?? 0 }})</span>
              </button>
            </div>
          </div>
          <button
            v-if="activeDegreeBucket"
            type="button"
            class="w-full rounded border border-border px-0.5 py-px text-[10px] leading-tight hover:bg-overlay"
            aria-label="Clear degree filter"
            @click="ge.clearDegreeBucket()"
          >
            Clear
          </button>
        </div>
        <div
          id="gi-kg-graph-minimap"
          v-show="minimapOpen"
          ref="minimapHost"
          class="pointer-events-auto absolute bottom-2 left-2 z-10 h-[7.5rem] w-[10.5rem] max-h-[min(13.5rem,35%)] max-w-[min(10.5rem,calc(100%-1rem))] overflow-hidden rounded border border-border bg-surface shadow-md"
          aria-label="Graph minimap"
        />
        <div
          class="graph-zoom-controls pointer-events-auto absolute bottom-2 right-2 z-[12] flex items-center gap-1 rounded border border-border bg-surface/95 px-1 py-0.5 shadow-md backdrop-blur-sm"
          role="toolbar"
          aria-label="Graph fit, zoom, and export"
        >
          <button
            type="button"
            class="rounded bg-primary px-2 py-1 text-xs font-medium text-primary-foreground hover:opacity-90"
            @click="fitAnimated"
          >
            Fit
          </button>
          <span class="text-[10px] text-muted opacity-60" aria-hidden="true">|</span>
          <button
            type="button"
            class="rounded border border-border px-2 py-1 text-xs hover:bg-overlay"
            aria-label="Zoom out"
            @click="zoomOut"
          >
            −
          </button>
          <span
            class="min-w-[2.5rem] text-center text-[10px] font-medium text-muted"
            title="Zoom level"
          >
            {{ zoomPercent }}%
          </span>
          <button
            type="button"
            class="rounded border border-border px-2 py-1 text-xs hover:bg-overlay"
            aria-label="Zoom in"
            @click="zoomIn"
          >
            +
          </button>
          <button
            type="button"
            class="rounded border border-border px-2 py-1 text-xs hover:bg-overlay"
            title="Reset zoom to 100% (does not change pan)"
            @click="zoomReset100"
          >
            100%
          </button>
          <span class="text-[10px] text-muted opacity-60" aria-hidden="true">|</span>
          <button
            type="button"
            class="rounded border border-border px-2 py-1 text-xs hover:bg-overlay"
            title="Full graph as PNG (2× scale)"
            @click="exportGraphPng"
          >
            Export PNG
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/*
 * With `container: '#…'`, the host never gets `.cytoscape-navigator`, so the
 * package CSS for `> img` / `> canvas` does not apply; keep the thumbnail
 * clipped to the inset panel.
 */
#gi-kg-graph-minimap :deep(img) {
  max-width: 100%;
  max-height: 100%;
}
</style>
