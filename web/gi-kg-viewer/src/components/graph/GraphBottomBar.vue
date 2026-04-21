<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import {
  GRAPH_LAYOUT_CYCLE_ORDER,
  type GraphLayoutName,
  useGraphExplorerStore,
} from '../../stores/graphExplorer'
import GraphStatusLine from './GraphStatusLine.vue'
const BOTTOM_BAR_COLLAPSED_KEY = 'ps_graph_bottom_bar_collapsed'

const props = withDefaults(
  defineProps<{
    zoomPercent: number
    searchHighlightCount: number
    preferredLayout: GraphLayoutName
    /** Lens presets + Reset (only when a full corpus graph is loaded). */
    showLensControls?: boolean
    /**
     * When false, **Gestures** lives on the stats strip (full merged graph). Keep
     * here for partial / non-corpus slices that omit the stats row.
     */
    showGesturesInBottomBar?: boolean
  }>(),
  { showLensControls: false, showGesturesInBottomBar: true },
)

const emit = defineEmits<{
  fit: []
  'zoom-in': []
  'zoom-out': []
  'zoom-reset': []
  'export-png': []
  'reopen-gestures': []
  relayout: []
  'cycle-layout': []
  'request-corpus-graph-sync': []
  'request-graph-full-reset': []
}>()

function onLensRequestReload(): void {
  emit('request-corpus-graph-sync')
}

const ge = useGraphExplorerStore()

const collapsed = ref(false)

function readCollapsedFromStorage(): boolean {
  try {
    return typeof localStorage !== 'undefined' && localStorage.getItem(BOTTOM_BAR_COLLAPSED_KEY) === '1'
  } catch {
    return false
  }
}

function persistCollapsed(v: boolean): void {
  try {
    if (typeof localStorage === 'undefined') return
    if (v) {
      localStorage.setItem(BOTTOM_BAR_COLLAPSED_KEY, '1')
    } else {
      localStorage.removeItem(BOTTOM_BAR_COLLAPSED_KEY)
    }
  } catch {
    /* ignore */
  }
}

watch(collapsed, (v) => {
  persistCollapsed(v)
})

watch(
  () => props.searchHighlightCount,
  (n) => {
    if (n > 0 && collapsed.value) {
      collapsed.value = false
    }
  },
)

function targetIsEditable(target: EventTarget | null): boolean {
  if (!target || !(target instanceof HTMLElement)) {
    return false
  }
  const tag = target.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
    return true
  }
  if (target.isContentEditable) {
    return true
  }
  return target.closest('[contenteditable="true"]') != null
}

function onGlobalKeydown(ev: KeyboardEvent): void {
  if (!ev.altKey || (ev.key !== 'b' && ev.key !== 'B')) return
  if (targetIsEditable(ev.target)) return
  ev.preventDefault()
  collapsed.value = !collapsed.value
}


onUnmounted(() => {
  window.removeEventListener('keydown', onGlobalKeydown, true)
})

const layoutCycleTitle = computed(() => {
  const long: Record<GraphLayoutName, string> = {
    cose: 'COSE force-directed',
    breadthfirst: 'Breadthfirst',
    circle: 'Circle',
    grid: 'Grid',
  }
  const order = [...GRAPH_LAYOUT_CYCLE_ORDER]
  const cur = props.preferredLayout
  const i = Math.max(0, order.indexOf(cur))
  const next = order[(i + 1) % order.length]!
  return `Current: ${long[cur]}. Click to switch to ${long[next]} and re-layout.`
})

function toggleCollapsed(): void {
  collapsed.value = !collapsed.value
}

function toggleMinimap(): void {
  ge.minimapOpen = !ge.minimapOpen
}

onMounted(() => {
  collapsed.value = readCollapsedFromStorage()
  if (props.searchHighlightCount > 0 && collapsed.value) {
    collapsed.value = false
  }
  window.addEventListener('keydown', onGlobalKeydown, true)
})
</script>

<template>
  <div data-testid="graph-bottom-bar" :aria-expanded="collapsed ? 'false' : 'true'">
    <div
      v-if="!collapsed"
      class="flex shrink-0 items-center gap-2 border-t border-border/80 bg-canvas py-0.5 pl-1 pr-1 text-surface-foreground"
    >
      <div
        class="flex shrink-0 flex-wrap items-center gap-0.5 py-0.5"
        data-testid="graph-bottom-bar-left"
      >
        <button
          type="button"
          class="rounded border border-border px-1 py-px text-[10px] leading-none hover:bg-overlay"
          :class="ge.minimapOpen ? 'border-primary/50 bg-primary/10 text-primary' : 'text-muted'"
          data-testid="graph-minimap-toggle"
          aria-label="Toggle minimap"
          title="Toggle minimap"
          @click="toggleMinimap"
        >
          ⊞
        </button>
        <button
          type="button"
          class="rounded px-1 py-px text-[10px] leading-tight text-surface-foreground hover:bg-overlay"
          data-testid="graph-relayout"
          @click="emit('relayout')"
        >
          Re-layout
        </button>
        <button
          type="button"
          class="flex max-w-[5.5rem] items-center gap-0.5 rounded border border-border px-1 py-px text-left text-[10px] leading-tight hover:bg-overlay"
          data-testid="graph-layout-cycle"
          :title="layoutCycleTitle"
          @click="emit('cycle-layout')"
        >
          <span aria-hidden="true">⟲</span>
          <span class="truncate">{{ preferredLayout }}</span>
        </button>
      </div>

      <div
        v-if="props.showLensControls"
        class="flex min-w-0 flex-1 justify-center border-r border-border/60 pr-1"
        data-testid="graph-bottom-bar-centre"
      >
        <GraphStatusLine
          variant="controls"
          embedded
          @request-reload="onLensRequestReload"
          @request-graph-full-reset="emit('request-graph-full-reset')"
        />
      </div>

      <div
        class="flex shrink-0 flex-nowrap items-center gap-0.5 py-0.5"
        :class="props.showLensControls ? '' : 'ml-auto'"
        data-testid="graph-bottom-bar-right"
        role="toolbar"
        :aria-label="
          props.showGesturesInBottomBar
            ? 'Graph fit, zoom, gestures, and export'
            : 'Graph fit, zoom, and export'
        "
      >
        <button
          type="button"
          class="rounded bg-primary px-1.5 py-px text-[10px] font-medium leading-tight text-primary-foreground hover:opacity-90"
          data-testid="graph-zoom-fit"
          aria-label="Fit graph"
          @click="emit('fit')"
        >
          Fit
        </button>
        <span class="text-[9px] text-muted opacity-50" aria-hidden="true">|</span>
        <button
          type="button"
          class="rounded border border-border px-1 py-px text-[10px] leading-none hover:bg-overlay"
          data-testid="graph-zoom-out"
          aria-label="Zoom out"
          @click="emit('zoom-out')"
        >
          −
        </button>
        <span
          class="min-w-[2rem] shrink-0 text-center text-[10px] font-medium leading-none text-muted"
          title="Zoom level"
        >
          {{ zoomPercent }}%
        </span>
        <button
          type="button"
          class="rounded border border-border px-1 py-px text-[10px] leading-none hover:bg-overlay"
          data-testid="graph-zoom-in"
          aria-label="Zoom in"
          @click="emit('zoom-in')"
        >
          +
        </button>
        <button
          type="button"
          class="rounded border border-border px-1 py-px text-[10px] leading-tight hover:bg-overlay"
          data-testid="graph-zoom-reset"
          title="Reset zoom to 100% and center the visible graph"
          @click="emit('zoom-reset')"
        >
          100%
        </button>
        <template v-if="props.showGesturesInBottomBar">
          <span class="text-[9px] text-muted opacity-50" aria-hidden="true">|</span>
          <button
            type="button"
            class="rounded border border-border px-1 py-px text-[10px] leading-tight hover:bg-overlay"
            data-testid="graph-gesture-overlay-reopen"
            aria-label="Show graph gestures help"
            @click="emit('reopen-gestures')"
          >
            Gestures
          </button>
        </template>
        <span class="text-[9px] text-muted opacity-50" aria-hidden="true">|</span>
        <button
          type="button"
          class="rounded border border-border px-1 py-px text-[10px] leading-tight hover:bg-overlay"
          title="Full graph as PNG (2× scale)"
          aria-label="Export PNG"
          data-testid="graph-export-png"
          @click="emit('export-png')"
        >
          PNG
        </button>
        <button
          type="button"
          class="rounded px-0.5 py-px text-[10px] leading-none text-muted hover:bg-overlay hover:text-surface-foreground"
          data-testid="graph-bottom-bar-toggle"
          aria-label="Collapse graph bar"
          @click="toggleCollapsed"
        >
          ⌄
        </button>
      </div>
    </div>
    <button
      v-else
      type="button"
      class="flex h-7 w-full shrink-0 cursor-pointer items-center justify-center border-t-2 border-border bg-elevated text-sm font-semibold leading-none text-surface-foreground hover:bg-overlay"
      title="Expand graph bar"
      data-testid="graph-bottom-bar-expand"
      aria-label="Expand graph bar"
      aria-expanded="false"
      @click="toggleCollapsed"
    >
      ⌃
    </button>
  </div>
</template>
