<script setup lang="ts">
import { computed, nextTick, onMounted, provide, ref, watch } from 'vue'
import type { SearchHit } from './api/searchApi'
import { useViewerKeyboard } from './composables/useViewerKeyboard'
import DashboardView from './components/dashboard/DashboardView.vue'
import DigestView from './components/digest/DigestView.vue'
import GraphTabPanel from './components/graph/GraphTabPanel.vue'
import LibraryView from './components/library/LibraryView.vue'
import LeftPanel from './components/shell/LeftPanel.vue'
import StatusBar from './components/shell/StatusBar.vue'
import SubjectRail from './components/shell/SubjectRail.vue'
import { useArtifactsStore } from './stores/artifacts'
import { useExploreStore } from './stores/explore'
import { useGraphExpansionStore } from './stores/graphExpansion'
import { useGraphExplorerStore } from './stores/graphExplorer'
import { useSearchStore } from './stores/search'
import { useShellStore } from './stores/shell'
import { useSubjectStore } from './stores/subject'
import { useThemeStore } from './stores/theme'
import {
  GRAPH_DEFAULT_EPISODE_CAP,
  selectRelPathsForGraphLoad,
} from './utils/graphEpisodeSelection'
import { sourceMetadataRelativePathFromSearchHit } from './utils/searchHitLibrary'
import { corpusGraphBaselineLoaderKey } from './corpusGraphBaseline'
import { StaleGeneration } from './utils/staleGeneration'

const LS_LEFT_PANEL_OPEN = 'ps_left_panel_open'

function readLeftPanelOpenPreference(): boolean {
  try {
    const v = localStorage.getItem(LS_LEFT_PANEL_OPEN)
    if (v === 'false') {
      return false
    }
  } catch {
    /* ignore */
  }
  return true
}

const shell = useShellStore()
const artifacts = useArtifactsStore()
const search = useSearchStore()
const explore = useExploreStore()
const theme = useThemeStore()
const subject = useSubjectStore()
const graphExplorer = useGraphExplorerStore()
const graphExpansion = useGraphExpansionStore()

const mainTab = ref<'digest' | 'library' | 'graph' | 'dashboard'>('digest')
const leftPanelRef = ref<{ focusQuery: () => void } | null>(null)
const graphCanvasRef = ref<{
  clearInteractionState: (opts?: { skipRedraw?: boolean }) => void
} | null>(null)
const isGraphTab = computed(() => mainTab.value === 'graph')

const leftOpen = ref(readLeftPanelOpenPreference())
const rightOpen = ref(true)

/** Collapsible rail seam tabs (minimal chrome). */
const railEdgeToggleTab = {
  left:
    'absolute left-full top-1/2 z-20 flex h-10 w-2.5 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-border/25 bg-canvas/90 text-muted backdrop-blur-sm motion-safe:transition-[color,background-color,box-shadow,border-color] hover:border-border/45 hover:bg-overlay hover:text-surface-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/35 focus-visible:ring-offset-1 focus-visible:ring-offset-canvas',
  right:
    'absolute right-full top-1/2 z-20 flex h-10 w-2.5 translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-border/25 bg-canvas/90 text-muted backdrop-blur-sm motion-safe:transition-[color,background-color,box-shadow,border-color] hover:border-border/45 hover:bg-overlay hover:text-surface-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/35 focus-visible:ring-offset-1 focus-visible:ring-offset-canvas',
} as const

watch(leftOpen, (open) => {
  try {
    localStorage.setItem(LS_LEFT_PANEL_OPEN, open ? 'true' : 'false')
  } catch {
    /* ignore */
  }
})

useViewerKeyboard({
  focusSearch: () => {
    leftOpen.value = true
    void nextTick(() => {
      leftPanelRef.value?.focusQuery()
    })
  },
  clearGraphFocus: () => {
    graphCanvasRef.value?.clearInteractionState()
  },
  isGraphTab,
})

/**
 * Switch to Graph without re-running corpus auto-sync when the merged graph is already loaded.
 * Tab switches alone must not replace the slice (episode “Open in graph” is highlight-only).
 */
function activateGraphTab(): void {
  mainTab.value = 'graph'
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus) {
    return
  }
  if (artifacts.manualGraphSelection) {
    return
  }
  if (artifacts.parsedList.length > 0) {
    return
  }
  void syncMergedGraphFromCorpusApi()
}

function onSwitchMainTab(tab: 'digest' | 'library' | 'graph' | 'dashboard'): void {
  if (tab === 'graph') {
    activateGraphTab()
    return
  }
  mainTab.value = tab
}

function onStatusBarLocalArtifactsLoaded(loaded: boolean): void {
  if (loaded) {
    activateGraphTab()
  }
}

function onCloseSubjectRail(): void {
  subject.clearSubject()
  graphCanvasRef.value?.clearInteractionState()
}

onMounted(() => {
  void shell.fetchHealth()
})

/** Run sibling merge when Graph is visible and load finished (covers tab switch during load). */
watch(
  () => [mainTab.value, artifacts.loading, artifacts.loadError] as const,
  async ([tab, loading, err]) => {
    if (tab !== 'graph' || loading || err) {
      return
    }
    await artifacts.maybeMergeClusterSiblingEpisodes(true)
  },
)

watch(
  () => shell.corpusPath,
  (p, old) => {
    subject.clearSubject()
    artifacts.setCorpusPath(p)
    const prev = old !== undefined ? String(old ?? '').trim() : null
    const next = String(p ?? '').trim()
    if (prev !== null && prev !== next) {
      graphExplorer.resetGraphLensForNewCorpus()
      artifacts.clearManualGraphSelection()
    }
  },
  { immediate: true },
)

const corpusGraphSyncGate = new StaleGeneration()

let corpusGraphSyncRunning = false
let corpusGraphSyncQueued = false

/**
 * When the API is healthy and a corpus path is set, list GI/KG via ``GET /api/artifacts``,
 * apply ``graphLens`` + episode cap, then load the merged graph. Skipped until the user opens
 * the Graph tab once (per docs/architecture/VIEWER_GRAPH_SPEC.md initial load). Offline / failed health: skip so file-picker loads stay intact.
 */
async function runCorpusGraphSyncBody(): Promise<void> {
  const gen = corpusGraphSyncGate.bump()
  artifacts.setCorpusPath(shell.corpusPath)
  const root = shell.corpusPath.trim()
  if (!root) {
    artifacts.clearSelection()
    return
  }
  if (!shell.healthStatus) {
    return
  }
  if (artifacts.manualGraphSelection) {
    return
  }
  if (mainTab.value === 'graph') {
    graphExplorer.markGraphTabOpenedOnce()
  }
  if (!graphExplorer.graphTabOpenedThisSession) {
    return
  }
  graphExplorer.seedFromCorpusLensIfNeeded()
  await artifacts.syncTopicClustersForCurrentCorpus()
  await shell.fetchArtifactList()
  if (corpusGraphSyncGate.isStale(gen)) {
    return
  }
  const rows = shell.artifactList.map((a) => ({
    relative_path: a.relative_path,
    kind: a.kind,
    publish_date: a.publish_date ?? '',
  }))
  const { selectedRelPaths, wasCapped } = selectRelPathsForGraphLoad(
    rows,
    graphExplorer.sinceYmd,
    GRAPH_DEFAULT_EPISODE_CAP,
    artifacts.topicClustersDoc,
  )
  graphExplorer.setLastAutoLoadCapped(wasCapped)
  if (selectedRelPaths.length === 0) {
    artifacts.clearSelection()
    await artifacts.syncTopicClustersForCurrentCorpus()
    return
  }
  artifacts.selectAllListed(selectedRelPaths)
  await artifacts.loadSelected()
}

async function syncMergedGraphFromCorpusApi(): Promise<void> {
  if (corpusGraphSyncRunning) {
    corpusGraphSyncQueued = true
    return
  }
  corpusGraphSyncRunning = true
  try {
    await runCorpusGraphSyncBody()
  } finally {
    corpusGraphSyncRunning = false
    if (corpusGraphSyncQueued) {
      corpusGraphSyncQueued = false
      void syncMergedGraphFromCorpusApi()
    }
  }
}

/**
 * Library/Digest **Open in graph** (before first Graph visit): same default merged load as opening
 * Graph, then callers merge episode GI/KG. Injected into episode + digest panels.
 */
async function ensureCorpusGraphBaselineForHandoff(): Promise<void> {
  graphExplorer.markGraphTabOpenedOnce()
  mainTab.value = 'graph'
  await syncMergedGraphFromCorpusApi()
}

provide(corpusGraphBaselineLoaderKey, ensureCorpusGraphBaselineForHandoff)

async function onCorpusGraphLensReload(): Promise<void> {
  graphExpansion.resetExpansionState()
  graphExpansion.invalidateCorpusBeyondHints()
  artifacts.clearManualGraphSelection()
  await syncMergedGraphFromCorpusApi()
}

/**
 * Graph status bar “Reset”: collapse cross-episode expansion bookkeeping, restore the same time slice as
 * first corpus auto-sync (15-episode cap), clear graph/subject focus, reload from API, then fit in
 * ``finishLayoutPass`` (no pending viewport preserve).
 */
async function onGraphCorpusFullReset(): Promise<void> {
  if (!shell.corpusPath.trim() || !shell.healthStatus) {
    return
  }
  graphCanvasRef.value?.clearInteractionState({ skipRedraw: true })
  graphExpansion.resetExpansionState()
  graphExpansion.invalidateCorpusBeyondHints()
  artifacts.clearManualGraphSelection()
  graphExplorer.resetSinceYmdToInitialCorpusSeed()
  await syncMergedGraphFromCorpusApi()
}

watch(
  () => [shell.corpusPath, shell.healthStatus] as const,
  () => {
    void syncMergedGraphFromCorpusApi()
  },
  { immediate: true },
)

function onLibraryFocusSearch(payload: {
  feed: string
  query: string
  since?: string
  feedDisplayTitle?: string
}): void {
  leftOpen.value = true
  search.applyLibrarySearchHandoff(payload.feed, payload.query, {
    since: payload.since,
    feedDisplayTitle: payload.feedDisplayTitle,
  })
  void nextTick(() => {
    leftPanelRef.value?.focusQuery()
  })
}

/** Graph Topic node detail: prefill semantic search (subject rail unchanged). */
function onGraphNodeTopicPrefillSearch(payload: { query: string }): void {
  const q = payload.query.trim()
  if (!q) return
  leftOpen.value = true
  search.applyLibrarySearchHandoff('', q)
  void nextTick(() => {
    leftPanelRef.value?.focusQuery()
  })
}

/** Graph Topic node detail: Explore + Topic contains filter. */
function onGraphNodeTopicOpenExploreFilter(payload: { topic: string }): void {
  const t = payload.topic.trim()
  if (!t) return
  leftOpen.value = true
  shell.setLeftPanelSurface('explore')
  explore.filters.topic = t
  explore.filters.speaker = ''
  explore.clearOutput()
}

/** Graph Person / Entity (person) detail: Explore + Speaker contains. */
function onGraphNodeSpeakerOpenExploreFilter(payload: { speaker: string }): void {
  const s = payload.speaker.trim()
  if (!s) return
  leftOpen.value = true
  shell.setLeftPanelSurface('explore')
  explore.filters.topic = ''
  explore.filters.speaker = s
  explore.clearOutput()
}

/** Graph Insight node detail: Explore + grounded/min-confidence filters. */
function onGraphNodeInsightOpenExploreFilters(payload: {
  groundedOnly: boolean
  minConfidence: number | null
}): void {
  leftOpen.value = true
  shell.setLeftPanelSurface('explore')
  explore.filters.topic = ''
  explore.filters.speaker = ''
  explore.filters.groundedOnly = payload.groundedOnly
  explore.filters.minConfidence =
    payload.minConfidence != null && Number.isFinite(payload.minConfidence)
      ? String(payload.minConfidence)
      : ''
  explore.clearOutput()
}

/** Digest row / topic hit: episode detail in the subject rail; stay on Digest. */
function onDigestOpenEpisodeInRail(payload: { metadata_relative_path: string }): void {
  subject.focusEpisode(payload.metadata_relative_path)
}

/** Search hit **L**: open episode in the subject rail (main tab unchanged). */
function onSearchOpenLibraryEpisode(payload: { metadata_relative_path: string }): void {
  subject.focusEpisode(payload.metadata_relative_path)
}

function onSearchOpenEpisodeSummary(hit: SearchHit): void {
  const rel = sourceMetadataRelativePathFromSearchHit(hit)
  if (rel) {
    subject.focusEpisode(rel)
  }
}

watch(
  () =>
    [subject.kind, subject.episodeMetadataPath, subject.graphNodeCyId] as const,
  () => {
    const ep = subject.episodeMetadataPath?.trim()
    const gn = subject.graphNodeCyId?.trim()
    if (subject.kind === 'episode' && ep) {
      rightOpen.value = true
    }
    if (subject.kind === 'graph-node' && gn) {
      rightOpen.value = true
    }
  },
)

</script>

<template>
  <div
    class="flex min-h-0 min-w-0 flex-col overflow-hidden bg-canvas text-canvas-foreground h-dvh max-h-dvh"
  >
    <header class="shrink-0 border-b border-border bg-surface px-4 py-2 shadow-sm">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <h1 class="text-lg font-semibold tracking-tight text-surface-foreground">
          Podcast Intelligence Platform
          <span class="ml-1 text-xs font-normal text-muted">v2</span>
        </h1>
        <div class="flex items-center gap-3">
          <nav
            class="flex gap-1 rounded border border-border bg-elevated p-0.5 text-xs font-medium"
            aria-label="Main views"
          >
            <button
              type="button"
              class="rounded px-3 py-1"
              :class="
                mainTab === 'digest'
                  ? 'bg-primary text-primary-foreground'
                  : 'text-elevated-foreground hover:bg-overlay'
              "
              @click="mainTab = 'digest'"
            >
              Digest
            </button>
            <button
              type="button"
              class="rounded px-3 py-1"
              :class="
                mainTab === 'library'
                  ? 'bg-primary text-primary-foreground'
                  : 'text-elevated-foreground hover:bg-overlay'
              "
              @click="mainTab = 'library'"
            >
              Library
            </button>
            <button
              type="button"
              class="rounded px-3 py-1"
              :class="
                mainTab === 'graph'
                  ? 'bg-primary text-primary-foreground'
                  : 'text-elevated-foreground hover:bg-overlay'
              "
              @click="activateGraphTab()"
            >
              Graph
            </button>
            <button
              type="button"
              class="rounded px-3 py-1"
              :class="
                mainTab === 'dashboard'
                  ? 'bg-primary text-primary-foreground'
                  : 'text-elevated-foreground hover:bg-overlay'
              "
              @click="mainTab = 'dashboard'"
            >
              Dashboard
            </button>
          </nav>
          <button
            type="button"
            class="flex items-center gap-1 rounded border border-border px-2 py-1 text-xs text-muted hover:bg-overlay hover:text-surface-foreground"
            :title="`Theme: ${theme.choice} (click to cycle)`"
            @click="theme.cycle()"
          >
            <svg
              v-if="theme.choice === 'light'"
              class="h-3.5 w-3.5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <circle cx="12" cy="12" r="5" />
              <line x1="12" y1="1" x2="12" y2="3" />
              <line x1="12" y1="21" x2="12" y2="23" />
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
              <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
              <line x1="1" y1="12" x2="3" y2="12" />
              <line x1="21" y1="12" x2="23" y2="12" />
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
              <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
            </svg>
            <svg
              v-else-if="theme.choice === 'dark'"
              class="h-3.5 w-3.5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
            </svg>
            <svg
              v-else
              class="h-3.5 w-3.5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <circle cx="12" cy="12" r="9" />
              <path d="M12 3a9 9 0 0 1 0 18" fill="currentColor" />
            </svg>
            <span class="hidden sm:inline">{{ theme.choice }}</span>
          </button>
          <span class="hidden text-[10px] text-muted sm:inline">
            <kbd class="rounded border border-border bg-surface px-1 font-mono text-surface-foreground">/</kbd>
            search
            ·
            <kbd class="rounded border border-border bg-surface px-1 font-mono text-surface-foreground">Esc</kbd>
            clear
          </span>
        </div>
      </div>
    </header>

    <div
      v-if="artifacts.siblingMergeError && artifacts.siblingMergeLine"
      class="shrink-0 border-b border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive"
      role="alert"
      data-testid="sibling-merge-error-banner"
    >
      <div class="flex flex-wrap items-start justify-between gap-2">
        <span class="min-w-0 flex-1 leading-snug">{{ artifacts.siblingMergeLine }}</span>
        <button
          type="button"
          class="shrink-0 rounded border border-destructive/50 px-2 py-0.5 text-[10px] font-medium hover:bg-destructive/10"
          data-testid="sibling-merge-error-dismiss"
          @click="artifacts.clearSiblingMergeBanner()"
        >
          Dismiss
        </button>
      </div>
    </div>

    <div class="flex min-h-0 flex-1 flex-col">
      <div class="flex min-h-0 flex-1">
      <!-- LEFT SIDEBAR (collapsible) -->
      <div
        class="relative z-10 flex min-h-0 min-w-0 shrink-0 flex-col border-r border-border bg-canvas transition-all"
        :class="leftOpen ? 'w-72' : 'w-8'"
      >
        <button
          type="button"
          :class="railEdgeToggleTab.left"
          :title="leftOpen ? 'Collapse left panel' : 'Expand left panel'"
          :aria-expanded="leftOpen"
          data-testid="left-rail-edge-toggle"
          @click="leftOpen = !leftOpen"
        >
          <svg class="h-3 w-3 transition-transform" :class="{ 'rotate-180': !leftOpen }" viewBox="0 0 12 12" fill="currentColor">
            <path d="M8 2L4 6l4 4z" />
          </svg>
        </button>
        <div
          v-if="!leftOpen"
          class="flex min-h-0 flex-1 flex-col items-center justify-center gap-4 py-2"
        >
          <button
            type="button"
            class="text-[10px] font-medium text-muted hover:text-surface-foreground"
            style="writing-mode: vertical-lr"
            title="Open left panel (Search and Explore — use Explore corpus → inside for GI explore)"
            @click="
              leftOpen = true;
              void nextTick(() => leftPanelRef?.focusQuery())
            "
          >
            Search / Explore
          </button>
        </div>
        <div v-show="leftOpen" class="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
          <div class="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden px-2 pb-4 pt-2">
            <LeftPanel
              ref="leftPanelRef"
              @go-graph="activateGraphTab()"
              @open-library-episode="onSearchOpenLibraryEpisode"
              @open-episode-summary="onSearchOpenEpisodeSummary"
            />
          </div>
        </div>
      </div>

      <!-- CENTER -->
      <div class="flex min-h-0 min-w-0 flex-1 flex-col overflow-x-hidden">
        <!-- Graph / Dashboard — flex column so tab roots (h-full / flex-1) receive height -->
        <div class="flex min-h-0 flex-1 flex-col">
          <keep-alive>
            <DigestView
              v-if="mainTab === 'digest'"
              class="h-full"
              @switch-main-tab="onSwitchMainTab($event)"
              @focus-search="onLibraryFocusSearch"
              @open-library-episode="onDigestOpenEpisodeInRail"
            />
          </keep-alive>
          <keep-alive>
            <LibraryView
              v-if="mainTab === 'library'"
              class="h-full"
              @switch-main-tab="onSwitchMainTab($event)"
              @focus-search="onLibraryFocusSearch"
            />
          </keep-alive>
          <div
            v-if="mainTab === 'dashboard'"
            class="h-full min-h-0 max-w-full flex-1 overflow-x-hidden overflow-y-auto p-3"
          >
            <DashboardView
              @go-graph="activateGraphTab()"
              @open-library="mainTab = 'library'"
              @open-digest="mainTab = 'digest'"
            />
          </div>
          <keep-alive>
            <GraphTabPanel
              v-if="mainTab === 'graph'"
              ref="graphCanvasRef"
              @request-corpus-graph-sync="onCorpusGraphLensReload"
              @request-graph-full-reset="onGraphCorpusFullReset"
            />
          </keep-alive>
        </div>
      </div>

      <!-- RIGHT SIDEBAR (collapsible) — subject rail only -->
      <div
        class="relative z-10 flex min-h-0 shrink-0 flex-col border-l border-border bg-canvas transition-all"
        :class="rightOpen ? 'w-96' : 'w-8'"
      >
        <button
          type="button"
          :class="railEdgeToggleTab.right"
          :title="rightOpen ? 'Collapse right panel' : 'Expand right panel'"
          :aria-expanded="rightOpen"
          data-testid="right-rail-edge-toggle"
          @click="rightOpen = !rightOpen"
        >
          <svg class="h-3 w-3 transition-transform" :class="{ 'rotate-180': !rightOpen }" viewBox="0 0 12 12" fill="currentColor">
            <path d="M4 2l4 4-4 4z" />
          </svg>
        </button>
        <div
          v-if="!rightOpen"
          class="flex min-h-0 flex-1 flex-col items-center justify-center gap-4 py-2"
        >
          <button
            type="button"
            class="text-[10px] font-medium text-muted hover:text-surface-foreground"
            style="writing-mode: vertical-lr"
            title="Open details panel (episode, graph selection, connections)"
            data-testid="rail-collapsed-subject"
            @click="rightOpen = true"
          >
            Details
          </button>
        </div>
        <div v-show="rightOpen" class="flex min-h-0 flex-1 flex-col overflow-hidden">
          <SubjectRail
            :main-tab="mainTab"
            @close-subject="onCloseSubjectRail"
            @go-graph="activateGraphTab()"
            @focus-search-handoff="onLibraryFocusSearch"
            @prefill-semantic-search="onGraphNodeTopicPrefillSearch"
            @open-explore-topic-filter="onGraphNodeTopicOpenExploreFilter"
            @open-explore-speaker-filter="onGraphNodeSpeakerOpenExploreFilter"
            @open-explore-insight-filters="onGraphNodeInsightOpenExploreFilters"
            @open-library-episode="onSearchOpenLibraryEpisode"
            @open-episode-summary="onSearchOpenEpisodeSummary"
            @switch-main-tab="onSwitchMainTab($event)"
          />
        </div>
      </div>
      </div>
      <StatusBar
        @local-artifacts-loaded="onStatusBarLocalArtifactsLoaded"
        @go-graph="activateGraphTab()"
      />
    </div>
  </div>
</template>
