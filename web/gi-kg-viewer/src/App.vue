<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useViewerKeyboard } from './composables/useViewerKeyboard'
import DashboardOverviewSection from './components/dashboard/DashboardOverviewSection.vue'
import DashboardView from './components/dashboard/DashboardView.vue'
import GraphCanvas from './components/graph/GraphCanvas.vue'
import ExplorePanel from './components/explore/ExplorePanel.vue'
import SearchPanel from './components/search/SearchPanel.vue'
import HelpTip from './components/shared/HelpTip.vue'
import DigestView from './components/digest/DigestView.vue'
import LibraryView from './components/library/LibraryView.vue'
import { useArtifactsStore } from './stores/artifacts'
import { useSearchStore } from './stores/search'
import { useShellStore } from './stores/shell'
import { useThemeStore } from './stores/theme'

const shell = useShellStore()
const artifacts = useArtifactsStore()
const search = useSearchStore()
const theme = useThemeStore()

const mainTab = ref<'digest' | 'library' | 'graph' | 'dashboard'>('digest')
const localFileInput = ref<HTMLInputElement | null>(null)
const searchPanelRef = ref<{ focusQuery: () => void } | null>(null)
const graphCanvasRef = ref<{ clearInteractionState: () => void } | null>(null)
const isGraphTab = computed(() => mainTab.value === 'graph')

const leftOpen = ref(true)
const leftTab = ref<'corpus' | 'api'>('corpus')
const rightOpen = ref(true)
const rightTab = ref<'search' | 'explore'>('search')

useViewerKeyboard({
  focusSearch: () => {
    rightOpen.value = true
    rightTab.value = 'search'
    searchPanelRef.value?.focusQuery()
  },
  clearGraphFocus: () => {
    graphCanvasRef.value?.clearInteractionState()
  },
  isGraphTab,
})

function triggerLocalFilePick(): void {
  localFileInput.value?.click()
}

async function onLocalFilesChange(ev: Event): Promise<void> {
  const el = ev.target as HTMLInputElement
  await artifacts.loadFromLocalFiles(el.files)
  el.value = ''
  if (artifacts.displayArtifact) {
    mainTab.value = 'graph'
  }
}

async function onLoadIntoGraphClick(): Promise<void> {
  await artifacts.loadSelected()
  if (artifacts.displayArtifact) {
    mainTab.value = 'graph'
  }
}

onMounted(() => {
  void shell.fetchHealth()
})

watch(
  () => shell.corpusPath,
  (p) => {
    artifacts.setCorpusPath(p)
  },
  { immediate: true },
)

/** Bumps when corpus path or health changes; stale async sync steps bail out. */
let corpusGraphSyncGen = 0

/**
 * When the API is healthy and a corpus path is set, list GI/KG via ``GET /api/artifacts``
 * and load all of them into the merged graph (same end state as **List** → **All** → **Load into graph**).
 * Offline / failed health: skip so file-picker loads stay intact.
 */
async function syncMergedGraphFromCorpusApi(): Promise<void> {
  const gen = ++corpusGraphSyncGen
  artifacts.setCorpusPath(shell.corpusPath)
  const root = shell.corpusPath.trim()
  if (!root) {
    artifacts.clearSelection()
    return
  }
  if (!shell.healthStatus) {
    return
  }
  await shell.fetchArtifactList()
  if (gen !== corpusGraphSyncGen) {
    return
  }
  const giKgRelPaths = shell.artifactList
    .filter((a) => a.kind === 'gi' || a.kind === 'kg')
    .map((a) => a.relative_path)
  if (giKgRelPaths.length === 0) {
    artifacts.clearSelection()
    return
  }
  artifacts.selectAllListed(giKgRelPaths)
  await artifacts.loadSelected()
}

watch(
  () => [shell.corpusPath, shell.healthStatus] as const,
  () => {
    void syncMergedGraphFromCorpusApi()
  },
  { immediate: true },
)

function onLibraryFocusSearch(payload: { feed: string; query: string }): void {
  rightOpen.value = true
  rightTab.value = 'search'
  search.applyLibrarySearchHandoff(payload.feed, payload.query)
  void nextTick(() => {
    searchPanelRef.value?.focusQuery()
  })
}

function onDigestOpenLibraryEpisode(payload: { metadata_relative_path: string }): void {
  shell.setPendingLibraryEpisode(payload.metadata_relative_path)
  mainTab.value = 'library'
}
</script>

<template>
  <div class="flex min-h-screen flex-col bg-canvas text-canvas-foreground">
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
              @click="mainTab = 'graph'"
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

    <div class="flex min-h-0 flex-1">
      <!-- LEFT SIDEBAR (collapsible) -->
      <div
        class="relative shrink-0 border-r border-border bg-canvas transition-all"
        :class="leftOpen ? 'w-72' : 'w-8'"
      >
        <button
          type="button"
          class="flex h-8 w-full items-center justify-center text-muted hover:text-surface-foreground"
          :title="leftOpen ? 'Collapse left panel' : 'Expand left panel'"
          @click="leftOpen = !leftOpen"
        >
          <svg class="h-3 w-3 transition-transform" :class="{ 'rotate-180': !leftOpen }" viewBox="0 0 12 12" fill="currentColor">
            <path d="M8 2L4 6l4 4z" />
          </svg>
        </button>
        <!-- Vertical labels when collapsed -->
        <div
          v-if="!leftOpen"
          class="flex flex-col items-center gap-4 pt-2"
        >
          <button
            type="button"
            class="text-[10px] font-medium text-muted hover:text-surface-foreground"
            style="writing-mode: vertical-lr"
            @click="leftOpen = true; leftTab = 'corpus'"
          >
            Corpus
          </button>
          <button
            type="button"
            class="text-[10px] font-medium text-muted hover:text-surface-foreground"
            style="writing-mode: vertical-lr"
            @click="leftOpen = true; leftTab = 'api'"
          >
            API+Data
          </button>
        </div>
        <div v-show="leftOpen" class="flex flex-col" style="max-height: calc(100vh - 6rem)">
          <nav
            class="mx-2 mt-1 flex gap-1 rounded border border-border bg-elevated p-0.5 text-xs font-medium"
            aria-label="Left panel tabs"
          >
            <button
              type="button"
              class="flex-1 rounded px-2 py-1 text-center"
              :class="
                leftTab === 'corpus'
                  ? 'bg-primary text-primary-foreground'
                  : 'text-elevated-foreground hover:bg-overlay'
              "
              @click="leftTab = 'corpus'"
            >
              Corpus
            </button>
            <button
              type="button"
              class="flex-1 rounded px-2 py-1 text-center"
              :class="
                leftTab === 'api'
                  ? 'bg-primary text-primary-foreground'
                  : 'text-elevated-foreground hover:bg-overlay'
              "
              @click="leftTab = 'api'"
            >
              API · Data
            </button>
          </nav>
          <div class="min-h-0 flex-1 overflow-y-auto px-2 pb-3 pt-2">
            <!-- Corpus tab -->
            <section v-show="leftTab === 'corpus'">
              <div class="mb-1.5 flex items-center gap-1">
                <h2 class="text-xs font-medium text-kg">
                  Corpus path
                </h2>
                <HelpTip>
                  Set the same folder you pass as
                  <code class="rounded bg-overlay px-0.5 text-[10px]">--output-dir</code>
                  to the pipeline (contains
                  <code class="text-[10px]">metadata/</code>
                  with your
                  <code class="text-[10px]">.gi.json</code> /
                  <code class="text-[10px]">.kg.json</code>). When the API is healthy, the viewer
                  lists artifacts and loads <strong>all</strong> GI/KG files into the merged graph
                  automatically (same as <strong>List</strong> → <strong>All</strong> →
                  <strong>Load into graph</strong>), like Digest/Library catalog refresh. Large
                  corpora may take a while.
                </HelpTip>
              </div>
              <input
                v-model="shell.corpusPath"
                type="text"
                class="w-full rounded border border-border bg-elevated px-2 py-1 text-xs text-elevated-foreground placeholder:text-muted"
                placeholder="/path/to/output"
                autocomplete="off"
              >
              <div class="mt-1.5 flex flex-wrap gap-1">
                <button
                  type="button"
                  class="rounded bg-primary px-2 py-0.5 text-[10px] font-medium text-primary-foreground hover:opacity-90 disabled:opacity-40"
                  :disabled="!shell.hasCorpusPath || shell.artifactsLoading"
                  @click="shell.fetchArtifactList()"
                >
                  List
                </button>
              </div>
              <p v-if="shell.artifactsLoading" class="mt-1 text-[10px] text-muted">
                Loading…
              </p>
              <template v-else>
                <div
                  v-if="shell.corpusHints.length"
                  class="mt-1.5 rounded border border-warning/40 bg-warning/10 px-2 py-1.5 text-[10px] text-surface-foreground"
                  role="status"
                >
                  <p class="font-medium text-warning">
                    Corpus path hint
                  </p>
                  <ul class="mt-0.5 list-inside list-disc text-muted">
                    <li v-for="(h, i) in shell.corpusHints" :key="i">
                      {{ h }}
                    </li>
                  </ul>
                </div>
                <div
                  v-if="shell.artifactList.length"
                  class="mt-1.5 space-y-1"
                >
                <div class="flex flex-wrap items-center gap-1">
                  <button
                    type="button"
                    class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
                    @click="artifacts.selectAllListed(shell.artifactList.map((a) => a.relative_path))"
                  >
                    All
                  </button>
                  <button
                    type="button"
                    class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
                    @click="artifacts.deselectAllListed()"
                  >
                    None
                  </button>
                  <button
                    type="button"
                    class="rounded border border-border px-2 py-0.5 text-[10px] font-medium hover:bg-overlay disabled:opacity-40"
                    :disabled="artifacts.selectedRelPaths.length === 0 || artifacts.loading"
                    @click="onLoadIntoGraphClick()"
                  >
                    {{ artifacts.loading ? 'Loading…' : 'Load into graph' }}
                  </button>
                </div>
                <div class="overflow-y-auto rounded border border-border bg-elevated p-1 text-[11px]">
                  <label
                    v-for="a in shell.artifactList"
                    :key="a.relative_path"
                    class="flex cursor-pointer items-start gap-1 py-0.5 hover:bg-overlay"
                  >
                    <input
                      type="checkbox"
                      class="mt-0.5 rounded border-border"
                      :checked="artifacts.selectedRelPaths.includes(a.relative_path)"
                      @change="artifacts.toggleSelection(a.relative_path)"
                    >
                    <span class="break-all">
                      <span :class="a.kind === 'gi' ? 'text-gi' : 'text-kg'">{{ a.kind }}</span>
                      {{ a.relative_path }}
                    </span>
                  </label>
                </div>
                </div>
                <p
                  v-else-if="shell.artifactCount !== null && shell.artifactCount === 0"
                  class="mt-1 text-[10px] text-muted"
                >
                  No artifacts found.
                </p>
              </template>
              <p v-if="shell.artifactsError" class="mt-1 text-[10px] text-danger">
                {{ shell.artifactsError }}
              </p>
              <p v-if="artifacts.loadError" class="mt-1 text-[10px] text-danger">
                {{ artifacts.loadError }}
              </p>
              <p v-if="artifacts.displayArtifact" class="mt-1 text-[10px] text-muted">
                {{ artifacts.displayArtifact.name }} ({{ artifacts.displayArtifact.nodes }} nodes)
              </p>
            </section>

            <!-- API · Data tab -->
            <section
              v-show="leftTab === 'api'"
              class="space-y-2"
            >
              <h2 class="text-xs font-medium text-surface-foreground">
                API
              </h2>
              <p class="text-[10px] leading-snug text-muted">
                Capability flags from
                <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">GET /api/health</code>
                (graph, search, index routes, catalog). FAISS availability is separate — see Data → Vector index.
              </p>
              <div class="rounded border border-border bg-elevated p-2 text-[10px]">
                <dl class="space-y-1">
                  <div class="flex justify-between gap-2">
                    <dt class="text-muted">
                      Health
                    </dt>
                    <dd
                      v-if="shell.healthStatus"
                      class="font-medium text-success"
                    >
                      {{ shell.healthStatusDisplay }}
                    </dd>
                    <dd
                      v-else-if="shell.healthError"
                      class="font-medium text-danger"
                    >
                      {{ shell.healthError }}
                    </dd>
                    <dd
                      v-else
                      class="text-muted"
                    >
                      Checking…
                    </dd>
                  </div>
                </dl>
                <dl
                  v-if="shell.healthStatus"
                  class="mt-1.5 space-y-1 border-t border-border/60 pt-1.5"
                >
                  <div class="flex justify-between gap-2">
                    <dt class="text-muted">
                      Artifacts (graph)
                    </dt>
                    <dd
                      :class="
                        shell.artifactsApiAvailable !== false ? 'text-success' : 'text-danger'
                      "
                    >
                      {{ shell.artifactsApiAvailable !== false ? 'Yes' : 'No' }}
                    </dd>
                  </div>
                  <div class="flex justify-between gap-2">
                    <dt class="text-muted">
                      Semantic search
                    </dt>
                    <dd
                      :class="shell.searchApiAvailable !== false ? 'text-success' : 'text-danger'"
                    >
                      {{ shell.searchApiAvailable !== false ? 'Yes' : 'No' }}
                    </dd>
                  </div>
                  <div class="flex justify-between gap-2">
                    <dt class="text-muted">
                      Graph explore
                    </dt>
                    <dd
                      :class="shell.exploreApiAvailable !== false ? 'text-success' : 'text-danger'"
                    >
                      {{ shell.exploreApiAvailable !== false ? 'Yes' : 'No' }}
                    </dd>
                  </div>
                  <div class="flex justify-between gap-2">
                    <dt class="text-muted">
                      Index routes
                    </dt>
                    <dd
                      :class="
                        shell.indexRoutesApiAvailable !== false ? 'text-success' : 'text-danger'
                      "
                    >
                      {{ shell.indexRoutesApiAvailable !== false ? 'Yes' : 'No' }}
                    </dd>
                  </div>
                  <div class="flex justify-between gap-2">
                    <dt class="text-muted">
                      Corpus metrics
                    </dt>
                    <dd
                      :class="
                        shell.corpusMetricsApiAvailable !== false ? 'text-success' : 'text-danger'
                      "
                    >
                      {{ shell.corpusMetricsApiAvailable !== false ? 'Yes' : 'No' }}
                    </dd>
                  </div>
                  <div class="flex justify-between gap-2">
                    <dt class="text-muted">
                      Library API
                    </dt>
                    <dd
                      :class="
                        shell.corpusLibraryApiAvailable ? 'text-success' : 'text-danger'
                      "
                    >
                      {{ shell.corpusLibraryApiAvailable ? 'Yes' : 'No' }}
                    </dd>
                  </div>
                  <div class="flex justify-between gap-2">
                    <dt class="text-muted">
                      Digest API
                    </dt>
                    <dd
                      :class="
                        shell.corpusDigestApiAvailable ? 'text-success' : 'text-danger'
                      "
                    >
                      {{ shell.corpusDigestApiAvailable ? 'Yes' : 'No' }}
                    </dd>
                  </div>
                  <div class="flex justify-between gap-2">
                    <dt class="text-muted">
                      Binary (covers)
                    </dt>
                    <dd
                      :class="
                        shell.corpusBinaryApiAvailable !== false ? 'text-success' : 'text-danger'
                      "
                    >
                      {{ shell.corpusBinaryApiAvailable !== false ? 'Yes' : 'No' }}
                    </dd>
                  </div>
                </dl>
                <p
                  v-if="shell.healthStatus && !shell.corpusLibraryApiAvailable"
                  class="mt-1.5 text-[10px] leading-snug text-danger"
                >
                  Corpus Library API not advertised — upgrade/restart the Python server (Library tab).
                </p>
                <button
                  type="button"
                  class="mt-1.5 rounded border border-border px-2 py-0.5 hover:bg-overlay"
                  @click="shell.fetchHealth()"
                >
                  Retry health
                </button>
              </div>
              <div
                v-if="shell.healthError"
                class="rounded border border-border bg-overlay p-1.5 text-[10px]"
              >
                <p class="mb-1 text-muted">
                  Load files directly (no API):
                </p>
                <input
                  ref="localFileInput"
                  type="file"
                  class="sr-only"
                  multiple
                  accept=".gi.json,.kg.json,application/json"
                  @change="onLocalFilesChange"
                >
                <button
                  type="button"
                  class="rounded border border-border px-2 py-0.5 hover:bg-canvas"
                  @click="triggerLocalFilePick"
                >
                  Choose files…
                </button>
              </div>
              <DashboardOverviewSection />
            </section>
          </div>
        </div>
      </div>

      <!-- CENTER -->
      <div class="flex min-w-0 flex-1 flex-col">
        <!-- Graph / Dashboard -->
        <div class="min-h-0 flex-1">
          <DigestView
            v-if="mainTab === 'digest'"
            class="h-full"
            @switch-main-tab="mainTab = $event"
            @focus-search="onLibraryFocusSearch"
            @open-library-episode="onDigestOpenLibraryEpisode"
          />
          <keep-alive>
            <LibraryView
              v-if="mainTab === 'library'"
              class="h-full"
              @switch-main-tab="mainTab = $event"
              @focus-search="onLibraryFocusSearch"
            />
          </keep-alive>
          <div
            v-if="mainTab === 'dashboard'"
            class="h-full overflow-y-auto p-3"
            style="max-height: calc(100vh - 5rem)"
          >
            <DashboardView />
          </div>
          <keep-alive>
            <GraphCanvas
              v-if="mainTab === 'graph' && artifacts.displayArtifact"
              ref="graphCanvasRef"
              class="h-full"
            />
          </keep-alive>
          <div
            v-if="mainTab === 'graph' && !artifacts.displayArtifact"
            class="flex h-full min-h-[280px] items-center justify-center rounded border border-dashed border-border bg-surface p-8 text-sm text-muted"
          >
            <span class="max-w-md text-center">
              With a healthy API, set <strong>Corpus path</strong> to auto-load all GI/KG; or use
              <strong>List</strong> and <strong>Load into graph</strong>. Offline: <strong>Choose
              files…</strong> on <strong>API · Data</strong>.
            </span>
          </div>
        </div>
      </div>

      <!-- RIGHT SIDEBAR (collapsible) -->
      <div
        class="relative shrink-0 border-l border-border bg-canvas transition-all"
        :class="rightOpen ? 'w-80' : 'w-8'"
      >
        <button
          type="button"
          class="flex h-8 w-full items-center justify-center text-muted hover:text-surface-foreground"
          :title="rightOpen ? 'Collapse right panel' : 'Expand right panel'"
          @click="rightOpen = !rightOpen"
        >
          <svg class="h-3 w-3 transition-transform" :class="{ 'rotate-180': !rightOpen }" viewBox="0 0 12 12" fill="currentColor">
            <path d="M4 2l4 4-4 4z" />
          </svg>
        </button>
        <!-- Vertical labels when collapsed -->
        <div
          v-if="!rightOpen"
          class="flex flex-col items-center gap-4 pt-2"
        >
          <button
            type="button"
            class="text-[10px] font-medium text-muted hover:text-surface-foreground"
            style="writing-mode: vertical-lr"
            @click="rightOpen = true; rightTab = 'search'"
          >
            Search
          </button>
          <button
            type="button"
            class="text-[10px] font-medium text-muted hover:text-surface-foreground"
            style="writing-mode: vertical-lr"
            @click="rightOpen = true; rightTab = 'explore'"
          >
            Explore
          </button>
        </div>
        <div v-show="rightOpen" class="flex flex-col" style="max-height: calc(100vh - 6rem)">
          <nav
            class="mx-2 mt-1 flex gap-1 rounded border border-border bg-elevated p-0.5 text-xs font-medium"
            aria-label="Right panel tabs"
          >
            <button
              type="button"
              class="flex-1 rounded px-2 py-1 text-center"
              :class="
                rightTab === 'search'
                  ? 'bg-primary text-primary-foreground'
                  : 'text-elevated-foreground hover:bg-overlay'
              "
              @click="rightTab = 'search'"
            >
              Search
            </button>
            <button
              type="button"
              class="flex-1 rounded px-2 py-1 text-center"
              :class="
                rightTab === 'explore'
                  ? 'bg-primary text-primary-foreground'
                  : 'text-elevated-foreground hover:bg-overlay'
              "
              @click="rightTab = 'explore'"
            >
              Explore
            </button>
          </nav>
          <div class="min-h-0 flex-1 overflow-y-auto px-2 pb-3 pt-2">
            <SearchPanel
              v-show="rightTab === 'search'"
              ref="searchPanelRef"
              @go-graph="mainTab = 'graph'"
            />
            <ExplorePanel
              v-show="rightTab === 'explore'"
              @go-graph="mainTab = 'graph'"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
