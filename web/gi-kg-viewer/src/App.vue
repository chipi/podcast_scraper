<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useViewerKeyboard } from './composables/useViewerKeyboard'
import DashboardView from './components/dashboard/DashboardView.vue'
import GraphCanvas from './components/graph/GraphCanvas.vue'
import ExplorePanel from './components/explore/ExplorePanel.vue'
import SearchPanel from './components/search/SearchPanel.vue'
import GraphFilters from './components/graph/GraphFilters.vue'
import GraphLegend from './components/graph/GraphLegend.vue'
import HelpTip from './components/shared/HelpTip.vue'
import CollapsibleSection from './components/shared/CollapsibleSection.vue'
import { useArtifactsStore } from './stores/artifacts'
import { useShellStore } from './stores/shell'
import { useThemeStore } from './stores/theme'

const shell = useShellStore()
const artifacts = useArtifactsStore()
const theme = useThemeStore()

const mainTab = ref<'graph' | 'dashboard'>('graph')
const localFileInput = ref<HTMLInputElement | null>(null)
const searchPanelRef = ref<{ focusQuery: () => void } | null>(null)
const graphCanvasRef = ref<{ clearInteractionState: () => void } | null>(null)
const isGraphTab = computed(() => mainTab.value === 'graph')

const leftOpen = ref(true)
const leftTab = ref<'corpus' | 'api'>('corpus')
const topOpen = ref(true)
const rightOpen = ref(true)
const rightTab = ref<'search' | 'explore'>('search')

const filtersRef = ref<InstanceType<typeof GraphFilters> | null>(null)
const legendRef = ref<InstanceType<typeof GraphLegend> | null>(null)

const topSummary = computed(() => {
  const parts: string[] = []
  if (filtersRef.value?.filterSummary) parts.push(filtersRef.value.filterSummary)
  if (legendRef.value?.legendSummary) parts.push(legendRef.value.legendSummary)
  return parts.join(' · ')
})

const leftSummary = computed(() => {
  const parts: string[] = []
  if (artifacts.displayArtifact) parts.push(`${artifacts.displayArtifact.nodes} nodes`)
  parts.push(shell.healthStatus ? 'API ok' : 'API offline')
  return parts.join(' · ')
})

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

function onLocalFilesChange(ev: Event): void {
  const el = ev.target as HTMLInputElement
  void artifacts.loadFromLocalFiles(el.files)
  el.value = ''
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
</script>

<template>
  <div class="flex min-h-screen flex-col bg-canvas text-canvas-foreground">
    <header class="shrink-0 border-b border-border bg-surface px-4 py-2 shadow-sm">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <h1 class="text-lg font-semibold tracking-tight text-surface-foreground">
          GI / KG Viewer
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
            API
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
              API
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
                  <code class="text-[10px]">.kg.json</code>).
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
              <div
                v-else-if="shell.artifactList.length"
                class="mt-1.5 space-y-1"
              >
                <div class="flex gap-1">
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
              <p v-else-if="shell.artifactCount !== null && shell.artifactCount === 0" class="mt-1 text-[10px] text-muted">
                No artifacts found.
              </p>
              <p v-if="shell.artifactsError" class="mt-1 text-[10px] text-danger">
                {{ shell.artifactsError }}
              </p>
              <button
                type="button"
                class="mt-1.5 w-full rounded border border-border px-2 py-1 text-[10px] font-medium hover:bg-overlay disabled:opacity-40"
                :disabled="artifacts.selectedRelPaths.length === 0 || artifacts.loading"
                @click="artifacts.loadSelected()"
              >
                {{ artifacts.loading ? 'Loading…' : 'Load into graph' }}
              </button>
              <p v-if="artifacts.loadError" class="mt-1 text-[10px] text-danger">
                {{ artifacts.loadError }}
              </p>
              <p v-if="artifacts.displayArtifact" class="mt-1 text-[10px] text-muted">
                {{ artifacts.displayArtifact.name }} ({{ artifacts.displayArtifact.nodes }} nodes)
              </p>
            </section>

            <!-- API tab -->
            <section v-show="leftTab === 'api'">
              <h2 class="mb-1.5 text-xs font-medium text-gi">
                API health
              </h2>
              <p v-if="shell.healthStatus" class="text-[10px] text-success">
                {{ shell.healthStatus }}
              </p>
              <p v-else-if="shell.healthError" class="text-[10px] text-danger">
                {{ shell.healthError }}
              </p>
              <p v-else class="text-[10px] text-muted">
                Checking…
              </p>
              <button
                type="button"
                class="mt-1 rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay"
                @click="shell.fetchHealth()"
              >
                Retry
              </button>
              <div
                v-if="shell.healthError"
                class="mt-2 rounded border border-border bg-overlay p-1.5 text-[10px]"
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
            </section>
          </div>
        </div>
      </div>

      <!-- CENTER -->
      <div class="flex min-w-0 flex-1 flex-col">
        <!-- TOP: Filters + Node types (collapsible) -->
        <div
          v-if="artifacts.displayArtifact && mainTab === 'graph'"
          class="shrink-0 border-b border-border"
        >
          <CollapsibleSection
            title="Filters & sources"
            :summary="topSummary"
            :default-open="true"
          >
            <div class="flex flex-wrap gap-4">
              <div class="min-w-0 flex-1">
                <GraphFilters ref="filtersRef" />
              </div>
              <div class="w-full max-w-[50%] shrink-0 border-l border-border pl-4 xl:w-auto">
                <GraphLegend ref="legendRef" />
              </div>
            </div>
          </CollapsibleSection>
        </div>

        <!-- Graph / Dashboard -->
        <div class="min-h-0 flex-1">
          <template v-if="mainTab === 'dashboard'">
            <div class="overflow-y-auto p-3" style="max-height: calc(100vh - 5rem)">
              <DashboardView />
            </div>
          </template>
          <template v-else>
            <GraphCanvas
              v-if="artifacts.displayArtifact"
              ref="graphCanvasRef"
              class="h-full"
            />
            <div
              v-else
              class="flex h-full min-h-[280px] items-center justify-center rounded border border-dashed border-border bg-surface p-8 text-sm text-muted"
            >
              List artifacts, select files, then "Load into graph".
            </div>
          </template>
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
          <svg class="h-3 w-3 transition-transform" :class="{ 'rotate-180': rightOpen }" viewBox="0 0 12 12" fill="currentColor">
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
