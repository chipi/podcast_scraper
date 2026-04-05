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

const shell = useShellStore()
const artifacts = useArtifactsStore()

const mainTab = ref<'graph' | 'dashboard'>('graph')
const localFileInput = ref<HTMLInputElement | null>(null)
const searchPanelRef = ref<{ focusQuery: () => void } | null>(null)
const graphCanvasRef = ref<{ clearInteractionState: () => void } | null>(null)
const isGraphTab = computed(() => mainTab.value === 'graph')

const leftOpen = ref(true)
const topOpen = ref(true)
const rightOpen = ref(true)

const filtersRef = ref<InstanceType<typeof GraphFilters> | null>(null)
const legendRef = ref<InstanceType<typeof GraphLegend> | null>(null)

const topSummary = computed(() => {
  const parts: string[] = []
  if (filtersRef.value?.filterSummary) parts.push(filtersRef.value.filterSummary)
  if (legendRef.value?.legendSummary) parts.push(legendRef.value.legendSummary)
  return parts.join(' · ')
})

const leftSummary = computed(() => {
  if (artifacts.displayArtifact) {
    return `${artifacts.displayArtifact.nodes} nodes`
  }
  return shell.healthStatus ? 'connected' : 'offline'
})

useViewerKeyboard({
  focusSearch: () => {
    rightOpen.value = true
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
            @click="leftOpen = true"
          >
            Corpus
          </button>
          <button
            type="button"
            class="text-[10px] font-medium text-muted hover:text-surface-foreground"
            style="writing-mode: vertical-lr"
            @click="leftOpen = true"
          >
            API
          </button>
        </div>
        <div
          v-show="leftOpen"
          class="space-y-3 overflow-y-auto px-2 pb-3"
          style="max-height: calc(100vh - 6rem)"
        >
          <section class="rounded border border-border bg-surface p-2.5">
            <div class="mb-1 flex items-center gap-1">
              <h2 class="text-xs font-medium text-kg">
                Corpus
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
              <div class="max-h-36 overflow-y-auto rounded border border-border bg-elevated p-1 text-[11px]">
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

          <!-- API health -->
          <section class="rounded border border-border bg-surface p-2.5">
            <h2 class="mb-1 text-xs font-medium text-gi">
              API
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
              class="mt-1.5 rounded border border-border bg-overlay p-1.5 text-[10px]"
            >
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
            @click="rightOpen = true"
          >
            Search
          </button>
          <button
            type="button"
            class="text-[10px] font-medium text-muted hover:text-surface-foreground"
            style="writing-mode: vertical-lr"
            @click="rightOpen = true"
          >
            Explore
          </button>
        </div>
        <div
          v-show="rightOpen"
          class="space-y-3 overflow-y-auto px-2 pb-3"
          style="max-height: calc(100vh - 6rem)"
        >
          <SearchPanel
            ref="searchPanelRef"
            @go-graph="mainTab = 'graph'"
          />
          <ExplorePanel @go-graph="mainTab = 'graph'" />
        </div>
      </div>
    </div>
  </div>
</template>
