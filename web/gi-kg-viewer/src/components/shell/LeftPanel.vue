<script setup lang="ts">
import { nextTick, ref } from 'vue'
import { storeToRefs } from 'pinia'
import type { SearchHit } from '../../api/searchApi'
import { useShellStore } from '../../stores/shell'
import ExplorePanel from '../explore/ExplorePanel.vue'
import SearchPanel from '../search/SearchPanel.vue'

const shell = useShellStore()
const { leftPanelSurface } = storeToRefs(shell)

const searchPanelRef = ref<{ focusQuery: () => void } | null>(null)

function enterExplore(): void {
  shell.setLeftPanelSurface('explore')
}

function backToSearch(): void {
  shell.setLeftPanelSurface('search')
}

defineEmits<{
  'go-graph': []
  'open-library-episode': [{ metadata_relative_path: string }]
  'open-episode-summary': [SearchHit]
}>()

defineExpose({
  focusQuery: () => {
    shell.setLeftPanelSurface('search')
    void nextTick(() => {
      void nextTick(() => {
        searchPanelRef.value?.focusQuery()
      })
    })
  },
})
</script>

<template>
  <div class="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
    <div
      class="relative min-h-0 flex-1 overflow-hidden"
      data-testid="left-panel-slide-host"
    >
      <div
        class="flex h-full min-h-0 w-[200%] flex-row transition-transform duration-300 ease-out motion-reduce:transition-none"
        :class="leftPanelSurface === 'explore' ? '-translate-x-1/2' : 'translate-x-0'"
      >
        <!-- Search mode: scrollable search card + fixed footer (stays above status bar) -->
        <div class="flex h-full min-h-0 w-1/2 min-w-0 shrink-0 flex-col gap-2 pr-1">
          <SearchPanel
            ref="searchPanelRef"
            class="min-h-0 min-w-0 flex-1"
            @go-graph="$emit('go-graph')"
            @open-library-episode="$emit('open-library-episode', $event)"
            @open-episode-summary="$emit('open-episode-summary', $event)"
          />
          <div
            class="shrink-0 rounded-lg border border-border bg-surface px-3 py-2"
            data-testid="left-panel-explore-footer"
          >
            <button
              type="button"
              data-testid="left-panel-enter-explore"
              class="flex w-full items-center gap-1.5 text-left text-xs text-muted hover:text-surface-foreground"
              aria-label="Open Explore corpus mode"
              @click="enterExplore"
            >
              <span class="min-w-0 flex-1">Explore corpus</span>
              <svg
                class="h-3.5 w-3.5 shrink-0 opacity-70"
                viewBox="0 0 12 12"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
              >
                <path
                  d="M2.25 6h6.5M7.25 3.25 10 6 7.25 8.75"
                  stroke="currentColor"
                  stroke-width="1.25"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                />
              </svg>
            </button>
          </div>
        </div>
        <!-- Explore mode: back to Search + GI explore UI -->
        <div class="flex h-full min-h-0 w-1/2 min-w-0 shrink-0 flex-col pl-1">
          <button
            type="button"
            data-testid="left-panel-back-search"
            class="mb-2 flex shrink-0 items-center gap-1.5 self-start text-xs text-muted hover:text-surface-foreground"
            aria-label="Back to semantic search"
            @click="backToSearch"
          >
            <svg
              class="h-3.5 w-3.5 shrink-0 opacity-70"
              viewBox="0 0 12 12"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              aria-hidden="true"
            >
              <path
                d="M9.75 6H3.25M4.75 3.25 2 6l2.75 2.75"
                stroke="currentColor"
                stroke-width="1.25"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
            <span>Search</span>
          </button>
          <div class="min-h-0 flex-1 overflow-x-hidden overflow-y-auto">
            <ExplorePanel @go-graph="$emit('go-graph')" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
