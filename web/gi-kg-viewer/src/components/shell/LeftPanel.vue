<script setup lang="ts">
/**
 * Left query column — Search compact-launcher only (UXS-005 §Compact launcher
 * (RFC-107) + Search v3 §S1). The former Explore mode + slide was retired: the
 * ``shell.leftPanelSurface = 'explore'`` state, the ``left-panel-enter-explore``
 * / ``left-panel-back-search`` test IDs, and the ``ExplorePanel`` component are
 * all gone. All Explore filters live on ``SearchFilterBar`` as chips; graph
 * handoffs that previously set Explore filters now set Search filters (App.vue).
 */
import { nextTick, ref } from 'vue'
import type { SearchHit } from '../../api/searchApi'
import SearchPanel from '../search/SearchPanel.vue'

const searchPanelRef = ref<{ focusQuery: () => void } | null>(null)

defineEmits<{
  'go-graph': []
  'open-library-episode': [{ metadata_relative_path: string }]
  'open-episode-summary': [SearchHit]
}>()

defineExpose({
  focusQuery: () => {
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
    <SearchPanel
      ref="searchPanelRef"
      class="min-h-0 min-w-0 flex-1"
      @go-graph="$emit('go-graph')"
      @open-library-episode="$emit('open-library-episode', $event)"
      @open-episode-summary="$emit('open-episode-summary', $event)"
    />
  </div>
</template>
