<script setup lang="ts">
import { ref } from 'vue'
import type { SearchHit } from '../../api/searchApi'
import ExplorePanel from '../explore/ExplorePanel.vue'
import SearchPanel from '../search/SearchPanel.vue'

const searchPanelRef = ref<{ focusQuery: () => void } | null>(null)

defineEmits<{
  'go-graph': []
  'open-library-episode': [{ metadata_relative_path: string }]
  'open-episode-summary': [SearchHit]
}>()

defineExpose({
  focusQuery: () => {
    searchPanelRef.value?.focusQuery()
  },
})
</script>

<template>
  <div class="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto">
    <SearchPanel
      ref="searchPanelRef"
      @go-graph="$emit('go-graph')"
      @open-library-episode="$emit('open-library-episode', $event)"
      @open-episode-summary="$emit('open-episode-summary', $event)"
    />
    <section class="rounded-lg border border-border bg-surface p-2">
      <h3 class="mb-1 px-1 text-[11px] font-medium text-muted">
        Explore
      </h3>
      <ExplorePanel @go-graph="$emit('go-graph')" />
    </section>
  </div>
</template>
