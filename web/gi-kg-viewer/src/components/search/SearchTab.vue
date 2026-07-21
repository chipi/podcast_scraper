<script setup lang="ts">
/**
 * Query Workspace — Search v3 §S2 + §S4-shell pivot. Full-width main-tab
 * surface that hosts the SearchPanel query UI. The Saved + Recent sidebar
 * moved to the app-level LeftPanel (§S4-shell) so the workspace can use the
 * whole main-tab area for query + results + future slices (S4 operator bar,
 * S5 enriched-answer hero).
 *
 * SearchPanel is the query-surface source of truth; this component is
 * intentionally a thin wrapper — slices S4/S5 attach their surfaces to it.
 */
import SearchPanel from './SearchPanel.vue'

const emit = defineEmits<{
  'go-graph': []
  'open-library-episode': [payload: { metadata_relative_path: string }]
  /**
   * Search v3 §S4a — result-set "On graph" operator; forwarded from
   * SearchPanel. App.vue is the graph handoff site.
   */
  'focus-set': [ids: string[]]
}>()
</script>

<template>
  <section
    class="flex h-full min-h-0 flex-col overflow-hidden p-3"
    data-testid="search-workspace"
    aria-label="Query workspace"
  >
    <SearchPanel
      class="min-h-0 min-w-0 flex-1"
      @go-graph="emit('go-graph')"
      @open-library-episode="emit('open-library-episode', $event)"
      @focus-set="(ids: string[]) => emit('focus-set', ids)"
    />
  </section>
</template>
