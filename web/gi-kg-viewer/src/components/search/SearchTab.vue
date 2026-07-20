<script setup lang="ts">
/**
 * Query Workspace — Search v3 §S2 (UXS-016). Full-width main-tab surface that
 * hosts the same query UI as the LeftPanel launcher (SearchPanel), but with the
 * shell tab's real estate. Renders when ``shellStore.mainTab === 'search'``.
 *
 * Composition today:
 *   [ WorkspaceHeader (in SearchPanel) ] [ Results (in SearchPanel) ]
 *   [ WorkspaceSidebar — Recent + Saved placeholders (S7 wires) ]
 *
 * Slices S4 (result-set operator bar) and S5 (enriched-answer hero) attach
 * their surfaces to this component; the wrapped SearchPanel is the query
 * surface source of truth.
 */
import { computed } from 'vue'
import { useUserPreferencesStore } from '../../stores/userPreferences'
import type { SearchHit } from '../../api/searchApi'
import SearchPanel from './SearchPanel.vue'

const emit = defineEmits<{
  'go-graph': []
  'open-library-episode': [payload: { metadata_relative_path: string }]
  'open-episode-summary': [hit: SearchHit]
}>()

/**
 * S7 placeholder: WorkspaceSidebar Recent section reads from USERPREFS-1
 * key ``search.recentQueries`` when it lands. Today the key is unset in most
 * corpora so the section shows the honest empty state.
 */
const userPrefs = useUserPreferencesStore()
const recentQueries = computed<Array<{ q: string; ts?: number }>>(() => {
  const raw = userPrefs.get<Array<{ q: string; ts?: number }>>('search.recentQueries')
  return Array.isArray(raw) ? raw.slice(0, 20) : []
})
</script>

<template>
  <section
    class="grid h-full min-h-0 grid-cols-[minmax(0,1fr)_18rem] gap-3 overflow-hidden p-3"
    data-testid="search-workspace"
    aria-label="Query workspace"
  >
    <!-- Main workspace column — hosts the Search UI (SearchPanel is the source
         of truth for query field + filter chips + results). Slices S4/S5 mount
         the operator bar + enriched-answer hero above / around SearchPanel. -->
    <div class="flex min-h-0 min-w-0 flex-col overflow-hidden">
      <SearchPanel
        class="min-h-0 min-w-0 flex-1"
        @go-graph="emit('go-graph')"
        @open-library-episode="emit('open-library-episode', $event)"
        @open-episode-summary="emit('open-episode-summary', $event)"
      />
    </div>

    <!-- Sidebar — Saved + Recent (Search v3 §S7 will populate the Saved
         section from USERPREFS-1 ``search.savedQueries``; Recent already
         reads from USERPREFS-1 ``search.recentQueries`` where set). -->
    <aside
      class="flex min-h-0 min-w-0 flex-col gap-3 overflow-y-auto rounded-lg border border-border bg-surface p-3"
      data-testid="workspace-sidebar"
      aria-label="Saved and recent queries"
    >
      <section aria-labelledby="workspace-sidebar-saved-heading">
        <h2
          id="workspace-sidebar-saved-heading"
          class="text-[10px] font-semibold uppercase tracking-wider text-muted"
        >
          Saved
        </h2>
        <p
          class="mt-2 text-xs text-muted"
          data-testid="workspace-sidebar-saved-empty"
        >
          Saved queries land in slice S7 (USERPREFS-1 ``search.savedQueries``).
        </p>
      </section>

      <section aria-labelledby="workspace-sidebar-recent-heading">
        <h2
          id="workspace-sidebar-recent-heading"
          class="text-[10px] font-semibold uppercase tracking-wider text-muted"
        >
          Recent
        </h2>
        <ul
          v-if="recentQueries.length"
          class="mt-2 flex flex-col gap-1"
          data-testid="workspace-sidebar-recent-list"
        >
          <li
            v-for="(entry, i) in recentQueries"
            :key="`${entry.q}-${entry.ts ?? i}`"
            class="truncate text-xs text-surface-foreground"
            :title="entry.q"
          >
            {{ entry.q }}
          </li>
        </ul>
        <p
          v-else
          class="mt-2 text-xs text-muted"
          data-testid="workspace-sidebar-recent-empty"
        >
          No recent queries yet.
        </p>
      </section>
    </aside>
  </section>
</template>
