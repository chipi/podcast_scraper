<script setup lang="ts">
/**
 * Left query column — Saved + Recent queries (Search v3 §S4-shell pivot).
 * The compact SearchPanel launcher retired: search now lives only in the
 * main-window Search tab. The left rail becomes a dedicated Saved + Recent
 * queries surface backed by USERPREFS-1 (server-mirrored per-user prefs).
 *
 * Data:
 *   - ``search.savedQueries`` — list of {q, ts?, label?}. Written by the
 *     command palette's "Save this query" action (see
 *     ``paletteCommands.session.save-query`` +
 *     ``useSavedQueriesStore.saveQuery``).
 *   - ``search.recentQueries`` — list of {q, ts?}. Written by the search
 *     store after every successful ``runSearch``. Both keys live in
 *     USERPREFS-1 so the panel mirrors across devices.
 *
 * Interaction:
 *   Clicking a row emits ``apply-query`` with the query string; the App
 *   host switches the main tab to Search + runs the query (same code path
 *   the command palette uses for "Open in Workspace").
 */
import { computed } from 'vue'
import { useUserPreferencesStore } from '../../stores/userPreferences'

interface SavedEntry {
  q: string
  ts?: number
  label?: string
}

interface RecentEntry {
  q: string
  ts?: number
}

const emit = defineEmits<{
  'apply-query': [q: string]
}>()

const userPrefs = useUserPreferencesStore()

const savedQueries = computed<SavedEntry[]>(() => {
  const raw = userPrefs.get<SavedEntry[]>('search.savedQueries')
  return Array.isArray(raw) ? raw : []
})

const recentQueries = computed<RecentEntry[]>(() => {
  const raw = userPrefs.get<RecentEntry[]>('search.recentQueries')
  return Array.isArray(raw) ? raw.slice(0, 20) : []
})

function apply(q: string): void {
  const term = q.trim()
  if (!term) return
  emit('apply-query', term)
}
</script>

<template>
  <aside
    class="flex min-h-0 min-w-0 flex-1 flex-col gap-3 overflow-y-auto bg-surface p-3"
    data-testid="left-panel-saved-queries"
    aria-label="Saved and recent queries"
  >
    <section aria-labelledby="left-panel-saved-heading">
      <h2
        id="left-panel-saved-heading"
        class="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted"
      >
        Saved
      </h2>
      <ul
        v-if="savedQueries.length"
        class="flex flex-col gap-1"
        data-testid="left-panel-saved-list"
      >
        <li v-for="(entry, i) in savedQueries" :key="`${entry.q}-${entry.ts ?? i}`">
          <button
            type="button"
            class="w-full truncate rounded px-2 py-1.5 text-left text-xs text-surface-foreground hover:bg-overlay"
            :title="entry.label ?? entry.q"
            @click="apply(entry.q)"
          >
            {{ entry.label ?? entry.q }}
          </button>
        </li>
      </ul>
      <p
        v-else
        class="text-xs text-muted"
        data-testid="left-panel-saved-empty"
      >
        No saved queries yet. Open the command palette (Cmd-K) and pick "Save this query" to keep one here.
      </p>
    </section>

    <section aria-labelledby="left-panel-recent-heading">
      <h2
        id="left-panel-recent-heading"
        class="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted"
      >
        Recent
      </h2>
      <ul
        v-if="recentQueries.length"
        class="flex flex-col gap-1"
        data-testid="left-panel-recent-list"
      >
        <li v-for="(entry, i) in recentQueries" :key="`${entry.q}-${entry.ts ?? i}`">
          <button
            type="button"
            class="w-full truncate rounded px-2 py-1.5 text-left text-xs text-surface-foreground hover:bg-overlay"
            :title="entry.q"
            @click="apply(entry.q)"
          >
            {{ entry.q }}
          </button>
        </li>
      </ul>
      <p
        v-else
        class="text-xs text-muted"
        data-testid="left-panel-recent-empty"
      >
        No recent queries yet.
      </p>
    </section>
  </aside>
</template>
