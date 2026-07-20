<script setup lang="ts">
/**
 * LibraryTab (UXS-015 / RFC-104) — hosts the Library tab's two browse modes.
 *
 * A mode toggle switches between **Shows** (shows-first ``ShowsBrowse``, default)
 * and **Episodes** (the existing episode-first ``LibraryView``, UXS-003).
 *
 * View mode is now a proper Pinia store with USERPREFS-1 write-through so the
 * choice syncs across devices (was previously component-local localStorage —
 * see docs/wip/USERPREFS-1.md § "Not shipped" migration item, and stores/
 * libraryViewMode.ts for the store definition).
 */
import { computed } from 'vue'
import ShowsBrowse from './ShowsBrowse.vue'
import LibraryView from './LibraryView.vue'
import { useLibraryViewModeStore } from '../../stores/libraryViewMode'

const emit = defineEmits<{
  'focus-search': [
    payload: { feed: string; query: string; since?: string; feedDisplayTitle?: string },
  ]
  'switch-main-tab': [tab: 'digest' | 'library' | 'graph' | 'dashboard']
}>()

const libraryViewMode = useLibraryViewModeStore()
const mode = computed({
  get: () => libraryViewMode.mode,
  set: (v) => libraryViewMode.setMode(v),
})
</script>

<template>
  <div class="flex h-full min-h-0 flex-col">
    <div class="flex shrink-0 items-center border-b border-default px-3 py-1.5">
      <div class="inline-flex rounded bg-overlay p-0.5 text-xs font-semibold" role="group" aria-label="Library view mode">
        <button
          type="button"
          data-testid="library-mode-shows"
          :aria-pressed="mode === 'shows'"
          class="rounded px-2 py-1 outline-none focus-visible:ring-2 focus-visible:ring-primary"
          :class="mode === 'shows' ? 'bg-overlay-2 text-primary' : 'text-muted'"
          @click="mode = 'shows'"
        >
          Shows
        </button>
        <button
          type="button"
          data-testid="library-mode-episodes"
          :aria-pressed="mode === 'episodes'"
          class="rounded px-2 py-1 outline-none focus-visible:ring-2 focus-visible:ring-primary"
          :class="mode === 'episodes' ? 'bg-overlay-2 text-primary' : 'text-muted'"
          @click="mode = 'episodes'"
        >
          Episodes
        </button>
      </div>
    </div>
    <div class="min-h-0 flex-1">
      <keep-alive>
        <ShowsBrowse v-if="mode === 'shows'" />
        <LibraryView
          v-else
          class="h-full"
          @switch-main-tab="emit('switch-main-tab', $event)"
          @focus-search="emit('focus-search', $event)"
        />
      </keep-alive>
    </div>
  </div>
</template>
