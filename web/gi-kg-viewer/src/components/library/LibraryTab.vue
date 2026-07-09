<script setup lang="ts">
/**
 * LibraryTab (UXS-015 / RFC-104) — hosts the Library tab's two browse modes.
 *
 * A mode toggle switches between **Shows** (shows-first ``ShowsBrowse``, default)
 * and **Episodes** (the existing episode-first ``LibraryView``, UXS-003). The mode
 * persists per browser (localStorage, matching theme/shell stores). ``LibraryView``'s
 * events are forwarded verbatim so App.vue's wiring is unchanged.
 */
import { ref, watch } from 'vue'
import ShowsBrowse from './ShowsBrowse.vue'
import LibraryView from './LibraryView.vue'

type Mode = 'shows' | 'episodes'
const STORAGE_KEY = 'gikg.library.mode'

const emit = defineEmits<{
  'focus-search': [
    payload: { feed: string; query: string; since?: string; feedDisplayTitle?: string },
  ]
  'switch-main-tab': [tab: 'digest' | 'library' | 'graph' | 'dashboard']
}>()

function readMode(): Mode {
  try {
    return localStorage.getItem(STORAGE_KEY) === 'episodes' ? 'episodes' : 'shows'
  } catch {
    return 'shows'
  }
}

const mode = ref<Mode>(readMode())
watch(mode, (v) => {
  try {
    localStorage.setItem(STORAGE_KEY, v)
  } catch {
    /* storage unavailable — non-fatal */
  }
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
