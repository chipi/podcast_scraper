<script setup lang="ts">
/**
 * Episode-scope chip — Search v3 §S6 (RFC-107 §S6). Visible ONLY when
 * ``search.filters.episodeId`` is set (i.e. after the
 * "Search within episode" rail launcher has fired). Renders the exact
 * ``episode_id`` on-screen so the user can see the scope is active, and
 * clicking the chip clears the filter to widen the search back out.
 *
 * Unlike the Topic / Speaker chips, this chip has NO popover — episode
 * scope is a corpus-stable id set by the rail, not a substring the user
 * types. When the filter is empty, the chip does not render (avoids
 * chip-bar noise on the default state).
 */
import { computed } from 'vue'
import { useSearchStore } from '../../../stores/search'

const search = useSearchStore()

const isActive = computed(() => Boolean(search.filters.episodeId?.trim()))

function clear(): void {
  search.filters.episodeId = ''
}
</script>

<template>
  <button
    v-if="isActive"
    type="button"
    class="inline-flex h-6 items-center rounded border border-primary bg-primary px-2 text-[11px] leading-none text-primary-foreground transition-colors hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
    data-testid="search-chip-episode"
    :aria-label="`Episode scope active: ${search.filters.episodeId}. Click to clear.`"
    :title="`Search results scoped to episode ${search.filters.episodeId}. Click to clear the scope.`"
    @click="clear"
  >
    Episode ✕
  </button>
</template>
