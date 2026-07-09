<script setup lang="ts">
/**
 * ShowsBrowse (UXS-015 / RFC-104) — shows-first sub-state for the Library tab.
 *
 * ``null`` selectedFeed → grid (``ShowsView``); a selection → ``ShowDetailView``
 * for that feed (replace-in-panel, house pattern — no modal / second backdrop).
 */
import { ref } from 'vue'
import type { CorpusFeedItem } from '../../api/corpusLibraryApi'
import ShowsView from './ShowsView.vue'
import ShowDetailView from './ShowDetailView.vue'

const selectedFeed = ref<CorpusFeedItem | null>(null)
</script>

<template>
  <ShowDetailView
    v-if="selectedFeed"
    :feed="selectedFeed"
    @back="selectedFeed = null"
  />
  <ShowsView v-else @select="selectedFeed = $event" />
</template>
