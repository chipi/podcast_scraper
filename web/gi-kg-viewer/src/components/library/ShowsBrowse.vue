<script setup lang="ts">
/**
 * ShowsBrowse (UXS-015 / RFC-104) — the Shows mode of the Library tab.
 *
 * Renders the shows grid; selecting a card opens that show in the **right subject
 * rail** (`subject.focusShow` → `ShowRailPanel`), consistent with how episodes
 * open — not replacing the grid in-panel. (The former in-panel `ShowDetailView`
 * is retired from this path; the file is kept but no longer rendered here.)
 */
import { useSubjectStore } from '../../stores/subject'
import type { CorpusFeedItem } from '../../api/corpusLibraryApi'
import ShowsView from './ShowsView.vue'

const subject = useSubjectStore()

function openShowInRail(feed: CorpusFeedItem): void {
  subject.focusShow(feed.feed_id, { uiTitle: feed.display_title?.trim() || feed.feed_id })
}
</script>

<template>
  <ShowsView @select="openShowInRail" />
</template>
