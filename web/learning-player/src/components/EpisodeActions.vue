<script setup lang="ts">
/**
 * The standard episode action row — favorite, download, queue — the MINIMUM set every episode
 * surface shows (UXS-014: define once, use everywhere). Before this, each surface hand-rolled its
 * own subset, so rails were missing download and Home's What's-new / Recommended were missing
 * favorite and download.
 *
 * Scope of the minimum:
 *  - **Favorite** (heart) and **Queue** — always, every surface.
 *  - **Download** — always included; it self-hides on web (`DownloadButton` is `v-if="native"`),
 *    so on the web PWA the row is favorite + queue and on native it is all three. Native-only by
 *    design, not omitted per-surface.
 *  - **Add-to-collection** is NOT here — it belongs on detail surfaces only, placed explicitly.
 *
 * `gap-3` holds the 32px targets at a non-overlapping pitch. Each button stops its own click
 * propagation, so the row is safe inside a card/tile whose body is a link.
 */
import FavoriteButton from './FavoriteButton.vue'
import DownloadButton from './DownloadButton.vue'
import QueueButton from './QueueButton.vue'

defineProps<{ slug: string }>()
</script>

<template>
  <div class="flex items-center gap-3">
    <FavoriteButton :item="{ kind: 'episode', ref: slug }" />
    <DownloadButton :slug="slug" />
    <QueueButton :slug="slug" />
  </div>
</template>
