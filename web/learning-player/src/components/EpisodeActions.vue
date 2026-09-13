<script setup lang="ts">
/**
 * The standard episode action row — favorite, queue, download, add-to-collection — the set every
 * episode surface shows (UXS-014: define once, use everywhere). Before this, each surface
 * hand-rolled its own subset, so rails were missing download and Home's What's-new / Recommended
 * were missing favorite and download.
 *
 * Scope + order (kept identical to EpisodeCard so grid and list never differ — the action count
 * must NOT change with the type of view, operator 2026-09-13):
 *  - **Favorite** (heart) and **Queue** — always, every surface.
 *  - **Download** — self-hides on web (`DownloadButton` is `v-if="native"`), so the web PWA row is
 *    favorite + queue + collection and native adds download. Native-only by design, not per-surface.
 *  - **Add-to-collection** — included here now (RFC-119). It was previously "detail surfaces only",
 *    which is exactly what made the browse GRID show 3 while the LIST showed 4.
 *
 * `gap-[12px]` (NOT `gap-3`) holds the 32px targets at a 44px pitch exactly — the app root is not
 * 16px, so `gap-3` measured 11.4px on a Pixel 7 and the invisible tap boxes overlapped by 0.6px
 * (touch-affordances guard). `flex-wrap` lets the row fold when a caller constrains its width (e.g.
 * the list card's narrow left column); it is a no-op where there is room. Each button stops its own
 * click propagation, so the row is safe inside a card/tile whose body is a link.
 */
import FavoriteButton from './FavoriteButton.vue'
import DownloadButton from './DownloadButton.vue'
import QueueButton from './QueueButton.vue'
import AddToCollectionButton from './AddToCollectionButton.vue'

defineProps<{ slug: string }>()
</script>

<template>
  <div class="flex flex-wrap items-center gap-[12px]" data-testid="episode-actions">
    <FavoriteButton :item="{ kind: 'episode', ref: slug }" />
    <QueueButton :slug="slug" />
    <DownloadButton :slug="slug" />
    <AddToCollectionButton :item="{ kind: 'episode', ref: slug }" />
    <!-- Extra, surface-specific controls in the same row (e.g. the queue's reorder ↑/↓). -->
    <slot />
  </div>
</template>
