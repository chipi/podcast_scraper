<script setup lang="ts">
/**
 * The standard episode action row — favorite, queue, download, add-to-collection — the set every
 * episode surface shows (UXS-014: define once, use everywhere). Before this, each surface
 * hand-rolled its own subset, so rails were missing download and Home's What's-new / Recommended
 * were missing favorite and download.
 *
 * Scope + order (kept identical everywhere so grid and list never differ — the action count must
 * NOT change with the type of view, operator 2026-09-13):
 *  - **Favorite** (heart) and **Queue** — primary, always inline.
 *  - **Download** and **Add-to-collection** — secondary, collapsed behind a **⋯ overflow**.
 *
 * Why the ⋯ (operator 2026-09-13): four inline controls are 176px of 44px targets, which wraps to a
 * second row under the artwork-width column of the list/grid card (the reported "fourth option on a
 * new row"). Shrinking the targets to fit is barred by the 44px floor (#1594) + the pitch guard, so
 * the fix is to collapse, not shrink. Two inline + ⋯ is 120px and fits one row. This also makes the
 * top-level count uniform at three across web AND native (download self-hides on web via
 * `DownloadButton`'s `v-if="native"`, so on web the ⋯ carries collection alone), rather than the old
 * web-3 / native-4 split — and it mirrors the player masthead's ⋯.
 *
 * `gap-[12px]` (NOT `gap-3`) holds the 32px targets at a 44px pitch exactly — the app root is not
 * 16px, so `gap-3` measured 11.4px on a Pixel 7 and the invisible tap boxes overlapped by 0.6px
 * (touch-affordances guard). `flex-wrap` is retained as a no-op safety: the full card now fits in one
 * row, but the compact `w-20` queue card (80px) still can't hold three targets, so the ⋯ folds there
 * rather than overflowing into the text. Each button stops its own click propagation, so the row is
 * safe inside a card/tile whose body is a link.
 */
import FavoriteButton from './FavoriteButton.vue'
import DownloadButton from './DownloadButton.vue'
import QueueButton from './QueueButton.vue'
import AddToCollectionButton from './AddToCollectionButton.vue'
import OverflowMenu from './OverflowMenu.vue'
import { useI18n } from 'vue-i18n'

defineProps<{ slug: string }>()
const { t } = useI18n()
</script>

<template>
  <div class="flex flex-wrap items-center gap-[12px]" data-testid="episode-actions">
    <FavoriteButton :item="{ kind: 'episode', ref: slug }" />
    <QueueButton :slug="slug" />
    <OverflowMenu :label="t('common.moreActions')">
      <template #default="{ close }">
        <!-- Download self-hides on web, so on web this menu carries collection alone. -->
        <DownloadButton :slug="slug" variant="menuitem" @activated="close" />
        <AddToCollectionButton :item="{ kind: 'episode', ref: slug }" variant="menuitem" />
      </template>
    </OverflowMenu>
    <!-- Extra, surface-specific controls in the same row (e.g. the queue's reorder ↑/↓). -->
    <slot />
  </div>
</template>
