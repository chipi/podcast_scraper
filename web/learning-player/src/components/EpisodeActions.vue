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
import ShareIcon from './ShareIcon.vue'
import OverflowMenu from './OverflowMenu.vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { useOnline } from '../composables/useOnline'
import { useDownloadsStore } from '../stores/downloads'
import { track } from '../services/analytics'
import { openShareSheet } from '../services/native'
import { copyText } from '../utils/clipboard'
import { shareUrl } from '../utils/shareLink'

defineProps<{
  slug: string
  /**
   * The episode has a Moments reel (insights): "Play moments" leads the ⋯ menu (operator
   * 2026-10-10) — one change reaches every card, tile, search result, Library and Queue row.
   */
  moments?: boolean
  /**
   * The row is sitting ON artwork rather than on the page background.
   *
   * Without this each control is a hairline border plus a muted glyph directly over a photo, so
   * legibility is left to whatever the artwork happens to be — over a light portrait they were
   * almost invisible (operator screenshot 2026-09-16). Each button gets its own shaded plate, so
   * contrast stops depending on the image underneath.
   *
   * Scoped to DIRECT-child buttons: all three controls are `<button>` roots, and the overflow PANEL
   * is teleported to `<body>`, so menu items can never inherit the plate.
   *
   * Text colour is deliberately NOT overridden — FavoriteButton and QueueButton signal their active
   * state through colour (`lp-fav--on`, `text-accent` when queued), and forcing white would flatten
   * "already saved / already queued" into "not".
   */
  overlay?: boolean
  /**
   * Drop the heart from the visible row, moving it into the ⋯ instead.
   *
   * For surfaces where "favourited" is true BY CONSTRUCTION — the Saved list — so the icon only
   * restates what the surface already says, while costing one of three slots in a 128px column and
   * wrapping the colour control onto a second row. Hiding it outright would remove the only way to
   * UNSAVE, so it becomes a menu item rather than disappearing (operator 2026-09-16).
   */
  hideFavorite?: boolean
  /**
   * Drop the queue toggle from the visible row, moving it into the ⋯ — the mirror of
   * `hideFavorite`, for the same reason.
   *
   * Set by the queue panel's **Recently played** list, where the point is to FIND and resume
   * something you heard, not to re-queue it (operator 2026-09-23). It also buys back a slot: the
   * compact card's column is 80px, which holds two targets, so with three the ⋯ wrapped onto its
   * own line under the artwork.
   */
  hideQueue?: boolean
  /**
   * Promote download OUT of the ⋯ and into the visible row.
   *
   * Set by **Up next**, where "is this on the device?" is the question you are asking — you are
   * looking at what you are about to play, possibly before losing signal. It was two taps and a
   * menu away, so the answer was invisible (operator 2026-09-23). Native-only by construction:
   * `DownloadButton` self-hides on web, so this adds nothing to a browser row.
   */
  showDownload?: boolean
  /**
   * Offer "Share" in the ⋯, sending the episode's public link (operator 2026-10-08: Saved's
   * episode groups had no way to share the episode). The title heads the share sheet. Opt-in, so the
   * rails' menus do not grow a control nobody asked for there.
   */
  shareTitle?: string
}>()
const { t } = useI18n()
const router = useRouter()
const { isOnline } = useOnline()
const downloads = useDownloadsStore()
function playMoments(slug: string): void {
  void router.push({ name: 'player', params: { slug }, query: { moments: '1' } })
}

/** The phone's share sheet with the episode link; where there is none, the link is copied. */
async function shareEpisode(slug: string, title: string): Promise<void> {
  const url = shareUrl('episode', slug)
  try {
    await openShareSheet(title, url)
    track('share', { target_kind: 'episode', method: 'native_sheet' })
  } catch {
    if (await copyText(url)) track('share', { target_kind: 'episode', method: 'copy_link' })
  }
}
</script>

<template>
  <div
    class="flex flex-wrap items-center gap-[12px]"
    :class="
      overlay
        ? '[&>button]:border-white/25 [&>button]:bg-black/55 [&>button]:shadow-lg'
        : undefined
    "
    data-testid="episode-actions"
  >
    <!-- Leading surface-specific control, taking the heart's place when it is hidden (Saved puts
         its colour picker here, so the row stays three wide and does not wrap). -->
    <slot name="lead" />
    <FavoriteButton v-if="!hideFavorite" :item="{ kind: 'episode', ref: slug }" />
    <QueueButton v-if="!hideQueue" :slug="slug" />
    <OverflowMenu :label="t('common.moreActions')">
      <template #default="{ close }">
        <button
          v-if="moments"
          type="button"
          role="menuitem"
          data-menuitem=""
          class="flex w-full flex-col items-start rounded-lg px-3 py-2 text-left text-sm font-bold transition"
          :class="isOnline || downloads.isDownloaded(slug) ? 'text-accent hover:bg-overlay' : 'text-disabled'"
          :disabled="!isOnline && !downloads.isDownloaded(slug)"
          data-testid="episode-play-moments"
          @click="close(); playMoments(slug)"
        >
          <span>▶ {{ t('moments.play_moments') }}</span>
          <span class="text-xs font-normal text-muted">{{
            isOnline || downloads.isDownloaded(slug) ? t('moments.play_moments_hint') : t('moments.needsConnection')
          }}</span>
        </button>
        <!-- Download self-hides on web, so on web this menu carries collection alone. Omitted when
             it is already inline — never offer the same control in two places. -->
        <DownloadButton v-if="!showDownload" :slug="slug" variant="menuitem" @activated="close" />
        <AddToCollectionButton :item="{ kind: 'episode', ref: slug }" variant="menuitem" />
        <!-- Only when it is NOT in the row above — never offer the same toggle in two places. -->
        <FavoriteButton
          v-if="hideFavorite"
          :item="{ kind: 'episode', ref: slug }"
          variant="menuitem"
        />
        <QueueButton v-if="hideQueue" :slug="slug" variant="menuitem" />
        <button
          v-if="shareTitle"
          type="button"
          role="menuitem"
          data-menuitem=""
          class="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-canvas-foreground transition hover:bg-overlay"
          data-testid="episode-share"
          @click="close(); shareEpisode(slug, shareTitle)"
        >
          <ShareIcon />{{ t('share.open') }}
        </button>
      </template>
    </OverflowMenu>
    <!-- Extra, surface-specific controls in the same row (e.g. the queue's reorder ↑/↓). -->
    <slot />
    <!-- Inline download LAST (operator 2026-09-23), so the two rows the artwork-width column wraps
         into read as a grid rather than as a row that overflowed: `♡ ⧉ ⋯` above `↑ ↓ ⬇`. Placed
         before the ⋯ it pushed the overflow onto the second line and left the first ending on a
         control that is not the "more" affordance, which is where the eye expects it. -->
    <DownloadButton v-if="showDownload" :slug="slug" />
  </div>
</template>
