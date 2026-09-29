<script setup lang="ts">
/**
 * Favorite (heart) toggle — the ONE shared affordance for saving any item (UXS-014: define once,
 * use everywhere). Renders for signed-out visitors too (#1590): saving requires auth, but hiding
 * the control hid the capability — tapping routes to sign-in and returns here. Stops click
 * propagation so it works on cards/links without triggering navigation.
 *
 * The heart means exactly ONE thing: a favourite, on a WHOLE object — an episode, a show, a topic,
 * a person. It is not the save mark for a fragment; that is the bookmark, and it lives in
 * `HighlightToggle`.
 *
 * There used to be a `controlled` variant, whose only caller was the Knowledge panel's insight
 * save. It rendered a heart while writing an insight HIGHLIGHT through the capture store — never
 * the favorites store, because favourite(insight) is banned (#1593) — so the heart meant a
 * favourite here and a highlight there, and announced "Save to favorites" either way. The operator
 * spotted it from the outside (2026-09-27: "we can favourite insights and bookmark parts of
 * transcript, feels inconsistent"). The variant is GONE rather than left available: as long as a
 * parent could own the state, the heart could quietly be reattached to a non-favourite store again,
 * which is the whole way this happened.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useFavoritesStore } from '../stores/favorites'
import { useSignInGate } from '../composables/useSignInGate'
import type { FavoriteAdd } from '../services/types'

const props = defineProps<{
  item: FavoriteAdd
  /**
   * `menuitem` renders this inside an `OverflowMenu` instead of as a standalone circular button.
   *
   * Used where the heart is true BY CONSTRUCTION and so carries no information as an icon — the
   * Saved list, where every row is saved by definition. There it was spending one of three slots in
   * a 128px column, which pushed the colour control onto a second row (operator 2026-09-16). It
   * still has to be REACHABLE, though: tapping it is the only way to unsave, so it moves into the ⋯
   * rather than disappearing. Mirrors DownloadButton / AddToCollectionButton's `menuitem` variant,
   * including `data-menuitem` so the menu's arrow-key roaming picks it up.
   */
  variant?: 'icon' | 'menuitem'
}>()

const { t } = useI18n()
const favorites = useFavoritesStore()

const isActive = computed(() => favorites.has(props.item.kind, props.item.ref))

const { isGated, gated } = useSignInGate()
const storeToggle = gated(() => favorites.toggle(props.item))

function onGatedClick(e: MouseEvent): void {
  e.preventDefault()
  e.stopPropagation()
  storeToggle()
}
</script>

<template>
  <!-- Rendered signed-out too (#1590) — see useSignInGate. -->
  <button
    type="button"
    :class="[
      variant === 'menuitem'
        ? 'flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-canvas-foreground transition hover:bg-overlay'
        : ['lp-fav lp-tap h-8 w-8 shrink-0 rounded-full border border-border text-base', { 'lp-fav--on': isActive }],
    ]"
    :data-menuitem="variant === 'menuitem' ? '' : undefined"
    :role="variant === 'menuitem' ? 'menuitem' : undefined"
    :aria-pressed="isGated || variant === 'menuitem' ? undefined : isActive"
    :aria-label="isGated ? t('auth.signInToSave') : isActive ? t('fav.remove') : t('fav.add')"
    data-testid="favorite-button"
    @click="onGatedClick"
  >
    <template v-if="variant === 'menuitem'">
      <span class="lp-fav shrink-0 text-base" :class="{ 'lp-fav--on': isActive }" aria-hidden="true">{{
        isActive ? '♥' : '♡'
      }}</span>
      <span>{{ isActive ? t('fav.remove') : t('fav.add') }}</span>
    </template>
    <!-- The glyph is DECORATION; the NAME sits beside it (2026-09-25, Android device tier).
         It was bare text, so Chromium used it as the button's accessible name: the control
         announced as "♡", and every name-based lookup — a screen reader's, a device test's — got a
         symbol instead of "Add to favourites". The `aria-label` above did not win.
         `AccessibleNameAuditTests` counts this class app-wide and fails on it. -->
    <template v-else>
      <span aria-hidden="true">{{ isActive ? '♥' : '♡' }}</span>
      <!-- The short form is enough here. The `label` disambiguation existed for the `controlled`
           variant, where many hearts sat on one panel; the remaining variants live in card rows
           that already provide surrounding context. `HighlightToggle` carries `label` for the case
           that needed it. -->
      <span class="sr-only">{{
        isGated ? t('auth.signInToSave') : isActive ? t('fav.remove') : t('fav.add')
      }}</span>
    </template>
  </button>
</template>
