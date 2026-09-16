<script setup lang="ts">
/**
 * Favorite (heart) toggle — the ONE shared affordance for saving any item (UXS-014: define once,
 * use everywhere). Renders for signed-out visitors too (#1590): saving requires auth, but hiding
 * the control hid the capability — tapping routes to sign-in and returns here. Stops click
 * propagation so it works on cards/links without triggering navigation.
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

const active = computed(() => favorites.has(props.item.kind, props.item.ref))

const { isGated, gated } = useSignInGate()
const toggle = gated(() => favorites.toggle(props.item))

function onGatedClick(e: MouseEvent): void {
  e.preventDefault()
  e.stopPropagation()
  toggle()
}
</script>

<template>
  <!-- Rendered signed-out too (#1590) — see useSignInGate. -->
  <button
    type="button"
    :class="
      variant === 'menuitem'
        ? 'flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-canvas-foreground transition hover:bg-overlay'
        : 'lp-fav lp-tap h-8 w-8 shrink-0 rounded-full border border-border text-base'
    "
    :data-menuitem="variant === 'menuitem' ? '' : undefined"
    :role="variant === 'menuitem' ? 'menuitem' : undefined"
    :aria-pressed="isGated || variant === 'menuitem' ? undefined : active"
    :aria-label="isGated ? t('auth.signInToSave') : active ? t('fav.remove') : t('fav.add')"
    data-testid="favorite-button"
    @click="onGatedClick"
  >
    <template v-if="variant === 'menuitem'">
      <span class="lp-fav shrink-0 text-base" :class="{ 'lp-fav--on': active }" aria-hidden="true">{{
        active ? '♥' : '♡'
      }}</span>
      <span>{{ active ? t('fav.remove') : t('fav.add') }}</span>
    </template>
    <template v-else>{{ active ? '♥' : '♡' }}</template>
  </button>
</template>
