<script setup lang="ts">
/**
 * Favorite (heart) toggle — the ONE shared affordance for saving any item (UXS-014: define once,
 * use everywhere). Renders for signed-out visitors too (#1590): saving requires auth, but hiding
 * the control hid the capability — tapping routes to sign-in and returns here. Stops click
 * propagation so it works on cards/links without triggering navigation.
 *
 * `controlled` variant: the parent owns the active state and the toggle side-effect. Used by
 * KnowledgePanel's insight save, which writes a highlight via the capture store (NOT the favorites
 * store — favorite(insight) is banned, #1593). The parent must pass `active` and listen for
 * `toggle`; `item` is optional (only the `label` field is used, for the accessible name).
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useFavoritesStore } from '../stores/favorites'
import { useSignInGate } from '../composables/useSignInGate'
import type { FavoriteAdd } from '../services/types'

const props = defineProps<{
  item?: FavoriteAdd
  /**
   * External active state for the `controlled` variant. Ignored by `icon` and `menuitem` variants,
   * which read state from the favorites store.
   */
  active?: boolean
  /**
   * Context label for the `controlled` variant's `sr-only` name. When provided, the accessible
   * name becomes e.g. "Save to favorites — Sleep consolidates memory." so multiple hearts on
   * one panel don't all announce identically (2026-09-25, Android device tier).
   * `icon` and `menuitem` variants ignore this; their items are disambiguated by surrounding context.
   */
  label?: string
  /**
   * `menuitem` renders this inside an `OverflowMenu` instead of as a standalone circular button.
   *
   * Used where the heart is true BY CONSTRUCTION and so carries no information as an icon — the
   * Saved list, where every row is saved by definition. There it was spending one of three slots in
   * a 128px column, which pushed the colour control onto a second row (operator 2026-09-16). It
   * still has to be REACHABLE, though: tapping it is the only way to unsave, so it moves into the ⋯
   * rather than disappearing. Mirrors DownloadButton / AddToCollectionButton's `menuitem` variant,
   * including `data-menuitem` so the menu's arrow-key roaming picks it up.
   *
   * `controlled` lets the parent own the active state and the toggle side-effect; the component
   * only renders correctly and announces correctly. Required when the item's persistence path is
   * not the favorites store (e.g. insight highlights, which go through the capture store).
   */
  variant?: 'icon' | 'menuitem' | 'controlled'
}>()

const emit = defineEmits<{ (e: 'toggle'): void }>()

const { t } = useI18n()
const favorites = useFavoritesStore()

// For `icon` / `menuitem`: derive state from the store. For `controlled`: the parent provides it.
const storeActive = computed(() => (props.item ? favorites.has(props.item.kind, props.item.ref) : false))
const isActive = computed(() => props.variant === 'controlled' ? (props.active ?? false) : storeActive.value)

const { isGated, gated } = useSignInGate()
// `item` is required for non-controlled variants; safe to assert here since the call branch
// only fires when `variant !== 'controlled'`, where the caller must provide `item`.
const storeToggle = gated(() => favorites.toggle(props.item!))

function onGatedClick(e: MouseEvent): void {
  e.preventDefault()
  e.stopPropagation()
  if (props.variant === 'controlled') {
    emit('toggle')
  } else {
    storeToggle()
  }
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
      <!-- The `label` prop appends context to the `sr-only` name in the `controlled` variant (e.g.
           the insight text), so several hearts on one panel don't all announce as the same string.
           `icon` / `menuitem` variants keep the short form; they live in card rows that already
           provide surrounding context. -->
      <span class="sr-only">{{
        isGated
          ? t('auth.signInToSave')
          : isActive
            ? (variant === 'controlled' && label ? `${t('fav.remove')} — ${label}` : t('fav.remove'))
            : (variant === 'controlled' && label ? `${t('fav.add')} — ${label}` : t('fav.add'))
      }}</span>
    </template>
  </button>
</template>
