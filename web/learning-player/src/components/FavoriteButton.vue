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

const props = defineProps<{ item: FavoriteAdd }>()
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
    class="lp-fav text-lg"
    :class="{ 'lp-fav--on': active }"
    :aria-pressed="isGated ? undefined : active"
    :aria-label="isGated ? t('auth.signInToSave') : active ? t('fav.remove') : t('fav.add')"
    @click="onGatedClick"
  >{{ active ? '♥' : '♡' }}</button>
</template>
