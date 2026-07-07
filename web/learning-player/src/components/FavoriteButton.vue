<script setup lang="ts">
/**
 * Favorite (heart) toggle — the ONE shared affordance for saving any item (UXS-014: define once,
 * use everywhere). Saving requires auth, so it only renders when signed in. Stops click
 * propagation so it works on cards/links without triggering navigation.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useFavoritesStore } from '../stores/favorites'
import { useAuthStore } from '../stores/auth'
import type { FavoriteAdd } from '../services/types'

const props = defineProps<{ item: FavoriteAdd }>()
const { t } = useI18n()
const favorites = useFavoritesStore()
const auth = useAuthStore()

const active = computed(() => favorites.has(props.item.kind, props.item.ref))

function onClick(e: MouseEvent): void {
  e.preventDefault()
  e.stopPropagation()
  void favorites.toggle(props.item)
}
</script>

<template>
  <button
    v-if="auth.isAuthenticated"
    type="button"
    class="lp-fav text-lg"
    :class="{ 'lp-fav--on': active }"
    :aria-pressed="active"
    :aria-label="active ? t('fav.remove') : t('fav.add')"
    @click="onClick"
  >{{ active ? '♥' : '♡' }}</button>
</template>
