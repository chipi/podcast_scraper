<script setup lang="ts">
/**
 * A show's actions behind ONE ⋯ — Follow, Save, Add to board (operator 2026-10-05).
 *
 * Search lists shows above episodes, and its episode cards carry a single ⋯ to the right of the
 * title. The show rows had no controls at all, so the two kinds on one page acted differently. This
 * puts every show action in the same ⋯, in the same place, so a show reads like an episode.
 *
 * Owns its Follow toggle (gated, like ShowTile) so a surface drops it in without wiring.
 */
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useSignInGate } from '../composables/useSignInGate'
import type { Podcast } from '../services/types'
import { useLibraryStore } from '../stores/library'
import AddToCollectionButton from './AddToCollectionButton.vue'
import FavoriteButton from './FavoriteButton.vue'
import FollowButton from './FollowButton.vue'
import OverflowMenu from './OverflowMenu.vue'

const props = defineProps<{ show: Podcast }>()
const { t } = useI18n()

const library = useLibraryStore()
const { isGated, gated } = useSignInGate()
const following = computed(() => library.has(props.show.feed_id))
const busy = ref(false)
const label = computed(() => props.show.title ?? props.show.feed_id)

// Gated for the reason ShowTile gives: the store reverts optimistically, so an ungated signed-out
// tap would flip, fire a 401 and flip back.
const toggleFollow = gated(async () => {
  busy.value = true
  try {
    await library.toggle(props.show.feed_id, { title: props.show.title })
  } finally {
    busy.value = false
  }
})
</script>

<template>
  <OverflowMenu :label="t('common.moreActions')">
    <template #default="{ close }">
      <FollowButton
        variant="menuitem"
        :following="following"
        :busy="busy"
        :gated="isGated"
        @toggle="toggleFollow(); close()"
      />
      <FavoriteButton :item="{ kind: 'show', ref: show.feed_id, label }" variant="menuitem" />
      <AddToCollectionButton :item="{ kind: 'show', ref: show.feed_id, title: label }" variant="menuitem" />
    </template>
  </OverflowMenu>
</template>
