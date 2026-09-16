<script setup lang="ts">
/**
 * Storyline card — MODAL presentation of {@link StorylineView} (the theme-cluster page), opened
 * "on top" from a topic card's storyline link instead of navigating away. Mirrors
 * {@link EntityCard} exactly: teleported to <body> so it escapes any clipped/transformed ancestor,
 * modal a11y (role/aria-modal + focus trap + restore focus + ESC/backdrop dismiss), and a
 * `?storyline=<anchorTopicId>` history entry so hardware/browser Back closes the card rather than
 * navigating the page underneath (see EntityCard's header for the full rationale of the three close
 * paths and why we go through the router, not raw pushState).
 *
 * The route param IS the anchor topic id; a storyline has no dedicated endpoint, so StorylineView
 * reconstructs the whole theme cluster from any member topic's card. The topic card passes its own
 * id, which is a member, so the same storyline resolves.
 */
import { ref } from "vue"
import StorylineView from "../views/StorylineView.vue"
import { useModalSheet } from "../composables/useModalSheet"

const props = withDefaults(
  defineProps<{
    id: string
    /**
     * Opened from ANOTHER sheet (a topic card), so it sits lower and lets the parent's kicker +
     * title stay visible. Opened from a page (Home, Discover) it is the only sheet on screen and
     * takes the full height — applying the stacked height there just made it short for no reason,
     * and made a parent+child pair exactly the same size so neither could peek (operator
     * 2026-09-16).
     */
    stacked?: boolean
  }>(),
  { stacked: false }
)
const emit = defineEmits<{ (e: "close"): void }>()

const dialogEl = ref<HTMLElement | null>(null)
useModalSheet(dialogEl, () => emit("close"), { key: "storyline", value: () => props.id })
</script>

<template>
  <Teleport to="body">
    <div class="lp-sheet-scrim" role="dialog" aria-modal="true" @click.self="emit('close')">
      <div
        ref="dialogEl"
        tabindex="-1"
        class="lp-sheet relative w-full max-w-lg overflow-hidden rounded-t-2xl bg-surface outline-none sm:rounded-2xl"
        :class="stacked ? 'lp-sheet--stacked' : undefined"
        data-testid="storyline-card"
      >
        <!-- The ✕ now rides StorylineView's action row (embedded), unified with the topic/person
             card — so it no longer floats over the header content. StorylineView emits `close`. -->
        <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          <StorylineView :id="id" embedded @close="emit('close')" />
        </div>
      </div>
    </div>
  </Teleport>
</template>
