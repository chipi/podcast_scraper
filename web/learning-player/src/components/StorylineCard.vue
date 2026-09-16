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
import { onUnmounted, ref } from "vue"
import StorylineView from "../views/StorylineView.vue"
import { useModalSheet } from "../composables/useModalSheet"
import { registerStackedSheet, sheetTeleportTarget } from "../composables/sheetStack"

const props = withDefaults(
  defineProps<{
    id: string
    /**
     * How many sheets this one is stacked ON TOP of. 0 = opened from a page (Home, Discover) and
     * the only sheet on screen, so it takes the full height — applying a stacked height there made
     * it short for no reason. Each level above 0 sits one peek lower, so the card beneath keeps its
     * kicker + title visible and a deep chain reads as a deck (operator 2026-09-16).
     */
    depth?: number
  }>(),
  { depth: 0 }
)
const emit = defineEmits<{ (e: "close"): void }>()

const dialogEl = ref<HTMLElement | null>(null)
useModalSheet(dialogEl, () => emit("close"), { key: "storyline", value: () => props.id })

// A layered card pins the whole stack's geometry for as long as it is on screen.
if (props.depth > 0) {
  const release = registerStackedSheet()
  onUnmounted(release)
}

// Resolved at mount: a sheet opened from inside the Knowledge Panel's modal <dialog> must render
// INSIDE it, or the panel's top layer hides it completely. See sheetTeleportTarget().
const teleportTarget = sheetTeleportTarget()
</script>

<template>
  <Teleport :to="teleportTarget">
    <div class="lp-sheet-scrim" role="dialog" aria-modal="true" @click.self="emit('close')">
      <div
        ref="dialogEl"
        tabindex="-1"
        class="lp-sheet relative w-full max-w-lg overflow-hidden rounded-t-2xl bg-surface outline-none sm:rounded-2xl"
        :class="depth > 0 ? 'lp-sheet--stacked' : undefined"
        :style="{ '--lp-depth': depth }"
        data-testid="storyline-card"
      >
        <!-- The ✕ now rides StorylineView's action row (embedded), unified with the topic/person
             card — so it no longer floats over the header content. StorylineView emits `close`. -->
        <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          <StorylineView :id="id" embedded :depth="depth" @close="emit('close')" />
        </div>
      </div>
    </div>
  </Teleport>
</template>
