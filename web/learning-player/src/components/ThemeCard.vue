<script setup lang="ts">
/**
 * Theme card — MODAL presentation of {@link ThemeView}, opened "on top" from a topic card's theme
 * link instead of navigating away. Mirrors {@link StorylineCard} exactly, because a theme and a
 * storyline are the same kind of object to a reader (a grouping of topics) and the gesture that
 * opens one must behave like the gesture that opens the other.
 *
 * Why a sheet rather than a route: the topic card can be rendered inside the Knowledge Panel, which
 * opens with `showModal()` and therefore sits in the browser's top layer. `router.push` from there
 * changes the page UNDERNEATH it and the tap looks completely dead — the defect that made the
 * storyline link route-free. A theme link added as a RouterLink would have reintroduced it.
 *
 * Unlike the storyline, the route param is the theme's OWN `tc:` id: a theme has a real endpoint,
 * so it does not have to be reconstructed from a member the way a storyline is from its anchor.
 */
import { onUnmounted, ref } from "vue"
import ThemeView from "../views/ThemeView.vue"
import { useModalSheet } from "../composables/useModalSheet"
import { registerStackedSheet, sheetTeleportTarget } from "../composables/sheetStack"

const props = withDefaults(
  defineProps<{
    id: string
    /** How many sheets this one is stacked ON TOP of. 0 = opened from a page, so it takes full height. */
    depth?: number
  }>(),
  { depth: 0 }
)
const emit = defineEmits<{ (e: "close"): void }>()

const dialogEl = ref<HTMLElement | null>(null)
useModalSheet(dialogEl, () => emit("close"), { key: "theme", value: () => props.id })

// A layered card pins the whole stack's geometry for as long as it is on screen.
if (props.depth > 0) {
  const release = registerStackedSheet()
  onUnmounted(release)
}

// A sheet opened from inside the Knowledge Panel's modal <dialog> must render INSIDE it, or the
// panel's top layer hides it completely. See sheetTeleportTarget().
const teleportTarget = sheetTeleportTarget()
</script>

<template>
  <Teleport :to="teleportTarget">
    <div class="lp-sheet-scrim" role="dialog" aria-modal="true" @click.self="emit('close')">
      <div
        ref="dialogEl"
        tabindex="-1"
        class="lp-sheet relative w-full max-w-lg lg:max-w-3xl overflow-hidden rounded-t-2xl bg-surface outline-none sm:rounded-2xl"
        :class="depth > 0 ? 'lp-sheet--stacked' : undefined"
        :style="{ '--lp-depth': depth }"
        data-testid="theme-card"
      >
        <!-- The ✕ rides ThemeView's header row (embedded), unified with the topic/person card. -->
        <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          <ThemeView :id="id" embedded :depth="depth" @close="emit('close')" />
        </div>
      </div>
    </div>
  </Teleport>
</template>
