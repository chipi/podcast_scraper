<script setup lang="ts">
/**
 * A compact colour control (RFC-121 ph. 4 / #2042) shared by every colour-markable saved surface —
 * highlights and, since B, favourited episodes + entities — so the gesture is identical everywhere
 * (UXS-014 "define once, apply app-wide").
 *
 * A single current-colour dot (an empty ring when unset) that expands the fixed palette inline on
 * tap. The dot keeps a 32px ring with `.lp-tap` growing the finger target to 44px (#1594), like the
 * other card actions; the expanded swatches are full 44px buttons with an inner dot — a 24px pitch
 * cannot hold 44px targets, so the button grows and the ink does not.
 */
import { onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { HIGHLIGHT_COLORS, swatchClass } from '../utils/highlightColors'

const props = defineProps<{ color: string | null | undefined }>()
const emit = defineEmits<{ pick: [token: string | null] }>()
const { t } = useI18n()

const rootEl = ref<HTMLElement | null>(null)
const open = ref(false)
function toggle(): void {
  open.value = !open.value
}
/** Tapping the active colour clears it; the picker closes on any pick. */
function pick(token: string): void {
  emit('pick', props.color === token ? null : token)
  open.value = false
}

// Dismiss the popover on outside-click or Escape (Fable-5 review nit) — listeners live only while
// open. Capture phase so an outside tap closes before it acts on whatever it hit.
function onDocPointer(e: Event): void {
  if (rootEl.value && !rootEl.value.contains(e.target as Node)) open.value = false
}
function onKey(e: KeyboardEvent): void {
  if (e.key === 'Escape') open.value = false
}
watch(open, (isOpen) => {
  const m = isOpen ? 'addEventListener' : 'removeEventListener'
  document[m]('click', onDocPointer, true)
  document[m]('keydown', onKey as EventListener)
})
onBeforeUnmount(() => {
  document.removeEventListener('click', onDocPointer, true)
  document.removeEventListener('keydown', onKey as EventListener)
})
</script>

<template>
  <div ref="rootEl" class="relative flex items-center">
    <button
      type="button"
      data-testid="saved-color"
      class="lp-tap flex h-8 w-8 items-center justify-center rounded-full transition hover:bg-overlay"
      :aria-label="t('highlights.colorPick')"
      :aria-expanded="open"
      @click="toggle"
    >
      <span
        class="h-4 w-4 rounded-full"
        :class="swatchClass(color) || 'border border-border'"
      />
    </button>
    <!-- The palette is an ABSOLUTE popover, not inline: expanding five 44px swatches in place would
         overflow the shared episode card's tight action cluster. Closes on pick or on re-tap. -->
    <div
      v-if="open"
      class="absolute right-0 top-full z-40 mt-1 flex items-center rounded-xl border border-border bg-surface p-1 shadow-lg"
    >
      <button
        v-for="c in HIGHLIGHT_COLORS"
        :key="c.token"
        type="button"
        data-testid="saved-swatch"
        class="flex h-11 w-11 items-center justify-center rounded-full transition"
        :aria-pressed="color === c.token"
        :aria-label="t('highlights.setColor', { color: t(c.labelKey) })"
        :title="t(c.labelKey)"
        @click="pick(c.token)"
      >
        <span
          class="h-3.5 w-3.5 rounded-full"
          :class="[c.swatch, color === c.token ? 'ring-2 ring-accent' : 'opacity-60']"
        />
      </button>
    </div>
  </div>
</template>
