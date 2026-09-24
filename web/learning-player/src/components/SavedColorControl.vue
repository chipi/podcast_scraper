<script setup lang="ts">
/**
 * A compact colour control (RFC-121 ph. 4 / #2042) shared by every colour-markable saved surface —
 * highlights and, since B, favourited episodes + entities — so the gesture is identical everywhere
 * (UXS-014 "define once, apply app-wide").
 *
 * A single current-colour dot (an empty ring when unset) that expands the fixed palette on tap. The
 * dot keeps a 32px ring with `.lp-tap` growing the finger target to 44px (#1594), like the other
 * card actions; the expanded swatches are full 44px buttons with an inner dot — a 24px pitch cannot
 * hold 44px targets, so the button grows and the ink does not.
 *
 * The palette is a TELEPORTED, viewport-clamped popover via the shared shell (useAnchoredMenu), like
 * add-to-collection and share. It used to be a bespoke `absolute right-0` panel, which — with the
 * dot sitting in a card's LEFT action cluster — ran the ~220px palette off the LEFT edge of the
 * screen where it could not be tapped (#2042 mobile regression). One rule for every menu now.
 */
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { HIGHLIGHT_COLORS, swatchClass } from '../utils/highlightColors'
import { useAnchoredMenu } from '../composables/useAnchoredMenu'

const props = defineProps<{ color: string | null | undefined }>()
const emit = defineEmits<{ pick: [token: string | null] }>()
const { t } = useI18n()

const triggerEl = ref<HTMLElement | null>(null)
const panelEl = ref<HTMLElement | null>(null)
const { open, toggle, close, teleportTarget } = useAnchoredMenu(triggerEl, panelEl, { align: 'end' })

/** Tapping the active colour clears it; the picker closes on any pick. */
function pick(token: string): void {
  emit('pick', props.color === token ? null : token)
  close(false)
}
</script>

<template>
  <div class="relative inline-flex items-center">
    <button
      ref="triggerEl"
      type="button"
      data-testid="saved-color"
      class="lp-tap flex h-8 w-8 items-center justify-center rounded-full transition hover:bg-overlay"
      :aria-label="t('highlights.colorPick')"
      aria-haspopup="true"
      :aria-expanded="open"
      @click.stop="toggle"
    >
      <span
        class="h-4 w-4 rounded-full"
        :class="swatchClass(color) || 'border border-border'"
      />
    </button>
    <Teleport :to="teleportTarget">
      <div
        v-if="open"
        ref="panelEl"
        class="invisible fixed left-0 top-0 z-50 flex items-center rounded-xl border border-border bg-surface p-1 shadow-lg"
        data-testid="saved-color-menu"
        @click.stop
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
    </Teleport>
  </div>
</template>
