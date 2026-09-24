<script setup lang="ts">
/**
 * OverflowMenu — the ONE `⋯` menu (UXS-014 "Item actions"). Where a surface has more actions than
 * fit inline, the primaries stay inline and the rest go here: one extra tap, never a dropped
 * capability. This is the single overflow pattern; do not hand-roll another (same rule as `Tabs`).
 *
 * Mechanics, and why:
 *  - **Shared popover shell (`useAnchoredMenu`).** Teleport-to-`<body>` + `position: fixed`, placed
 *    by `anchorPanel` so it is clamped inside the viewport and cannot open off a screen edge — the
 *    same shell the share and add-to-collection menus use, so all three position identically.
 *    Escapes clipped/transformed ancestors and the z-index hack a card-local `absolute` popup needs.
 *  - **Slot-composed.** Callers drop their own action buttons (mark-as-played, share, add-note, an
 *    add-to-collection control) in the default slot; the shell owns placement and dismissal and
 *    passes `close` so an item can dismiss after acting.
 *  - **a11y:** `aria-haspopup="menu"` / `aria-expanded` trigger; `role="menu"` panel; focus moves to
 *    the first item on open; ↑/↓/Home/End roam the items; Escape closes and restores focus to the
 *    trigger; an outside pointer dismisses, scroll/resize re-place the panel against the trigger.
 */
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useAnchoredMenu } from '../composables/useAnchoredMenu'

const props = withDefaults(defineProps<{ label?: string }>(), { label: '' })
const { t } = useI18n()

const triggerEl = ref<HTMLElement | null>(null)
const panelEl = ref<HTMLElement | null>(null)

function items(): HTMLElement[] {
  const el = panelEl.value
  if (!el) return []
  return Array.from(el.querySelectorAll<HTMLElement>('[data-menuitem]'))
}

// Shared popover shell (teleport + fixed + viewport-clamped placement + dismissal). This component
// adds only the menu-specific keyboard roaming; focus lands on the first item once placed.
const { open, toggle, close, teleportTarget } = useAnchoredMenu(triggerEl, panelEl, { align: 'end' }, {
  onOpened: () => items()[0]?.focus(),
})

function onKeydown(e: KeyboardEvent): void {
  const list = items()
  if (!list.length) return
  const i = list.indexOf(document.activeElement as HTMLElement)
  if (e.key === 'ArrowDown') {
    e.preventDefault()
    list[(i + 1) % list.length]?.focus()
  } else if (e.key === 'ArrowUp') {
    e.preventDefault()
    list[(i - 1 + list.length) % list.length]?.focus()
  } else if (e.key === 'Home') {
    e.preventDefault()
    list[0]?.focus()
  } else if (e.key === 'End') {
    e.preventDefault()
    list[list.length - 1]?.focus()
  } else if (e.key === 'Escape') {
    e.preventDefault()
    close()
  }
}

defineExpose({ close: () => close(false) })
</script>

<template>
  <button
    ref="triggerEl"
    type="button"
    class="lp-tap flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border text-muted transition hover:text-canvas-foreground"
    :class="{ 'text-canvas-foreground': open }"
    aria-haspopup="menu"
    :aria-expanded="open"
    :aria-label="label || t('common.moreActions')"
    :title="label || t('common.moreActions')"
    data-testid="overflow-trigger"
    @click.stop.prevent="toggle"
  >
    <svg viewBox="0 0 24 24" fill="currentColor" class="h-4 w-4" aria-hidden="true">
      <circle cx="5" cy="12" r="2" /><circle cx="12" cy="12" r="2" /><circle cx="19" cy="12" r="2" />
    </svg>
  </button>

  <Teleport :to="teleportTarget">
    <div
      v-if="open"
      ref="panelEl"
      role="menu"
      :aria-label="label || t('common.moreActions')"
      class="invisible fixed left-0 top-0 z-50 min-w-44 max-w-[calc(100vw-1rem)] rounded-xl border border-border bg-surface p-1 shadow-lg"
      data-testid="overflow-menu"
      @keydown="onKeydown"
    >
      <!-- Callers pass their action buttons; each must carry `data-menuitem` (for keyboard roaming)
           and `role="menuitem"`. `close` dismisses the menu after an item acts. -->
      <slot :close="() => close(false)" />
    </div>
  </Teleport>
</template>
