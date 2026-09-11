<script setup lang="ts">
/**
 * OverflowMenu — the ONE `⋯` menu (UXS-014 "Item actions"). Where a surface has more actions than
 * fit inline, the primaries stay inline and the rest go here: one extra tap, never a dropped
 * capability. This is the single overflow pattern; do not hand-roll another (same rule as `Tabs`).
 *
 * Mechanics, and why:
 *  - **Teleported to `<body>` + fixed positioning from the trigger rect.** Escapes clipped/
 *    transformed ancestors and avoids the stacking-context z-index hack a card-local `absolute`
 *    popup needs (a sibling card's controls otherwise paint over an open menu). z-50 per UXS-014.
 *  - **Slot-composed.** Callers drop their own action buttons (mark-as-played, share, add-note, an
 *    add-to-collection control) in the default slot; the menu owns the shell, a11y and dismissal and
 *    passes `close` so an item can dismiss after acting.
 *  - **a11y:** `aria-haspopup="menu"` / `aria-expanded` trigger; `role="menu"` panel; focus moves to
 *    the first item on open; ↑/↓/Home/End roam the items; Escape closes and restores focus to the
 *    trigger; an outside pointer, scroll, or resize closes it (a fixed panel would otherwise strand).
 */
import { nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

const props = withDefaults(defineProps<{ label?: string }>(), { label: '' })
const { t } = useI18n()

const open = ref(false)
const triggerEl = ref<HTMLElement | null>(null)
const panelEl = ref<HTMLElement | null>(null)
const pos = ref<{ top: number; right: number }>({ top: 0, right: 0 })

function items(): HTMLElement[] {
  const el = panelEl.value
  if (!el) return []
  return Array.from(el.querySelectorAll<HTMLElement>('[data-menuitem]'))
}

function place(): void {
  const el = triggerEl.value
  if (!el) return
  const r = el.getBoundingClientRect()
  // Hang from the trigger's right edge, below it. `right` is measured from the viewport's right so
  // the menu grows leftward and stays put without knowing its own width.
  pos.value = { top: Math.round(r.bottom + 4), right: Math.round(window.innerWidth - r.right) }
}

async function openMenu(): Promise<void> {
  place()
  open.value = true
  await nextTick()
  items()[0]?.focus()
}

function closeMenu(restoreFocus = true): void {
  if (!open.value) return
  open.value = false
  if (restoreFocus) triggerEl.value?.focus()
}

function toggle(): void {
  if (open.value) closeMenu()
  else void openMenu()
}

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
    closeMenu()
  }
}

// A pointer outside both trigger and panel dismisses. Registered only while open. The close is
// deferred to a microtask so we never mutate state (and re-patch the teleport) mid-dispatch of a
// native capture-phase event — Vue's own handlers already schedule this way.
function onDocPointer(e: PointerEvent): void {
  const target = e.target as Node
  if (triggerEl.value?.contains(target) || panelEl.value?.contains(target)) return
  queueMicrotask(() => closeMenu(false))
}
// A fixed panel would drift on scroll/resize; close rather than chase.
function onViewportChange(): void {
  queueMicrotask(() => closeMenu(false))
}
function bind(on: boolean): void {
  if (on) {
    document.addEventListener('pointerdown', onDocPointer as EventListener, true)
    window.addEventListener('scroll', onViewportChange, true)
    window.addEventListener('resize', onViewportChange)
  } else {
    document.removeEventListener('pointerdown', onDocPointer as EventListener, true)
    window.removeEventListener('scroll', onViewportChange, true)
    window.removeEventListener('resize', onViewportChange)
  }
}
// (De)register global dismiss listeners only while open.
watch(open, (v) => bind(v))
onBeforeUnmount(() => bind(false))

defineExpose({ close: () => closeMenu(false) })
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

  <Teleport to="body">
    <div
      v-if="open"
      ref="panelEl"
      role="menu"
      :aria-label="label || t('common.moreActions')"
      class="fixed z-50 min-w-44 max-w-[calc(100vw-1rem)] rounded-xl border border-border bg-surface p-1 shadow-lg"
      :style="{ top: `${pos.top}px`, right: `${pos.right}px` }"
      data-testid="overflow-menu"
      @keydown="onKeydown"
    >
      <!-- Callers pass their action buttons; each must carry `data-menuitem` (for keyboard roaming)
           and `role="menuitem"`. `close` dismisses the menu after an item acts. -->
      <slot :close="() => closeMenu(false)" />
    </div>
  </Teleport>
</template>
