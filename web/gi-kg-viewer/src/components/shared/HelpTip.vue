<script setup lang="ts">
import { computed, nextTick, onUnmounted, ref, toRef, watch } from 'vue'

const DEFAULT_BUTTON_CLASS =
  'inline-flex h-4 w-4 items-center justify-center rounded-full border border-border text-[10px] font-bold leading-none text-muted hover:bg-overlay hover:text-surface-foreground'

const props = withDefaults(
  defineProps<{
    /** Popover max width hint (px); wider panels fit troubleshooting copy. */
    prefWidth?: number
    buttonAriaLabel?: string
    /** Trigger label (one character is typical: ``?`` help, ``E`` ids). */
    buttonText?: string
    /**
     * Optional full Tailwind class string for the trigger button.
     * When set (e.g. canvas chip next to **E**), replaces the default round **?** style.
     */
    buttonClass?: string
    /**
     * Where to mount the popover (Teleport ``to``). Default ``body``. Pass the parent
     * ``<dialog>`` element when this trigger lives inside ``showModal()`` so the panel
     * mounts inside the dialog subtree (top layer) instead of ``body`` (where it would
     * sit behind the modal).
     */
    teleportTo?: string | Element | null
  }>(),
  {
    prefWidth: 256,
    buttonAriaLabel: 'Help',
    buttonText: '?',
    teleportTo: null,
  },
)

const prefWidthPx = toRef(props, 'prefWidth')

const teleportTarget = computed((): string | Element => {
  const t = props.teleportTo
  if (t == null) {
    return 'body'
  }
  if (typeof t === 'string' && !t.trim()) {
    return 'body'
  }
  return t
})

const open = ref(false)
const triggerRef = ref<HTMLButtonElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)

const panelStyle = ref<Record<string, string>>({
  top: '0px',
  left: '0px',
  maxWidth: '256px',
  maxHeight: 'min(70vh, 24rem)',
})

const PAD = 8

function updatePosition(): void {
  const btn = triggerRef.value
  const panel = panelRef.value
  if (!btn) return

  const vw = window.innerWidth
  const vh = window.innerHeight
  const r = btn.getBoundingClientRect()
  const pref = Math.max(160, prefWidthPx.value)
  const maxW = Math.max(PAD * 2, Math.min(pref, vw - 2 * PAD))
  const maxH = Math.max(PAD * 2, Math.min(Math.floor(vh * 0.72), 384))

  panelStyle.value = {
    ...panelStyle.value,
    maxWidth: `${maxW}px`,
    maxHeight: `${maxH}px`,
  }

  if (!panel) return

  const box = panel.getBoundingClientRect()
  const w = box.width
  const h = box.height

  // Prefer to the right of the trigger; if it overflows, place to the left
  let left = r.right + PAD
  if (left + w > vw - PAD) {
    left = r.left - w - PAD
  }
  // Still overflowing (narrow viewport): pin inside viewport
  if (left + w > vw - PAD) {
    left = vw - w - PAD
  }
  left = Math.max(PAD, left)

  // Vertically align with trigger top, then clamp
  let top = r.top
  if (top + h > vh - PAD) {
    top = vh - h - PAD
  }
  top = Math.max(PAD, top)

  panelStyle.value = {
    top: `${top}px`,
    left: `${left}px`,
    maxWidth: `${maxW}px`,
    maxHeight: `${maxH}px`,
  }
}

function schedulePosition(): void {
  void nextTick(() => {
    updatePosition()
    requestAnimationFrame(() => {
      updatePosition()
      requestAnimationFrame(() => updatePosition())
    })
  })
}

function onDocPointerDown(ev: PointerEvent): void {
  const t = ev.target as Node
  if (triggerRef.value?.contains(t)) return
  if (panelRef.value?.contains(t)) return
  open.value = false
}

function toggle(): void {
  open.value = !open.value
}

watch(open, (v) => {
  if (v) {
    schedulePosition()
    document.addEventListener('pointerdown', onDocPointerDown, true)
    window.addEventListener('scroll', updatePosition, true)
    window.addEventListener('resize', updatePosition)
  } else {
    document.removeEventListener('pointerdown', onDocPointerDown, true)
    window.removeEventListener('scroll', updatePosition, true)
    window.removeEventListener('resize', updatePosition)
  }
})

onUnmounted(() => {
  document.removeEventListener('pointerdown', onDocPointerDown, true)
  window.removeEventListener('scroll', updatePosition, true)
  window.removeEventListener('resize', updatePosition)
})
</script>

<template>
  <span class="inline-flex">
    <button
      ref="triggerRef"
      type="button"
      :class="buttonClass ?? DEFAULT_BUTTON_CLASS"
      :aria-label="buttonAriaLabel"
      :aria-expanded="open"
      @click.stop="toggle"
    >
      {{ buttonText }}
    </button>
    <Teleport :to="teleportTarget">
      <div
        v-if="open"
        ref="panelRef"
        class="fixed z-[10000] min-w-0 select-text overflow-y-auto break-words rounded border border-border bg-elevated p-2.5 text-xs leading-relaxed text-elevated-foreground shadow-xl box-border"
        :style="panelStyle"
        role="tooltip"
      >
        <slot />
      </div>
    </Teleport>
  </span>
</template>
