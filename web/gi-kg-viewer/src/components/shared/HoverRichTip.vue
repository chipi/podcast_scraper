<script setup lang="ts">
import { computed, nextTick, onUnmounted, ref, useId, watch } from 'vue'

defineOptions({ inheritAttrs: false })

const props = withDefaults(
  defineProps<{
    /** When false, no hover panel (pass-through wrapper only). */
    active?: boolean
    /** Popover max width hint (px). */
    prefWidth?: number
  }>(),
  {
    active: true,
    prefWidth: 320,
  },
)

const uid = useId()
const tooltipId = `hover-rich-tip-${uid}`

const anchorRef = ref<HTMLElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)
const open = ref(false)

const panelStyle = ref<Record<string, string>>({
  top: '0px',
  left: '0px',
  maxWidth: '320px',
})

const PAD = 8
const SHOW_DELAY_MS = 360
const HIDE_DELAY_MS = 140

let showTimer: ReturnType<typeof setTimeout> | null = null
let hideTimer: ReturnType<typeof setTimeout> | null = null

function clearShowTimer(): void {
  if (showTimer) {
    clearTimeout(showTimer)
    showTimer = null
  }
}

function clearHideTimer(): void {
  if (hideTimer) {
    clearTimeout(hideTimer)
    hideTimer = null
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

function updatePosition(): void {
  const anchor = anchorRef.value
  const panel = panelRef.value
  const vw = window.innerWidth
  const vh = window.innerHeight
  const pref = Math.max(200, props.prefWidth)
  const maxW = Math.max(PAD * 2, Math.min(pref, vw - 2 * PAD))

  panelStyle.value = {
    ...panelStyle.value,
    maxWidth: `${maxW}px`,
  }

  if (!anchor) {
    return
  }
  const r = anchor.getBoundingClientRect()
  if (!panel) {
    return
  }

  const box = panel.getBoundingClientRect()
  const w = box.width
  const h = box.height

  let top = r.bottom + PAD
  if (top + h > vh - PAD) {
    top = Math.max(PAD, r.top - h - PAD)
  }

  let left = r.left + (r.width - w) / 2
  if (left + w > vw - PAD) {
    left = vw - w - PAD
  }
  left = Math.max(PAD, left)

  panelStyle.value = {
    top: `${top}px`,
    left: `${left}px`,
    maxWidth: `${maxW}px`,
  }
}

function openNow(): void {
  if (!props.active) {
    return
  }
  clearHideTimer()
  clearShowTimer()
  open.value = true
  schedulePosition()
}

function openDelayed(): void {
  if (!props.active) {
    return
  }
  clearHideTimer()
  clearShowTimer()
  showTimer = setTimeout(() => {
    showTimer = null
    open.value = true
    schedulePosition()
  }, SHOW_DELAY_MS)
}

function closeDelayed(): void {
  clearShowTimer()
  clearHideTimer()
  hideTimer = setTimeout(() => {
    hideTimer = null
    open.value = false
  }, HIDE_DELAY_MS)
}

function onAnchorPointerEnter(): void {
  if (!props.active) {
    return
  }
  openDelayed()
}

function onAnchorPointerLeave(): void {
  closeDelayed()
}

function onPanelPointerEnter(): void {
  clearHideTimer()
}

function onPanelPointerLeave(): void {
  closeDelayed()
}

function onAnchorFocusIn(ev: FocusEvent): void {
  if (!props.active) {
    return
  }
  if (!anchorRef.value?.contains(ev.target as Node)) {
    return
  }
  openNow()
}

function onAnchorFocusOut(ev: FocusEvent): void {
  const next = ev.relatedTarget as Node | null
  if (panelRef.value?.contains(next)) {
    return
  }
  closeDelayed()
}

function onEscape(ev: KeyboardEvent): void {
  if (ev.key !== 'Escape') {
    return
  }
  clearShowTimer()
  clearHideTimer()
  open.value = false
}

const describedBy = computed((): string | undefined => {
  if (!props.active || !open.value) {
    return undefined
  }
  return tooltipId
})

watch(open, (v) => {
  if (v) {
    document.addEventListener('keydown', onEscape, true)
    window.addEventListener('scroll', updatePosition, true)
    window.addEventListener('resize', updatePosition)
  } else {
    document.removeEventListener('keydown', onEscape, true)
    window.removeEventListener('scroll', updatePosition, true)
    window.removeEventListener('resize', updatePosition)
  }
})

watch(
  () => props.active,
  (v) => {
    if (!v) {
      clearShowTimer()
      clearHideTimer()
      open.value = false
    }
  },
)

onUnmounted(() => {
  clearShowTimer()
  clearHideTimer()
  document.removeEventListener('keydown', onEscape, true)
  window.removeEventListener('scroll', updatePosition, true)
  window.removeEventListener('resize', updatePosition)
})
</script>

<template>
  <div
    ref="anchorRef"
    v-bind="$attrs"
    :aria-describedby="describedBy"
    @pointerenter="onAnchorPointerEnter"
    @pointerleave="onAnchorPointerLeave"
    @focusin.capture="onAnchorFocusIn"
    @focusout.capture="onAnchorFocusOut"
  >
    <slot />
    <Teleport to="body">
      <div
        v-if="active && open"
        :id="tooltipId"
        ref="panelRef"
        class="fixed z-[10000] min-w-0 select-text overflow-visible break-words rounded-md border border-border bg-elevated px-2 py-1.5 text-xs leading-snug text-elevated-foreground shadow-xl box-border"
        :style="panelStyle"
        role="tooltip"
        @pointerenter="onPanelPointerEnter"
        @pointerleave="onPanelPointerLeave"
      >
        <slot name="panel" />
      </div>
    </Teleport>
  </div>
</template>
