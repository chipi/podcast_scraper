<script setup lang="ts">
/**
 * Horizontal card rail with the modern carousel affordance (UXS-012): native swipe on touch,
 * scroll-snap for a polished glide, and chevron page-controls that fade in on desktop — no
 * exposed scrollbar. Edge gradients hint there's more; the controls disable at each end.
 *
 * Mobile-first & a11y: the controls are `lg:`-only (touch users swipe); the track is keyboard
 * scrollable and the buttons carry i18n aria-labels. Pass cards via the default slot.
 */
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
const track = ref<HTMLUListElement | null>(null)
const atStart = ref(true)
const atEnd = ref(false)

function update(): void {
  const el = track.value
  if (!el) return
  atStart.value = el.scrollLeft <= 1
  atEnd.value = el.scrollLeft + el.clientWidth >= el.scrollWidth - 1
}

function page(dir: 1 | -1): void {
  const el = track.value
  if (!el) return
  el.scrollBy({ left: dir * Math.round(el.clientWidth * 0.85), behavior: 'smooth' })
}

let ro: ResizeObserver | null = null
onMounted(() => {
  update()
  track.value?.addEventListener('scroll', update, { passive: true })
  if (typeof ResizeObserver !== 'undefined' && track.value) {
    ro = new ResizeObserver(update)
    ro.observe(track.value)
  }
})
onBeforeUnmount(() => {
  track.value?.removeEventListener('scroll', update)
  ro?.disconnect()
})
</script>

<template>
  <div class="group relative">
    <!-- Left edge fade -->
    <div
      class="pointer-events-none absolute inset-y-0 left-0 z-10 w-10 bg-gradient-to-r from-canvas to-transparent transition-opacity"
      :class="atStart ? 'opacity-0' : 'opacity-100'"
    />
    <button
      type="button"
      class="absolute -left-3 top-[38%] z-20 hidden h-11 w-11 -translate-y-1/2 items-center justify-center rounded-full border border-border bg-surface text-2xl leading-none text-canvas-foreground shadow-lg transition hover:scale-105 hover:border-accent hover:bg-elevated sm:flex"
      :class="atStart ? 'pointer-events-none opacity-0' : 'opacity-100'"
      :aria-label="t('rail.prev')"
      :aria-disabled="atStart"
      @click="page(-1)"
    >
      ‹
    </button>

    <ul ref="track" class="lp-rail flex snap-x snap-mandatory gap-3 overflow-x-auto pb-2">
      <slot />
    </ul>

    <!-- Right edge fade -->
    <div
      class="pointer-events-none absolute inset-y-0 right-0 z-10 w-10 bg-gradient-to-l from-canvas to-transparent transition-opacity"
      :class="atEnd ? 'opacity-0' : 'opacity-100'"
    />
    <button
      type="button"
      class="absolute -right-3 top-[38%] z-20 hidden h-11 w-11 -translate-y-1/2 items-center justify-center rounded-full border border-border bg-surface text-2xl leading-none text-canvas-foreground shadow-lg transition hover:scale-105 hover:border-accent hover:bg-elevated sm:flex"
      :class="atEnd ? 'pointer-events-none opacity-0' : 'opacity-100'"
      :aria-label="t('rail.next')"
      :aria-disabled="atEnd"
      @click="page(1)"
    >
      ›
    </button>
  </div>
</template>

<style scoped>
/* Hide the native scrollbar — the chevron controls + edge fades are the affordance. */
.lp-rail {
  scrollbar-width: none;
  scroll-behavior: smooth;
}
.lp-rail::-webkit-scrollbar {
  display: none;
}
.lp-rail > :deep(*) {
  scroll-snap-align: start;
}
</style>
