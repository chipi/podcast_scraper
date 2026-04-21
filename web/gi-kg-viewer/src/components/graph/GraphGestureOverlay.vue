<script setup lang="ts">
import { nextTick, onMounted, onUnmounted, ref, watch } from 'vue'

const STORAGE_KEY = 'ps_graph_hints_seen'

/** Expandable / expanded seed rings — matches `cyGraphStylesheet.ts`. */
const EXPAND_RING_TEAL = '#14b8a6'
const EXPAND_RING_BLUE = '#748ffc'

const props = defineProps<{ hasNodes: boolean }>()
const emit = defineEmits<{ dismissed: [] }>()

const manualOpen = ref(false)
const visible = ref(false)
const dismissButtonRef = ref<HTMLButtonElement | null>(null)
const overlayRootRef = ref<HTMLDivElement | null>(null)

function storageDismissed(): boolean {
  return typeof localStorage !== 'undefined' && localStorage.getItem(STORAGE_KEY) === '1'
}

let escListener: ((e: KeyboardEvent) => void) | null = null

function detachEsc(): void {
  if (escListener) {
    window.removeEventListener('keydown', escListener, true)
    escListener = null
  }
}

function attachEsc(): void {
  detachEsc()
  escListener = (e: KeyboardEvent) => {
    if (e.key !== 'Escape') return
    const root = overlayRootRef.value
    if (!root) return
    const ae = document.activeElement
    if (ae && ae !== document.body && !root.contains(ae)) {
      return
    }
    e.preventDefault()
    dismiss()
  }
  window.addEventListener('keydown', escListener, true)
}

watch(visible, async (v) => {
  if (!v) {
    detachEsc()
    return
  }
  await nextTick()
  attachEsc()
})

onUnmounted(detachEsc)

function shouldAutoShow(): boolean {
  return props.hasNodes && !storageDismissed()
}

function focusDismiss(): void {
  void nextTick(() => {
    requestAnimationFrame(() => dismissButtonRef.value?.focus())
  })
}

function tryAutoOpen(): void {
  if (manualOpen.value) return
  if (!shouldAutoShow()) {
    visible.value = false
    return
  }
  visible.value = true
  focusDismiss()
}

watch(
  () => props.hasNodes,
  (has) => {
    if (!has) {
      visible.value = false
      manualOpen.value = false
      return
    }
    tryAutoOpen()
  },
)

onMounted(() => {
  tryAutoOpen()
})

function dismiss(): void {
  visible.value = false
  manualOpen.value = false
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(STORAGE_KEY, '1')
  }
  emit('dismissed')
}

function reopen(): void {
  if (!props.hasNodes) return
  manualOpen.value = true
  visible.value = true
  focusDismiss()
}

defineExpose({ reopen })
</script>

<template>
  <Transition name="ps-gesture-fade">
    <div
      v-if="visible"
      class="absolute inset-0 z-[9] flex items-center justify-center p-3"
      style="background: color-mix(in srgb, var(--ps-canvas) 65%, transparent)"
      data-testid="graph-gesture-overlay"
      @click.self="dismiss"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="graph-gesture-dialog-title"
        class="max-w-[20rem] cursor-default rounded-sm border border-border bg-elevated p-4 shadow-md"
        @click.stop
      >
        <h2
          id="graph-gesture-dialog-title"
          class="mb-3 text-sm font-semibold text-surface-foreground"
        >
          Graph gestures
        </h2>

        <ul class="flex list-none flex-col gap-2 p-0">
          <li class="flex gap-3 text-xs">
            <svg
              class="mt-0.5 h-4 w-4 shrink-0 text-muted"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              aria-hidden="true"
            >
              <path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z" />
            </svg>
            <span class="shrink-0 font-mono text-surface-foreground">Click</span>
            <span class="min-w-0 text-muted">Open node details</span>
          </li>
          <li class="flex gap-3 text-xs">
            <svg
              class="mt-0.5 h-4 w-4 shrink-0 text-muted"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              aria-hidden="true"
            >
              <rect x="4" y="6" width="16" height="12" rx="1" />
              <path d="M8 10h8M8 14h5" />
            </svg>
            <span class="shrink-0 font-mono text-surface-foreground">Alt + B</span>
            <span class="min-w-0 text-muted">Toggle the graph bottom bar (collapse / expand)</span>
          </li>
          <li class="flex gap-3 text-xs">
            <svg
              class="mt-0.5 h-4 w-4 shrink-0 text-muted"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="9" />
              <path d="M8 12h8M12 8v8" />
            </svg>
            <span class="shrink-0 font-mono text-surface-foreground">Shift + dbl-click</span>
            <span class="min-w-0 text-muted">Expand 1-hop neighbourhood</span>
          </li>
          <li class="flex gap-3 text-xs">
            <svg
              class="mt-0.5 h-4 w-4 shrink-0 text-muted"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              aria-hidden="true"
            >
              <rect x="3" y="3" width="18" height="18" rx="2" />
            </svg>
            <span class="shrink-0 font-mono text-surface-foreground">Shift + drag</span>
            <span class="min-w-0 text-muted">Box zoom / select</span>
          </li>
          <li class="flex gap-3 text-xs">
            <svg
              class="mt-0.5 h-4 w-4 shrink-0 text-muted"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="9" />
              <path d="M12 8v8M8 12h8" />
            </svg>
            <span class="shrink-0 font-mono text-surface-foreground">Dbl-click</span>
            <span class="min-w-0 text-muted">Load more episodes for this node</span>
          </li>
          <li class="flex gap-3 text-xs">
            <svg
              class="mt-0.5 h-4 w-4 shrink-0 text-muted"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="9" />
              <path d="M8 12h8" />
            </svg>
            <span class="shrink-0 font-mono text-surface-foreground">Dbl-click again</span>
            <span class="min-w-0 text-muted">Collapse loaded episodes</span>
          </li>
        </ul>

        <div class="mt-3 border-t border-border pt-3">
          <div class="flex items-center gap-2 text-xs text-muted">
            <span
              class="inline-block h-2.5 w-2.5 shrink-0 rounded-full border-2"
              :style="{ borderColor: EXPAND_RING_TEAL, backgroundColor: 'transparent' }"
              aria-hidden="true"
            />
            <span>More episodes available (teal ring)</span>
          </div>
          <div class="mt-1 flex items-center gap-2 text-xs text-muted">
            <span
              class="inline-block h-2.5 w-2.5 shrink-0 rounded-full border-2"
              :style="{ borderColor: EXPAND_RING_BLUE, backgroundColor: 'transparent' }"
              aria-hidden="true"
            />
            <span>Episodes loaded from here (blue ring)</span>
          </div>
        </div>

        <div class="mt-3 flex justify-end">
          <button
            ref="dismissButtonRef"
            type="button"
            class="rounded border border-border bg-primary px-3 py-1 text-xs font-medium text-primary-foreground hover:opacity-90"
            data-testid="graph-gesture-overlay-dismiss"
            aria-label="Dismiss graph gesture hints"
            @click="dismiss"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
/* Theme-agnostic veil when `color-mix` is unavailable (older engines). */
.ps-gesture-overlay-root {
  background-color: rgba(0, 0, 0, 0.35);
}
@supports (background: color-mix(in srgb, red 50%, blue)) {
  .ps-gesture-overlay-root {
    background: color-mix(in srgb, var(--ps-canvas) 65%, transparent);
  }
}

.ps-gesture-fade-enter-active,
.ps-gesture-fade-leave-active {
  transition: opacity 150ms ease;
}
.ps-gesture-fade-enter-from,
.ps-gesture-fade-leave-to {
  opacity: 0;
}
@media (prefers-reduced-motion: reduce) {
  .ps-gesture-fade-enter-active,
  .ps-gesture-fade-leave-active {
    transition-duration: 0ms;
  }
}
</style>
