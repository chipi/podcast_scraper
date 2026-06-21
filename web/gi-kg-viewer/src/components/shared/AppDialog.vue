<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

let appDialogIdSeq = 0

/**
 * Canonical modal dialog for the viewer (#695). Wraps the native ``<dialog>``
 * element so every modal gets the same chrome, backdrop, Esc/backdrop dismissal
 * and focus trap (browser-native via ``showModal()``) without re-implementing
 * the pattern that StatusBar / TranscriptViewerDialog grew independently.
 *
 * Controlled via ``v-model:open``. Esc, backdrop click, and the Close button all
 * emit ``update:open=false`` so the parent stays the single source of truth.
 */
const props = withDefaults(
  defineProps<{
    open: boolean
    title: string
    subtitle?: string | null
    /** ``data-testid`` for the dialog element (so specs can target it). */
    testid?: string
    /** Tailwind width classes; default is a comfortable medium modal. */
    widthClass?: string
    /** Tailwind max-height classes; the body scrolls within this. */
    maxHeightClass?: string
    /** Close when the backdrop (outside the panel) is clicked. */
    closeOnBackdrop?: boolean
    /** Body wrapper classes; override when the content manages its own scroll
     *  (e.g. a two-pane rail layout) instead of a single scrolling column. */
    bodyClass?: string
    /** ``data-testid`` for the Close button (back-compat with existing specs). */
    closeTestid?: string
  }>(),
  {
    subtitle: null,
    testid: undefined,
    widthClass: 'w-[min(100%,48rem)]',
    maxHeightClass: 'max-h-[min(92vh,48rem)]',
    closeOnBackdrop: true,
    bodyClass: 'min-h-0 flex-1 overflow-y-auto',
    closeTestid: 'app-dialog-close',
  },
)

const emit = defineEmits<{ 'update:open': [boolean] }>()

const dialogRef = ref<HTMLDialogElement | null>(null)
const titleId = `app-dialog-title-${(appDialogIdSeq += 1)}`

function syncOpen(next: boolean): void {
  const el = dialogRef.value
  if (!el) return
  if (next && !el.open) {
    el.showModal()
  } else if (!next && el.open) {
    el.close()
  }
}

watch(
  () => props.open,
  (next) => {
    syncOpen(next)
  },
)

// Apply the initial state once the <dialog> ref exists (the watcher above can't
// run immediately — dialogRef is still null during setup).
onMounted(() => {
  syncOpen(props.open)
})

function requestClose(): void {
  if (props.open) {
    emit('update:open', false)
  }
}

/** Native ``close`` fires for Esc, programmatic close, and backdrop forms. */
function onNativeClose(): void {
  requestClose()
}

function onBackdropClick(e: MouseEvent): void {
  if (!props.closeOnBackdrop) return
  // Only the dialog element itself is the backdrop; inner content stops here.
  if (e.target === dialogRef.value) {
    requestClose()
  }
}

onBeforeUnmount(() => {
  // Ensure we never leak an open modal if the host unmounts while shown.
  const el = dialogRef.value
  if (el?.open) el.close()
})
</script>

<template>
  <dialog
    ref="dialogRef"
    :data-testid="testid"
    :class="[
      widthClass,
      maxHeightClass,
      'overflow-hidden rounded-lg border border-border bg-surface p-0 text-surface-foreground shadow-xl [&::backdrop]:bg-black/40',
    ]"
    :aria-labelledby="titleId"
    @click="onBackdropClick"
    @close="onNativeClose"
  >
    <!-- Never put display:flex on <dialog> itself — it overrides the UA
         display:none for the closed state. Wrap instead. -->
    <div :class="['flex flex-col', maxHeightClass]">
      <div class="flex shrink-0 items-start justify-between gap-3 border-b border-border px-4 py-2">
        <div class="min-w-0 flex-1 pr-2">
          <h2
            :id="titleId"
            class="truncate text-sm font-semibold text-surface-foreground"
          >
            {{ title }}
          </h2>
          <p
            v-if="subtitle"
            class="mt-0.5 truncate text-[11px] text-muted"
          >
            {{ subtitle }}
          </p>
          <slot name="header" />
        </div>
        <div class="flex shrink-0 items-center gap-2">
          <slot name="header-actions" />
          <button
            type="button"
            class="shrink-0 rounded border border-border px-2 py-1 text-xs hover:bg-overlay"
            :data-testid="closeTestid"
            @click="requestClose"
          >
            Close
          </button>
        </div>
      </div>

      <div :class="bodyClass">
        <slot />
      </div>

      <div
        v-if="$slots.footer"
        class="flex shrink-0 items-center justify-between gap-2 border-t border-border px-4 py-2"
      >
        <slot name="footer" />
      </div>
    </div>
  </dialog>
</template>
