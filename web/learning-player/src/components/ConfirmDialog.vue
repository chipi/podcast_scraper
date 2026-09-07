<script setup lang="ts">
/**
 * A confirmation step in front of the deletes that destroy authored content (#1594).
 *
 * ## What this is for, and what it is deliberately NOT for
 *
 * Four controls in the app deleted something on a single tap with no way back: delete a collection,
 * remove an item from one, delete a highlight, delete a note. Three of those destroy something the
 * user WROTE or CURATED, and none of them can be reconstructed from anywhere else in the app.
 *
 * Removing an item from a collection is the fourth, and it does NOT get a dialog. The item itself
 * survives — only a membership row goes — and re-adding it is two taps from the same screen. A
 * confirm there would be friction spent on nothing, which is how confirm dialogs become the thing
 * everyone clicks through without reading, including on the three that matter.
 *
 * ## Why confirm rather than undo
 *
 * Undo is the better pattern where an exact restore is possible. It is not possible here:
 * `createCollection(name)` mints a NEW id, so "undoing" a deleted collection would produce a
 * different board that happens to share a name — every reference to the old id would still be
 * broken, and the undo would be a lie in the one case the user cares most about. Highlights and
 * notes have the same shape, plus a tombstone the client cannot see the semantics of. Confirming
 * before the fact is honest about what the app can actually do.
 *
 * ## Why a native `<dialog>`
 *
 * The same reasoning as the Knowledge Panel (S9): `showModal()` supplies the top layer, the focus
 * trap, Escape-to-close and an inert background from the browser. Reimplementing those in userland
 * is where a11y bugs live, and a confirm that Escape cannot dismiss is a trap in the literal sense.
 *
 * ## The caller's testid falls through; it is not a prop
 *
 * `<dialog>` is the single root, so `data-testid` from the call site lands on it automatically.
 * That is not a style preference: the surface-map guard scans component source for a literal
 * `data-testid="..."`, so a testid passed as `testid="x"` and re-bound here would be invisible to
 * it — the map would document selectors the guard cannot confirm anything renders.
 *
 * For the same reason there is no comment above the root element. A leading comment makes the
 * template a multi-root fragment, and Vue does not auto-inherit attributes onto a fragment: the
 * testid would silently stop falling through. Production builds strip comments and dev builds do
 * not, so the symptom appears in tests and disappears in the bundle.
 *
 * ## Focus lands on CANCEL, not Confirm
 *
 * A dialog that opens with the destructive action focused turns "tap, tap" muscle memory into a
 * delete — it would add a step without adding a decision.
 */
import { nextTick, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

const props = withDefaults(
  defineProps<{
    open: boolean
    /** The question. Short, and it should name the thing being destroyed. */
    title: string
    /** What is actually lost. Omit when the title already says it. */
    body?: string
    /** The verb, e.g. "Delete collection" — never a bare "OK". */
    confirmLabel: string
  }>(),
  { body: '' },
)

const emit = defineEmits<{ (e: 'confirm'): void; (e: 'cancel'): void }>()

const { t } = useI18n()
const el = ref<HTMLDialogElement | null>(null)
const cancelEl = ref<HTMLButtonElement | null>(null)

watch(
  () => props.open,
  async (open) => {
    const d = el.value
    if (!d) return
    if (open) {
      if (!d.open) d.showModal()
      await nextTick()
      cancelEl.value?.focus()
    } else if (d.open) {
      d.close()
    }
  },
)

/**
 * `close` fires for Escape and for the backdrop as well as our own buttons, so it is the only
 * place that can guarantee the parent's `open` flag follows the dialog's real state. Without it,
 * Escape would leave the dialog closed and the parent believing it is still open — and the next
 * delete would never show a confirmation at all.
 */
function onClose(): void {
  if (props.open) emit('cancel')
}
</script>

<template>
  <dialog
    ref="el"
    class="max-w-[20rem] rounded-xl border border-border bg-surface p-5 text-canvas-foreground backdrop:bg-black/50"
    @close="onClose"
  >
    <h2 class="text-base font-bold">{{ title }}</h2>
    <p v-if="body" class="mt-2 text-sm text-muted">{{ body }}</p>

    <div class="mt-5 flex justify-end gap-3">
      <button
        ref="cancelEl"
        type="button"
        data-testid="confirm-cancel"
        class="flex h-11 items-center rounded-full px-4 text-sm font-bold text-muted transition hover:text-canvas-foreground"
        @click="emit('cancel')"
      >
        {{ t('common.cancel') }}
      </button>
      <button
        type="button"
        data-testid="confirm-accept"
        class="flex h-11 items-center rounded-full bg-danger px-4 text-sm font-bold text-white transition"
        @click="emit('confirm')"
      >
        {{ confirmLabel }}
      </button>
    </div>
  </dialog>
</template>
