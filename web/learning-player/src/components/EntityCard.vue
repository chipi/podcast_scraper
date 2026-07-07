<script setup lang="ts">
/**
 * Entity card — MODAL presentation of {@link EntityCardBody} (UXS-014: a modal is only opened from a
 * page-level surface, e.g. the Search results). Inside a panel we instead render EntityCardBody
 * INLINE (replace-in-panel), never stacking a second backdrop. Teleported to <body> so it covers the
 * viewport (escapes any clipped/transformed ancestor); modal a11y = role/aria-modal + focus trap +
 * restore focus + ESC/backdrop dismiss.
 */
import { nextTick, onMounted, onUnmounted, ref } from 'vue'
import EntityCardBody from './EntityCardBody.vue'

defineProps<{ kind: 'person' | 'topic'; id: string }>()
const emit = defineEmits<{ (e: 'close'): void }>()

const dialogEl = ref<HTMLElement | null>(null)
let restoreFocus: HTMLElement | null = null

function focusables(): HTMLElement[] {
  if (!dialogEl.value) return []
  const sel = 'a[href], button:not([disabled]), input, [tabindex]:not([tabindex="-1"])'
  return Array.from(dialogEl.value.querySelectorAll<HTMLElement>(sel))
}

function onKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape') {
    emit('close')
    return
  }
  if (e.key !== 'Tab') return
  const items = focusables()
  if (items.length === 0) return
  const first = items[0]
  const last = items[items.length - 1]
  if (e.shiftKey && document.activeElement === first) {
    e.preventDefault()
    last.focus()
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault()
    first.focus()
  }
}

onMounted(() => {
  restoreFocus = document.activeElement as HTMLElement | null
  window.addEventListener('keydown', onKeydown)
  void nextTick(() => (focusables()[0] ?? dialogEl.value)?.focus())
})
onUnmounted(() => {
  window.removeEventListener('keydown', onKeydown)
  restoreFocus?.focus?.()
})
</script>

<template>
  <Teleport to="body">
    <div
      class="fixed inset-0 z-50 flex items-end justify-center bg-black/40 sm:items-center"
      role="dialog"
      aria-modal="true"
      @click.self="emit('close')"
    >
      <div
        ref="dialogEl"
        tabindex="-1"
        class="flex max-h-[85vh] w-full max-w-lg flex-col overflow-hidden rounded-t-2xl bg-surface outline-none sm:rounded-2xl"
      >
        <EntityCardBody variant="overlay" :kind="kind" :id="id" @close="emit('close')" />
      </div>
    </div>
  </Teleport>
</template>
