<script setup lang="ts">
/**
 * Interests picker (PRD-043 FR4 / 3.5) — a dismissible modal to choose interest *clusters*
 * (the top corpus themes by prevalence). Saved per-user; the discovery feed re-ranks Home by
 * these when the personalization flag is on. Pure overlay: backdrop / ESC / ✕ dismiss, focus
 * trap, restore focus on close.
 */
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { getTopClusters, getUserInterests, putUserInterests } from '../services/api'
import type { InterestCluster } from '../services/types'

const emit = defineEmits<{ (e: 'close'): void; (e: 'saved', ids: string[]): void }>()
const { t } = useI18n()

const clusters = ref<InterestCluster[]>([])
const selected = ref<Set<string>>(new Set())
const loading = ref(true)
const saving = ref(false)

const hasClusters = computed(() => clusters.value.length > 0)

function toggle(id: string): void {
  const next = new Set(selected.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  selected.value = next
}

async function save(): Promise<void> {
  saving.value = true
  try {
    const ids = clusters.value.map((c) => c.id).filter((id) => selected.value.has(id))
    const stored = await putUserInterests(ids)
    emit('saved', stored)
    emit('close')
  } catch {
    saving.value = false // keep the modal open so the user can retry
  }
}

// --- modal a11y (focus trap + restore) ---
const dialogEl = ref<HTMLElement | null>(null)
let restoreFocus: HTMLElement | null = null

function focusables(): HTMLElement[] {
  if (!dialogEl.value) return []
  return Array.from(
    dialogEl.value.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), input, [tabindex]:not([tabindex="-1"])',
    ),
  )
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

onMounted(async () => {
  restoreFocus = document.activeElement as HTMLElement | null
  window.addEventListener('keydown', onKeydown)
  const [tops, current] = await Promise.all([
    getTopClusters(12).catch(() => [] as InterestCluster[]),
    getUserInterests().catch(() => [] as string[]),
  ])
  clusters.value = tops
  selected.value = new Set(current)
  loading.value = false
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
    :aria-label="t('interests.title')"
    @click.self="emit('close')"
  >
    <div
      ref="dialogEl"
      tabindex="-1"
      class="flex max-h-[85vh] w-full max-w-lg flex-col rounded-t-2xl bg-surface outline-none sm:rounded-2xl"
    >
      <header class="flex items-center justify-between gap-2 border-b border-border px-4 py-3">
        <span class="min-w-0">
          <span class="block font-display text-lg font-bold">{{ t('interests.title') }}</span>
          <span class="lp-kicker block">{{ t('interests.subtitle') }}</span>
        </span>
        <button type="button" class="text-muted" :aria-label="t('interests.close')" @click="emit('close')">✕</button>
      </header>

      <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <p v-if="loading" class="text-sm text-muted">{{ t('interests.loading') }}</p>
        <p v-else-if="!hasClusters" class="text-sm text-muted">{{ t('interests.empty') }}</p>
        <div v-else class="flex flex-wrap gap-2">
          <button
            v-for="c in clusters"
            :key="c.id"
            type="button"
            :aria-pressed="selected.has(c.id)"
            class="rounded-full border px-3 py-1.5 text-sm transition"
            :class="
              selected.has(c.id)
                ? 'border-accent bg-accent text-accent-foreground'
                : 'border-border bg-overlay text-topic hover:bg-elevated'
            "
            @click="toggle(c.id)"
          >{{ c.label }}</button>
        </div>
      </div>

      <footer class="flex items-center justify-end gap-2 border-t border-border px-4 py-3">
        <button type="button" class="rounded-full px-4 py-2 text-sm font-bold text-muted" @click="emit('close')">
          {{ t('interests.cancel') }}
        </button>
        <button
          type="button"
          :disabled="saving || loading"
          class="rounded-full bg-accent px-5 py-2 text-sm font-bold text-accent-foreground disabled:opacity-50"
          @click="save"
        >
          {{ saving ? t('interests.saving') : t('interests.save') }}
        </button>
      </footer>
    </div>
  </div>
  </Teleport>
</template>
