<script setup lang="ts">
/**
 * Storyline card (#9) — a storyline is a THEME CLUSTER (topics discussed together), so tapping one
 * used to open the anchor topic's entity card, whose headline was a DIFFERENT topic name than the
 * storyline the user tapped — confusing. This sheet is titled with the storyline itself and lists
 * its member topics; tapping a member opens THAT topic's entity card.
 *
 * The members come from the anchor topic's card (`theme_sibling_topics` + the anchor) — there is no
 * dedicated storyline-members endpoint, and the anchor's theme cluster IS the storyline. Modal shell
 * mirrors EntityCard (teleport to body, focus trap, ESC/backdrop dismiss, restore focus).
 */
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { getTopicCard } from '../services/api'

/** Only id + label are needed to list + open a member topic. */
type Member = { id: string; label: string }

const props = defineProps<{ id: string; label: string; anchorTopicId: string }>()
const emit = defineEmits<{ (e: 'close'): void; (e: 'open-topic', id: string): void }>()

const { t } = useI18n()

const loading = ref(true)
const failed = ref(false)
const topics = ref<Member[]>([])

onMounted(async () => {
  restoreFocus = document.activeElement as HTMLElement | null
  window.addEventListener('keydown', onKeydown)
  void nextTick(() => (focusables()[0] ?? dialogEl.value)?.focus())
  try {
    const card = await getTopicCard(props.anchorTopicId)
    // The anchor is the storyline's representative; its theme siblings are the rest. De-dupe by id
    // so the anchor isn't listed twice if the API already includes it among the siblings.
    const members: Member[] = [
      { id: card.id, label: card.label },
      ...(card.theme_sibling_topics ?? []).map((tp) => ({ id: tp.id, label: tp.label })),
    ]
    const seen = new Set<string>()
    topics.value = members.filter((tp) => tp.id && !seen.has(tp.id) && seen.add(tp.id))
  } catch {
    failed.value = true
  } finally {
    loading.value = false
  }
})

const count = computed(() => topics.value.length)

// --- modal a11y (mirrors EntityCard) ---
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
      data-testid="storyline-card"
      @click.self="emit('close')"
    >
      <div
        ref="dialogEl"
        tabindex="-1"
        class="flex max-h-[92dvh] w-full max-w-lg flex-col overflow-hidden rounded-t-2xl bg-surface outline-none sm:max-h-[85dvh] sm:rounded-2xl"
      >
        <header class="flex items-start justify-between gap-3 border-b border-border px-5 py-4">
          <div class="min-w-0">
            <p class="text-xs font-bold uppercase tracking-wide text-muted">
              {{ t('home.storylines') }}
            </p>
            <h2 class="font-display text-xl font-extrabold tracking-tight text-canvas-foreground">
              {{ label }}
            </h2>
            <p v-if="!loading && !failed" class="mt-0.5 text-sm text-muted">
              {{ t('home.storylineSheetCount', count, { named: { count } }) }}
            </p>
          </div>
          <button
            type="button"
            class="shrink-0 rounded-full px-2 py-1 text-lg leading-none text-muted transition hover:bg-overlay"
            :aria-label="t('nav.back')"
            data-testid="storyline-card-close"
            @click="emit('close')"
          >
            ✕
          </button>
        </header>

        <div class="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          <p v-if="loading" class="text-sm text-muted">{{ t('home.storylineSheetLoading') }}</p>
          <p v-else-if="failed || count === 0" class="text-sm text-muted">
            {{ t('home.storylineSheetEmpty') }}
          </p>
          <ul v-else class="flex flex-col gap-2">
            <li v-for="tp in topics" :key="tp.id">
              <button
                type="button"
                class="flex w-full items-center justify-between gap-3 rounded-xl border border-border bg-canvas px-4 py-3 text-left transition hover:bg-overlay"
                data-testid="storyline-topic-row"
                @click="emit('open-topic', tp.id)"
              >
                <span class="min-w-0 truncate font-semibold text-canvas-foreground">{{ tp.label }}</span>
                <span class="shrink-0 text-muted" aria-hidden="true">›</span>
              </button>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </Teleport>
</template>
