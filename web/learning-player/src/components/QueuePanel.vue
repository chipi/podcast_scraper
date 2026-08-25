<script setup lang="ts">
/**
 * Queue & Recently-played panel (#1838) — the transport lists, surfaced FROM THE PLAYER (a queue
 * button on the mini/full player) instead of buried in Library. Two sections:
 *
 * - **Up next** — the play queue (reorder / remove / play), reusing QueueView.
 * - **Recently played** — playback history; the point is to *find/resume* something you heard, not
 *   to re-queue it, so tapping a row opens the player (it does not add to the queue).
 *
 * Modal bottom-sheet shell mirrors EntityCard (teleport, focus trap, ESC / backdrop dismiss).
 */
import { nextTick, onMounted, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import QueueView from '../views/QueueView.vue'
import EpisodeCard from './EpisodeCard.vue'
import { getEpisode, getPlaybackList } from '../services/api'
import type { EpisodeDetail } from '../services/types'
import { summaryFromDetail } from '../utils/episode'

const emit = defineEmits<{ (e: 'close'): void }>()
const { t } = useI18n()

const recent = ref<EpisodeDetail[]>([])
const recentLoading = ref(true)

onMounted(async () => {
  restoreFocus = document.activeElement as HTMLElement | null
  window.addEventListener('keydown', onKeydown)
  void nextTick(() => (focusables()[0] ?? dialogEl.value)?.focus())
  try {
    const positions = await getPlaybackList().catch(() => [])
    const hydrated = await Promise.all(
      positions.slice(0, 30).map((p) => getEpisode(p.slug).catch(() => null)),
    )
    recent.value = hydrated.filter((d): d is EpisodeDetail => !!d)
  } finally {
    recentLoading.value = false
  }
})

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
      data-testid="queue-panel"
      @click.self="emit('close')"
    >
      <div
        ref="dialogEl"
        tabindex="-1"
        class="flex max-h-[92dvh] w-full max-w-lg flex-col overflow-hidden rounded-t-2xl bg-surface outline-none sm:max-h-[85dvh] sm:rounded-2xl"
      >
        <header class="flex items-center justify-between gap-3 border-b border-border px-5 py-4">
          <h2 class="font-display text-xl font-extrabold tracking-tight text-canvas-foreground">
            {{ t('queue.title') }}
          </h2>
          <button
            type="button"
            class="shrink-0 rounded-full px-2 py-1 text-lg leading-none text-muted transition hover:bg-overlay"
            :aria-label="t('nav.back')"
            data-testid="queue-panel-close"
            @click="emit('close')"
          >
            ✕
          </button>
        </header>

        <div class="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          <!-- Up next -->
          <section class="mb-6">
            <h3 class="lp-kicker mb-2">{{ t('queue.upNext') }}</h3>
            <QueueView hide-title />
          </section>

          <!-- Recently played — resume, don't re-queue. -->
          <section>
            <h3 class="lp-kicker mb-2">{{ t('queue.recentlyPlayed') }}</h3>
            <p v-if="recentLoading" class="text-sm text-muted">{{ t('catalog.loading') }}</p>
            <p v-else-if="!recent.length" class="text-sm text-muted">{{ t('queue.recentEmpty') }}</p>
            <div v-else class="flex flex-col" data-testid="queue-panel-recent" @click="emit('close')">
              <EpisodeCard v-for="d in recent" :key="d.slug" :episode="summaryFromDetail(d)" />
            </div>
          </section>
        </div>
      </div>
    </div>
  </Teleport>
</template>
