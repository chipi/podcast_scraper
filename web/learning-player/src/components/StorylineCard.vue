<script setup lang="ts">
/**
 * Storyline card (#9) — a storyline is a THEME CLUSTER (topics discussed together). Tapping one used
 * to open the anchor topic's entity card (a different name than the storyline → confusing). This
 * sheet is titled with the storyline itself and gives a short analytical overview: how many topics /
 * episodes, the voices (people) involved, and the member topics — rather than just a flat list.
 *
 * All of it comes from the anchor topic's card (`theme_sibling_topics`, `related_people`,
 * `episode_count`) — there is no dedicated storyline endpoint, and the anchor's theme cluster IS the
 * storyline. Modal shell mirrors EntityCard (teleport, focus trap, ESC/backdrop dismiss).
 */
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { getTopicCard } from '../services/api'

type Member = { id: string; label: string }
type Voice = { id: string; label: string; role?: string | null }

const props = defineProps<{ id: string; label: string; anchorTopicId: string }>()
const emit = defineEmits<{
  (e: 'close'): void
  (e: 'open-topic', id: string): void
  (e: 'open-person', id: string): void
}>()

const { t } = useI18n()

const loading = ref(true)
const failed = ref(false)
const topics = ref<Member[]>([])
const people = ref<Voice[]>([])
const episodeCount = ref(0)

onMounted(async () => {
  restoreFocus = document.activeElement as HTMLElement | null
  window.addEventListener('keydown', onKeydown)
  void nextTick(() => (focusables()[0] ?? dialogEl.value)?.focus())
  try {
    const card = await getTopicCard(props.anchorTopicId)
    // Anchor + its theme siblings = the storyline's topics. De-dupe (the API may include the anchor).
    const members: Member[] = [
      { id: card.id, label: card.label },
      ...(card.theme_sibling_topics ?? []).map((tp) => ({ id: tp.id, label: tp.label })),
    ]
    const seen = new Set<string>()
    topics.value = members.filter((tp) => tp.id && !seen.has(tp.id) && seen.add(tp.id))
    people.value = (card.related_people ?? []).map((p) => ({ id: p.id, label: p.name, role: p.role }))
    episodeCount.value = card.episode_count ?? 0
  } catch {
    failed.value = true
  } finally {
    loading.value = false
  }
})

const topicCount = computed(() => topics.value.length)

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
              {{ t('home.storylineSheetCount', topicCount, { named: { count: topicCount } }) }}
              <template v-if="episodeCount">
                {{ t('home.storylineEpisodeCount', episodeCount, { named: { count: episodeCount } }) }}
              </template>
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
          <p v-else-if="failed || topicCount === 0" class="text-sm text-muted">
            {{ t('home.storylineSheetEmpty') }}
          </p>
          <template v-else>
            <!-- Voices: the people this storyline is discussed by — the analytical "who" at a glance. -->
            <section v-if="people.length" class="mb-5">
              <h3 class="lp-kicker mb-2">{{ t('home.storylineVoices') }}</h3>
              <div class="flex flex-wrap gap-1.5">
                <button
                  v-for="p in people"
                  :key="p.id"
                  type="button"
                  class="inline-flex items-center gap-1 rounded-full bg-overlay px-2.5 py-1 text-xs font-semibold text-person transition hover:opacity-80"
                  data-testid="storyline-person"
                  @click="emit('open-person', p.id)"
                >
                  {{ p.label }}
                  <span v-if="p.role" class="text-[9px] font-bold uppercase tracking-wide text-muted"
                    >{{ p.role }}</span
                  >
                </button>
              </div>
            </section>

            <!-- Topics: compact grid, not full-width giant rows. -->
            <section>
              <h3 class="lp-kicker mb-2">{{ t('home.storylineTopicsHeading') }}</h3>
              <ul class="grid grid-cols-2 gap-2">
                <li v-for="tp in topics" :key="tp.id">
                  <button
                    type="button"
                    class="flex w-full items-center justify-between gap-1 rounded-lg border border-border bg-canvas px-3 py-2 text-left text-sm font-semibold text-canvas-foreground transition hover:bg-overlay"
                    data-testid="storyline-topic-row"
                    @click="emit('open-topic', tp.id)"
                  >
                    <span class="min-w-0 truncate">{{ tp.label }}</span>
                    <span class="shrink-0 text-muted" aria-hidden="true">›</span>
                  </button>
                </li>
              </ul>
            </section>
          </template>
        </div>
      </div>
    </div>
  </Teleport>
</template>
