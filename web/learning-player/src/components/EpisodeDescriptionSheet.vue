<script setup lang="ts">
/**
 * The publisher's own episode description, in full (operator 2026-10-10).
 *
 * Episode cards show it clamped; the player never showed it at all. Opened from the player's
 * Description pill, beside Episode notes. Links the publisher wrote out are tappable (`linkify`);
 * they open through `openExternal`, because `window.open` does nothing in the iOS WebView.
 *
 * A native <dialog> + `showModal()` for the same reasons the notes sheet uses one: focus trap,
 * Escape, an inert page behind it. Bottom sheet on phones, a centred card at lg.
 *
 * Laid out exactly like the Brief (operator 2026-10-10, "full consistency"): the grab handle (pull
 * it down to close), a header named as its door on the obi — "About" — with ✕, then the show and
 * the episode title opening the content, then the description.
 */
import { computed, nextTick, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import CloseIcon from './CloseIcon.vue'
import { useSheetDrag } from '../composables/useSheetDrag'
import { linkify } from '../services/linkify'
import { openExternal } from '../services/native'

const props = defineProps<{
  open: boolean
  title: string
  description: string
  showTitle?: string | null
}>()
const emit = defineEmits<{ (e: 'close'): void }>()

const { t } = useI18n()
const el = ref<HTMLDialogElement | null>(null)
const segments = computed(() => linkify(props.description))
const panelEl = ref<HTMLElement | null>(null)
const handleDrag = useSheetDrag(panelEl, () => emit('close'))

watch(
  [() => props.open, el],
  async ([open]) => {
    await nextTick()
    const d = el.value
    if (!d) return
    if (open && !d.open) d.showModal()
    else if (!open && d.open) d.close()
  },
  { immediate: true },
)

function onBackdropClick(e: MouseEvent): void {
  if (e.target === el.value) emit('close')
}
</script>

<template>
  <dialog
    ref="el"
    data-testid="episode-description"
    :aria-label="t('player.about')"
    class="fixed inset-x-0 bottom-0 top-[var(--lp-sheet-top)] m-0 h-[calc(100dvh-var(--lp-sheet-top))] max-h-none w-full max-w-none border-0 bg-transparent p-0 text-canvas-foreground backdrop:bg-black/50 lg:inset-0 lg:m-auto lg:h-fit lg:max-h-[80dvh] lg:max-w-2xl"
    @close="emit('close')"
    @click="onBackdropClick"
  >
    <div
      ref="panelEl"
      class="flex h-full flex-col overflow-hidden rounded-t-2xl border-t border-border bg-surface pb-[env(safe-area-inset-bottom)] lg:max-h-[80dvh] lg:rounded-2xl lg:border lg:pb-0"
    >
      <!-- The Brief's grab handle: pull it down to close (phones; ✕ is the accessible close). -->
      <div
        class="flex h-6 shrink-0 touch-none items-center justify-center lg:hidden"
        aria-hidden="true"
        data-testid="episode-description-handle"
        v-bind="handleDrag"
      >
        <span class="h-1.5 w-10 rounded-full bg-border"></span>
      </div>
      <header class="flex items-center justify-between border-b border-border px-4 py-3">
        <span class="font-display text-lg font-bold">{{ t('player.about') }}</span>
        <button
          type="button"
          class="lp-nav shrink-0"
          data-testid="episode-description-close"
          :aria-label="t('player.descriptionClose')"
          @click="emit('close')"
        >
          <CloseIcon />
        </button>
      </header>
      <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <!-- The same opening as the Brief's: which show, which episode. -->
        <section class="mb-4">
          <p v-if="showTitle" class="lp-kicker lp-show-name mb-0.5 text-muted" :title="showTitle">
            {{ showTitle }}
          </p>
          <h2
            class="font-display text-xl font-bold leading-tight text-canvas-foreground"
            data-testid="episode-description-episode"
          >
            {{ title }}
          </h2>
        </section>
        <p
          class="whitespace-pre-line break-words text-sm leading-relaxed"
          data-testid="episode-description-text"
        ><template v-for="(s, i) in segments" :key="i"><a
              v-if="s.href"
              :href="s.href"
              target="_blank"
              rel="noopener noreferrer"
              class="break-all text-accent underline"
              data-testid="episode-description-link"
              @click.prevent="openExternal(s.href)"
            >{{ s.text }}</a><template v-else>{{ s.text }}</template></template></p>
      </div>
    </div>
  </dialog>
</template>
