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
 */
import { computed, nextTick, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import CloseIcon from './CloseIcon.vue'
import { linkify } from '../services/linkify'
import { openExternal } from '../services/native'

const props = defineProps<{ open: boolean; title: string; description: string }>()
const emit = defineEmits<{ (e: 'close'): void }>()

const { t } = useI18n()
const el = ref<HTMLDialogElement | null>(null)
const segments = computed(() => linkify(props.description))

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
    :aria-label="t('player.descriptionTitle')"
    class="fixed inset-x-0 bottom-0 top-[var(--lp-sheet-top)] m-0 h-[calc(100dvh-var(--lp-sheet-top))] max-h-none w-full max-w-none border-0 bg-transparent p-0 text-canvas-foreground backdrop:bg-black/50 lg:inset-0 lg:m-auto lg:h-fit lg:max-h-[80dvh] lg:max-w-2xl"
    @close="emit('close')"
    @click="onBackdropClick"
  >
    <div
      class="flex h-full flex-col overflow-hidden rounded-t-2xl border-t border-border bg-canvas pb-[env(safe-area-inset-bottom)] lg:max-h-[80dvh] lg:rounded-2xl lg:border lg:pb-0"
    >
      <header class="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
        <div class="min-w-0">
          <span class="font-display text-lg font-bold">{{ t('player.descriptionTitle') }}</span>
          <p class="text-xs text-muted" data-testid="episode-description-episode">{{ title }}</p>
        </div>
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
