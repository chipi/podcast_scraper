<script setup lang="ts">
/**
 * The language an episode or show is in, as a compact squared chip with the uppercase code
 * (PRD-047 FR7.2, MULTILINGUAL_ARC_V2 V2-C.1).
 *
 * Renders NOTHING in two cases, both deliberate:
 * * **The corpus is single-language** (FR7.5). Every badge would read the same code — the constant
 *   #2115 removed. `useCorpusLanguages` answers that once for the whole page.
 * * **The language is unknown.** Omitted, never guessed: a wrong code is worse than none.
 *
 * Squared, not a pill, so it does not read as one of the rounded status chips (Played, Pending)
 * that sit beside it. The visible text is the code; the accessible name is the language's NAME in
 * the UI locale ("Spanish", not "ES"), since a screen reader spelling out two letters says nothing.
 *
 * `overlay` is for placement over artwork: the same dark plate the action buttons use there, so
 * contrast never depends on whatever picture happens to be underneath.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { primaryLanguage, useCorpusLanguages } from '../composables/useCorpusLanguages'

const props = defineProps<{ lang?: string | null; overlay?: boolean }>()
const { t, locale } = useI18n()
const { badgeShown } = useCorpusLanguages()

const code = computed(() => primaryLanguage(props.lang))

const name = computed(() => {
  if (!code.value) return ''
  try {
    return new Intl.DisplayNames([locale.value], { type: 'language' }).of(code.value) ?? code.value
  } catch {
    // An invalid tag throws RangeError; the code itself is still the honest label.
    return code.value.toUpperCase()
  }
})
</script>

<template>
  <span
    v-if="badgeShown(code)"
    role="img"
    :aria-label="t('language.badgeLabel', { name })"
    :title="name"
    data-testid="language-badge"
    :data-lang="code"
    class="inline-flex w-fit shrink-0 items-center rounded-[3px] border px-1 text-[10px] font-bold uppercase leading-4 tracking-wide"
    :class="
      overlay
        ? 'border-white/25 bg-black/55 text-white shadow-lg backdrop-blur-sm'
        : 'border-border text-muted'
    "
  >{{ code }}</span>
</template>
