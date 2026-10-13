<script setup lang="ts">
/**
 * "Translated from Spanish" under a quote, take or moment whose text is a machine translation
 * (D-36 slice b, operator 2026-10-13). A quote can be passed on as somebody's words, so a
 * translated one says so. Renders nothing when `lang` is absent: a native quote carries no label.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

const props = defineProps<{ lang?: string | null; inline?: boolean }>()
const { t, locale } = useI18n()

const name = computed(() => {
  const code = props.lang?.trim()
  if (!code) return ''
  try {
    return new Intl.DisplayNames([locale.value], { type: 'language' }).of(code) ?? code
  } catch {
    // An invalid tag throws RangeError; the code itself is still the honest label.
    return code.toUpperCase()
  }
})
</script>

<template>
  <component
    :is="inline ? 'span' : 'p'"
    v-if="name"
    class="text-xs italic text-muted"
    :class="inline ? 'ml-1' : 'mt-1'"
    data-testid="translated-from"
    :data-lang="lang"
  >{{ t('language.translatedFrom', { name }) }}</component>
</template>
