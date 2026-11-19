<script setup lang="ts">
/**
 * A show's language as a squared chip with the uppercase code — the operator library's half of
 * V2-C.1 (the consumer player has its own, i18n-backed twin). The caller decides whether badges are
 * worth showing at all (only in a multilingual corpus, PRD-047 FR7.5); this renders nothing for an
 * unknown language rather than guessing.
 */
import { computed } from 'vue'
import { primaryLanguage } from '../../utils/language'

const props = defineProps<{ lang?: string | null }>()

const code = computed(() => primaryLanguage(props.lang))
const name = computed(() => {
  if (!code.value) return ''
  try {
    return new Intl.DisplayNames(['en'], { type: 'language' }).of(code.value) ?? code.value
  } catch {
    return code.value.toUpperCase()
  }
})
</script>

<template>
  <span
    v-if="code"
    role="img"
    :aria-label="`Language: ${name}`"
    :title="name"
    data-testid="language-badge"
    :data-lang="code"
    class="inline-flex w-fit shrink-0 items-center rounded-[3px] border border-border px-1 text-[10px] font-bold uppercase leading-4 tracking-wide text-muted"
  >{{ code }}</span>
</template>
