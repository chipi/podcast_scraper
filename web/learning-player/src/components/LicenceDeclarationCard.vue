<script setup lang="ts">
/**
 * One licence or copyright declaration exactly as a feed carries it (#2325): what kind of
 * declaration it is, whether it covers the whole show or one episode, what it applies to, the text
 * verbatim, its licence link, a recognised licence's conditions and any note. Shared by the show's
 * and the episode's licence pages so both read the same.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { LicenceDeclaration } from '../services/types'

const props = defineProps<{ declaration: LicenceDeclaration }>()
const { t } = useI18n()

function yesNo(v: boolean): string {
  return v ? t('licence.yes') : t('licence.no')
}
const conditions = computed(() => {
  const k = props.declaration.known
  if (!k) return []
  return [
    t('licence.conditions.commercial', { value: yesNo(k.commercial_use) }),
    t('licence.conditions.adaptations', { value: yesNo(k.adaptations) }),
    t('licence.conditions.attribution', { value: yesNo(k.attribution) }),
    t('licence.conditions.shareAlike', { value: yesNo(k.share_alike) }),
  ]
})
</script>

<template>
  <article class="mb-4 rounded-xl bg-elevated p-4" data-testid="podcast-licence-declaration">
    <p class="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">
      {{ t(`licence.mechanism.${declaration.mechanism}`) }} · {{ t(`licence.level.${declaration.level}`) }}
    </p>
    <p class="mb-2 text-xs text-muted">{{ t(`licence.scope.${declaration.asset_scope}`) }}</p>
    <p class="whitespace-pre-wrap break-words text-sm" data-testid="podcast-licence-raw">{{ declaration.raw }}</p>
    <p v-if="declaration.url" class="mt-2 text-sm">
      {{ t('licence.link') }}:
      <a :href="declaration.url" target="_blank" rel="noopener noreferrer" class="break-all underline">{{
        declaration.url
      }}</a>
    </p>
    <div v-if="declaration.known" class="mt-2 text-sm">
      <p>{{ t('licence.recognised', { name: declaration.known.name }) }}</p>
      <ul class="mt-1 list-disc pl-5 text-muted">
        <li v-for="c in conditions" :key="c">{{ c }}</li>
      </ul>
    </div>
    <ul v-if="declaration.notes.length" class="mt-2 list-disc pl-5 text-xs text-muted">
      <li v-for="n in declaration.notes" :key="n">{{ n }}</li>
    </ul>
  </article>
</template>
