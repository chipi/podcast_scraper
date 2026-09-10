<script setup lang="ts">
/**
 * Placeholder About/legal page — 3rd-party software, Privacy policy, Terms of use (operator
 * 2026-09-09). Empty content FOR NOW: the routes, titles and back-nav exist so Settings can link to
 * them and the real copy drops in later without any wiring change. Support is a link, not a page, so
 * it is not routed here.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'

const props = defineProps<{ page: string }>()
const { t } = useI18n()

const TITLES: Record<string, string> = {
  'third-party': 'about.thirdParty',
  privacy: 'about.privacy',
  terms: 'about.terms',
}
const titleKey = computed(() => TITLES[props.page] ?? 'about.thirdParty')
</script>

<template>
  <section class="mx-auto max-w-2xl px-4 pb-8 pt-4" data-testid="about-page">
    <RouterLink
      :to="{ name: 'settings' }"
      class="mb-4 inline-flex items-center gap-1 text-sm font-medium text-muted no-underline transition hover:text-canvas-foreground"
    >
      ‹ {{ t('settings.title') }}
    </RouterLink>
    <h1 class="mb-4 font-display text-3xl font-extrabold tracking-tight" data-testid="about-page-title">
      {{ t(titleKey) }}
    </h1>
    <p class="text-sm text-muted">{{ t('about.placeholder') }}</p>
  </section>
</template>
