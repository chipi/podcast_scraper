<script setup lang="ts">
/**
 * About/legal pages — 3rd-party software, Privacy policy, Terms of use (operator
 * 2026-09-09). All three carry real content now: the privacy policy (#2210), the terms of use (first
 * version, 2026-10-05) and the third-party list, regenerated on every build. Support is a link, not
 * a page, so it is not routed here.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import BackIcon from '../components/BackIcon.vue'
import PrivacyPolicy from '../components/PrivacyPolicy.vue'
import TermsOfUse from '../components/TermsOfUse.vue'
import ThirdPartySoftware from '../components/ThirdPartySoftware.vue'

const props = defineProps<{ page: string }>()
const { t } = useI18n()

const TITLES: Record<string, string> = {
  'third-party': 'about.thirdParty',
  privacy: 'about.privacy',
  terms: 'about.terms',
}
// A known slug titles the page; anything else (only reachable by hand-typed URL) falls back to the
// generic "About", never to a specific page's title that would misdescribe the content.
const titleKey = computed(() => TITLES[props.page] ?? 'about.title')

/**
 * Back goes where you came FROM (2026-10-05). It always said "‹ Settings", which was right while
 * Settings was the only way in — but the sign-in page's footer now links Terms and Privacy too, and
 * from there "Settings" sent a visitor without an account somewhere they had never been. With an
 * in-app history entry behind this page it returns there; opened directly (a shared link, `/terms`)
 * it falls back to Settings as before.
 */
const router = useRouter()
const cameFromApp = typeof window !== 'undefined' && Boolean(window.history.state?.back)
function goBack(e: MouseEvent): void {
  if (!cameFromApp) return
  e.preventDefault()
  router.back()
}
</script>

<template>
  <section class="lp-page lp-focus pb-8 pt-4" data-testid="about-page">
    <RouterLink
      :to="{ name: 'settings' }"
      class="mb-4 inline-flex items-center gap-1 text-sm font-medium text-muted no-underline transition hover:text-canvas-foreground"
      data-testid="about-page-back"
      @click="goBack"
    >
      <BackIcon /> {{ cameFromApp ? t('nav.back') : t('settings.title') }}
    </RouterLink>
    <h1 class="mb-4 font-display text-3xl font-extrabold tracking-tight" data-testid="about-page-title">
      {{ t(titleKey) }}
    </h1>
    <PrivacyPolicy v-if="page === 'privacy'" />
    <TermsOfUse v-else-if="page === 'terms'" />
    <ThirdPartySoftware v-else-if="page === 'third-party'" />
    <p v-else class="text-sm text-muted">{{ t('about.placeholder') }}</p>
  </section>
</template>
