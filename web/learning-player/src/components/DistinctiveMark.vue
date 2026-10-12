<script setup lang="ts">
/**
 * The mark on a show-page pill that is distinctive to the show (operator 2026-10-12): a small
 * drawn star — drawn, not an emoji, which rendered as an empty box on Android (2026-09-16) — with
 * its meaning as hidden text, so the pill's spoken name says it too.
 */
import { useI18n } from 'vue-i18n'

/** `decorative`: beside words that already say it (the band's note), no hidden text. */
defineProps<{ decorative?: boolean }>()
const { t } = useI18n()
</script>

<template>
  <svg
    viewBox="0 0 24 24"
    class="mr-1 h-3 w-3 shrink-0"
    fill="currentColor"
    aria-hidden="true"
    data-testid="ps-distinctive-mark"
  >
    <path d="M12 2.5l2.9 6.1 6.6.8-4.9 4.6 1.3 6.6L12 17.3l-5.9 3.3 1.3-6.6-4.9-4.6 6.6-.8z" />
  </svg>
  <span v-if="!decorative" class="sr-only">{{ t('podcast.sigDistinctiveName') }}</span>
</template>
