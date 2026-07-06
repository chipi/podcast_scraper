<script setup lang="ts">
/** Keyboard-first skip link (UXS-011 a11y): jumps focus to #main, visible only on focus. */
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
</script>

<template>
  <a
    href="#main"
    class="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded-md focus:bg-elevated focus:px-4 focus:py-2 focus:text-surface-foreground"
  >
    {{ t('nav.skipToContent') }}
  </a>
</template>
