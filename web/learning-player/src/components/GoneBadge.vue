<script setup lang="ts">
/** "No longer available" on a stored reference the corpus no longer has (#2345). */
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
</script>

<template>
  <span class="text-xs font-semibold text-muted" data-testid="gone-badge">{{ t('common.noLongerAvailable') }}</span>
</template>
