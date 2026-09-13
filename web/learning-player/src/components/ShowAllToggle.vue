<script setup lang="ts">
/**
 * The "Show all (N) / Show less" control under a capped Library section (#2042 follow-up). Kept as
 * one component so every section reads identically and the label stays translated in one place.
 */
import { useI18n } from 'vue-i18n'

defineProps<{ expanded: boolean; count: number }>()
defineEmits<{ toggle: [] }>()
const { t } = useI18n()
</script>

<template>
  <button
    type="button"
    class="mt-2 text-xs font-bold text-accent transition hover:opacity-80"
    data-testid="show-all-toggle"
    :aria-expanded="expanded"
    @click="$emit('toggle')"
  >
    {{ expanded ? t('library.showLess') : t('library.showAll', { count }) }}
  </button>
</template>
