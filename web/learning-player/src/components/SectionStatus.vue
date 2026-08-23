<script setup lang="ts">
/**
 * The loading and error halves of a section's lifecycle (#1591).
 *
 * One component so every section fails the same way. Before this, error styling was inconsistent
 * for the same class of failure — `text-muted` in one view, `text-danger` in another — and **no
 * error state anywhere in the app offered a retry**. A user whose request failed had no move except
 * reloading the page.
 *
 * Renders nothing when ready; the caller owns the success and empty cases, since only the caller
 * knows whether its emptiness is actionable.
 */
import { useI18n } from 'vue-i18n'
import type { SectionPhase } from '../composables/useSectionState'

defineProps<{
  phase: SectionPhase
  /** Skeleton rows to show while loading. Match the section's real shape so nothing jumps. */
  rows?: number
}>()

const emit = defineEmits<{ (e: 'retry'): void }>()
const { t } = useI18n()
</script>

<template>
  <div v-if="phase === 'loading'" class="space-y-2" :aria-busy="true" data-testid="section-loading">
    <div
      v-for="i in rows ?? 2"
      :key="i"
      class="h-14 animate-pulse rounded-xl bg-elevated"
      aria-hidden="true"
    />
    <span class="sr-only">{{ t('section.loading') }}</span>
  </div>

  <div
    v-else-if="phase === 'error'"
    class="flex flex-wrap items-center gap-3 rounded-xl border border-border px-4 py-3"
    role="status"
    data-testid="section-error"
  >
    <p class="text-sm text-muted">{{ t('section.error') }}</p>
    <button
      type="button"
      class="rounded-full border border-border px-3 py-1 text-xs font-bold text-canvas-foreground transition hover:bg-overlay"
      data-testid="section-retry"
      @click="emit('retry')"
    >
      {{ t('section.retry') }}
    </button>
  </div>
</template>
