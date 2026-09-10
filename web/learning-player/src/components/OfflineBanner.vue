<script setup lang="ts">
/**
 * App-level offline indicator (F1.2) — a slim bar shown whenever the device reports offline, so the
 * listener always has ONE obvious signal that what they see is saved/cached. Complements, does not
 * replace, the per-section `StaleNotice` (operator decision 2026-09-09).
 *
 * `navigator.onLine` is coarse — it means "no network interface", not "nothing works" — so the copy
 * says *offline*, not *broken*: cached content still renders beneath this bar. `role="status"` so a
 * screen reader announces the state change without stealing focus.
 */
import { useI18n } from 'vue-i18n'
import { useOnline } from '../composables/useOnline'

const { t } = useI18n()
const { isOnline } = useOnline()
</script>

<template>
  <Transition name="offline-fade">
    <div
      v-if="!isOnline"
      role="status"
      data-testid="offline-banner"
      class="flex items-center justify-center gap-2 border-b border-border bg-elevated px-4 py-1.5 text-center text-xs font-semibold text-muted"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" class="h-3.5 w-3.5 shrink-0" aria-hidden="true">
        <path d="M1 1l22 22M16.72 11.06A10.94 10.94 0 0 1 19 12.55M5 12.55a10.94 10.94 0 0 1 5.17-2.39M10.71 5.05A16 16 0 0 1 22.58 9M1.42 9a15.91 15.91 0 0 1 4.7-2.88M8.53 16.11a6 6 0 0 1 6.95 0M12 20h.01" />
      </svg>
      {{ t('app.offline') }}
    </div>
  </Transition>
</template>

<style scoped>
/* The bar slides in from the top edge; it never covers content, only pushes it down. */
.offline-fade-enter-active,
.offline-fade-leave-active {
  transition:
    opacity 0.2s ease,
    transform 0.2s ease;
}
.offline-fade-enter-from,
.offline-fade-leave-to {
  opacity: 0;
  transform: translateY(-100%);
}
</style>
