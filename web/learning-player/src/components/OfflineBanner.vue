<script setup lang="ts">
/**
 * App-level offline indicator (F1.2) — a slim bar shown whenever the device reports offline, so the
 * listener always has ONE obvious signal that what they see is saved/cached. Complements, does not
 * replace, the per-section `StaleNotice` (operator decision 2026-09-09).
 *
 * `navigator.onLine` is coarse — it means "no network interface", not "nothing works" — so the copy
 * says *offline*, not *broken*: cached content still renders beneath this bar. The `role="status"`
 * live region is a PERSISTENT wrapper (always in the DOM), with only the bar toggled inside it — a
 * live region must exist before its content changes for a screen reader to announce it, so putting
 * the role on the `v-if` node itself would miss the very transition it exists to announce.
 */
import { useI18n } from 'vue-i18n'
import { useOnline } from '../composables/useOnline'

const { t } = useI18n()
const { isOnline } = useOnline()
</script>

<template>
  <div role="status" aria-live="polite">
    <Transition name="offline-fade">
      <div
        v-if="!isOnline"
        data-testid="offline-banner"
        class="flex items-center justify-center gap-2 border-b border-border bg-elevated px-4 py-1.5 text-center text-xs font-semibold text-muted"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" class="h-3.5 w-3.5 shrink-0" aria-hidden="true">
          <path d="M1 1l22 22M16.72 11.06A10.94 10.94 0 0 1 19 12.55M5 12.55a10.94 10.94 0 0 1 5.17-2.39M10.71 5.05A16 16 0 0 1 22.58 9M1.42 9a15.91 15.91 0 0 1 4.7-2.88M8.53 16.11a6 6 0 0 1 6.95 0M12 20h.01" />
        </svg>
        {{ t('app.offline') }}
      </div>
    </Transition>
  </div>
</template>

<style scoped>
/* Slide in from the top edge on ENTER; fade out on LEAVE. The bar is in normal flow, so animating
 * `transform` on leave would snap the freed height when the node is removed — leave fades opacity
 * only, which reads clean without the jump. */
.offline-fade-enter-active {
  transition:
    opacity 0.2s ease,
    transform 0.2s ease;
}
.offline-fade-leave-active {
  transition: opacity 0.15s ease;
}
.offline-fade-enter-from {
  opacity: 0;
  transform: translateY(-100%);
}
.offline-fade-leave-to {
  opacity: 0;
}
</style>
