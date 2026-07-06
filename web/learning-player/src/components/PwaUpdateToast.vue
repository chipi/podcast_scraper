<script setup lang="ts">
/**
 * Update-available toast — anchored to bottom-right, only visible when a
 * new service worker is waiting. Non-blocking; dismissable.
 *
 * The button explicitly says "Reload" (not "OK" / "Update") so the user
 * knows the page will refresh — matching the guide's #2 lesson that the
 * update path must be visible and understood, not silent.
 */
import { useI18n } from 'vue-i18n'
import { usePwaUpdate } from '../composables/usePwaUpdate'

const { t } = useI18n()
const { needRefresh, applyUpdate, dismissUpdate } = usePwaUpdate()
</script>

<template>
  <div
    v-if="needRefresh"
    role="status"
    aria-live="polite"
    class="fixed bottom-4 right-4 z-50 flex max-w-sm items-center gap-3 rounded-lg border border-border bg-surface px-4 py-3 shadow-lg"
    data-testid="pwa-update-toast"
  >
    <div class="flex-1">
      <p class="text-sm font-semibold text-canvas-foreground">
        {{ t('pwa.updateAvailable.title') }}
      </p>
      <p class="text-xs text-muted">
        {{ t('pwa.updateAvailable.body') }}
      </p>
    </div>
    <button
      type="button"
      class="rounded-full bg-accent px-3 py-1.5 text-xs font-bold text-accent-foreground"
      data-testid="pwa-update-reload"
      @click="applyUpdate"
    >
      {{ t('pwa.updateAvailable.reload') }}
    </button>
    <button
      type="button"
      class="text-xs text-muted underline"
      data-testid="pwa-update-dismiss"
      @click="dismissUpdate"
    >
      {{ t('pwa.updateAvailable.dismiss') }}
    </button>
  </div>
</template>
