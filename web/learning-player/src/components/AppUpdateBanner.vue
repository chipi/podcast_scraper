<script setup lang="ts">
/**
 * App-update banner (UXS-011, wave-I.6) — a non-blocking bottom banner shown on NATIVE when the
 * server's released `player_version` is newer than this baked build. Web updates flow through the
 * service worker instead (`PwaUpdateToast`), so this never shows on web.
 *
 * When a store URL is configured it offers an Update button to it; pre-launch (no store URL yet)
 * the banner is informational — a truthful "update available" without a dead link. Dismissable.
 */
import { onMounted } from 'vue'
import { useI18n } from 'vue-i18n'

import { useAppUpdate } from '../composables/useAppUpdate'

const { t } = useI18n()
const { updateAvailable, latestVersion, dismissed, dismiss, storeUrl, check } = useAppUpdate()

onMounted(check)
</script>

<template>
  <div
    v-if="updateAvailable && !dismissed"
    role="status"
    aria-live="polite"
    class="fixed bottom-4 left-1/2 z-50 flex w-[calc(100vw-2rem)] max-w-sm -translate-x-1/2 items-center gap-3 rounded-lg border border-border bg-surface px-4 py-3 shadow-lg"
    data-testid="app-update-banner"
  >
    <div class="min-w-0 flex-1">
      <p class="text-sm font-semibold text-canvas-foreground">{{ t('appUpdate.title') }}</p>
      <p class="text-xs text-muted">{{ t('appUpdate.body', { version: latestVersion }) }}</p>
    </div>
    <a
      v-if="storeUrl"
      :href="storeUrl"
      target="_blank"
      rel="noopener"
      class="shrink-0 rounded-full bg-accent px-3 py-1.5 text-xs font-bold text-accent-foreground no-underline"
      data-testid="app-update-action"
    >{{ t('appUpdate.action') }}</a>
    <button
      type="button"
      class="shrink-0 text-xs text-muted underline"
      data-testid="app-update-dismiss"
      @click="dismiss"
    >{{ t('appUpdate.dismiss') }}</button>
  </div>
</template>
