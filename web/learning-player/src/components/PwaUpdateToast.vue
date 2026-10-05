<script setup lang="ts">
/**
 * Update-available toast — anchored to bottom-right, only visible when a
 * new service worker is waiting. Non-blocking; dismissable.
 *
 * WEB ONLY. In the native app a store update leaves a new worker waiting too, so this showed on the
 * first launch after every install — over the tab bar, where a tap on Library landed on the toast
 * and did nothing (2026-10-05). Native updates arrive with the build and have `AppUpdateBanner`;
 * the waiting worker takes over on the next launch by itself.
 *
 * Lifted clear of the tab bar (phones) and the mini-player (when one is loaded) — the same numbers
 * App.vue reserves as the page's bottom padding — so it never sits on a control.
 *
 * The button explicitly says "Reload" (not "OK" / "Update") so the user
 * knows the page will refresh — matching the guide's #2 lesson that the
 * update path must be visible and understood, not silent.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { usePwaUpdate } from '../composables/usePwaUpdate'
import { isNative } from '../services/native'
import { usePlayerStore } from '../stores/player'
import { track } from '../services/analytics'

const { t } = useI18n()
const { needRefresh, applyUpdate, dismissUpdate } = usePwaUpdate()
const native = isNative()
const player = usePlayerStore()
const lift = computed(() =>
  player.currentSlug
    ? 'bottom-[calc(8.25rem+env(safe-area-inset-bottom))] sm:bottom-[calc(6rem+env(safe-area-inset-bottom))]'
    : 'bottom-[calc(4.25rem+env(safe-area-inset-bottom))] sm:bottom-4',
)

// #2267. `kind: 'pwa_toast'` — the native banner is a separate surface and reports its own. Whether
// people take updates matters for the beta specifically: a tester on a stale build reports bugs
// that are already fixed, and the guide asks the operator to log which build each person saw.
function onAccept(): void {
  track('update_prompt', { kind: 'pwa_toast', action: 'accepted' })
  applyUpdate()
}
function onDismiss(): void {
  track('update_prompt', { kind: 'pwa_toast', action: 'dismissed' })
  dismissUpdate()
}
</script>

<template>
  <div
    v-if="needRefresh && !native"
    role="status"
    aria-live="polite"
    :class="lift"
    class="fixed right-4 z-50 flex max-w-sm items-center gap-3 rounded-lg border border-border bg-surface px-4 py-3 shadow-lg"
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
      @click="onAccept"
    >
      {{ t('pwa.updateAvailable.reload') }}
    </button>
    <button
      type="button"
      class="text-xs text-muted underline"
      data-testid="pwa-update-dismiss"
      @click="onDismiss"
    >
      {{ t('pwa.updateAvailable.dismiss') }}
    </button>
  </div>
</template>
