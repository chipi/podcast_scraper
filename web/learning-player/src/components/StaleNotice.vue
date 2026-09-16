<script setup lang="ts">
/**
 * One line at the top of a page whose sections are showing what they had last time (#1909).
 *
 * ## Why one notice and not eight
 *
 * With no network Home rendered five identical "Couldn't load this right now" cards — the app
 * saying one thing five times, in the place where the content should have been. Now the rails keep
 * their content and this says the one true thing once, above them, where a page-level fact belongs.
 *
 * ## Why it must carry the retry
 *
 * A stale section no longer renders an error card, so it no longer offers its own "Try again". If
 * this did not carry one, the change would have traded a wall of noise for a page with no way to
 * refresh — quieter, and worse. The retry is the whole reason this is a control and not a caption.
 *
 * Deliberately calm: muted, no icon, no colour. It reports a condition the user can do nothing
 * about except retry, and an alarm-coloured banner over content that is perfectly readable would
 * overstate it.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useOnline } from '../composables/useOnline'

defineProps<{ busy?: boolean }>()
defineEmits<{ (e: 'retry'): void }>()

const { t } = useI18n()
const { offlineReason } = useOnline()

/**
 * Say the true thing for the reason we are actually offline (2026-09-16).
 *
 * The one sentence this used to show — "we couldn't reach the server" — is simply false in the
 * forced case: the app deliberately did not ask, and the server is fine.
 */
const message = computed(() => {
  switch (offlineReason.value) {
    case 'forced':
      return t('home.staleForced')
    case 'network':
      return t('home.staleNetwork')
    case 'server':
      return t('home.staleServer')
    default:
      // Not offline by any signal, yet a section is stale: a request really was made and really
      // failed, so the original wording is the honest one.
      return t('home.stale')
  }
})

/**
 * Retry is hidden in FORCED offline only. The read gate refuses the request while the switch is on,
 * so the button could never succeed — an affordance that cannot work is worse than none. Every
 * other reason keeps it, including `server`, where retrying is exactly the right move.
 */
const canRetry = computed(() => offlineReason.value !== 'forced')
</script>

<template>
  <div
    class="mb-4 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-border px-3 py-2"
    data-testid="stale-notice"
    role="status"
  >
    <p class="text-sm text-muted">{{ message }}</p>
    <button
      v-if="canRetry"
      type="button"
      class="shrink-0 rounded-full border border-border px-3 py-1 text-xs font-bold text-canvas-foreground transition hover:bg-overlay disabled:opacity-50"
      data-testid="stale-retry"
      :disabled="busy"
      @click="$emit('retry')"
    >
      {{ busy ? t('home.staleRetrying') : t('home.staleRetry') }}
    </button>
  </div>
</template>
