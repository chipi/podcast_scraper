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
import { useI18n } from 'vue-i18n'

defineProps<{ busy?: boolean }>()
defineEmits<{ (e: 'retry'): void }>()

const { t } = useI18n()
</script>

<template>
  <div
    class="mb-4 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-border px-3 py-2"
    data-testid="stale-notice"
    role="status"
  >
    <p class="text-sm text-muted">{{ t('home.stale') }}</p>
    <button
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
