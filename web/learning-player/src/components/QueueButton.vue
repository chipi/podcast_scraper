<script setup lang="ts">
/**
 * The ONE add/remove-to-queue control — same icon + behaviour everywhere (EpisodeCard, Home rails,
 * Recommended). Renders for signed-out visitors too (#1590) — the queue is a capability worth
 * knowing about; tapping while signed out routes to sign-in and returns here. `@click.stop.prevent`
 * so it queues the episode instead of following a surrounding card link.
 *
 * Works OFFLINE (#1925). It used to be disabled whenever the queue was `stale` — a cached copy —
 * because every mutation went through a whole-list PUT that would have deleted the server's queue.
 * Add and remove are item-level and idempotent now, so an offline tap is queued in the outbox and
 * replayed; only REORDERING still needs a live list.
 */
import { useI18n } from 'vue-i18n'
import { useQueueStore } from '../stores/queue'
import { useSignInGate } from '../composables/useSignInGate'

const props = defineProps<{ slug: string }>()
const { t } = useI18n()
const queue = useQueueStore()

const { isGated, gated } = useSignInGate()
const onClick = gated(async () => {
  // The action reports whether the write survived (#1906); the gate's handler type is void.
  await queue.toggle(props.slug)
})
</script>

<template>
  <!-- Rendered signed-out too (#1590): hiding it left visitors with no evidence the queue exists.
       Tapping while signed out routes to sign-in and comes back here. -->
  <button
    type="button"
    class="relative z-30 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border"
    :class="queue.has(slug) ? 'border-accent text-accent' : 'border-border text-muted hover:text-canvas-foreground'"
    :aria-pressed="isGated ? undefined : queue.has(slug)"
    :aria-label="isGated ? t('auth.signInToQueue') : queue.has(slug) ? t('queue.remove') : t('queue.add')"
    :title="isGated ? t('auth.signInToQueue') : queue.has(slug) ? t('queue.remove') : t('queue.add')"
    @click.stop.prevent="onClick"
  >
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true">
      <template v-if="queue.has(slug)"><path d="M20 6 9 17l-5-5" /></template>
      <template v-else>
        <path d="M11 12H3" /><path d="M16 6H3" /><path d="M16 18H3" /><path d="M18 9v6" /><path d="M21 12h-6" />
      </template>
    </svg>
  </button>
</template>
