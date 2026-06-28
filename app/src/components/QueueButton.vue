<script setup lang="ts">
/**
 * The ONE add/remove-to-queue control — same icon + behaviour everywhere (EpisodeCard, Home rails,
 * Recommended). Auth-gated (hidden signed-out). `@click.stop.prevent` so it queues the episode
 * instead of following a surrounding card link.
 */
import { useI18n } from 'vue-i18n'
import { useAuthStore } from '../stores/auth'
import { useQueueStore } from '../stores/queue'

defineProps<{ slug: string }>()
const { t } = useI18n()
const auth = useAuthStore()
const queue = useQueueStore()
</script>

<template>
  <button
    v-if="auth.isAuthenticated"
    type="button"
    class="relative z-30 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border"
    :class="queue.has(slug) ? 'border-accent text-accent' : 'border-border text-muted hover:text-canvas-foreground'"
    :aria-pressed="queue.has(slug)"
    :aria-label="queue.has(slug) ? t('queue.remove') : t('queue.add')"
    :title="queue.has(slug) ? t('queue.remove') : t('queue.add')"
    @click.stop.prevent="queue.toggle(slug)"
  >
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true">
      <template v-if="queue.has(slug)"><path d="M20 6 9 17l-5-5" /></template>
      <template v-else>
        <path d="M11 12H3" /><path d="M16 6H3" /><path d="M16 18H3" /><path d="M18 9v6" /><path d="M21 12h-6" />
      </template>
    </svg>
  </button>
</template>
