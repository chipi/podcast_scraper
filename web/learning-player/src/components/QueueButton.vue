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

const props = withDefaults(
  defineProps<{
    slug: string
    /**
     * `menuitem` renders this inside an `OverflowMenu` instead of as a circular icon button.
     *
     * Mirrors FavoriteButton / DownloadButton's variant, `data-menuitem` included so the menu's
     * arrow-key roaming picks it up. Used by Recently played, where the queue toggle is demoted
     * rather than dropped — it is the wrong PRIMARY action for a history list, but removing it
     * outright would take away the only way to queue something you just heard.
     */
    variant?: 'icon' | 'menuitem'
  }>(),
  { variant: 'icon' },
)
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
    :class="
      props.variant === 'menuitem'
        ? 'flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-canvas-foreground transition hover:bg-overlay'
        : [
            'lp-tap z-30 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border',
            queue.has(slug)
              ? 'border-accent text-accent'
              : 'border-border text-muted hover:text-canvas-foreground',
          ]
    "
    :data-menuitem="props.variant === 'menuitem' ? '' : undefined"
    :role="props.variant === 'menuitem' ? 'menuitem' : undefined"
    :aria-pressed="isGated || props.variant === 'menuitem' ? undefined : queue.has(slug)"
    :aria-label="isGated ? t('auth.signInToQueue') : queue.has(slug) ? t('queue.remove') : t('queue.add')"
    :title="props.variant === 'menuitem' ? undefined : isGated ? t('auth.signInToQueue') : queue.has(slug) ? t('queue.remove') : t('queue.add')"
    @click.stop.prevent="onClick"
  >
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4 shrink-0" aria-hidden="true">
      <!-- Queued: the SAME list metaphor as "add" but with a check instead of a plus, so it reads
           "in your queue" rather than a bare ✓ that looks like "selected/done" (operator 2026-09-14). -->
      <template v-if="queue.has(slug)">
        <path d="M13 6H3" /><path d="M13 12H3" /><path d="M13 18H3" /><path d="M15 16l2 2 4-4" />
      </template>
      <template v-else>
        <path d="M11 12H3" /><path d="M16 6H3" /><path d="M16 18H3" /><path d="M18 9v6" /><path d="M21 12h-6" />
      </template>
    </svg>
    <!-- The menu is a list of words; an icon alone there would be the only unlabelled row. -->
    <span v-if="props.variant === 'menuitem'">{{
      queue.has(slug) ? t('queue.remove') : t('queue.add')
    }}</span>
    <!-- The icon variant needs the same name, visually hidden (2026-09-25, Android device tier).
         The svg above is correctly `aria-hidden`, and with nothing else inside, Chromium left the
         button with NO usable name — which is why a device test could not find "Add to queue" on a
         page that plainly showed it, and reported the control as absent on Android. It was there;
         it had no name to be found by. A screen reader had the same problem, silently. -->
    <span v-else class="sr-only">{{
      isGated ? t('auth.signInToQueue') : queue.has(slug) ? t('queue.remove') : t('queue.add')
    }}</span>
  </button>
</template>
