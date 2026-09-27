<script setup lang="ts">
/**
 * "You've heard this" — the marker an episode list needs and did not have (operator 2026-09-23).
 *
 * Reported from the queue: the recently-played list showed finished episodes looking exactly like
 * unheard ones, so the one thing that list exists to tell you was the one thing it did not say.
 *
 * Deliberately a LABELLED check, not a bare tick. A tick alone is decoration that has to be
 * learned, it carries nothing for a screen reader without a title attribute nobody reads, and this
 * app already uses a bare check for "reviewed" on Revisit cards — two meanings, one glyph. The word
 * costs about thirty pixels.
 *
 * Muted, not accent: this is settled history, not something to act on. Accent is reserved for the
 * live/actionable state (what is playing, what is due), and a row of bright green ticks down a
 * finished list would out-shout the episodes the listener still has to get to.
 *
 * Renders NOTHING when unplayed — an "Unplayed" badge on every row would be noise on the common
 * case, and absence already reads as unplayed.
 */
import { useI18n } from 'vue-i18n'
import { usePlayed } from '../composables/usePlayed'

const props = defineProps<{ slug: string }>()
const { t } = useI18n()
const { isPlayed } = usePlayed()
</script>

<template>
  <span
    v-if="isPlayed(props.slug)"
    class="inline-flex w-fit shrink-0 items-center gap-1 rounded-full bg-overlay px-2 py-0.5 text-xs font-semibold text-muted"
    data-testid="played-badge"
  >
    <!-- `aria-hidden`: the label beside it already says it, and an unlabelled path announced as
         "graphic" adds a second, worse reading of the same fact. -->
    <svg
      class="h-3 w-3"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      stroke-width="2.5"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path d="M3 8.5 6.5 12 13 4.5" />
    </svg>
    {{ t('status.played') }}
  </span>
</template>
