<script setup lang="ts">
/**
 * The capture bookmark — filled when the thing is saved, outlined when it is not.
 *
 * The path is the one `TranscriptList` has drawn since captures existed; this component exists so
 * the two cannot drift (UXS-014: define once, use everywhere).
 *
 * NOT the heart. The heart (`FavoriteButton`) saves episodes, people, topics, shows and storylines;
 * this bookmark saves captures — marked moments, quotes, kept insights. `types.ts` states the split
 * outright: "an insight is a capture, saved via the highlights path, never a favorite"
 * (RFC-121 / #1593). Offering a heart to unsave a capture would name the wrong save.
 *
 * Drawn rather than a character, per `CloseIcon`: "🔖" is the same bet that already lost on device.
 */
withDefaults(defineProps<{ size?: number; filled?: boolean }>(), { size: 16, filled: false })
</script>

<template>
  <svg
    viewBox="0 0 24 24"
    :fill="filled ? 'currentColor' : 'none'"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    :width="size"
    :height="size"
    class="shrink-0"
    aria-hidden="true"
  >
    <path d="M6 3h12a1 1 0 0 1 1 1v17l-7-4-7 4V4a1 1 0 0 1 1-1z" />
  </svg>
</template>
