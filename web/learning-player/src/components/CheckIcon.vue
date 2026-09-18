<script setup lang="ts">
/**
 * A tick — "I have reviewed this".
 *
 * Drawn rather than the character "✓" (U+2713), for the reason `CloseIcon` records: "✕" was a
 * character until it rendered as a tofu box in the iOS UI font, on every sheet and modal at once,
 * and drawing it was the only fix that held. A tick is from the same Unicode neighbourhood and is
 * not worth re-running the experiment on a device.
 *
 * Same viewBox, stroke weight and cap style as `CloseIcon` so the two sit level beside each other
 * in a row of controls.
 */
withDefaults(defineProps<{ size?: number }>(), { size: 16 })
</script>

<template>
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="2.5"
    stroke-linecap="round"
    stroke-linejoin="round"
    :width="size"
    :height="size"
    class="shrink-0"
    aria-hidden="true"
  >
    <path d="M4 12.5l5.5 5.5L20 6.5" />
  </svg>
</template>
