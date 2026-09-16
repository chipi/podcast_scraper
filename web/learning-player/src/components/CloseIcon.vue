<script setup lang="ts">
/**
 * The close ✕, as an inline SVG.
 *
 * It WAS the character "✕" (U+2715). That glyph is absent from the iOS UI font and rendered as a
 * tofu box on device — confirmed by cropping a device screenshot, 2026-09-16 — so every sheet,
 * panel and modal showed "?" where its dismiss control should be. Adding emoji/symbol faces to the
 * font stacks did NOT fix it; drawing the glyph is the only treatment that has held (the same
 * conclusion reached for the avatar pencil and the streak flame).
 *
 * One component rather than several copied SVGs, so the next person changes it once.
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
    :width="size"
    :height="size"
    class="shrink-0"
    aria-hidden="true"
  >
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
</template>
