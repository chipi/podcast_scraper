<script setup lang="ts">
/**
 * The back chevron, as an inline SVG — the twin of {@link CloseIcon}, same box and stroke.
 *
 * Back used to be the character "‹" (U+2039) at body size. It rendered, but as a sliver a few
 * pixels wide next to the drawn 16px ✕ the episode-notes panel closes with, so the two dismiss
 * controls looked like they belonged to different apps (operator 2026-10-04).
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
    <path d="M15 6l-6 6 6 6" />
  </svg>
</template>
