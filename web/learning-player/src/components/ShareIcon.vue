<script setup lang="ts">
/**
 * The share glyph — three joined nodes, the one the episode page's share control has always drawn
 * (operator 2026-10-07: every share in the app wears it). Highlights used a bare "↗", which reads
 * as "open externally", not "share". Drawn rather than a character, per `CloseIcon`.
 */
withDefaults(defineProps<{ size?: number }>(), { size: 16 })
</script>

<template>
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    :width="size"
    :height="size"
    class="shrink-0"
    aria-hidden="true"
    data-testid="share-icon"
  >
    <circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" />
    <path d="m8.6 10.5 6.8-4M8.6 13.5l6.8 4" />
  </svg>
</template>
