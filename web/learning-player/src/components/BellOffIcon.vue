<script setup lang="ts">
/**
 * The notifications bell with a line through it — "stop asking me about this".
 *
 * Deliberately the SAME bell the masthead uses (`NotificationsBell`, identical path data) with a
 * slash added, so the icon inherits a meaning the user has already learned on this app rather than
 * teaching a new one (operator 2026-09-18).
 *
 * Drawn, not a character. `CloseIcon` documents why: the "✕" glyph was absent from the iOS UI font
 * and rendered as a tofu box on device, and adding symbol faces to the font stack did not fix it.
 * Any new icon here follows that conclusion instead of rediscovering it — a "🔕" or "⊘" would be
 * the same bet that already lost.
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
  >
    <!-- The masthead bell, unchanged. -->
    <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
    <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
    <!-- The slash. Corner to corner so it reads as "off" at 16px, where a short stroke through the
         middle of the bell just looks like part of the drawing. -->
    <path d="M3 3l18 18" />
  </svg>
</template>
