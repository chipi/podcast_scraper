<script setup lang="ts">
/**
 * ProfileAvatar — the account's picture. Renders the photo when `src` is supplied (the OAuth
 * `picture` from `/me`, or a user upload — Area E); otherwise falls back to INITIALS on a
 * deterministic hue derived from the name, so every surface shows a stable mark, never a broken
 * image, when there is no photo.
 */
import { computed, ref, watch } from "vue"

const props = withDefaults(
  defineProps<{
    name?: string | null
    email?: string | null
    src?: string | null
    size?: number
    /** Circle for small avatars (the default everywhere); square (rounded) for a large, prominent
     *  portrait like the person card, where a circle crops too much of the face. */
    shape?: "circle" | "square"
  }>(),
  { name: null, email: null, src: null, size: 32, shape: "circle" }
)

// A broken photo (expired OAuth URL, deleted upload) falls back to initials rather than the
// browser's broken-image glyph. Reset when the src changes so a new upload gets a fresh try.
const failed = ref(false)
watch(
  () => props.src,
  () => (failed.value = false)
)
const showImg = computed(() => Boolean(props.src) && !failed.value)

const initials = computed(() => {
  const source = (props.name || props.email || "").trim()
  if (!source) return "?"
  const parts = source.split(/[\s@._-]+/).filter(Boolean)
  const first = parts[0]?.[0] ?? ""
  const second = parts.length > 1 ? parts[1]?.[0] ?? "" : ""
  return (first + second).toUpperCase() || "?"
})

// Deterministic hue from the name so the same account always gets the same colour.
const hue = computed(() => {
  const s = props.name || props.email || "?"
  let h = 0
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360
  return h
})
</script>

<template>
  <span
    class="inline-flex shrink-0 items-center justify-center overflow-hidden bg-elevated font-bold text-canvas-foreground"
    :class="shape === 'square' ? 'rounded-xl' : 'rounded-full'"
    :style="{ width: `${size}px`, height: `${size}px`, fontSize: `${Math.round(size * 0.4)}px` }"
    data-testid="profile-avatar"
    aria-hidden="true"
  >
    <img
      v-if="showImg"
      :src="src!"
      alt=""
      class="h-full w-full object-cover"
      @error="failed = true"
    />
    <span
      v-else
      class="flex h-full w-full items-center justify-center"
      :style="{ backgroundColor: `hsl(${hue} 45% 30%)` }"
      >{{ initials }}</span
    >
  </span>
</template>
