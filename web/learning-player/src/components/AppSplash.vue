<script setup lang="ts">
/**
 * Web launch overlay (native shell). Shown ON TOP of the app for a brief, deliberate moment after
 * the native launch splash hands off — so (a) the branded splash is actually seen, (b) a build is
 * identifiable via the version line, and (c) the app + its first data load underneath while it's up,
 * so the home screen is mostly-populated by the time it fades. Uses the SAME image as the native
 * splash (public/splash.jpg) for a seamless native→web transition. Dismissed by the parent.
 */
defineProps<{ version: string }>()
const bg = `${import.meta.env.BASE_URL}splash.jpg`
</script>

<template>
  <div
    class="fixed inset-0 z-[100] bg-canvas bg-cover bg-center"
    :style="{ backgroundImage: `url(${bg})` }"
    aria-hidden="true"
  >
    <div
      class="absolute inset-x-0 bottom-0 pb-[max(2.1rem,calc(env(safe-area-inset-bottom)+0.9rem))] text-center text-xs font-semibold uppercase tracking-[0.22em] text-canvas-foreground/70"
    >
      {{ version }}
    </div>
  </div>
</template>
