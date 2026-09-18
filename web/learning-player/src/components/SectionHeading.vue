<script setup lang="ts">
/**
 * One section header for the whole page (UXS-014: define once, use everywhere).
 *
 * Home had grown three of them by hand: a kicker above the title on some sections and beside it on
 * others, three different title fonts (`h1` display, `h2` display, `h2 lp-section`), and nothing
 * stopping either line from wrapping to a second row. The result read as several designs sharing a
 * page (operator 2026-09-18).
 *
 * The contract:
 *
 *   KICKER            small caps, muted, ONE line, truncated — a count or a date, never a
 *                     restatement of the title beneath it
 *   Title   [action]  `lp-section`, ONE line, truncated; the action (a "See all" link) sits at the
 *                     far right of the same row
 *
 * The kicker goes ABOVE, not beside. Beside, it competes with the action link for the same edge and
 * the two collide on a narrow column; above, the title always starts at the same x on every
 * section, which is what makes a page of stacked sections look deliberate.
 *
 * Both lines truncate rather than wrap. A heading that reflows to two rows shifts everything under
 * it and makes otherwise identical sections different heights.
 */
defineProps<{
  title: string
  /** A count or a date. Omit it rather than passing a label that repeats the title. */
  kicker?: string | null
}>()
</script>

<template>
  <div class="mb-3">
    <p v-if="kicker" class="lp-kicker truncate" data-testid="section-kicker">{{ kicker }}</p>
    <div class="flex items-baseline gap-3">
      <h2 class="lp-section min-w-0 truncate" data-testid="section-title">{{ title }}</h2>
      <!-- `ms-auto` on the slot rather than on the caller's element, so every action lands in the
           same place whether or not the caller remembered to push it right. -->
      <div v-if="$slots.action" class="ms-auto shrink-0"><slot name="action" /></div>
    </div>
  </div>
</template>
