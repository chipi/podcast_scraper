<script setup lang="ts" generic="T extends string">
/**
 * A labelled two-or-more-way switch (operator 2026-10-07): Trends' "You | Everyone" and
 * "Rising | Most talked about". They were icon-only buttons — a person and a chart — and a beta
 * tester could not tell what they did or whether the person one did anything at all. Words, both
 * options visible, the chosen one filled.
 *
 * Buttons with `aria-pressed` inside a labelled group: each option is a choice you can see and press,
 * which is what a reader expects; a tablist would promise panels that do not exist.
 */
defineProps<{
  options: { value: T; label: string; testid?: string }[]
  /** The group's accessible name, e.g. "Whose trends". */
  label: string
}>()
const model = defineModel<T>({ required: true })
</script>

<template>
  <div role="group" :aria-label="label" class="inline-flex shrink-0 rounded-full border border-border p-0.5">
    <button
      v-for="o in options"
      :key="o.value"
      type="button"
      class="lp-tap whitespace-nowrap rounded-full px-3 py-1 text-xs font-bold transition"
      :class="model === o.value ? 'bg-accent text-accent-foreground' : 'text-muted hover:text-canvas-foreground'"
      :aria-pressed="model === o.value"
      :data-testid="o.testid"
      @click="model = o.value"
    >{{ o.label }}</button>
  </div>
</template>
