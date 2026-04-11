<script setup lang="ts">
import type { MetricRow } from '../../utils/metrics'

withDefaults(
  defineProps<{
    title: string
    rows: MetricRow[]
    /** Tighter typography for nested panels (e.g. left rail Data card). */
    dense?: boolean
    /** Omit outer border/surface — parent provides the card (e.g. left rail elevated panel). */
    unframed?: boolean
    /** Parent supplies the section heading (e.g. card header row with actions). */
    hideTitle?: boolean
  }>(),
  { dense: false, unframed: false, hideTitle: false },
)
</script>

<template>
  <div
    class="text-surface-foreground"
    :class="
      unframed
        ? dense
          ? 'text-[10px]'
          : 'text-xs'
        : dense
          ? 'rounded border border-border bg-surface p-2 text-[10px]'
          : 'rounded border border-border bg-surface p-3 text-xs'
    "
  >
    <h3
      v-if="!hideTitle"
      class="mb-2 font-semibold"
      :class="dense ? 'text-xs' : 'text-sm'"
    >
      {{ title }}
    </h3>
    <dl
      class="space-y-1"
      :class="dense ? 'text-[10px]' : 'text-xs'"
    >
      <div
        v-for="(r, i) in rows"
        :key="`${r.k}-${i}`"
        class="flex gap-2 border-b border-border/60 py-1 last:border-0"
      >
        <dt class="w-[40%] shrink-0 font-medium text-muted">
          {{ r.k }}
        </dt>
        <dd class="min-w-0 break-words text-surface-foreground">
          {{ r.v }}
        </dd>
      </div>
    </dl>
  </div>
</template>
