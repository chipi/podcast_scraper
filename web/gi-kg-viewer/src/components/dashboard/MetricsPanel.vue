<script setup lang="ts">
import type { MetricRow } from '../../utils/metrics'

defineProps<{
  title: string
  rows: MetricRow[]
}>()
</script>

<template>
  <div class="rounded border border-border bg-surface p-3 text-surface-foreground">
    <h3 class="mb-2 text-sm font-semibold">
      {{ title }}
    </h3>
    <dl class="space-y-1 text-xs">
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
