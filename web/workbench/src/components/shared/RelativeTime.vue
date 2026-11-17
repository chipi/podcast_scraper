<script setup lang="ts">
import { ref, watch } from 'vue'
import { formatAbsoluteUtc, formatRelativeShort } from '../../utils/relativeTime'

/**
 * Compact relative timestamp ("in 3h 12m") with the absolute UTC time on hover.
 * Used for scheduled-jobs `next_run_at` (#709); `nowMs` is captured per render
 * (refreshed when the value changes) — no background ticker.
 */
const props = withDefaults(
  defineProps<{ iso: string | null | undefined; testid?: string }>(),
  { testid: 'relative-time' },
)

const nowMs = ref(Date.now())
watch(
  () => props.iso,
  () => {
    nowMs.value = Date.now()
  },
)
</script>

<template>
  <span
    :data-testid="testid"
    :title="formatAbsoluteUtc(iso) || undefined"
  >{{ formatRelativeShort(iso, nowMs) }}</span>
</template>
