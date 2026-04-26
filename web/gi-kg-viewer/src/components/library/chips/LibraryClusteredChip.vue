<script setup lang="ts">
/**
 * Library "Clustered" toggle chip (#669). No popover — a single click
 * flips ``topicClusterOnly``. Active state uses the filled chip style.
 */
import { computed } from 'vue'

const props = defineProps<{
  modelValue: boolean
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: boolean): void
}>()

const isActive = computed(() => props.modelValue)

function toggle(): void {
  emit('update:modelValue', !props.modelValue)
}
</script>

<template>
  <button
    type="button"
    class="inline-flex h-6 items-center rounded border px-2 text-[11px] leading-none transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
    :class="
      isActive
        ? 'border-primary bg-primary/15 font-medium text-surface-foreground'
        : 'border-border/70 text-muted hover:bg-overlay'
    "
    data-testid="library-chip-clustered"
    :aria-pressed="isActive"
    aria-label="Toggle clustered episodes only"
    @click="toggle"
  >
    {{ isActive ? 'Clustered ✓' : 'Clustered' }}
  </button>
</template>
