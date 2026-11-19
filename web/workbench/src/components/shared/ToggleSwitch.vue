<script setup lang="ts">
/**
 * Accessible on/off switch (`role="switch"`). Reusable primitive introduced for
 * the scheduled-jobs enable/disable toggle (#709); replaces the ad-hoc
 * `aria-pressed` buttons copied around the codebase.
 */
const props = withDefaults(
  defineProps<{
    modelValue: boolean
    disabled?: boolean
    /** Accessible name (e.g. "Enable nightly sweep"). */
    label?: string
    testid?: string
  }>(),
  { disabled: false, label: '', testid: 'toggle-switch' },
)

const emit = defineEmits<{ 'update:modelValue': [boolean] }>()

function toggle(): void {
  if (!props.disabled) {
    emit('update:modelValue', !props.modelValue)
  }
}
</script>

<template>
  <button
    type="button"
    role="switch"
    :aria-checked="modelValue"
    :aria-label="label || undefined"
    :disabled="disabled"
    :data-testid="testid"
    class="relative inline-flex h-4 w-7 shrink-0 items-center rounded-full border border-border transition-colors disabled:cursor-not-allowed disabled:opacity-40"
    :class="modelValue ? 'bg-primary' : 'bg-overlay'"
    @click="toggle"
  >
    <span
      class="inline-block h-3 w-3 rounded-full bg-surface shadow-sm transition-transform"
      :class="modelValue ? 'translate-x-3.5' : 'translate-x-0.5'"
    />
  </button>
</template>
