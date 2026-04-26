<script setup lang="ts">
/**
 * Explore text chip (#671) — generic substring popover. Bound to a single
 * ``string`` v-model (Topic or Speaker substring). Active when non-empty.
 */
import { computed, ref, watch } from 'vue'
import { useFilterChipPopover } from '../../../composables/useFilterChipPopover'

const props = withDefaults(
  defineProps<{
    modelValue: string
    label: string
    chipTestid: string
    popoverTestid: string
    placeholder?: string
    /** If false, chip is disabled (no toggle, dimmed). */
    enabled?: boolean
    /** Tooltip when disabled. */
    disabledTitle?: string
  }>(),
  { enabled: true, placeholder: '', disabledTitle: '' },
)

const emit = defineEmits<{
  (e: 'update:modelValue', v: string): void
  (e: 'submit'): void
}>()

const anchorRef = ref<HTMLButtonElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)
const { open, toggle, close } = useFilterChipPopover(anchorRef, panelRef)

const draft = ref(props.modelValue)
watch(
  () => props.modelValue,
  (v) => {
    draft.value = v
  },
)

const isActive = computed(() => props.modelValue.trim().length > 0)

const chipLabel = computed(() => {
  if (!isActive.value) return `${props.label} ▾`
  const v = props.modelValue.trim()
  const trim = v.length > 18 ? `${v.slice(0, 17)}…` : v
  return `${props.label}: ${trim} ▾`
})

function commit(): void {
  emit('update:modelValue', draft.value)
}

function clear(): void {
  draft.value = ''
  emit('update:modelValue', '')
}

function commitAndSubmit(): void {
  commit()
  close()
  emit('submit')
}
</script>

<template>
  <div class="relative inline-flex items-center">
    <button
      ref="anchorRef"
      type="button"
      class="inline-flex h-6 items-center rounded border px-2 text-[11px] leading-none transition-colors hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary disabled:opacity-50"
      :class="
        isActive
          ? 'border-border font-medium text-surface-foreground'
          : 'border-border/70 text-muted'
      "
      :data-testid="props.chipTestid"
      :aria-expanded="open"
      aria-haspopup="dialog"
      :aria-label="`${props.label} contains`"
      :disabled="!props.enabled"
      :title="props.enabled ? undefined : props.disabledTitle"
      @click="toggle"
    >
      {{ chipLabel }}
    </button>
    <div
      v-show="open"
      ref="panelRef"
      role="dialog"
      :aria-label="`${props.label} contains`"
      :data-testid="props.popoverTestid"
      class="absolute left-0 top-full z-[40] mt-1 w-56 rounded-sm border border-border bg-elevated p-2 shadow-md"
    >
      <label class="block text-[10px] font-semibold uppercase tracking-wider text-muted">
        {{ props.label }} contains
      </label>
      <input
        v-model="draft"
        type="text"
        class="mt-1 w-full rounded border border-border bg-elevated px-2 py-1 text-xs text-surface-foreground"
        :placeholder="props.placeholder"
        :data-testid="`${props.chipTestid}-input`"
        @blur="commit"
        @keydown.enter.prevent="commitAndSubmit"
      >
      <button
        v-if="isActive"
        type="button"
        class="mt-2 w-full rounded border border-border px-2 py-1 text-[11px] text-primary hover:bg-overlay"
        :data-testid="`${props.chipTestid}-clear`"
        @click="clear(); close()"
      >
        Clear
      </button>
    </div>
  </div>
</template>
