<script setup lang="ts">
/**
 * Shared date filter chip. Emits a YYYY-MM-DD lower bound (or empty for
 * "all time") via v-model. Used by Library (#669), Digest (#670), and
 * Search (#671) — each surface wires its own state ref into the model.
 *
 * Behaviour:
 *   - Chip label: "Date ▾" when default; "Date: Last 7d / 30d / 90d ▾"
 *     for preset matches; "Date: ≥ YYYY-MM-DD ▾" for custom values.
 *   - Popover: "All time" / "7d" / "30d" / "90d" presets + a YYYY-MM-DD
 *     custom input + a "Clear" affordance when non-default.
 *   - Pointerdown / Escape close behaviour comes from
 *     ``useFilterChipPopover``.
 */
import { computed, ref, watch } from 'vue'
import {
  inferCorpusLensPreset,
  localYmdDaysAgo,
} from '../../utils/localCalendarDate'
import { useFilterChipPopover } from '../../composables/useFilterChipPopover'

const props = withDefaults(
  defineProps<{
    /** Lower-bound YYYY-MM-DD; empty string = "all time". */
    modelValue: string
    /** Visible label prefix. Default "Date". */
    label?: string
    /** ``data-testid`` on the chip ``<button>``. */
    chipTestid?: string
    /** ``data-testid`` on the popover ``<div role="dialog">``. */
    popoverTestid?: string
  }>(),
  {
    label: 'Date',
    chipTestid: 'date-chip',
    popoverTestid: 'date-popover',
  },
)

const emit = defineEmits<{
  (e: 'update:modelValue', v: string): void
}>()

const { open, anchorRef, panelRef, toggle, close } = useFilterChipPopover()

const preset = computed(() => inferCorpusLensPreset(props.modelValue))

const chipLabel = computed(() => {
  switch (preset.value) {
    case 'all':
      return `${props.label} ▾`
    case '7':
      return `${props.label}: Last 7d ▾`
    case '30':
      return `${props.label}: Last 30d ▾`
    case '90':
      return `${props.label}: Last 90d ▾`
    default:
      return `${props.label}: ≥ ${props.modelValue} ▾`
  }
})

const isActive = computed(() => preset.value !== 'all')

const customInput = ref(props.modelValue)

watch(
  () => props.modelValue,
  (v) => {
    customInput.value = v
  },
)

function setAll(): void {
  emit('update:modelValue', '')
}

function setPreset(days: 7 | 30 | 90): void {
  emit('update:modelValue', localYmdDaysAgo(days))
}

function commitCustom(): void {
  const v = customInput.value.trim()
  if (!/^\d{4}-\d{2}-\d{2}$/.test(v) && v !== '') return
  emit('update:modelValue', v)
}

function presetButtonClass(active: boolean): string {
  const base
    = 'inline-flex items-center rounded border px-2 py-0.5 text-[11px] leading-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary'
  return active
    ? `${base} border-primary bg-primary/10 font-medium text-surface-foreground`
    : `${base} border-border text-muted hover:bg-overlay`
}
</script>

<template>
  <div class="relative inline-flex items-center">
    <button
      ref="anchorRef"
      type="button"
      class="inline-flex h-6 items-center rounded border px-2 text-[11px] leading-none transition-colors hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
      :class="
        isActive
          ? 'border-border font-medium text-surface-foreground'
          : 'border-border/70 text-muted'
      "
      :data-testid="props.chipTestid"
      :aria-expanded="open"
      aria-haspopup="dialog"
      :aria-label="`${props.label} filter`"
      @click="toggle"
    >
      {{ chipLabel }}
    </button>
    <div
      v-show="open"
      ref="panelRef"
      role="dialog"
      :aria-label="`${props.label} filter`"
      :data-testid="props.popoverTestid"
      class="absolute left-0 top-full z-[40] mt-1 min-w-[14rem] rounded-sm border border-border bg-elevated p-2 shadow-md"
    >
      <div class="flex flex-wrap items-center gap-1">
        <button
          type="button"
          :class="presetButtonClass(preset === 'all')"
          @click="setAll"
        >
          All time
        </button>
        <button
          type="button"
          :class="presetButtonClass(preset === '7')"
          @click="setPreset(7)"
        >
          7d
        </button>
        <button
          type="button"
          :class="presetButtonClass(preset === '30')"
          @click="setPreset(30)"
        >
          30d
        </button>
        <button
          type="button"
          :class="presetButtonClass(preset === '90')"
          @click="setPreset(90)"
        >
          90d
        </button>
      </div>
      <div class="mt-2 border-t border-border pt-2">
        <label class="block text-[10px] font-semibold uppercase tracking-wider text-muted">
          Custom
        </label>
        <input
          v-model="customInput"
          type="date"
          class="mt-1 w-full rounded border border-border bg-elevated px-2 py-1 text-xs text-surface-foreground"
          aria-label="Custom date (YYYY-MM-DD)"
          :data-testid="`${props.chipTestid}-custom`"
          @keydown.enter="commitCustom(); close()"
          @blur="commitCustom"
        >
      </div>
      <button
        v-if="isActive"
        type="button"
        class="mt-2 w-full rounded border border-border px-2 py-1 text-[11px] text-primary hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        @click="setAll(); close()"
      >
        Clear
      </button>
    </div>
  </div>
</template>
