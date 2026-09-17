<script setup lang="ts">
/**
 * Multi-select "kind" chips — none selected means All.
 *
 * This markup existed twice, verbatim, in `SavedFilterBar` and `SearchView`, and the Boards tab
 * wanted a third copy (operator 2026-09-17). Type filtering is one concept, so it is one component:
 * a new surface passes its options and a testid prefix instead of re-deriving the chip styling and
 * the toggle logic.
 *
 * "All" is an explicit chip rather than an implicit empty state so clearing a filter is one tap,
 * and selection is a plain array so the caller can persist or watch it.
 */
import { useI18n } from "vue-i18n"

export interface TypeFilterOption {
  key: string
  label: string
}

const props = defineProps<{
  /** Selected keys. Empty = All. */
  modelValue: string[]
  options: TypeFilterOption[]
  /** Testid stem: yields `<prefix>-filter`, `<prefix>-all`, `<prefix>-<key>`. */
  testidPrefix: string
  /** Overrides the "All" chip wording where a surface needs different words. */
  allLabel?: string
}>()

const emit = defineEmits<{ "update:modelValue": [string[]] }>()

const { t } = useI18n()

function toggle(key: string): void {
  emit(
    "update:modelValue",
    props.modelValue.includes(key)
      ? props.modelValue.filter((k) => k !== key)
      : [...props.modelValue, key]
  )
}
</script>

<template>
  <div
    class="flex flex-wrap items-center gap-2"
    :data-testid="`${testidPrefix}-filter`"
  >
    <button
      type="button"
      class="rounded-full border px-3 py-1 text-xs font-semibold transition"
      :class="
        modelValue.length === 0
          ? 'border-accent bg-accent/10 text-accent'
          : 'border-border text-muted hover:text-canvas-foreground'
      "
      :aria-pressed="modelValue.length === 0"
      :data-testid="`${testidPrefix}-all`"
      @click="emit('update:modelValue', [])"
    >
      {{ allLabel ?? t("library.savedFilterAllTypes") }}
    </button>
    <button
      v-for="ty in options"
      :key="ty.key"
      type="button"
      class="rounded-full border px-3 py-1 text-xs font-semibold transition"
      :class="
        modelValue.includes(ty.key)
          ? 'border-accent bg-accent/10 text-accent'
          : 'border-border text-muted hover:text-canvas-foreground'
      "
      :aria-pressed="modelValue.includes(ty.key)"
      :data-testid="`${testidPrefix}-${ty.key}`"
      @click="toggle(ty.key)"
    >
      {{ ty.label }}
    </button>
  </div>
</template>
