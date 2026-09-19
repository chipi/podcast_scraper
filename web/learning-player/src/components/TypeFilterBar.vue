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
 *
 * ONE ROW, scrolled horizontally — never wrapped (operator 2026-09-19). It used to `flex-wrap`, so
 * adding a fifth kind dropped the last chip onto a line of its own and pushed the content below it
 * down. Since this component now backs four surfaces (Following, Saved, Notes, Search) that was
 * four places to notice it. The idiom is the Knowledge Panel's insight-type strip: `overflow-x-auto`
 * with the scrollbar hidden and `shrink-0` on every chip, so they run off the edge instead of
 * squashing or reflowing.
 */
import { useI18n } from "vue-i18n"

export interface TypeFilterOption {
  key: string
  label: string
  /** Optional tally shown after the label — how many items this filter would leave. */
  count?: number
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
    class="flex items-center gap-2 overflow-x-auto pb-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
    :data-testid="`${testidPrefix}-filter`"
  >
    <button
      type="button"
      class="shrink-0 rounded-full border px-3 py-1 text-xs font-semibold transition"
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
      class="shrink-0 rounded-full border px-3 py-1 text-xs font-semibold transition"
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
      <!-- Tabular so the chips do not twitch as counts change under a search. -->
      <span v-if="ty.count != null" class="ml-1 font-mono tabular-nums opacity-70">{{
        ty.count
      }}</span>
    </button>
  </div>
</template>
