<script setup lang="ts">
/**
 * The Saved tab's filter bar (RFC-121 ph. 3 / #2042) — lifted to the top of Saved so it governs
 * every section at once, instead of a colour filter buried inside the Highlights list.
 *
 * Three controls: **type** chips (which saved kinds to show — multi-select, none selected = all),
 * a collapsed **colour** filter (the always-on swatch strip is now behind one "Colour" toggle), and
 * a **sort** select. Type chips render only for kinds present and colour swatches only for colours
 * in use, so the bar never offers a filter that would empty the list — same presence rule as the
 * sections themselves (#1962 single-empty-state).
 *
 * Presentational — two-way binds via v-model; the parent owns the state and applies it.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { HIGHLIGHT_COLORS } from '../utils/highlightColors'

const types = defineModel<string[]>('types', { default: () => [] })
const color = defineModel<string | null>('color', { default: null })
const sort = defineModel<string>('sort', { default: 'recent' })
const search = defineModel<string>('search', { default: '' })

const props = defineProps<{
  /** The saved kinds that actually have items, in display order. */
  availableTypes: { key: string; label: string }[]
  /** Colour tokens in use across saved items — the swatch strip renders only these. */
  colorsPresent: string[]
  /** Sort options for the select; defaults to the Saved set (episode / recent / colour). */
  sortOptions?: { value: string; label: string }[]
  /** Search-box placeholder (a type-to-filter over every section). */
  searchPlaceholder?: string
}>()

const { t } = useI18n()

const colorOptions = computed(() => HIGHLIGHT_COLORS.filter((c) => props.colorsPresent.includes(c.token)))
const resolvedSortOptions = computed(
  () =>
    props.sortOptions ?? [
      { value: 'episode', label: t('library.savedSortEpisode') },
      { value: 'recent', label: t('library.savedSortRecent') },
      { value: 'color', label: t('library.savedSortColor') },
    ],
)

function toggleType(key: string): void {
  types.value = types.value.includes(key)
    ? types.value.filter((k) => k !== key)
    : [...types.value, key]
}
function pickColor(token: string): void {
  color.value = color.value === token ? null : token
}
const hasFilters = computed(
  () => types.value.length > 0 || color.value !== null || search.value.trim() !== '',
)
function clearAll(): void {
  types.value = []
  color.value = null
  search.value = ''
}
</script>

<template>
  <div v-if="availableTypes.length" class="mb-5 flex flex-col gap-3" data-testid="saved-filter-bar">
    <!-- Type-to-filter search across every section — the primary find tool at 100+ items (#2042).
         A match is never hidden behind a section cap: the parent lifts caps while this is non-empty. -->
    <input
      v-model="search"
      type="search"
      :placeholder="searchPlaceholder ?? t('library.searchSaved')"
      :aria-label="searchPlaceholder ?? t('library.searchSaved')"
      data-testid="saved-search"
      class="lp-search w-full rounded-full border border-border bg-surface px-4 py-2 text-sm text-canvas-foreground outline-none focus:border-accent"
    />
    <!-- Type chips: none selected = All. "All" is an explicit chip so clearing is one tap. -->
    <div class="flex flex-wrap items-center gap-2">
      <button
        type="button"
        class="rounded-full border px-3 py-1 text-xs font-semibold transition"
        :class="types.length === 0
          ? 'border-accent bg-accent/10 text-accent'
          : 'border-border text-muted hover:text-canvas-foreground'"
        :aria-pressed="types.length === 0"
        data-testid="saved-type-all"
        @click="types = []"
      >{{ t('library.savedFilterAllTypes') }}</button>
      <button
        v-for="ty in availableTypes"
        :key="ty.key"
        type="button"
        class="rounded-full border px-3 py-1 text-xs font-semibold transition"
        :class="types.includes(ty.key)
          ? 'border-accent bg-accent/10 text-accent'
          : 'border-border text-muted hover:text-canvas-foreground'"
        :aria-pressed="types.includes(ty.key)"
        :data-testid="`saved-type-${ty.key}`"
        @click="toggleType(ty.key)"
      >{{ ty.label }}</button>
    </div>

    <!-- Colour filter (collapsed) + sort, on one row. -->
    <div class="flex flex-wrap items-center gap-x-4 gap-y-2">
      <div v-if="colorOptions.length" class="flex flex-wrap items-center gap-1">
        <span class="mr-1 text-xs text-muted">{{ t('library.savedFilterColor') }}</span>
        <button
          v-for="c in colorOptions"
          :key="c.token"
          type="button"
          data-testid="saved-filter-swatch"
          class="flex h-11 w-11 items-center justify-center rounded-full transition"
          :aria-pressed="color === c.token"
          :aria-label="t('library.savedFilterColorOnly', { color: t(c.labelKey) })"
          :title="t(c.labelKey)"
          @click="pickColor(c.token)"
        >
          <span
            class="h-4 w-4 rounded-full ring-offset-1 ring-offset-canvas transition"
            :class="[c.swatch, color === c.token ? 'ring-2 ring-accent' : 'opacity-70']"
          />
        </button>
      </div>

      <label class="flex items-center gap-1.5 text-xs text-muted">
        {{ t('library.savedSort') }}
        <select
          v-model="sort"
          :aria-label="t('library.savedSort')"
          data-testid="saved-sort"
          class="rounded-full border border-border bg-surface px-3 py-1.5 text-xs font-semibold text-canvas-foreground outline-none focus:border-accent"
        >
          <option v-for="o in resolvedSortOptions" :key="o.value" :value="o.value">{{ o.label }}</option>
        </select>
      </label>

      <button
        v-if="hasFilters"
        type="button"
        class="text-xs font-semibold text-accent"
        data-testid="saved-filter-clear"
        @click="clearAll"
      >{{ t('library.savedFilterClear') }}</button>
    </div>
  </div>
</template>
