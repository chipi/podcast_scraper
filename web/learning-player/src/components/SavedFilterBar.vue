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

const props = withDefaults(
  defineProps<{
    /** The saved kinds that actually have items, in display order. */
    availableTypes: { key: string; label: string }[]
    /**
     * Whether the colour strip renders at all.
     *
     * Following uses this same bar but nothing there CAN carry a colour — colour is a property of a
     * saved item. It used to opt out by passing an empty `colorsPresent`, which worked only while
     * the strip was data-driven; now that the palette is always shown, opting out has to be said
     * rather than implied.
     */
    showColors?: boolean
    /** Sort options for the select; defaults to the shared Recent / A–Z set used by both tabs. */
    sortOptions?: { value: string; label: string }[]
    /** Search-box placeholder (a type-to-filter over every section). */
    searchPlaceholder?: string
  }>(),
  { showColors: true }
)

const { t } = useI18n()

/**
 * The WHOLE palette, always — not only the colours currently in use (operator 2026-09-17).
 *
 * Rendering just the present colours meant the control's size changed with the data: one saved amber
 * highlight produced a single lone dot, which reads as a broken or disabled control rather than a
 * colour filter. The full strip shows what the feature IS, and a colour with nothing behind it
 * simply filters to empty — which is a legible answer, not a dead end.
 */
const colorOptions = HIGHLIGHT_COLORS
// One sort model across Following AND Saved (#2042): Recent (default) or A–Z. Per-episode grouping
// of highlights is structural and unaffected — sort only orders the groups + the flat lists.
const resolvedSortOptions = computed(
  () =>
    props.sortOptions ?? [
      { value: 'recent', label: t('library.sortRecent') },
      { value: 'title', label: t('library.sortAz') },
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
    <!-- Search + sort share ONE row (operator 2026-09-17). The search was full-width with sort
         stranded on the row below, which spent a whole line on a control two words wide. The field
         only needs enough room to read a query back, so it flexes and the select takes its natural
         width beside it.
         Type-to-filter search across every section — the primary find tool at 100+ items (#2042).
         A match is never hidden behind a section cap: the parent lifts caps while this is non-empty. -->
    <div class="flex items-center gap-2">
      <input
        v-model="search"
        type="search"
        :placeholder="searchPlaceholder ?? t('library.searchSaved')"
        :aria-label="searchPlaceholder ?? t('library.searchSaved')"
        data-testid="saved-search"
        class="lp-search min-w-0 flex-1 rounded-full border border-border bg-surface px-4 py-2 text-sm text-canvas-foreground outline-none focus:border-accent"
      />
      <select
        v-model="sort"
        :aria-label="t('library.savedSort')"
        data-testid="saved-sort"
        class="shrink-0 rounded-full border border-border bg-surface px-3 py-1.5 text-xs font-semibold text-canvas-foreground outline-none focus:border-accent"
      >
        <option v-for="o in resolvedSortOptions" :key="o.value" :value="o.value">{{ o.label }}</option>
      </select>
    </div>
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

    <!-- Colour swatches + sort on ONE row (operator): the "Colour"/"Sort" text labels are dropped —
         the swatches read as colours and the select shows its value — so the swatches (44px targets)
         and the sort select fit a single line instead of wrapping to two. Sort is pushed right. -->
    <div class="flex flex-wrap items-center gap-2">
      <div
        v-if="showColors"
        class="flex flex-wrap items-center gap-1"
        role="group"
        :aria-label="t('library.savedFilterColor')"
      >
        <!-- "Any colour" leads the strip as an EMPTY ring (operator 2026-09-17), so clearing a
             colour is the same gesture in the same place as choosing one. Previously the only way
             back was the separate "Clear" link, which also dropped the type and search filters —
             one control undoing three things the user did not ask to undo. Mirrors the "All" chip
             that leads the type row. -->
        <button
          type="button"
          data-testid="saved-filter-swatch-any"
          class="flex h-11 w-11 items-center justify-center rounded-full transition"
          :aria-pressed="color === null"
          :aria-label="t('library.savedFilterColorAny')"
          :title="t('library.savedFilterColorAny')"
          @click="color = null"
        >
          <span
            class="h-4 w-4 rounded-full border border-border ring-offset-1 ring-offset-canvas transition"
            :class="color === null ? 'ring-2 ring-accent' : 'opacity-70'"
          />
        </button>
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

      <button
        v-if="hasFilters"
        type="button"
        class="ml-auto text-xs font-semibold text-accent"
        data-testid="saved-filter-clear"
        @click="clearAll"
      >{{ t('library.savedFilterClear') }}</button>
    </div>
  </div>
</template>
