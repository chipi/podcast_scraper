<script setup lang="ts">
/**
 * List toolbar (UXS-014) — the ONE shared search / filter / sort / view header for big lists.
 *
 * Both Browse tabs (Episodes → CatalogView, Shows → ShowBrowseView) render THIS, so the two read as
 * one design and cannot drift. Tab-specific bits are props: the sort options, the optional facet
 * (Episodes: All / insights; Shows: category), the search placeholder, and testid overrides.
 *
 * Layout (operator 2026-09-14): ONE row — a WIDE search that fills the space, then three COMPACT
 * controls via {@link ToolbarMenu} (a small trigger + a vertical option menu, not a native select
 * that reserves width for its widest option): the filter as a chip showing the current value, the
 * sort as a ↑↓ circle, and the list/grid view as a circle showing the active view. Collapsing sort
 * and view to little circles is what buys the search its width back. An optional `from` chip (which
 * episodes: all shows / following / mine) sits first after the search (operator 2026-10-10).
 *
 * On a phone the search is a 🔍 circle (operator 2026-10-15, "one row and minimal"): tapped, it
 * takes the whole row with the keyboard up and the chips step aside; × (or Escape) closes it. While
 * it holds text it stays open, so a filtered list never hides what filters it. From `lg` up there is
 * room for both: the field is always there and the circle never is.
 *
 * Presentational — two-way-binds via v-model.
 */
import { computed, nextTick, ref } from "vue"
import { useI18n } from "vue-i18n"
import ToolbarMenu from "./ToolbarMenu.vue"

const search = defineModel<string>("search", { default: "" })
const sort = defineModel<string>("sort", { default: "" })
const filter = defineModel<string>("filter", { default: "all" })
/** A second facet, the language (operator 2026-10-10), shown only when the host passes options. */
const language = defineModel<string>("language", { default: "" })
const view = defineModel<"list" | "grid">("view", { default: "list" })
const from = defineModel<string>("from", { default: "all" })

const { t } = useI18n()
const props = withDefaults(
  defineProps<{
    sortOptions: { value: string; label: string }[]
    filterOptions?: { value: string; label: string }[]
    languageOptions?: { value: string; label: string }[]
    fromOptions?: { value: string; label: string; short: string }[]
    fromLabel?: string
    fromTestid?: string
    count?: string
    searchPlaceholder?: string
    searchTestid?: string
    sortTestid?: string
    filterTestid?: string
    languageTestid?: string
    viewTestid?: string
  }>(),
  {
    searchTestid: "list-toolbar-search",
    sortTestid: "list-toolbar-sort",
    filterTestid: "list-toolbar-filter",
    languageTestid: "list-toolbar-language",
    viewTestid: "list-toolbar-view",
  }
)

const expanded = ref(false)
const searchOpen = computed(() => expanded.value || search.value.trim() !== "")
const input = ref<HTMLInputElement | null>(null)
async function openSearch(): Promise<void> {
  expanded.value = true
  await nextTick()
  input.value?.focus()
}
/** Left empty, the search folds back to its circle; left holding text, it stays open. */
function onBlur(): void {
  if (search.value.trim() === "") expanded.value = false
}
function closeSearch(): void {
  search.value = ""
  expanded.value = false
}
const viewOptions = computed(() => [
  { value: "list", label: t("list.viewList") },
  { value: "grid", label: t("list.viewGrid") },
])
</script>

<template>
  <div class="mb-4 flex items-center gap-1.5">
    <!-- Phone, search closed: the 🔍 circle, the same 36px circle as sort and view. -->
    <button
      v-if="!searchOpen"
      type="button"
      class="lp-tap flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-border text-muted transition hover:text-canvas-foreground lg:hidden"
      :aria-label="searchPlaceholder ?? t('list.search')"
      :data-testid="`${searchTestid}-open`"
      @click="openSearch"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" class="h-4 w-4" aria-hidden="true">
        <circle cx="11" cy="11" r="7" /><path d="M20 20l-3.5-3.5" />
      </svg>
    </button>
    <!-- Below lg the browser's own clear button is hidden: the × beside the field clears AND closes,
         and two × side by side read as a mistake. -->
    <div class="min-w-0 flex-1 items-center gap-1.5" :class="searchOpen ? 'flex' : 'hidden lg:flex'">
      <input
        ref="input"
        v-model="search"
        type="search"
        :placeholder="searchPlaceholder ?? t('list.search')"
        :aria-label="searchPlaceholder ?? t('list.search')"
        :data-testid="searchTestid"
        class="lp-search min-w-0 flex-1 rounded-full border border-border bg-surface px-4 py-2 text-sm text-canvas-foreground outline-none focus:border-accent max-lg:[&::-webkit-search-cancel-button]:hidden"
        @keydown.escape="closeSearch"
        @blur="onBlur"
      />
      <button
        v-if="searchOpen"
        type="button"
        class="lp-tap flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-muted transition hover:text-canvas-foreground lg:hidden"
        :aria-label="t('list.searchClose')"
        :data-testid="`${searchTestid}-close`"
        @click="closeSearch"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" class="h-4 w-4" aria-hidden="true">
          <path d="M6 6l12 12M18 6L6 18" />
        </svg>
      </button>
    </div>
    <!-- The chips: right-aligned beside the 🔍 circle; stepped aside on a phone while it searches. -->
    <div class="items-center gap-1.5" :class="searchOpen ? 'hidden lg:flex' : 'ml-auto flex lg:ml-0'">

    <ToolbarMenu
      v-if="fromOptions && fromOptions.length"
      v-model="from"
      variant="pill"
      align="right"
      :options="fromOptions"
      :menu-label="fromLabel ?? t('list.filter')"
      :testid="fromTestid"
    />

    <!-- Language facet (optional) — the same chip, offered by the host only when its list spans
         more than one language. -->
    <ToolbarMenu
      v-if="languageOptions && languageOptions.length"
      v-model="language"
      variant="pill"
      align="right"
      :options="languageOptions"
      :menu-label="t('list.language')"
      :testid="props.languageTestid"
    />

    <!-- Filter facet (optional) — a chip that shows the CURRENT value ("All"), sized to it, not to
         the widest option like a native select. -->
    <ToolbarMenu
      v-if="filterOptions && filterOptions.length"
      v-model="filter"
      variant="pill"
      align="right"
      :options="filterOptions"
      :menu-label="t('list.filter')"
      :testid="props.filterTestid"
    />

    <!-- Sort → a ↑↓ circle; the menu shows the options with the active one ticked. -->
    <ToolbarMenu
      v-model="sort"
      align="right"
      :options="sortOptions"
      :menu-label="t('list.sort')"
      :testid="props.sortTestid"
    >
      <template #icon>
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          class="h-4 w-4"
          aria-hidden="true"
        >
          <path d="M7 4v16M7 4l-3 3M7 4l3 3M17 20V4M17 20l-3-3M17 20l3-3" />
        </svg>
      </template>
    </ToolbarMenu>

    <!-- List/grid → one circle showing the ACTIVE view; tap reveals both to switch. -->
    <ToolbarMenu
      :model-value="view"
      align="right"
      :options="viewOptions"
      :menu-label="t('list.view')"
      :testid="props.viewTestid"
      @update:model-value="(v) => (view = v as 'list' | 'grid')"
    >
      <template #icon>
        <svg
          v-if="view === 'grid'"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          class="h-4 w-4"
          aria-hidden="true"
        >
          <path d="M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z" />
        </svg>
        <svg
          v-else
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          class="h-4 w-4"
          aria-hidden="true"
        >
          <path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />
        </svg>
      </template>
    </ToolbarMenu>
    </div>
  </div>
</template>
