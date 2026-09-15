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
 * and view to little circles is what buys the search its width back.
 *
 * Presentational — two-way-binds via v-model.
 */
import { computed } from "vue"
import { useI18n } from "vue-i18n"
import ToolbarMenu from "./ToolbarMenu.vue"

const search = defineModel<string>("search", { default: "" })
const sort = defineModel<string>("sort", { default: "" })
const filter = defineModel<string>("filter", { default: "all" })
const view = defineModel<"list" | "grid">("view", { default: "list" })

const { t } = useI18n()
const props = withDefaults(
  defineProps<{
    sortOptions: { value: string; label: string }[]
    filterOptions?: { value: string; label: string }[]
    count?: string
    searchPlaceholder?: string
    searchTestid?: string
    sortTestid?: string
    filterTestid?: string
    viewTestid?: string
  }>(),
  {
    searchTestid: "list-toolbar-search",
    sortTestid: "list-toolbar-sort",
    filterTestid: "list-toolbar-filter",
    viewTestid: "list-toolbar-view",
  }
)

const viewOptions = computed(() => [
  { value: "list", label: t("list.viewList") },
  { value: "grid", label: t("list.viewGrid") },
])
</script>

<template>
  <div class="mb-4 flex items-center gap-1.5">
    <input
      v-model="search"
      type="search"
      :placeholder="searchPlaceholder ?? t('list.search')"
      :aria-label="searchPlaceholder ?? t('list.search')"
      :data-testid="searchTestid"
      class="lp-search min-w-0 flex-1 rounded-full border border-border bg-surface px-4 py-2 text-sm text-canvas-foreground outline-none focus:border-accent"
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
</template>
