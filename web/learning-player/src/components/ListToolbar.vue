<script setup lang="ts">
/**
 * List toolbar (UXS-014) — the ONE shared filter / sort / search header for big lists.
 *
 * ## Always visible (#2004 item 10)
 *
 * This used to collapse behind a "Sort & filter" pill, on the reasoning that it keeps lists clean.
 * The Shows tab, two tabs away, shows its filter field and sort select outright — so the same job
 * had two different interaction models depending on which tab you were on, and the collapsed one
 * hid the fact that filtering was possible at all.
 *
 * Marko chose the Shows pattern. The controls are now laid out and styled to match
 * `ShowBrowseView` exactly (a `flex-1` filter field, compact selects beside it) so the two tabs read
 * as one design rather than two takes on it.
 *
 * The third control is the reason this stayed a shared component rather than being hand-rolled a
 * second time: Episodes has a filter (All / With insights) that Shows does not, and flattening the
 * layout must not quietly drop it.
 *
 * Presentational — two-way-binds via v-model.
 */
import { useI18n } from 'vue-i18n'

const search = defineModel<string>('search', { default: '' })
const sort = defineModel<string>('sort', { default: 'newest' })
const filter = defineModel<string>('filter', { default: 'all' })
const show = defineModel<string>('show', { default: '' })

const { t } = useI18n()
withDefaults(
  defineProps<{ showFilter?: boolean; count?: string; shows?: { id: string; label: string }[] }>(),
  { showFilter: true, shows: () => [] },
)

</script>

<template>
  <div class="mb-4">
    <!--
      The filter field takes its OWN ROW on a phone (#2004 item 10 follow-up).

      Laid out as one wrapping row it shared the line with up to three selects, and as the `flex-1`
      child it absorbed all the leftover — the placeholder rendered as "Filte". The selects are
      compact and fixed; the field is the one control that needs width, so it gets the row.
    -->
    <div class="flex flex-wrap items-center gap-2">
      <input
        v-model="search"
        type="search"
        :placeholder="t('list.search')"
        :aria-label="t('list.search')"
        data-testid="list-toolbar-search"
        class="h-9 w-full rounded-full border border-border bg-surface px-4 text-sm text-canvas-foreground outline-none placeholder:text-muted focus:border-accent sm:w-auto sm:min-w-0 sm:flex-1"
      />
      <select
        v-model="sort"
        :aria-label="t('list.sort')"
        data-testid="list-toolbar-sort"
        class="h-9 shrink-0 rounded-full border border-border bg-surface px-3 text-sm font-semibold text-canvas-foreground outline-none focus:border-accent"
      >
        <option value="newest">{{ t('list.sortNewest') }}</option>
        <option value="oldest">{{ t('list.sortOldest') }}</option>
        <option value="title">{{ t('list.sortTitle') }}</option>
      </select>
      <select
        v-if="showFilter"
        v-model="filter"
        :aria-label="t('list.filter')"
        data-testid="list-toolbar-filter"
        class="h-9 shrink-0 rounded-full border border-border bg-surface px-3 text-sm font-semibold text-canvas-foreground outline-none focus:border-accent"
      >
        <option value="all">{{ t('list.filterAll') }}</option>
        <option value="insights">{{ t('list.filterInsights') }}</option>
      </select>
      <select
        v-if="shows.length"
        v-model="show"
        :aria-label="t('list.allShows')"
        data-testid="list-toolbar-show"
        class="h-9 max-w-[10rem] shrink-0 rounded-full border border-border bg-surface px-3 text-sm font-semibold text-canvas-foreground outline-none focus:border-accent"
      >
        <option value="">{{ t('list.allShows') }}</option>
        <option v-for="s in shows" :key="s.id" :value="s.id">{{ s.label }}</option>
      </select>
      <span v-if="count" class="shrink-0 text-xs text-muted">{{ count }}</span>
    </div>
  </div>
</template>
