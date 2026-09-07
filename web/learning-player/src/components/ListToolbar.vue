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
  <!--
    ONE line: filter left, sort right — exactly the Shows row (#2004 item 10).

    My first attempt kept all four controls and let them wrap, which produced three lines. Four
    controls do not fit one line on a phone, so the row is the two that matter and the classes are
    copied from `ShowBrowseView` rather than re-derived, so the two tabs cannot drift.

    The insights and per-show selects are NOT rendered here any more — see the note on #2004.
  -->
  <div class="mb-4 flex flex-wrap items-center gap-2">
    <input
      v-model="search"
      type="search"
      :placeholder="t('list.search')"
      :aria-label="t('list.search')"
      data-testid="list-toolbar-search"
      class="min-w-0 flex-1 rounded-full border border-border bg-surface px-4 py-2 text-sm text-canvas-foreground outline-none focus:border-accent"
    />
    <select
      v-model="sort"
      :aria-label="t('list.sort')"
      data-testid="list-toolbar-sort"
      class="shrink-0 rounded-full border border-border bg-surface px-3 py-2 text-sm font-semibold text-canvas-foreground outline-none focus:border-accent"
    >
      <option value="newest">{{ t('list.sortNewest') }}</option>
      <option value="oldest">{{ t('list.sortOldest') }}</option>
      <option value="title">{{ t('list.sortTitle') }}</option>
    </select>
  </div>
</template>
