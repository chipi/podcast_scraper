<script setup lang="ts">
import { computed } from 'vue'
import type { CoverageFeedItem } from '../../api/corpusCoverageApi'
import { feedCoverageColumns } from '../../extensions'

const props = defineProps<{
  rows: CoverageFeedItem[]
  feedsIndexed: string[]
}>()

const emit = defineEmits<{
  'select-feed': [feedId: string]
}>()

const indexedSet = computed(() => new Set(props.feedsIndexed.map((x) => x.trim()).filter(Boolean)))

const extraColumns = feedCoverageColumns.list()

function share(n: number, d: number): number {
  return d ? n / d : 0
}

/** Lowest KG share first (the server orders by another layer), then feed id. */
const sortedRows = computed(() =>
  [...props.rows].sort(
    (a, b) => share(a.with_kg, a.total) - share(b.with_kg, b.total) || a.feed_id.localeCompare(b.feed_id),
  ),
)

function pct(n: number, d: number): number {
  if (!d) {
    return 0
  }
  return Math.round((n / d) * 100)
}

const insight = computed(() => {
  const r = sortedRows.value[0]
  if (!r || !r.total || r.with_kg >= r.total) {
    return undefined
  }
  const kg = pct(r.with_kg, r.total)
  const missing = r.total - r.with_kg
  return `Feed ${r.display_title} has lowest KG coverage at ${kg}% — ${missing} episode${missing === 1 ? '' : 's'} without KG artifacts.`
})
</script>

<template>
  <div
    class="rounded border border-border bg-surface p-3 text-surface-foreground"
    data-testid="feed-coverage-table"
  >
    <h3 class="mb-2 text-sm font-semibold">
      Feed coverage (lowest KG first)
    </h3>
    <div
      v-if="rows.length === 0"
      class="text-xs text-muted"
    >
      No feeds in catalog.
    </div>
    <table
      v-else
      class="w-full border-collapse text-left text-xs"
    >
      <thead>
        <tr class="border-b border-border text-[10px] uppercase tracking-wide text-muted">
          <th class="py-1 pr-2">
            Feed
          </th>
          <th class="py-1 pr-2">
            Episodes
          </th>
          <th
            v-for="col in extraColumns"
            :key="col.id"
            class="py-1 pr-2"
          >
            {{ col.label }}
          </th>
          <th class="py-1 pr-2">
            KG
          </th>
          <th class="py-1">
            In index
          </th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="r in sortedRows"
          :key="r.feed_id"
          class="border-b border-border/60 hover:bg-overlay/40"
          data-testid="feed-coverage-row"
        >
          <td
            class="cursor-pointer py-1.5 pr-2 font-medium"
            title="Open Corpus Library filtered to this feed"
            @click.stop="emit('select-feed', r.feed_id)"
          >
            {{ r.display_title }}
          </td>
          <td
            class="cursor-pointer py-1.5 pr-2 tabular-nums"
            title="Open Corpus Library filtered to this feed"
            @click.stop="emit('select-feed', r.feed_id)"
          >
            {{ r.total }}
          </td>
          <td
            v-for="col in extraColumns"
            :key="col.id"
            class="cursor-pointer py-1.5 pr-2"
            :data-testid="`feed-coverage-cell-${col.id}`"
            :title="`${col.noun} for this feed — click to open Library with this feed (${pct(col.count(r), r.total)}%)`"
            @click.stop="emit('select-feed', r.feed_id)"
          >
            <div class="flex items-center gap-1">
              <div
                class="h-2 w-10 shrink-0 overflow-hidden rounded bg-overlay"
              >
                <div
                  class="h-full"
                  :class="col.barClass"
                  :style="{ width: `${pct(col.count(r), r.total)}%` }"
                />
              </div>
              <span class="tabular-nums text-muted">{{ pct(col.count(r), r.total) }}%</span>
            </div>
          </td>
          <td
            class="cursor-pointer py-1.5 pr-2"
            :title="`KG coverage for this feed — click to open Library with this feed (${pct(r.with_kg, r.total)}%)`"
            @click.stop="emit('select-feed', r.feed_id)"
          >
            <div class="flex items-center gap-1">
              <div
                class="h-2 w-10 shrink-0 overflow-hidden rounded bg-overlay"
              >
                <div
                  class="h-full bg-kg"
                  :style="{ width: `${pct(r.with_kg, r.total)}%` }"
                />
              </div>
              <span class="tabular-nums text-muted">{{ pct(r.with_kg, r.total) }}%</span>
            </div>
          </td>
          <td
            class="cursor-pointer py-1.5 text-muted"
            title="Open Corpus Library filtered to this feed"
            @click.stop="emit('select-feed', r.feed_id)"
          >
            {{ indexedSet.has(r.feed_id) ? 'Yes' : '—' }}
          </td>
        </tr>
      </tbody>
    </table>
    <p
      v-if="insight"
      class="mt-2 text-[11px] text-muted"
    >
      {{ insight }}
    </p>
  </div>
</template>
