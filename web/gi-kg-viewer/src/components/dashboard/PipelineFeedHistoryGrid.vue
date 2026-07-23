<script setup lang="ts">
/**
 * UXS-006 §6.5 — Feed processing history heatmap grid.
 *
 * Rows = feeds, columns = the last five runs (oldest → newest). Cells
 * are ``success`` / ``warning`` / ``danger`` fills with ✓ / ⚠ / ✗ glyphs
 * (colour is not the only signal). Silently hides when the backend
 * response is single-feed or has not yet shipped the ``by_feed``
 * breakdown (per UXS-006 §9 open question — tracked as a follow-up).
 */
import { computed } from 'vue'
import type { CorpusRunFeedOutcomeItem, CorpusRunSummaryItem } from '../../api/corpusMetricsApi'

const props = defineProps<{
  runs: CorpusRunSummaryItem[]
}>()

type FeedRow = {
  feedId: string
  displayTitle: string
  cells: (CorpusRunFeedOutcomeItem | null)[]
}

/** Newest-first chronological ordering of runs that carry ``by_feed`` data. */
const runsWithBreakdown = computed(() => {
  return props.runs
    .filter((r) => Array.isArray(r.by_feed) && (r.by_feed?.length ?? 0) > 0)
    .filter((r) => (r.created_at?.length ?? 0) >= 10)
    .sort((a, b) => (a.created_at ?? '').localeCompare(b.created_at ?? ''))
    .slice(-5)
})

/** Distinct feed ids that appear in any of the visible runs (stable order). */
const feedIds = computed(() => {
  const seen = new Map<string, string>()
  for (const r of runsWithBreakdown.value) {
    for (const f of r.by_feed ?? []) {
      if (!seen.has(f.feed_id)) {
        seen.set(f.feed_id, (f.display_title ?? '').trim() || f.feed_id)
      }
    }
  }
  return [...seen.entries()]
    .map(([id, title]) => ({ id, title }))
    .sort((a, b) => a.title.localeCompare(b.title))
})

const rows = computed<FeedRow[]>(() =>
  feedIds.value.map(({ id, title }) => ({
    feedId: id,
    displayTitle: title,
    cells: runsWithBreakdown.value.map(
      (r) => (r.by_feed ?? []).find((f) => f.feed_id === id) ?? null,
    ),
  })),
)

/** UXS-006 §6.5 hides the grid when the corpus is single-feed. */
const isMultiFeed = computed(() => feedIds.value.length >= 2)
const shouldRender = computed(
  () => runsWithBreakdown.value.length > 0 && isMultiFeed.value,
)

const insightLine = computed(() => {
  if (!shouldRender.value) return ''
  const windowLen = runsWithBreakdown.value.length
  let worstFeed: string | null = null
  let worstFailures = 0
  for (const row of rows.value) {
    const fails = row.cells.filter((c) => c?.status === 'failed').length
    if (fails > worstFailures) {
      worstFailures = fails
      worstFeed = row.displayTitle
    }
  }
  if (worstFailures >= 2 && worstFeed) {
    return `Feed ${worstFeed} failed ${worstFailures} of last ${windowLen} runs`
  }
  return `All feeds succeeded in last ${windowLen} runs`
})

function cellClass(cell: CorpusRunFeedOutcomeItem | null): string {
  if (!cell) return 'bg-surface text-muted'
  if (cell.status === 'failed') return 'bg-danger/25 text-danger'
  if (cell.status === 'partial') return 'bg-warning/25 text-warning'
  return 'bg-success/25 text-success'
}

function cellGlyph(cell: CorpusRunFeedOutcomeItem | null): string {
  if (!cell) return '·'
  if (cell.status === 'failed') return '✗'
  if (cell.status === 'partial') return '⚠'
  return '✓'
}

function cellTitle(row: FeedRow, cell: CorpusRunFeedOutcomeItem | null, runIdx: number): string {
  const run = runsWithBreakdown.value[runIdx]
  const when = (run?.created_at ?? '').slice(0, 10) || '—'
  if (!cell) return `${row.displayTitle} · ${when} · no data`
  const o = cell.ok ?? 0
  const f = cell.failed ?? 0
  const s = cell.skipped ?? 0
  return `${row.displayTitle} · ${when} · ${cell.status} (ok ${o} · failed ${f} · skipped ${s})`
}

function runHeaderLabel(r: CorpusRunSummaryItem): string {
  return (r.created_at ?? '').slice(0, 10) || '—'
}
</script>

<template>
  <section
    v-if="shouldRender"
    class="rounded border border-border bg-surface p-3 text-surface-foreground"
    data-testid="pipeline-feed-history-grid"
    aria-label="Feed processing history"
  >
    <h3 class="mb-2 text-sm font-semibold">
      Feed history
    </h3>
    <p
      class="mb-3 text-[10px] leading-snug text-muted"
      data-testid="pipeline-feed-history-grid-insight"
    >
      {{ insightLine }}
    </p>
    <div class="overflow-x-auto">
      <table class="border-separate border-spacing-1 text-[10px]">
        <thead>
          <tr>
            <th
              scope="col"
              class="text-left font-normal text-muted"
            >
              &nbsp;
            </th>
            <th
              v-for="r in runsWithBreakdown"
              :key="r.relative_path"
              scope="col"
              class="whitespace-nowrap px-1 text-center font-normal text-muted"
            >
              {{ runHeaderLabel(r) }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in rows"
            :key="row.feedId"
          >
            <th
              scope="row"
              class="max-w-[10rem] truncate pr-2 text-left text-xs font-normal text-surface-foreground"
              :title="row.displayTitle"
            >
              {{ row.displayTitle }}
            </th>
            <td
              v-for="(cell, i) in row.cells"
              :key="i"
              class="h-7 w-7 rounded text-center align-middle text-[11px] font-semibold"
              :class="cellClass(cell)"
              :title="cellTitle(row, cell, i)"
              data-testid="pipeline-feed-history-cell"
            >
              {{ cellGlyph(cell) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
