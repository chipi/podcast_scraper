<script setup lang="ts">
/**
 * UXS-006 §6.5 — Feed processing history heatmap grid.
 *
 * Rows = feeds, columns = the last five distinct run timestamps (oldest →
 * newest, aggregated across feeds). Each ``run.json`` on the server is
 * scoped to a single (feed, timestamp), so the flat ``runs[]`` array is
 * grouped here into a matrix keyed by ``feed_id`` × truncated
 * ``created_at``. Cells are ``success`` / ``warning`` / ``danger`` fills
 * with ✓ / ⚠ / ✗ glyphs (colour is not the only signal).
 *
 * Silently hides when the corpus is single-feed or when no run carries a
 * ``feed_id`` (legacy flat layouts predating #1269).
 */
import { computed } from 'vue'
import type { CorpusRunSummaryItem } from '../../api/corpusMetricsApi'
import type { CorpusFeedItem } from '../../api/corpusLibraryApi'

const props = withDefaults(
  defineProps<{
    runs: CorpusRunSummaryItem[]
    /** Optional feed catalog for display-title lookup. Grid falls back to feed_id. */
    feeds?: readonly CorpusFeedItem[]
  }>(),
  { feeds: () => [] as readonly CorpusFeedItem[] },
)

type Status = 'succeeded' | 'partial' | 'failed'
type Cell = {
  status: Status
  ok: number
  failed: number
  skipped: number
  run: CorpusRunSummaryItem
}
type FeedRow = {
  feedId: string
  displayTitle: string
  cells: (Cell | null)[]
}

function statusOf(r: CorpusRunSummaryItem): Status {
  const ok = r.episode_outcomes?.ok ?? 0
  const failed = r.episode_outcomes?.failed ?? 0
  const skipped = r.episode_outcomes?.skipped ?? 0
  const total = ok + failed + skipped
  if (total > 0 && failed > 0 && ok + skipped > 0) return 'partial'
  if (failed > 0 || (total === 0 && (r.errors_total ?? 0) > 0)) return 'failed'
  return 'succeeded'
}

function dayKey(r: CorpusRunSummaryItem): string {
  const ca = (r.created_at ?? '').trim()
  if (ca.length >= 10) return ca.slice(0, 10)
  return r.run_id ?? r.relative_path
}

const feedTitleById = computed(() => {
  const m = new Map<string, string>()
  for (const f of props.feeds ?? []) {
    const title = (f.display_title ?? '').trim()
    if (f.feed_id) m.set(f.feed_id, title || f.feed_id)
  }
  return m
})

/** Runs that carry ``feed_id`` and a usable ``created_at`` prefix. */
const usableRuns = computed(() =>
  props.runs.filter(
    (r) => (r.feed_id ?? '').trim() !== '' && (r.created_at?.length ?? 0) >= 10,
  ),
)

/** Distinct run-day columns across all feeds, oldest → newest, capped at 5. */
const runDays = computed(() => {
  const seen = new Set<string>()
  for (const r of usableRuns.value) seen.add(dayKey(r))
  return [...seen].sort((a, b) => a.localeCompare(b)).slice(-5)
})

const feedIds = computed(() => {
  const seen = new Set<string>()
  for (const r of usableRuns.value) seen.add(r.feed_id ?? '')
  return [...seen]
})

const rows = computed<FeedRow[]>(() => {
  const days = runDays.value
  return feedIds.value
    .map((id) => {
      const title = feedTitleById.value.get(id) ?? id
      const cells: (Cell | null)[] = days.map((d) => {
        const runsHere = usableRuns.value.filter(
          (r) => r.feed_id === id && dayKey(r) === d,
        )
        if (runsHere.length === 0) return null
        const merged: Cell = {
          status: 'succeeded',
          ok: 0,
          failed: 0,
          skipped: 0,
          run: runsHere[0]!,
        }
        for (const r of runsHere) {
          merged.ok += r.episode_outcomes?.ok ?? 0
          merged.failed += r.episode_outcomes?.failed ?? 0
          merged.skipped += r.episode_outcomes?.skipped ?? 0
        }
        const anyFailed = runsHere.some((r) => statusOf(r) === 'failed')
        const anyPartial = runsHere.some((r) => statusOf(r) === 'partial')
        if (anyFailed) merged.status = 'failed'
        else if (anyPartial) merged.status = 'partial'
        else merged.status = 'succeeded'
        return merged
      })
      return { feedId: id, displayTitle: title, cells }
    })
    .sort((a, b) => a.displayTitle.localeCompare(b.displayTitle))
})

const isMultiFeed = computed(() => feedIds.value.length >= 2)
const shouldRender = computed(() => runDays.value.length > 0 && isMultiFeed.value)

const insightLine = computed(() => {
  if (!shouldRender.value) return ''
  const windowLen = runDays.value.length
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

function cellClass(cell: Cell | null): string {
  if (!cell) return 'bg-surface text-muted'
  if (cell.status === 'failed') return 'bg-danger/25 text-danger'
  if (cell.status === 'partial') return 'bg-warning/25 text-warning'
  return 'bg-success/25 text-success'
}

function cellGlyph(cell: Cell | null): string {
  if (!cell) return '·'
  if (cell.status === 'failed') return '✗'
  if (cell.status === 'partial') return '⚠'
  return '✓'
}

function cellTitle(row: FeedRow, cell: Cell | null, day: string): string {
  if (!cell) return `${row.displayTitle} · ${day} · no data`
  return `${row.displayTitle} · ${day} · ${cell.status} (ok ${cell.ok} · failed ${cell.failed} · skipped ${cell.skipped})`
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
              v-for="d in runDays"
              :key="d"
              scope="col"
              class="whitespace-nowrap px-1 text-center font-normal text-muted"
            >
              {{ d }}
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
              :title="cellTitle(row, cell, runDays[i] ?? '')"
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
