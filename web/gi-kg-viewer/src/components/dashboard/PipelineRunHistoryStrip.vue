<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { CorpusRunSummaryItem } from '../../api/corpusMetricsApi'
import { formatDashboardRunDurationSeconds } from '../../utils/formatDuration'

const props = withDefaults(
  defineProps<{
    runs: CorpusRunSummaryItem[]
    /** When true, omit outer card chrome and heading (parent provides tab + border). */
    embedded?: boolean
    /** Prefer this ``run.json`` relative path in the detail card (from feed → run links). */
    highlightRelativePath?: string | null
  }>(),
  { embedded: false, highlightRelativePath: null },
)

/** Newest-first list (same ordering as ``GET /api/corpus/runs/summary``). */
const sortedRuns = computed(() => {
  const list = [...props.runs]
  list.sort((a, b) => {
    const ca = a.created_at ?? ''
    const cb = b.created_at ?? ''
    const c = cb.localeCompare(ca)
    if (c !== 0) {
      return c
    }
    return b.relative_path.localeCompare(a.relative_path)
  })
  return list
})

const selectedRelativePath = ref('')
const runFilter = ref('')

/** When there are many runs, list only the newest N in the dropdown until the user filters. */
const DEFAULT_SELECT_CAP = 120
/** Hard cap on visible list rows when filtering huge lists (in-memory). */
const FILTER_RESULTS_CAP = 250

const filteredRuns = computed(() => {
  const base = sortedRuns.value
  const q = runFilter.value.trim().toLowerCase()
  if (!q) {
    if (base.length > DEFAULT_SELECT_CAP) {
      return base.slice(0, DEFAULT_SELECT_CAP)
    }
    return base
  }
  return base
    .filter((r) => {
      const path = r.relative_path.toLowerCase()
      const id = (r.run_id ?? '').toLowerCase()
      const ca = (r.created_at ?? '').toLowerCase()
      return path.includes(q) || id.includes(q) || ca.includes(q)
    })
    .slice(0, FILTER_RESULTS_CAP)
})

/** Keep the current selection visible even if it falls outside the filtered/capped window. */
const selectOptions = computed(() => {
  const fr = filteredRuns.value
  const sel = selectedRelativePath.value.trim()
  if (!sel) {
    return fr
  }
  const hit = sortedRuns.value.find((r) => r.relative_path === sel)
  if (!hit) {
    return fr
  }
  if (fr.some((r) => r.relative_path === sel)) {
    return fr
  }
  return [hit, ...fr]
})

const runsPickerHint = computed(() => {
  const total = sortedRuns.value.length
  const q = runFilter.value.trim()
  if (total === 0) {
    return ''
  }
  if (!q && total > DEFAULT_SELECT_CAP) {
    return `Showing newest ${DEFAULT_SELECT_CAP.toLocaleString()} of ${total.toLocaleString()} in the list. Use the filter to search by path, run id, or date.`
  }
  if (q && filteredRuns.value.length >= FILTER_RESULTS_CAP) {
    return `Showing first ${FILTER_RESULTS_CAP.toLocaleString()} matches; narrow the filter if needed.`
  }
  return ''
})

watch(
  () => [props.runs, props.highlightRelativePath] as const,
  ([list, hpRaw]) => {
    if (!list.length) {
      selectedRelativePath.value = ''
      return
    }
    const hp = hpRaw?.trim() ?? ''
    if (hp) {
      const hit = list.find((r) => r.relative_path === hp)
      if (hit) {
        selectedRelativePath.value = hit.relative_path
        return
      }
    }
    const cur = selectedRelativePath.value.trim()
    if (cur && list.some((r) => r.relative_path === cur)) {
      return
    }
    selectedRelativePath.value = sortedRuns.value[0]?.relative_path ?? ''
  },
  { immediate: true },
)

type RunOutcomeKey = 'succeeded' | 'partial' | 'failed'

function runOutcomeKey(r: CorpusRunSummaryItem): RunOutcomeKey {
  const failed = r.episode_outcomes?.failed ?? 0
  const ok = r.episode_outcomes?.ok ?? 0
  const skipped = r.episode_outcomes?.skipped ?? 0
  const tot = failed + ok + skipped
  if (tot > 0 && failed > 0 && ok + skipped > 0) {
    return 'partial'
  }
  if (failed > 0 || (tot === 0 && (r.errors_total ?? 0) > 0)) {
    return 'failed'
  }
  return 'succeeded'
}

const selectedRun = computed(() => {
  const p = selectedRelativePath.value.trim()
  if (!p) {
    return null
  }
  return props.runs.find((r) => r.relative_path === p) ?? null
})

const insight = computed(() => {
  const list = props.runs
  if (list.length === 0) {
    return undefined
  }
  let ok = 0
  let partial = 0
  let failed = 0
  for (const r of list) {
    const f = r.episode_outcomes?.failed ?? 0
    const o = r.episode_outcomes?.ok ?? 0
    const s = r.episode_outcomes?.skipped ?? 0
    const t = f + o + s
    if (t > 0 && f > 0 && o + s > 0) {
      partial += 1
    } else if (f > 0) {
      failed += 1
    } else {
      ok += 1
    }
  }
  const n = list.length.toLocaleString()
  if (partial === 0 && failed === 0) {
    return `All ${n} loaded runs succeeded`
  }
  return `${n} loaded runs: ${ok.toLocaleString()} success, ${partial.toLocaleString()} partial, ${failed.toLocaleString()} failed`
})

function shortRunLabel(r: CorpusRunSummaryItem): string {
  const id = r.run_id?.trim()
  if (id) {
    return id.length > 12 ? `${id.slice(0, 8)}…` : id
  }
  const parts = r.relative_path.split('/').filter(Boolean)
  const tail = parts[parts.length - 1] || r.relative_path
  if (tail === 'run.json' && parts.length >= 2) {
    const folder = parts[parts.length - 2] ?? ''
    return folder.length > 14 ? `${folder.slice(0, 12)}…` : folder
  }
  return tail.length > 18 ? `${tail.slice(0, 16)}…` : tail
}

/** One line per run row (newest-first list). */
function runOptionLabel(r: CorpusRunSummaryItem): string {
  const when = formatRunCreated(r.created_at) || '—'
  const ep = (r.episodes_scraped_total ?? 0).toLocaleString()
  const st = runOutcomeKey(r)
  const path = r.relative_path.length > 72 ? `${r.relative_path.slice(0, 70)}…` : r.relative_path
  return `${when} · ${ep} ep · ${st} · ${path}`
}

function formatRunCreated(iso: string | null | undefined): string {
  if (!iso?.trim()) {
    return ''
  }
  const ms = Date.parse(iso)
  if (Number.isNaN(ms)) {
    return iso.trim()
  }
  try {
    return new Intl.DateTimeFormat(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(ms)
  } catch {
    return iso.trim()
  }
}

function fmtStageSec(sec: number | null | undefined): string {
  return formatDashboardRunDurationSeconds(sec) || '—'
}

function hasStageTimes(r: CorpusRunSummaryItem): boolean {
  return [r.time_scraping_seconds, r.time_parsing_seconds, r.time_normalizing_seconds, r.time_io_and_waiting_seconds].some(
    (x) => x != null && Number.isFinite(x) && (x as number) >= 0,
  )
}

function episodeOutcomeLine(r: CorpusRunSummaryItem): string {
  const o = r.episode_outcomes?.ok ?? 0
  const f = r.episode_outcomes?.failed ?? 0
  const s = r.episode_outcomes?.skipped ?? 0
  if (o + f + s === 0) {
    return ''
  }
  return `${o.toLocaleString()} ok · ${f.toLocaleString()} failed · ${s.toLocaleString()} skipped`
}

function runListOutcomeBarClass(r: CorpusRunSummaryItem): string {
  const k = runOutcomeKey(r)
  if (k === 'partial') {
    return 'bg-warning'
  }
  if (k === 'failed') {
    return 'bg-danger'
  }
  return 'bg-success'
}

function onRunListKeydown(e: KeyboardEvent): void {
  const opts = selectOptions.value
  if (!opts.length) {
    return
  }
  const cur = selectedRelativePath.value.trim()
  let i = opts.findIndex((r) => r.relative_path === cur)
  if (i < 0) {
    i = 0
  }
  if (e.key === 'ArrowDown') {
    e.preventDefault()
    i = Math.min(i + 1, opts.length - 1)
    selectedRelativePath.value = opts[i]!.relative_path
  } else if (e.key === 'ArrowUp') {
    e.preventDefault()
    i = Math.max(i - 1, 0)
    selectedRelativePath.value = opts[i]!.relative_path
  } else if (e.key === 'Home') {
    e.preventDefault()
    selectedRelativePath.value = opts[0]!.relative_path
  } else if (e.key === 'End') {
    e.preventDefault()
    selectedRelativePath.value = opts[opts.length - 1]!.relative_path
  }
}
</script>

<template>
  <div
    :class="
      embedded
        ? 'text-surface-foreground'
        : 'rounded border border-border bg-surface p-3 text-surface-foreground'
    "
    data-testid="pipeline-run-history-strip"
  >
    <h3
      v-if="!embedded"
      class="mb-2 text-sm font-semibold"
    >
      Run history
    </h3>
    <p
      v-if="!sortedRuns.length"
      class="text-[10px] text-muted leading-snug"
    >
      No <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">run.json</code> summaries found for this corpus yet.
    </p>
    <template v-else>
      <div
        class="flex flex-col gap-3 lg:flex-row lg:items-stretch lg:gap-4"
        data-testid="pipeline-run-history-layout"
      >
        <div class="flex min-h-0 min-w-0 flex-col gap-2 lg:max-w-[min(100%,40rem)] lg:flex-[0_1_38%]">
          <input
            id="pipeline-run-history-filter"
            v-model="runFilter"
            type="search"
            autocomplete="off"
            aria-label="Filter runs by path, run id, or date"
            placeholder="Path, run id, or date substring…"
            class="box-border min-h-9 w-full rounded border border-border bg-surface px-2 py-1.5 font-mono text-[10px] text-surface-foreground placeholder:text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-primary"
            data-testid="pipeline-run-history-filter"
          >
          <div
            id="pipeline-run-history-select"
            role="listbox"
            tabindex="0"
            aria-label="Runs"
            class="box-border flex min-h-[33dvh] max-h-[33dvh] w-full flex-1 flex-col overflow-hidden rounded border border-border bg-surface focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-primary"
            data-testid="pipeline-run-history-select"
            @keydown="onRunListKeydown"
          >
              <div class="min-h-0 flex-1 overflow-y-auto">
                <button
                  v-for="r in selectOptions"
                  :key="r.relative_path"
                  type="button"
                  role="option"
                  :aria-selected="selectedRelativePath === r.relative_path"
                  class="flex w-full items-stretch gap-2 border-b border-border/40 text-left last:border-b-0 hover:bg-overlay/80"
                  :class="
                    selectedRelativePath === r.relative_path
                      ? 'bg-primary/12 ring-1 ring-inset ring-primary/35'
                      : ''
                  "
                  @click="selectedRelativePath = r.relative_path"
                >
                  <span
                    class="w-1 shrink-0 self-stretch"
                    :class="runListOutcomeBarClass(r)"
                    aria-hidden="true"
                  />
                  <span class="min-w-0 flex-1 py-1 pr-2 font-mono text-[10px] leading-snug text-surface-foreground">
                    {{ runOptionLabel(r) }}
                  </span>
                </button>
              </div>
            </div>
          <p
            v-if="runsPickerHint"
            class="text-[9px] leading-snug text-muted"
            data-testid="pipeline-run-history-picker-hint"
          >
            {{ runsPickerHint }}
          </p>
        </div>
        <div class="flex min-h-0 min-w-0 flex-1 flex-col gap-2 lg:border-l lg:border-border lg:pl-4">
          <p
            v-if="selectedRun && insight"
            class="flex flex-wrap items-baseline gap-x-1 gap-y-0.5 text-[10px] leading-snug text-muted"
            data-testid="pipeline-run-history-summary-line"
          >
            <span>{{ insight }}</span>
            <span aria-hidden="true">·</span>
            <span class="font-mono text-surface-foreground">{{ shortRunLabel(selectedRun) }}</span>
            <span aria-hidden="true">·</span>
            <span>{{ (selectedRun.episodes_scraped_total ?? 0).toLocaleString() }} ep</span>
            <span aria-hidden="true">·</span>
            <span>wall {{ formatDashboardRunDurationSeconds(selectedRun.run_duration_seconds ?? null) || '—' }}</span>
            <template v-if="formatRunCreated(selectedRun.created_at)">
              <span aria-hidden="true">·</span>
              <span>{{ formatRunCreated(selectedRun.created_at) }}</span>
            </template>
          </p>
          <p
            v-else-if="insight"
            class="text-[10px] text-muted"
            data-testid="pipeline-run-history-summary-line-collapsed"
          >
            {{ insight }}
          </p>
          <div
            v-if="selectedRun"
            class="min-h-0 flex-1 space-y-1 overflow-y-auto rounded border border-border bg-overlay p-2 text-[10px] text-muted lg:min-h-[33dvh]"
            data-testid="pipeline-run-history-detail"
          >
        <p class="break-all">
          <span class="text-surface-foreground">Path:</span>
          {{ selectedRun.relative_path }}
        </p>
        <p v-if="episodeOutcomeLine(selectedRun)">
          <span class="text-surface-foreground">Episodes:</span>
          {{ episodeOutcomeLine(selectedRun) }}
        </p>
        <p v-else>
          <span class="text-surface-foreground">Episodes scraped:</span>
          {{ (selectedRun.episodes_scraped_total ?? 0).toLocaleString() }}
        </p>
        <p v-if="(selectedRun.errors_total ?? 0) > 0">
          <span class="text-surface-foreground">Errors:</span>
          {{ (selectedRun.errors_total ?? 0).toLocaleString() }}
        </p>
        <p
          v-if="selectedRun.gi_artifacts_generated != null || selectedRun.kg_artifacts_generated != null"
        >
          <span class="text-surface-foreground">Artifacts:</span>
          GI {{ selectedRun.gi_artifacts_generated ?? '—' }}
          · KG {{ selectedRun.kg_artifacts_generated ?? '—' }}
        </p>
        <template v-if="hasStageTimes(selectedRun)">
          <p class="text-surface-foreground">
            Stage time (metrics)
          </p>
          <ul class="list-inside list-disc pl-0.5 text-[9px] leading-snug">
            <li v-if="selectedRun.time_scraping_seconds != null">
              Scraping {{ fmtStageSec(selectedRun.time_scraping_seconds) }}
            </li>
            <li v-if="selectedRun.time_parsing_seconds != null">
              Parsing {{ fmtStageSec(selectedRun.time_parsing_seconds) }}
            </li>
            <li v-if="selectedRun.time_normalizing_seconds != null">
              Normalizing {{ fmtStageSec(selectedRun.time_normalizing_seconds) }}
            </li>
            <li v-if="selectedRun.time_io_and_waiting_seconds != null">
              I/O and wait {{ fmtStageSec(selectedRun.time_io_and_waiting_seconds) }}
            </li>
          </ul>
        </template>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>
