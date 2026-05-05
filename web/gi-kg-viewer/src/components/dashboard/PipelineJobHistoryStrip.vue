<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { listPipelineJobs, pipelineJobLogUrl, type PipelineJobRow } from '../../api/jobsApi'
import { useShellStore } from '../../stores/shell'
import PipelineJobExplorePanel from './PipelineJobExplorePanel.vue'

const emit = defineEmits<{
  'open-run-history': [payload: { relativePath: string }]
}>()

withDefaults(
  defineProps<{
    embedded?: boolean
  }>(),
  { embedded: false },
)

const TERMINAL = new Set(['succeeded', 'failed', 'cancelled', 'stale'])

const shell = useShellStore()
const rows = ref<PipelineJobRow[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const selectedJobId = ref('')
const jobFilter = ref('')

/** When there are many jobs, list only the newest N until the user filters. */
const DEFAULT_SELECT_CAP = 120
/** Hard cap on visible list rows when filtering huge lists (in-memory). */
const FILTER_RESULTS_CAP = 250

const root = computed(() => shell.corpusPath.trim())
const canFetch = computed(
  () => Boolean(root.value && shell.healthStatus && shell.jobsApiAvailable),
)

/** Finished HTTP jobs, newest first (same idea as Run history). */
const finishedJobs = computed(() => {
  const term = rows.value.filter((j) => TERMINAL.has(String(j.status).toLowerCase()))
  const list = [...term]
  list.sort((a, b) => {
    const ca = a.created_at ?? ''
    const cb = b.created_at ?? ''
    const c = cb.localeCompare(ca)
    if (c !== 0) {
      return c
    }
    return b.job_id.localeCompare(a.job_id)
  })
  return list
})

const filteredJobs = computed(() => {
  const base = finishedJobs.value
  const q = jobFilter.value.trim().toLowerCase()
  if (!q) {
    if (base.length > DEFAULT_SELECT_CAP) {
      return base.slice(0, DEFAULT_SELECT_CAP)
    }
    return base
  }
  return base
    .filter((j) => {
      const id = j.job_id.toLowerCase()
      const st = String(j.status).toLowerCase()
      const ca = (j.created_at ?? '').toLowerCase()
      const cmd = (j.argv_summary ?? '').toLowerCase()
      const ct = (j.command_type ?? '').toLowerCase()
      return (
        id.includes(q) ||
        st.includes(q) ||
        ca.includes(q) ||
        cmd.includes(q) ||
        ct.includes(q)
      )
    })
    .slice(0, FILTER_RESULTS_CAP)
})

/** Keep the current selection visible even if it falls outside the filtered/capped window. */
const selectOptions = computed(() => {
  const fj = filteredJobs.value
  const sel = selectedJobId.value.trim()
  if (!sel) {
    return fj
  }
  const hit = finishedJobs.value.find((j) => j.job_id === sel)
  if (!hit) {
    return fj
  }
  if (fj.some((j) => j.job_id === sel)) {
    return fj
  }
  return [hit, ...fj]
})

const jobsPickerHint = computed(() => {
  const total = finishedJobs.value.length
  const q = jobFilter.value.trim()
  if (total === 0) {
    return ''
  }
  if (!q && total > DEFAULT_SELECT_CAP) {
    return `Showing newest ${DEFAULT_SELECT_CAP.toLocaleString()} of ${total.toLocaleString()} in the list. Use the filter to search by id, status, or command.`
  }
  if (q && filteredJobs.value.length >= FILTER_RESULTS_CAP) {
    return `Showing first ${FILTER_RESULTS_CAP.toLocaleString()} matches; narrow the filter if needed.`
  }
  return ''
})

watch(
  finishedJobs,
  (list) => {
    if (!list.length) {
      selectedJobId.value = ''
      return
    }
    const cur = selectedJobId.value.trim()
    if (cur && list.some((j) => j.job_id === cur)) {
      return
    }
    selectedJobId.value = list[0]?.job_id ?? ''
  },
  { immediate: true },
)

function jobStatusKey(j: PipelineJobRow): string {
  return String(j.status).toLowerCase()
}

function jobListStatusBarClass(j: PipelineJobRow): string {
  const st = jobStatusKey(j)
  if (st === 'succeeded') {
    return 'bg-success'
  }
  if (st === 'failed') {
    return 'bg-danger'
  }
  if (st === 'stale') {
    return 'bg-warning'
  }
  if (st === 'cancelled') {
    return 'bg-sky-500 dark:bg-sky-400'
  }
  return 'bg-muted'
}

/** One line per job row (newest-first list). */
function jobOptionLabel(j: PipelineJobRow): string {
  const when = formatJobCreated(j.created_at) || '—'
  const st = jobStatusKey(j)
  const id = j.job_id.length > 14 ? `${j.job_id.slice(0, 12)}…` : j.job_id
  const raw = (j.argv_summary ?? '').trim() || j.command_type || '—'
  const cmd = raw.length > 52 ? `${raw.slice(0, 50)}…` : raw
  return `${when} · ${st} · ${id} · ${cmd}`
}

function formatJobCreated(iso: string | null | undefined): string {
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

function onJobListKeydown(e: KeyboardEvent): void {
  const opts = selectOptions.value
  if (!opts.length) {
    return
  }
  const cur = selectedJobId.value.trim()
  let i = opts.findIndex((j) => j.job_id === cur)
  if (i < 0) {
    i = 0
  }
  if (e.key === 'ArrowDown') {
    e.preventDefault()
    i = Math.min(i + 1, opts.length - 1)
    selectedJobId.value = opts[i]!.job_id
  } else if (e.key === 'ArrowUp') {
    e.preventDefault()
    i = Math.max(i - 1, 0)
    selectedJobId.value = opts[i]!.job_id
  } else if (e.key === 'Home') {
    e.preventDefault()
    selectedJobId.value = opts[0]!.job_id
  } else if (e.key === 'End') {
    e.preventDefault()
    selectedJobId.value = opts[opts.length - 1]!.job_id
  }
}

const selectedJob = computed(() => {
  const id = selectedJobId.value.trim()
  if (!id) {
    return null
  }
  return finishedJobs.value.find((j) => j.job_id === id) ?? null
})

const insight = computed(() => {
  const list = finishedJobs.value
  if (list.length === 0) {
    return undefined
  }
  let ok = 0
  let bad = 0
  let cancelled = 0
  for (const j of list) {
    const st = String(j.status).toLowerCase()
    if (st === 'succeeded') {
      ok += 1
    } else if (st === 'cancelled') {
      cancelled += 1
    } else {
      bad += 1
    }
  }
  if (bad === 0 && cancelled === 0) {
    return `All ${list.length} finished jobs succeeded`
  }
  return `${list.length} finished jobs: ${ok} ok, ${bad} failed/stale, ${cancelled} cancelled`
})

function shortId(id: string): string {
  return id.length > 10 ? `${id.slice(0, 8)}…` : id
}

function formatDurationMs(ms: number): string {
  const sec = Math.max(0, Math.floor(ms / 1000))
  if (sec < 60) {
    return `${sec}s`
  }
  const m = Math.floor(sec / 60)
  const rs = sec % 60
  if (m < 60) {
    return `${m}m ${rs}s`
  }
  const h = Math.floor(m / 60)
  const rm = m % 60
  return `${h}h ${rm}m`
}

function jobDurationLabel(j: PipelineJobRow): string {
  if (!j.started_at) {
    return '—'
  }
  const s = Date.parse(j.started_at)
  if (Number.isNaN(s)) {
    return '—'
  }
  const endIso = j.ended_at
  const e = endIso ? Date.parse(endIso) : Date.now()
  if (Number.isNaN(e)) {
    return '—'
  }
  return formatDurationMs(e - s)
}

async function refresh(): Promise<void> {
  if (!canFetch.value) {
    rows.value = []
    return
  }
  loading.value = true
  error.value = null
  try {
    const res = await listPipelineJobs(root.value)
    rows.value = Array.isArray(res.jobs) ? res.jobs : []
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    rows.value = []
  } finally {
    loading.value = false
  }
}

watch(
  () => [root.value, shell.healthStatus, shell.jobsApiAvailable] as const,
  () => {
    void refresh()
  },
  { immediate: true },
)
</script>

<template>
  <div
    :class="
      embedded
        ? 'text-surface-foreground'
        : 'rounded border border-border bg-surface p-3 text-surface-foreground'
    "
    data-testid="pipeline-job-history-strip"
  >
    <h3
      v-if="!embedded"
      class="mb-2 text-sm font-semibold"
    >
      Job history
    </h3>
    <p
      v-if="!shell.jobsApiAvailable && shell.healthStatus"
      class="mb-2 text-[10px] text-muted"
    >
      Jobs API is off (enable <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">--enable-jobs-api</code> on serve).
    </p>
    <p
      v-if="error"
      class="mb-2 text-[10px] text-danger"
    >
      {{ error }}
    </p>
    <p
      v-if="loading && !rows.length"
      class="text-[10px] text-muted"
    >
      Loading…
    </p>
    <p
      v-else-if="!finishedJobs.length && canFetch"
      class="text-[10px] text-muted leading-snug"
    >
      No finished HTTP pipeline jobs in the registry yet. Use <strong class="text-surface-foreground">Jobs</strong> to queue a run.
    </p>
    <template v-else-if="finishedJobs.length">
      <div
        class="flex flex-col gap-3 lg:flex-row lg:items-stretch lg:gap-4"
        data-testid="pipeline-job-history-layout"
      >
        <div class="flex min-h-0 min-w-0 flex-col gap-2 lg:max-w-[min(100%,40rem)] lg:flex-[0_1_38%]">
          <input
            id="pipeline-job-history-filter"
            v-model="jobFilter"
            type="search"
            autocomplete="off"
            aria-label="Filter jobs by id, status, time, or command"
            placeholder="Job id, status, time, or command substring…"
            class="box-border min-h-9 w-full rounded border border-border bg-surface px-2 py-1.5 font-mono text-[10px] text-surface-foreground placeholder:text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-primary"
            data-testid="pipeline-job-history-filter"
          >
          <div
            id="pipeline-job-history-select"
            role="listbox"
            tabindex="0"
            aria-label="Finished jobs"
            class="box-border flex min-h-[33dvh] max-h-[33dvh] w-full flex-1 flex-col overflow-hidden rounded border border-border bg-surface focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-primary"
            data-testid="pipeline-job-history-select"
            @keydown="onJobListKeydown"
          >
            <div class="min-h-0 flex-1 overflow-y-auto">
              <button
                v-for="j in selectOptions"
                :key="j.job_id"
                type="button"
                role="option"
                :aria-selected="selectedJobId === j.job_id"
                class="flex w-full items-stretch gap-2 border-b border-border/40 text-left last:border-b-0 hover:bg-overlay/80"
                :class="
                  selectedJobId === j.job_id
                    ? 'bg-primary/12 ring-1 ring-inset ring-primary/35'
                    : ''
                "
                data-testid="pipeline-job-history-row"
                @click="selectedJobId = j.job_id"
              >
                <span
                  class="w-1 shrink-0 self-stretch"
                  :class="jobListStatusBarClass(j)"
                  aria-hidden="true"
                />
                <span class="min-w-0 flex-1 py-1 pr-2 font-mono text-[10px] leading-snug text-surface-foreground">
                  {{ jobOptionLabel(j) }}
                </span>
              </button>
            </div>
          </div>
          <p
            v-if="jobsPickerHint"
            class="text-[9px] leading-snug text-muted"
            data-testid="pipeline-job-history-picker-hint"
          >
            {{ jobsPickerHint }}
          </p>
        </div>
        <div class="flex min-h-0 min-w-0 flex-1 flex-col gap-2 lg:border-l lg:border-border lg:pl-4">
          <p
            v-if="selectedJob && insight"
            class="flex flex-wrap items-baseline gap-x-1 gap-y-0.5 text-[10px] leading-snug text-muted"
            data-testid="pipeline-job-history-summary-line"
          >
            <span>{{ insight }}</span>
            <span aria-hidden="true">·</span>
            <span class="font-mono text-surface-foreground">{{ shortId(selectedJob.job_id) }}</span>
            <template v-if="selectedJob.exit_code != null">
              <span aria-hidden="true">·</span>
              <span>exit {{ selectedJob.exit_code }}</span>
            </template>
            <span aria-hidden="true">·</span>
            <span>wall {{ jobDurationLabel(selectedJob) }}</span>
            <span aria-hidden="true">·</span>
            <a
              class="shrink-0 font-medium text-primary underline decoration-dotted underline-offset-2 hover:decoration-solid"
              :href="pipelineJobLogUrl(root, selectedJob.job_id)"
              target="_blank"
              rel="noopener noreferrer"
              data-testid="pipeline-job-history-log-link"
              :title="selectedJob.log_relpath ? `Open log (${selectedJob.log_relpath})` : 'Open job log in new tab'"
            >Log</a>
          </p>
          <p
            v-else-if="insight"
            class="text-[10px] text-muted"
            data-testid="pipeline-job-history-summary-line-collapsed"
          >
            {{ insight }}
          </p>
          <PipelineJobExplorePanel
            v-if="selectedJob"
            class="min-h-0 flex-1 lg:min-h-0"
            :job="selectedJob"
            :corpus-path="root"
            @open-run-history="emit('open-run-history', $event)"
          />
        </div>
      </div>
    </template>
  </div>
</template>
