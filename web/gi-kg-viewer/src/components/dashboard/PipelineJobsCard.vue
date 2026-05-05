<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import {
  cancelPipelineJob,
  listPipelineJobs,
  pipelineJobLogUrl,
  reconcilePipelineJobs,
  submitPipelineJob,
  type PipelineJobRow,
} from '../../api/jobsApi'
import { usePageVisible } from '../../composables/usePageVisible'
import { useShellStore } from '../../stores/shell'
import PipelineJobExplorePanel from './PipelineJobExplorePanel.vue'

const props = withDefaults(
  defineProps<{
    /** When true, omit outer card chrome and title (parent provides tab + border). */
    embedded?: boolean
    /** When true, list only queued/running rows (finished jobs live on Job history tab). */
    activeJobsOnly?: boolean
  }>(),
  { embedded: false, activeJobsOnly: false },
)

const emit = defineEmits<{
  'open-run-history': [payload: { relativePath: string }]
  /** Embedded Jobs empty state: parent switches Pipeline sub-tab to Job history. */
  'go-to-job-history': []
}>()

const shell = useShellStore()
const { pageVisible } = usePageVisible()

const jobs = ref<PipelineJobRow[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const reconcileToast = ref<string | null>(null)
const runToast = ref<string | null>(null)
/** Bumps once per second while any job is running so elapsed time updates without spamming the API. */
const nowTick = ref(Date.now())

const root = computed(() => shell.corpusPath.trim())

const canUseJobs = computed(
  () => Boolean(root.value && shell.healthStatus && shell.jobsApiAvailable),
)

const hasActiveJobs = computed(() =>
  jobs.value.some((j) => j.status === 'running' || j.status === 'queued'),
)

const displayJobs = computed(() => {
  if (!props.activeJobsOnly) {
    return jobs.value
  }
  return jobs.value.filter((j) => j.status === 'queued' || j.status === 'running')
})

/** Embedded dashboard: show empty / hint copy on the same row as Run / Reconcile / Refresh. */
const embeddedToolbarLeadKind = computed<'loading' | 'none' | 'hint' | null>(() => {
  if (!props.embedded || !canUseJobs.value) {
    return null
  }
  if (loading.value && !jobs.value.length) {
    return 'loading'
  }
  if (!displayJobs.value.length && !jobs.value.length) {
    return 'none'
  }
  if (!displayJobs.value.length && jobs.value.length && props.activeJobsOnly) {
    return 'hint'
  }
  return null
})

let pollTimer: ReturnType<typeof setTimeout> | null = null
let clockTimer: ReturnType<typeof setInterval> | null = null
/** Counts consecutive quiet refreshes with an unchanged job snapshot (GH-743 backoff). */
let stableJobPollTicks = 0
let lastJobsFingerprint = ''

function jobsFingerprint(list: PipelineJobRow[]): string {
  return list
    .map(
      (j) =>
        `${j.job_id}:${j.status}:${j.queue_position ?? ''}:${j.pid ?? ''}:${j.started_at ?? ''}:${j.ended_at ?? ''}`,
    )
    .join('|')
}

function clearPollTimer(): void {
  if (pollTimer !== null) {
    clearTimeout(pollTimer)
    pollTimer = null
  }
}

function stopTimers(): void {
  clearPollTimer()
  if (clockTimer) {
    clearInterval(clockTimer)
    clockTimer = null
  }
}

/** Fast while registry rows churn; slower when the snapshot is stable (GH-743). */
function nextJobListPollDelayMs(): number {
  return stableJobPollTicks >= 3 ? 12_000 : 2_500
}

function scheduleJobListPoll(): void {
  clearPollTimer()
  if (!canUseJobs.value || !hasActiveJobs.value || !pageVisible.value) {
    return
  }
  pollTimer = setTimeout(() => {
    void refresh({ quiet: true })
  }, nextJobListPollDelayMs())
}

function syncTimers(): void {
  stopTimers()
  if (!canUseJobs.value) {
    stableJobPollTicks = 0
    lastJobsFingerprint = ''
    return
  }
  if (!hasActiveJobs.value) {
    stableJobPollTicks = 0
    lastJobsFingerprint = ''
    return
  }
  if (pageVisible.value) {
    clockTimer = setInterval(() => {
      nowTick.value = Date.now()
    }, 1000)
  }
  scheduleJobListPoll()
}

async function refresh(opts?: { quiet?: boolean }): Promise<void> {
  if (!canUseJobs.value) {
    jobs.value = []
    stopTimers()
    return
  }
  if (!opts?.quiet) {
    loading.value = true
  }
  error.value = null
  try {
    const res = await listPipelineJobs(root.value)
    jobs.value = Array.isArray(res.jobs) ? res.jobs : []
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    jobs.value = []
  } finally {
    if (!opts?.quiet) {
      loading.value = false
    }
    const fp = jobsFingerprint(jobs.value)
    if (opts?.quiet) {
      if (fp === lastJobsFingerprint) {
        stableJobPollTicks += 1
      } else {
        stableJobPollTicks = 0
      }
    } else {
      stableJobPollTicks = 0
    }
    lastJobsFingerprint = fp
    syncTimers()
  }
}

async function onRun(): Promise<void> {
  if (!canUseJobs.value) {
    return
  }
  loading.value = true
  error.value = null
  runToast.value = null
  try {
    const acc = await submitPipelineJob(root.value)
    runToast.value = `Queued job ${shortId(acc.job_id)} (${acc.status})`
    await refresh()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

async function onReconcile(): Promise<void> {
  if (!canUseJobs.value) {
    return
  }
  loading.value = true
  error.value = null
  reconcileToast.value = null
  try {
    const r = await reconcilePipelineJobs(root.value)
    reconcileToast.value =
      r.updated > 0 ? `Reconciled ${r.updated} job(s)` : 'Registry already consistent'
    await refresh()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

async function onCancel(jobId: string): Promise<void> {
  if (!canUseJobs.value) {
    return
  }
  loading.value = true
  error.value = null
  try {
    await cancelPipelineJob(root.value, jobId)
    await refresh()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

function shortId(id: string): string {
  return id.length > 10 ? `${id.slice(0, 8)}…` : id
}

function canCancel(row: PipelineJobRow): boolean {
  return row.status === 'queued' || row.status === 'running'
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

/** Wall-clock span from ``started_at`` to ``ended_at`` (or ``now`` while still running). */
function elapsedLabel(row: PipelineJobRow, now: number): string {
  if (!row.started_at) {
    return row.status === 'queued' ? 'Waiting to start…' : '—'
  }
  const s = Date.parse(row.started_at)
  if (Number.isNaN(s)) {
    return '—'
  }
  let e: number
  if (row.status === 'running') {
    e = now
  } else if (row.ended_at) {
    const p = Date.parse(row.ended_at)
    e = Number.isNaN(p) ? now : p
  } else {
    e = now
  }
  return formatDurationMs(e - s)
}

function formatClock(iso: string | null): string {
  if (!iso) {
    return ''
  }
  const t = Date.parse(iso)
  if (Number.isNaN(t)) {
    return ''
  }
  return new Date(t).toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
  })
}

function statusClass(status: string): string {
  switch (status) {
    case 'running':
      return 'text-warning font-medium'
    case 'queued':
      return 'text-muted font-medium'
    case 'succeeded':
      return 'text-success font-medium'
    case 'failed':
    case 'stale':
      return 'text-danger font-medium'
    case 'cancelled':
      return 'text-muted font-medium'
    default:
      return 'text-muted'
  }
}

function truncate(s: string, max: number): string {
  const t = s.trim()
  if (t.length <= max) {
    return t
  }
  return `${t.slice(0, max - 1)}…`
}

function jobLogUrl(jobId: string): string {
  return pipelineJobLogUrl(root.value, jobId)
}

watch(
  () => [root.value, shell.healthStatus, shell.jobsApiAvailable] as const,
  () => {
    void refresh()
  },
  { immediate: true },
)

watch(pageVisible, (vis) => {
  if (vis && canUseJobs.value && hasActiveJobs.value) {
    void refresh({ quiet: true })
  } else {
    syncTimers()
  }
})

onUnmounted(() => {
  stopTimers()
})
</script>

<template>
  <div
    :class="
      props.embedded
        ? 'text-sm'
        : 'rounded border border-border bg-surface p-3 text-sm'
    "
    data-testid="pipeline-jobs-card"
  >
    <div
      class="mb-2 flex flex-wrap items-center gap-x-3 gap-y-2"
      :class="
        props.embedded
          ? embeddedToolbarLeadKind
            ? 'justify-between'
            : 'justify-end'
          : 'justify-between'
      "
    >
      <h3
        v-if="!props.embedded"
        class="text-[10px] font-semibold uppercase tracking-wider text-muted"
      >
        Jobs
      </h3>
      <p
        v-else-if="embeddedToolbarLeadKind === 'loading'"
        class="min-w-0 flex-1 text-[10px] text-muted"
      >
        Loading…
      </p>
      <div
        v-else-if="embeddedToolbarLeadKind === 'none'"
        class="min-w-0 flex-1 text-[10px] text-muted"
      >
        No jobs yet. Run queues a CLI pipeline for this corpus.
      </div>
      <p
        v-else-if="embeddedToolbarLeadKind === 'hint'"
        class="min-w-0 flex-1 text-[10px] text-muted leading-snug"
      >
        No queued or running jobs. Open the
        <button
          type="button"
          class="inline cursor-pointer border-0 bg-transparent p-0 align-baseline font-semibold text-primary underline decoration-dotted underline-offset-2 hover:decoration-solid focus-visible:rounded-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
          data-testid="pipeline-jobs-go-to-job-history"
          @click="emit('go-to-job-history')"
        >Job history</button>
        tab for finished jobs.
      </p>
      <div class="flex shrink-0 flex-wrap gap-1">
        <button
          type="button"
          class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-50"
          :disabled="!canUseJobs || loading"
          data-testid="pipeline-jobs-run"
          @click="void onRun()"
        >
          Run
        </button>
        <button
          type="button"
          class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-50"
          :disabled="!canUseJobs || loading"
          @click="void onReconcile()"
        >
          Reconcile
        </button>
        <button
          type="button"
          class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-50"
          :disabled="!canUseJobs || loading"
          @click="void refresh()"
        >
          Refresh
        </button>
      </div>
    </div>
    <p
      v-if="!shell.jobsApiAvailable && shell.healthStatus"
      class="mb-2 text-[10px] text-muted"
    >
      Jobs API is off on this server (start <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">podcast serve --enable-jobs-api</code>).
    </p>
    <p
      v-if="error"
      class="mb-2 text-[10px] text-danger"
    >
      {{ error }}
    </p>
    <p
      v-if="reconcileToast"
      class="mb-2 text-[10px] text-success"
    >
      {{ reconcileToast }}
    </p>
    <p
      v-if="runToast"
      class="mb-2 text-[10px] text-success"
    >
      {{ runToast }}
    </p>
    <p
      v-if="hasActiveJobs"
      class="mb-2 text-[10px] text-muted leading-snug"
    >
      Auto-refresh while a job is queued or running (about every 2.5s when the registry is changing, then slower when stable). Polling pauses when this browser tab is in the background. Elapsed time updates every second while the tab is visible.
    </p>
    <p
      v-if="!props.embedded && loading && !jobs.length"
      class="text-[10px] text-muted"
    >
      Loading…
    </p>
    <div
      v-else-if="!props.embedded && !displayJobs.length && canUseJobs && !jobs.length"
      class="text-[10px] text-muted"
    >
      No jobs yet. Run queues a CLI pipeline for this corpus.
    </div>
    <p
      v-else-if="!props.embedded && !displayJobs.length && canUseJobs && jobs.length && props.activeJobsOnly"
      class="text-[10px] text-muted leading-snug"
    >
      No queued or running jobs. Open the
      <button
        type="button"
        class="inline cursor-pointer border-0 bg-transparent p-0 align-baseline font-semibold text-primary underline decoration-dotted underline-offset-2 hover:decoration-solid focus-visible:rounded-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
        data-testid="pipeline-jobs-go-to-job-history"
        @click="emit('go-to-job-history')"
      >Job history</button>
      tab for finished jobs.
    </p>
    <ul
      v-else-if="displayJobs.length"
      class="space-y-2 text-[10px]"
    >
      <li
        v-for="j in displayJobs"
        :key="j.job_id"
        class="rounded border border-border/60 bg-elevated/40 px-2 py-1.5"
      >
        <div class="flex flex-wrap items-center justify-between gap-2">
          <span class="font-mono text-muted">{{ shortId(j.job_id) }}</span>
          <span :class="statusClass(j.status)">{{ j.status }}</span>
          <div class="ml-auto flex flex-wrap items-center gap-1.5 text-[9px] text-muted">
            <span v-if="j.queue_position != null && j.status === 'queued'">
              queue #{{ j.queue_position }}
            </span>
            <span v-if="j.pid">pid {{ j.pid }}</span>
            <span v-if="j.exit_code != null && j.status !== 'running' && j.status !== 'queued'">
              exit {{ j.exit_code }}
            </span>
            <button
              v-if="canCancel(j)"
              type="button"
              class="rounded border border-border px-1.5 py-0.5 hover:bg-overlay disabled:opacity-50"
              :disabled="loading"
              @click="onCancel(j.job_id)"
            >
              Cancel
            </button>
          </div>
        </div>
        <PipelineJobExplorePanel
          :job="j"
          :corpus-path="root"
          @open-run-history="emit('open-run-history', $event)"
        />
        <div class="mt-1 space-y-0.5 text-[9px] leading-snug text-muted">
          <p v-if="j.started_at">
            Started {{ formatClock(j.started_at) }}
            <span v-if="j.ended_at && j.status !== 'running'"> · Ended {{ formatClock(j.ended_at) }}</span>
            · Runtime {{ elapsedLabel(j, nowTick) }}
          </p>
          <p v-else-if="j.status === 'queued'">
            {{ elapsedLabel(j, nowTick) }}
          </p>
          <p v-if="j.log_relpath" class="break-all font-mono text-[8px] opacity-90">
            <span class="text-muted">Log:</span>
            <a
              class="ml-1 text-surface-foreground underline decoration-dotted underline-offset-2 hover:decoration-solid"
              :href="jobLogUrl(j.job_id)"
              target="_blank"
              rel="noopener noreferrer"
              data-testid="pipeline-job-log-link"
              :title="`Open log in new tab (${j.log_relpath})`"
            >{{ j.log_relpath }}</a>
          </p>
          <p v-if="j.error_reason" class="text-danger">
            {{ truncate(j.error_reason, 160) }}
          </p>
        </div>
      </li>
    </ul>
  </div>
</template>
