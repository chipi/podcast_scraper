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
import { useShellStore } from '../../stores/shell'
import { pipelineJobRunDetailsText } from '../../utils/pipelineJobRunDetailsText'

const props = withDefaults(
  defineProps<{
    /** When true, omit outer card chrome and title (parent provides tab + border). */
    embedded?: boolean
    /** When true, list only queued/running rows (finished jobs live on Job history tab). */
    activeJobsOnly?: boolean
  }>(),
  { embedded: false, activeJobsOnly: false },
)

const shell = useShellStore()

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

let pollTimer: ReturnType<typeof setInterval> | null = null
let clockTimer: ReturnType<typeof setInterval> | null = null

function stopTimers(): void {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
  if (clockTimer) {
    clearInterval(clockTimer)
    clockTimer = null
  }
}

function syncTimers(): void {
  stopTimers()
  if (!canUseJobs.value) {
    return
  }
  if (hasActiveJobs.value) {
    clockTimer = setInterval(() => {
      nowTick.value = Date.now()
    }, 1000)
    pollTimer = setInterval(() => {
      void refresh({ quiet: true })
    }, 4000)
  }
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
      class="mb-2 flex flex-wrap items-center gap-2"
      :class="props.embedded ? 'justify-end' : 'justify-between'"
    >
      <h3
        v-if="!props.embedded"
        class="text-[10px] font-semibold uppercase tracking-wider text-muted"
      >
        Jobs
      </h3>
      <div class="flex flex-wrap gap-1">
        <button
          type="button"
          class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-50"
          :disabled="!canUseJobs || loading"
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
      Auto-refresh every 4s while a job is queued or running; elapsed time updates every second.
    </p>
    <p
      v-if="loading && !jobs.length"
      class="text-[10px] text-muted"
    >
      Loading…
    </p>
    <div
      v-else-if="!displayJobs.length && canUseJobs && !jobs.length"
      class="text-[10px] text-muted"
    >
      No jobs yet. Run queues a CLI pipeline for this corpus.
    </div>
    <p
      v-else-if="!displayJobs.length && canUseJobs && jobs.length && props.activeJobsOnly"
      class="text-[10px] text-muted leading-snug"
    >
      No queued or running jobs. Open the <strong class="text-surface-foreground">Job history</strong> tab for finished HTTP pipeline jobs.
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
        <details class="mt-1">
          <summary
            class="cursor-pointer list-none text-[9px] text-muted hover:text-surface-foreground [&::-webkit-details-marker]:hidden"
            title="Exact subprocess argv and paths from the job registry"
          >
            <span class="rounded border border-border/80 bg-overlay/30 px-1.5 py-0.5 font-medium text-surface-foreground">
              Command line and paths
            </span>
          </summary>
          <pre
            class="mt-1 w-full max-w-full overflow-x-auto whitespace-pre-wrap break-all rounded border border-border bg-canvas p-2 text-left font-mono text-[9px] leading-snug text-canvas-foreground"
            role="region"
            :aria-label="`Job ${j.job_id} run details`"
          >{{ pipelineJobRunDetailsText(j) }}</pre>
        </details>
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
