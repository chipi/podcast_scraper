<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { listPipelineJobs, pipelineJobLogUrl, type PipelineJobRow } from '../../api/jobsApi'
import { useShellStore } from '../../stores/shell'
import { pipelineJobRunDetailsText } from '../../utils/pipelineJobRunDetailsText'

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
const selectedIdx = ref(-1)

const root = computed(() => shell.corpusPath.trim())
const canFetch = computed(
  () => Boolean(root.value && shell.healthStatus && shell.jobsApiAvailable),
)

const strip = computed(() => {
  const term = rows.value.filter((j) => TERMINAL.has(String(j.status).toLowerCase()))
  const chrono = [...term].sort((a, b) => (a.created_at ?? '').localeCompare(b.created_at ?? ''))
  return chrono.slice(-10)
})

watch(
  strip,
  (s) => {
    if (s.length) {
      selectedIdx.value = s.length - 1
    } else {
      selectedIdx.value = -1
    }
  },
  { immediate: true },
)

function dotClass(j: PipelineJobRow): string {
  const st = String(j.status).toLowerCase()
  if (st === 'succeeded') {
    return 'text-success'
  }
  if (st === 'failed' || st === 'stale') {
    return 'text-danger'
  }
  if (st === 'cancelled') {
    return 'text-muted'
  }
  return 'text-muted'
}

function toggle(i: number): void {
  selectedIdx.value = selectedIdx.value === i ? -1 : i
}

const selectedJob = computed(() => {
  const list = strip.value
  const i = selectedIdx.value
  if (i < 0 || i >= list.length) {
    return null
  }
  return list[i] ?? null
})

const insight = computed(() => {
  const list = strip.value
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
    return `All ${list.length} jobs finished successfully`
  }
  return `Last ${list.length} jobs: ${ok} ok, ${bad} failed/stale, ${cancelled} cancelled`
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
      v-else-if="!strip.length && canFetch"
      class="text-[10px] text-muted leading-snug"
    >
      No finished HTTP pipeline jobs in the registry yet. Use <strong class="text-surface-foreground">Jobs</strong> to queue a run.
    </p>
    <template v-else-if="strip.length">
      <div class="flex flex-wrap items-center gap-1">
        <button
          v-for="(j, i) in strip"
          :key="j.job_id"
          type="button"
          :title="`${j.created_at ?? ''} · ${j.status}`"
          :class="[
            dotClass(j),
            i === strip.length - 1 ? 'text-[12px] leading-none' : 'text-[10px] leading-none',
            selectedIdx === i ? 'ring-1 ring-primary ring-offset-1 rounded-full' : '',
          ]"
          data-testid="pipeline-job-history-dot"
          @click="toggle(i)"
        >
          ●
        </button>
      </div>
      <div
        v-if="selectedJob"
        class="mt-2 rounded border border-border bg-overlay p-2 text-[10px] text-muted"
      >
        <p>
          <span class="text-surface-foreground">Job:</span>
          <span class="font-mono">{{ shortId(selectedJob.job_id) }}</span>
          <span class="ml-1 font-medium capitalize text-surface-foreground">{{ selectedJob.status }}</span>
        </p>
        <p v-if="selectedJob.exit_code != null">
          <span class="text-surface-foreground">Exit:</span> {{ selectedJob.exit_code }}
        </p>
        <p>
          <span class="text-surface-foreground">Wall time:</span> {{ jobDurationLabel(selectedJob) }}
        </p>
        <p v-if="selectedJob.error_reason" class="text-danger">
          {{ selectedJob.error_reason }}
        </p>
        <p v-if="selectedJob.log_relpath">
          <span class="text-surface-foreground">Log:</span>
          <a
            class="ml-1 break-all font-mono text-surface-foreground underline decoration-dotted underline-offset-2 hover:decoration-solid"
            :href="pipelineJobLogUrl(root, selectedJob.job_id)"
            target="_blank"
            rel="noopener noreferrer"
          >{{ selectedJob.log_relpath }}</a>
        </p>
        <pre
          class="mt-2 w-full max-w-full overflow-x-auto whitespace-pre-wrap break-all rounded border border-border bg-canvas p-2 text-left font-mono text-[9px] leading-snug text-canvas-foreground"
        >{{ pipelineJobRunDetailsText(selectedJob) }}</pre>
      </div>
      <p
        v-if="insight"
        class="mt-2 text-[11px] text-muted"
      >
        {{ insight }}
      </p>
    </template>
  </div>
</template>
