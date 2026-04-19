<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import {
  cancelPipelineJob,
  listPipelineJobs,
  reconcilePipelineJobs,
  submitPipelineJob,
  type PipelineJobRow,
} from '../../api/jobsApi'
import { useShellStore } from '../../stores/shell'

const shell = useShellStore()

const jobs = ref<PipelineJobRow[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const reconcileToast = ref<string | null>(null)

const root = computed(() => shell.corpusPath.trim())

const canUseJobs = computed(
  () => Boolean(root.value && shell.healthStatus && shell.jobsApiAvailable),
)

async function refresh(): Promise<void> {
  if (!canUseJobs.value) {
    jobs.value = []
    return
  }
  loading.value = true
  error.value = null
  try {
    const res = await listPipelineJobs(root.value)
    jobs.value = Array.isArray(res.jobs) ? res.jobs : []
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    jobs.value = []
  } finally {
    loading.value = false
  }
}

async function onRun(): Promise<void> {
  if (!canUseJobs.value) {
    return
  }
  loading.value = true
  error.value = null
  try {
    await submitPipelineJob(root.value)
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
    class="rounded border border-border bg-surface p-3 text-sm"
    data-testid="pipeline-jobs-card"
  >
    <div class="mb-2 flex flex-wrap items-center justify-between gap-2">
      <h3 class="text-[10px] font-semibold uppercase tracking-wider text-muted">
        HTTP pipeline jobs
      </h3>
      <div class="flex flex-wrap gap-1">
        <button
          type="button"
          class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-50"
          :disabled="!canUseJobs || loading"
          @click="onRun"
        >
          Run
        </button>
        <button
          type="button"
          class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-50"
          :disabled="!canUseJobs || loading"
          @click="onReconcile"
        >
          Reconcile
        </button>
        <button
          type="button"
          class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-50"
          :disabled="!canUseJobs || loading"
          @click="refresh"
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
      v-if="loading && !jobs.length"
      class="text-[10px] text-muted"
    >
      Loading…
    </p>
    <div
      v-else-if="!jobs.length && canUseJobs"
      class="text-[10px] text-muted"
    >
      No jobs yet. Run queues a CLI pipeline for this corpus.
    </div>
    <ul
      v-else-if="jobs.length"
      class="max-h-48 space-y-1 overflow-y-auto text-[10px]"
    >
      <li
        v-for="j in jobs"
        :key="j.job_id"
        class="flex flex-wrap items-center justify-between gap-2 rounded border border-border/60 bg-elevated/40 px-2 py-1"
      >
        <span class="font-mono text-muted">{{ shortId(j.job_id) }}</span>
        <span class="text-muted">{{ j.status }}</span>
        <button
          v-if="canCancel(j)"
          type="button"
          class="ml-auto rounded border border-border px-1.5 py-0.5 text-[9px] hover:bg-overlay disabled:opacity-50"
          :disabled="loading"
          @click="onCancel(j.job_id)"
        >
          Cancel
        </button>
      </li>
    </ul>
  </div>
</template>
