<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import {
  fetchPipelineJobLogTail,
  formatJobHttpErrorMessage,
  isLivePipelineJobStatus,
  listPipelineJobs,
  pipelineJobLogUrl,
} from '../../api/jobsApi'
import { usePipelineJobLogStore } from '../../stores/pipelineJobLog'
import { useShellStore } from '../../stores/shell'
import { usePageVisible } from '../../composables/usePageVisible'
import { copyTextToClipboard } from '../../utils/clipboard'
import { StaleGeneration } from '../../utils/staleGeneration'
import { textSearchSegments } from '../../utils/textSearch'
import AppDialog from '../shared/AppDialog.vue'

/**
 * In-app pipeline job log viewer (#695). Replaces the "open log in new tab"
 * download links across the pipeline surfaces: shows the tail inline, auto-
 * refreshes while the job is live, and keeps a "Download full log" escape hatch.
 */
const store = usePipelineJobLogStore()
const shell = useShellStore()
const { pageVisible } = usePageVisible()

const TAIL_SIZE_OPTIONS = [
  { label: '16 KB', bytes: 16_384 },
  { label: '64 KB', bytes: 65_536 },
  { label: '256 KB', bytes: 262_144 },
] as const

const maxBytes = ref<number>(65_536)
const loading = ref(false)
const text = ref('')
const truncated = ref(false)
const errorText = ref<string | null>(null)
const copyState = ref<'idle' | 'copied' | 'failed'>('idle')
const bodyEl = ref<HTMLElement | null>(null)
const search = ref('')
const activeMatch = ref(0)

const searchResult = computed(() => textSearchSegments(text.value, search.value))
const matchCount = computed(() => searchResult.value.matchCount)

watch([search, text], () => {
  // Reset to the first hit whenever the query or log content changes.
  activeMatch.value = 0
})

async function scrollActiveMatchIntoView(): Promise<void> {
  await nextTick()
  bodyEl.value
    ?.querySelector(`[data-match-idx="${activeMatch.value}"]`)
    ?.scrollIntoView({ block: 'center' })
}

function stepMatch(delta: number): void {
  if (matchCount.value === 0) return
  activeMatch.value = (activeMatch.value + delta + matchCount.value) % matchCount.value
  void scrollActiveMatchIntoView()
}

const gate = new StaleGeneration()
let pollTimer: ReturnType<typeof setTimeout> | null = null
let stableTicks = 0
let lastFingerprint = ''
let copyResetTimer: ReturnType<typeof setTimeout> | null = null

const target = computed(() => store.target)
const jobId = computed(() => target.value?.jobId ?? '')
const shortJobId = computed(() => {
  const id = jobId.value
  return id.length > 12 ? `${id.slice(0, 8)}…${id.slice(-3)}` : id
})
/** Live status is refreshed by the poll loop (not just the open-time snapshot),
 *  so auto-refresh stops on the real running→terminal transition (#695). */
const liveStatus = ref('')
const isLive = computed(() => isLivePipelineJobStatus(liveStatus.value))
const downloadUrl = computed(() =>
  target.value ? pipelineJobLogUrl(target.value.corpusPath, target.value.jobId) : '#',
)

function clearPollTimer(): void {
  if (pollTimer !== null) {
    clearTimeout(pollTimer)
    pollTimer = null
  }
}

function fingerprint(s: string): string {
  return s ? `${s.length}:${s.slice(-400)}` : '0'
}

function nextDelayMs(): number {
  return stableTicks >= 3 ? 10_000 : 3_000
}

async function scrollToBottom(): Promise<void> {
  await nextTick()
  const el = bodyEl.value
  if (el) {
    el.scrollTop = el.scrollHeight
  }
}

/** One-shot fetch (manual refresh, open, or tail-size change). */
async function loadTail(opts: { scroll: boolean }): Promise<void> {
  const t = target.value
  if (!t || !t.corpusPath.trim() || !t.jobId.trim()) {
    return
  }
  const seq = gate.bump()
  loading.value = true
  errorText.value = null
  try {
    const res = await fetchPipelineJobLogTail(t.corpusPath, t.jobId, maxBytes.value)
    if (gate.isStale(seq)) {
      return
    }
    text.value = res.text
    truncated.value = res.truncated
    lastFingerprint = fingerprint(res.text)
    stableTicks = 0
    if (opts.scroll) {
      void scrollToBottom()
    }
  } catch (e) {
    if (gate.isCurrent(seq)) {
      errorText.value = formatJobHttpErrorMessage(e instanceof Error ? e.message : String(e))
    }
  } finally {
    if (gate.isCurrent(seq)) {
      loading.value = false
    }
  }
}

/** Background poll loop while the job is live; backs off and stops when idle. */
function schedulePoll(): void {
  clearPollTimer()
  if (!pageVisible.value || !shell.jobsApiAvailable || !isLive.value || !store.open) {
    return
  }
  const t = target.value
  if (!t) {
    return
  }
  pollTimer = setTimeout(() => {
    // Refresh the tail AND the job's status, so polling stops on the real
    // running→terminal transition rather than a byte-stable heuristic.
    void Promise.allSettled([
      fetchPipelineJobLogTail(t.corpusPath, t.jobId, maxBytes.value),
      listPipelineJobs(t.corpusPath),
    ])
      .then(([tailRes, jobsRes]) => {
        if (tailRes.status === 'fulfilled') {
          const res = tailRes.value
          const fp = fingerprint(res.text)
          stableTicks = fp === lastFingerprint ? stableTicks + 1 : 0
          lastFingerprint = fp
          const atBottom = isScrolledToBottom()
          text.value = res.text
          truncated.value = res.truncated
          if (atBottom) {
            void scrollToBottom()
          }
        }
        if (jobsRes.status === 'fulfilled') {
          const row = jobsRes.value.jobs.find((j) => j.job_id === t.jobId)
          if (row) {
            liveStatus.value = row.status
          }
        }
      })
      .finally(() => {
        schedulePoll()
      })
  }, nextDelayMs())
}

function isScrolledToBottom(): boolean {
  const el = bodyEl.value
  if (!el) {
    return true
  }
  return el.scrollHeight - el.scrollTop - el.clientHeight < 24
}

async function onRefresh(): Promise<void> {
  await loadTail({ scroll: false })
  schedulePoll()
}

async function onCopy(): Promise<void> {
  if (copyResetTimer !== null) {
    clearTimeout(copyResetTimer)
  }
  const ok = await copyTextToClipboard(text.value)
  copyState.value = ok ? 'copied' : 'failed'
  copyResetTimer = setTimeout(() => {
    copyState.value = 'idle'
  }, 1_500)
}

function onOpenChange(next: boolean): void {
  if (!next) {
    store.close()
  }
}

function resetState(): void {
  text.value = ''
  truncated.value = false
  errorText.value = null
  copyState.value = 'idle'
  search.value = ''
  activeMatch.value = 0
  stableTicks = 0
  lastFingerprint = ''
}

// Open / change of target: reset, load, start polling.
watch(
  () => [store.open, target.value?.jobId, target.value?.corpusPath] as const,
  ([open]) => {
    clearPollTimer()
    if (!open || !target.value) {
      return
    }
    resetState()
    liveStatus.value = target.value.status
    void loadTail({ scroll: true }).then(() => schedulePoll())
  },
  { immediate: true },
)

// Tail-size change: refetch with the new byte budget (only when the dialog is open).
watch(maxBytes, () => {
  if (!store.open || !target.value) {
    return
  }
  void loadTail({ scroll: true }).then(() => schedulePoll())
})

// Pause/resume polling with tab visibility (reuses the shared GH-743 signal).
watch(pageVisible, (visible) => {
  if (visible) {
    schedulePoll()
  } else {
    clearPollTimer()
  }
})

onBeforeUnmount(() => {
  clearPollTimer()
  if (copyResetTimer !== null) {
    clearTimeout(copyResetTimer)
  }
})

defineExpose({ loadTail, onRefresh })
</script>

<template>
  <AppDialog
    :open="store.open"
    title="Pipeline job log"
    :subtitle="target ? `${shortJobId} · ${target.status}${target.logRelpath ? ` · ${target.logRelpath}` : ''}` : null"
    testid="pipeline-job-log-dialog"
    close-testid="pipeline-job-log-close"
    width-class="w-[min(100%,54rem)]"
    max-height-class="max-h-[min(90vh,46rem)]"
    @update:open="onOpenChange"
  >
    <template #header-actions>
      <label class="flex items-center gap-1 text-[11px] text-muted">
        <span class="sr-only">Tail size</span>
        <select
          v-model.number="maxBytes"
          class="rounded border border-border bg-canvas px-1 py-0.5 text-[11px] text-surface-foreground"
          data-testid="pipeline-job-log-tail-size"
        >
          <option
            v-for="opt in TAIL_SIZE_OPTIONS"
            :key="opt.bytes"
            :value="opt.bytes"
          >
            {{ opt.label }}
          </option>
        </select>
      </label>
      <button
        type="button"
        class="rounded border border-border px-2 py-1 text-xs hover:bg-overlay disabled:opacity-50"
        data-testid="pipeline-job-log-refresh"
        :disabled="loading"
        @click="onRefresh"
      >
        {{ loading ? 'Refreshing…' : 'Refresh' }}
      </button>
      <button
        type="button"
        class="rounded border border-border px-2 py-1 text-xs hover:bg-overlay"
        data-testid="pipeline-job-log-copy"
        @click="onCopy"
      >
        {{ copyState === 'copied' ? 'Copied' : copyState === 'failed' ? 'Copy failed' : 'Copy' }}
      </button>
    </template>

    <div class="flex h-full min-h-0 flex-col">
      <p
        v-if="truncated"
        class="shrink-0 border-b border-border bg-elevated/40 px-4 py-1.5 text-[10px] text-muted"
        data-testid="pipeline-job-log-truncated-hint"
      >
        Showing the last {{ Math.round(maxBytes / 1024) }} KB — download the full log for earlier output.
      </p>
      <p
        v-if="errorText"
        class="shrink-0 px-4 py-2 text-xs text-danger"
        data-testid="pipeline-job-log-error"
      >
        {{ errorText }}
      </p>
      <div class="flex shrink-0 items-center gap-1 border-b border-border px-3 py-1">
        <input
          v-model="search"
          type="text"
          placeholder="Find in log…"
          class="min-w-0 flex-1 rounded border border-border bg-canvas px-2 py-0.5 text-[11px] text-surface-foreground placeholder:text-muted"
          data-testid="pipeline-job-log-search"
          @keydown.enter.prevent="stepMatch(1)"
          @keydown.shift.enter.prevent="stepMatch(-1)"
        >
        <span
          v-if="search"
          class="shrink-0 text-[10px] tabular-nums text-muted"
          data-testid="pipeline-job-log-search-count"
        >{{ matchCount ? activeMatch + 1 : 0 }}/{{ matchCount }}</span>
        <button
          type="button"
          class="shrink-0 rounded border border-border px-1.5 py-0.5 text-[11px] hover:bg-overlay disabled:opacity-40"
          data-testid="pipeline-job-log-search-prev"
          :disabled="matchCount === 0"
          aria-label="Previous match"
          @click="stepMatch(-1)"
        >↑</button>
        <button
          type="button"
          class="shrink-0 rounded border border-border px-1.5 py-0.5 text-[11px] hover:bg-overlay disabled:opacity-40"
          data-testid="pipeline-job-log-search-next"
          :disabled="matchCount === 0"
          aria-label="Next match"
          @click="stepMatch(1)"
        >↓</button>
      </div>
      <pre
        ref="bodyEl"
        class="min-h-0 flex-1 overflow-auto whitespace-pre-wrap break-words bg-canvas/80 px-4 py-3 font-mono text-[11px] leading-snug text-surface-foreground"
        data-testid="pipeline-job-log-body"
      ><template v-if="search && text"><template v-for="(seg, i) in searchResult.segments" :key="i"><mark
        v-if="seg.match"
        :data-match-idx="seg.matchIndex"
        :class="seg.matchIndex === activeMatch ? 'bg-warning text-canvas' : 'bg-warning/30 text-surface-foreground'"
      >{{ seg.text }}</mark><template v-else>{{ seg.text }}</template></template></template><template v-else>{{ text || (loading ? 'Loading…' : 'No log output yet.') }}</template></pre>
    </div>

    <template #footer>
      <span class="text-[10px] text-muted">
        <span v-if="isLive">Auto-refreshing while job is live</span>
        <span v-else>Job is in a terminal state — auto-refresh off</span>
      </span>
      <a
        class="rounded border border-border px-2 py-1 text-xs font-medium text-primary hover:bg-overlay"
        :href="downloadUrl"
        target="_blank"
        rel="noopener noreferrer"
        data-testid="pipeline-job-log-download"
      >Download full log</a>
    </template>
  </AppDialog>
</template>
