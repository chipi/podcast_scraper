<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import type { PipelineJobRow } from '../../api/jobsApi'
import {
  fetchPipelineJobLogTail,
  formatJobHttpErrorMessage,
  pipelineJobLogUrl,
} from '../../api/jobsApi'
import {
  fetchCorpusManifest,
  fetchCorpusRunSummaryDocument,
  fetchCorpusRunsSummary,
  type CorpusManifestDocument,
  type CorpusRunSummaryItem,
} from '../../api/corpusMetricsApi'
import { useShellStore } from '../../stores/shell'
import { pipelineJobRunDetailsText } from '../../utils/pipelineJobRunDetailsText'
import { extractStructuredSummariesFromLogTail } from '../../utils/pipelineJobLogSummary'
import {
  buildCorpusLikeDocumentView,
  buildMultiFeedBatchView,
  partitionBatchIncidentRows,
  type CorpusLikeDocumentView,
  type LabelValueRow,
} from '../../utils/humanizeJsonDocument'
import { feedRowRunRelativePaths } from '../../utils/feedRunLinking'
import { usePageVisible } from '../../composables/usePageVisible'

const SUMMARY_KV_COLS = 3

const emit = defineEmits<{
  'open-run-history': [payload: { relativePath: string }]
}>()

function chunkLabelValueRows(rows: LabelValueRow[], perRow: number): LabelValueRow[][] {
  if (perRow < 1) {
    return rows.length ? [rows] : []
  }
  const out: LabelValueRow[][] = []
  for (let i = 0; i < rows.length; i += perRow) {
    out.push(rows.slice(i, i + perRow))
  }
  return out
}

const props = defineProps<{
  job: PipelineJobRow
  corpusPath: string
}>()

type ExploreTab = 'summary' | 'metrics' | 'details'

const shell = useShellStore()
const { pageVisible } = usePageVisible()
const tab = ref<ExploreTab>('summary')
const loading = ref(false)
const tailText = ref('')
const tailTruncated = ref(false)
const tailErr = ref<string | null>(null)
const corpusDoc = ref<Record<string, unknown> | null>(null)
const corpusDocErr = ref<string | null>(null)
const runsSummary = ref<CorpusRunSummaryItem[]>([])
const manifestDoc = ref<CorpusManifestDocument | null>(null)

let tailPollTimer: ReturnType<typeof setTimeout> | null = null
let tailPollStableTicks = 0
let lastTailPollFingerprint = ''

function isLiveJobStatus(): boolean {
  const s = String(props.job.status).toLowerCase()
  return s === 'running' || s === 'queued'
}

function tailPollFingerprint(text: string): string {
  if (!text) {
    return '0'
  }
  return `${text.length}:${text.slice(-400)}`
}

function clearTailPollTimer(): void {
  if (tailPollTimer !== null) {
    clearTimeout(tailPollTimer)
    tailPollTimer = null
  }
}

function nextTailPollDelayMs(): number {
  return tailPollStableTicks >= 3 ? 10_000 : 3_000
}

function scheduleTailPoll(): void {
  clearTailPollTimer()
  if (!pageVisible.value || !shell.jobsApiAvailable || !isLiveJobStatus()) {
    return
  }
  const p = props.corpusPath.trim()
  const jid = props.job.job_id.trim()
  if (!p || !jid) {
    return
  }
  tailPollTimer = setTimeout(() => {
    void fetchPipelineJobLogTail(p, jid)
      .then((res) => {
        const fp = tailPollFingerprint(res.text)
        if (fp === lastTailPollFingerprint) {
          tailPollStableTicks += 1
        } else {
          tailPollStableTicks = 0
        }
        lastTailPollFingerprint = fp
        tailText.value = res.text
        tailTruncated.value = res.truncated
      })
      .catch(() => {
        /* keep prior tail on transient errors */
      })
      .finally(() => {
        if (isLiveJobStatus()) {
          scheduleTailPoll()
        }
      })
  }, nextTailPollDelayMs())
}

const structured = computed(() => extractStructuredSummariesFromLogTail(tailText.value))

const corpusView = computed(() => buildCorpusLikeDocumentView(corpusDoc.value))

const logBatchView = computed(() => buildMultiFeedBatchView(structured.value.multiFeedBatch))

const logCorpusLineView = computed((): CorpusLikeDocumentView | null => {
  const s = structured.value.corpusMultiFeedSummary
  if (!s || typeof s !== 'object' || Array.isArray(s)) {
    return null
  }
  return buildCorpusLikeDocumentView(s as Record<string, unknown>)
})

const metricsHasContent = computed(() => {
  const b = logBatchView.value
  const c = logCorpusLineView.value
  return (
    (b.metaRows.length > 0 || (b.feedsTable?.rows.length ?? 0) > 0) ||
    (c != null &&
      (c.corpusParentRow != null ||
        c.metaRows.length > 0 ||
        c.incidentRows.length > 0 ||
        (c.feedsTable?.rows.length ?? 0) > 0))
  )
})

const summaryFeedRowRunPaths = computed(() =>
  feedRowRunRelativePaths(
    corpusView.value.feedsRowFeedUrls,
    runsSummary.value,
    manifestDoc.value?.feeds,
  ),
)

function openRunHistoryForRow(relativePath: string): void {
  const rp = relativePath.trim()
  if (!rp) {
    return
  }
  emit('open-run-history', { relativePath: rp })
}

const corpusMetaChunks = computed(() =>
  chunkLabelValueRows(corpusView.value.metaRows, SUMMARY_KV_COLS),
)
const corpusIncidentParts = computed(() => partitionBatchIncidentRows(corpusView.value.incidentRows))
const corpusIncidentCompactChunks = computed(() =>
  chunkLabelValueRows(corpusIncidentParts.value.compactRows, SUMMARY_KV_COLS),
)

async function load(): Promise<void> {
  clearTailPollTimer()
  tailPollStableTicks = 0
  lastTailPollFingerprint = ''
  loading.value = true
  tailErr.value = null
  corpusDocErr.value = null
  corpusDoc.value = null
  tailText.value = ''
  tailTruncated.value = false
  runsSummary.value = []
  manifestDoc.value = null
  const p = props.corpusPath.trim()
  const jid = props.job.job_id.trim()

  const tailP =
    shell.jobsApiAvailable && jid
      ? fetchPipelineJobLogTail(p, jid).catch((e: unknown) => {
          tailErr.value = formatJobHttpErrorMessage(e instanceof Error ? e.message : String(e))
          return null
        })
      : Promise.resolve(null)

  const corpusP =
    shell.corpusMetricsApiAvailable && p
      ? fetchCorpusRunSummaryDocument(p).catch((e: unknown) => {
          corpusDocErr.value = e instanceof Error ? e.message : String(e)
          return null
        })
      : Promise.resolve(null)

  const runsP =
    shell.corpusMetricsApiAvailable && p
      ? fetchCorpusRunsSummary(p).catch(() => null)
      : Promise.resolve(null)

  const manifestP =
    shell.corpusMetricsApiAvailable && p
      ? fetchCorpusManifest(p).catch(() => null)
      : Promise.resolve(null)

  const [tailRes, doc, runsRes, man] = await Promise.all([tailP, corpusP, runsP, manifestP])
  if (tailRes) {
    tailText.value = tailRes.text
    tailTruncated.value = tailRes.truncated
  }
  corpusDoc.value = doc
  runsSummary.value = Array.isArray(runsRes?.runs) ? runsRes!.runs : []
  manifestDoc.value = man
  loading.value = false
  if (isLiveJobStatus()) {
    lastTailPollFingerprint = tailPollFingerprint(tailText.value)
    scheduleTailPoll()
  }
}

watch(
  () =>
    [props.job.job_id, props.corpusPath, shell.corpusMetricsApiAvailable, shell.jobsApiAvailable] as const,
  () => {
    tab.value = 'summary'
    void load()
  },
  { immediate: true },
)

watch(pageVisible, () => {
  if (pageVisible.value && isLiveJobStatus() && shell.jobsApiAvailable) {
    tailPollStableTicks = 0
    scheduleTailPoll()
  } else {
    clearTailPollTimer()
  }
})

onUnmounted(() => {
  clearTailPollTimer()
})
</script>

<template>
  <div class="mt-2 space-y-2">
    <nav
      class="flex flex-wrap gap-1 border-b border-border pb-1"
      role="tablist"
      aria-label="Job log and metrics"
    >
      <button
        type="button"
        role="tab"
        :aria-selected="tab === 'summary'"
        class="rounded px-2 py-0.5 text-[9px] font-medium"
        :class="
          tab === 'summary'
            ? 'bg-primary text-primary-foreground'
            : 'text-muted hover:bg-overlay'
        "
        data-testid="pipeline-job-explore-tab-summary"
        @click="tab = 'summary'"
      >
        Summary
      </button>
      <button
        type="button"
        role="tab"
        :aria-selected="tab === 'metrics'"
        class="rounded px-2 py-0.5 text-[9px] font-medium"
        :class="
          tab === 'metrics'
            ? 'bg-primary text-primary-foreground'
            : 'text-muted hover:bg-overlay'
        "
        data-testid="pipeline-job-explore-tab-metrics"
        @click="tab = 'metrics'"
      >
        Metrics
      </button>
      <button
        type="button"
        role="tab"
        :aria-selected="tab === 'details'"
        class="rounded px-2 py-0.5 text-[9px] font-medium"
        :class="
          tab === 'details'
            ? 'bg-primary text-primary-foreground'
            : 'text-muted hover:bg-overlay'
        "
        data-testid="pipeline-job-explore-tab-details"
        @click="tab = 'details'"
      >
        Command & paths
      </button>
    </nav>

    <!-- Summary: corpus_run_summary.json only -->
    <div
      v-if="tab === 'summary'"
      class="min-h-0 space-y-1.5 text-[9px] leading-snug text-muted"
      role="tabpanel"
    >
      <p v-if="loading" class="text-muted">
        Loading summary…
      </p>
      <template v-else>
        <p class="text-[8px] text-muted">
          <strong class="text-surface-foreground">corpus_run_summary.json</strong> at corpus root
          (latest batch; may be from a different run if another job finished later).
        </p>
        <div
          v-if="corpusDoc && !corpusDocErr"
          class="space-y-1.5 rounded border border-border/60 bg-canvas/50 p-1.5 text-canvas-foreground"
        >
          <p class="font-medium text-surface-foreground">
            Corpus batch
          </p>
          <div
            v-if="corpusView.corpusParentRow"
            class="kv-corpus-parent-row mb-1 rounded border border-border/50 bg-overlay/25 px-1 py-0.5"
          >
            <div class="text-[7px] font-medium uppercase tracking-wide text-muted">
              {{ corpusView.corpusParentRow.label }}
            </div>
            <p class="break-all font-mono text-[8px] leading-snug text-surface-foreground">
              {{ corpusView.corpusParentRow.value }}
            </p>
          </div>
          <table
            v-if="corpusView.metaRows.length"
            class="kv-compact-table kv-compact-meta w-full border-collapse text-left text-[8px]"
          >
            <tbody>
              <tr
                v-for="(group, gi) in corpusMetaChunks"
                :key="gi"
                class="align-top"
              >
                <template
                  v-for="(row, ri) in group"
                  :key="`${gi}-${ri}-${row.label}`"
                >
                  <th
                    scope="row"
                    class="kv-compact-th text-muted"
                  >
                    {{ row.label }}
                  </th>
                  <td class="kv-compact-td font-mono text-surface-foreground">
                    {{ row.value }}
                  </td>
                </template>
              </tr>
            </tbody>
          </table>
          <div
            v-if="corpusView.incidentRows.length"
            class="border-t border-border/40 pt-1.5"
          >
            <p class="mb-0.5 font-medium text-surface-foreground">
              Batch incidents
            </p>
            <table
              v-if="corpusIncidentCompactChunks.length"
              class="kv-compact-table kv-compact-incidents w-full border-collapse text-left text-[8px]"
            >
              <tbody>
                <tr
                  v-for="(group, gi) in corpusIncidentCompactChunks"
                  :key="gi"
                  class="align-top"
                >
                  <template
                    v-for="(row, ri) in group"
                    :key="`${gi}-${ri}-${row.label}`"
                  >
                    <th
                      scope="row"
                      class="kv-compact-th text-muted"
                    >
                      {{ row.label }}
                    </th>
                    <td class="kv-compact-td font-mono text-surface-foreground">
                      {{ row.value }}
                    </td>
                  </template>
                </tr>
              </tbody>
            </table>
            <table
              v-if="corpusIncidentParts.longRows.length"
              class="kv-incident-long-table mt-1 w-full border-collapse text-left text-[8px]"
            >
              <tbody>
                <tr
                  v-for="row in corpusIncidentParts.longRows"
                  :key="row.label"
                  class="align-top"
                >
                  <th
                    scope="row"
                    class="kv-incident-long-th text-muted"
                  >
                    {{ row.label }}
                  </th>
                  <td class="kv-incident-long-td font-mono text-surface-foreground">
                    {{ row.value }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div
            v-if="corpusView.feedsTable"
            class="border-t border-border/40 pt-1.5"
          >
            <p class="mb-1 font-medium text-surface-foreground">
              Feeds
            </p>
            <div class="max-h-32 overflow-auto rounded border border-border">
              <table class="w-full border-collapse text-left text-[8px]">
                <thead class="sticky top-0 bg-overlay">
                  <tr>
                    <th
                      v-for="h in corpusView.feedsTable.headers"
                      :key="h"
                      class="border-b border-border px-1 py-0.5 font-medium text-muted"
                    >
                      {{ h }}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(r, ri) in corpusView.feedsTable.rows"
                    :key="ri"
                    class="border-b border-border/40 last:border-0"
                  >
                    <td
                      v-for="(c, ci) in r"
                      :key="ci"
                      class="px-1 py-0.5 align-top font-mono text-[8px] text-surface-foreground"
                    >
                      <button
                        v-if="ci === 0 && summaryFeedRowRunPaths[ri]"
                        type="button"
                        class="max-w-full cursor-pointer break-all text-left text-primary underline decoration-dotted underline-offset-2 hover:decoration-solid"
                        :title="`Open Run history · ${summaryFeedRowRunPaths[ri]}`"
                        data-testid="summary-feed-run-history-link"
                        @click="openRunHistoryForRow(summaryFeedRowRunPaths[ri]!)"
                      >
                        {{ c }}
                      </button>
                      <span v-else>{{ c }}</span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <p v-else-if="corpusDocErr" class="text-[8px] text-danger">
          corpus_run_summary: {{ corpusDocErr }}
        </p>
        <p v-else-if="!shell.corpusMetricsApiAvailable" class="text-[8px] text-muted">
          Corpus metrics API is off — open the Metrics tab for log-derived fields.
        </p>
      </template>
    </div>

    <!-- Metrics: structured log payloads + optional raw tail -->
    <div
      v-else-if="tab === 'metrics'"
      class="min-h-[6rem] space-y-2 text-[9px] leading-snug text-muted"
      role="tabpanel"
    >
      <p v-if="loading" class="text-muted">
        Loading metrics…
      </p>
      <template v-else>
        <p class="text-[8px] text-muted">
          Parsed from the end of this job’s log (multi-feed batch / corpus summary lines).
        </p>

        <div
          v-if="logBatchView.metaRows.length > 0 || logBatchView.feedsTable"
          class="space-y-2 rounded border border-border/60 bg-canvas/50 p-2 text-canvas-foreground"
        >
          <p class="font-medium text-surface-foreground">
            multi_feed_batch
          </p>
          <dl
            v-if="logBatchView.metaRows.length"
            class="grid grid-cols-[minmax(0,10rem)_1fr] gap-x-2 gap-y-0.5"
          >
            <template
              v-for="row in logBatchView.metaRows"
              :key="row.label"
            >
              <dt class="text-muted">
                {{ row.label }}
              </dt>
              <dd class="break-words font-mono text-[8px] text-surface-foreground">
                {{ row.value }}
              </dd>
            </template>
          </dl>
          <div v-if="logBatchView.feedsTable">
            <p class="mb-1 text-[8px] font-medium text-surface-foreground">
              Feeds
            </p>
            <div class="max-h-36 overflow-auto rounded border border-border">
              <table class="w-full border-collapse text-left text-[8px]">
                <thead class="sticky top-0 bg-overlay">
                  <tr>
                    <th
                      v-for="h in logBatchView.feedsTable.headers"
                      :key="h"
                      class="border-b border-border px-1 py-0.5 font-medium text-muted"
                    >
                      {{ h }}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(r, ri) in logBatchView.feedsTable.rows"
                    :key="ri"
                    class="border-b border-border/40 last:border-0"
                  >
                    <td
                      v-for="(c, ci) in r"
                      :key="ci"
                      class="px-1 py-0.5 align-top font-mono text-surface-foreground"
                    >
                      {{ c }}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div
          v-if="
            logCorpusLineView &&
              (logCorpusLineView.corpusParentRow ||
                logCorpusLineView.metaRows.length ||
                logCorpusLineView.incidentRows.length ||
                logCorpusLineView.feedsTable)
          "
          class="space-y-2 rounded border border-border/60 bg-canvas/50 p-2 text-canvas-foreground"
        >
          <p class="font-medium text-surface-foreground">
            corpus_multi_feed_summary (log line)
          </p>
          <div
            v-if="logCorpusLineView.corpusParentRow"
            class="kv-corpus-parent-row rounded border border-border/50 bg-overlay/25 px-1 py-0.5"
          >
            <div class="text-[7px] font-medium uppercase tracking-wide text-muted">
              {{ logCorpusLineView.corpusParentRow.label }}
            </div>
            <p class="break-all font-mono text-[8px] leading-snug text-surface-foreground">
              {{ logCorpusLineView.corpusParentRow.value }}
            </p>
          </div>
          <dl
            v-if="logCorpusLineView.metaRows.length"
            class="grid grid-cols-[minmax(0,10rem)_1fr] gap-x-2 gap-y-0.5"
          >
            <template
              v-for="row in logCorpusLineView.metaRows"
              :key="row.label"
            >
              <dt class="text-muted">
                {{ row.label }}
              </dt>
              <dd class="break-words font-mono text-[8px] text-surface-foreground">
                {{ row.value }}
              </dd>
            </template>
          </dl>
          <dl
            v-if="logCorpusLineView.incidentRows.length"
            class="grid grid-cols-[minmax(0,10rem)_1fr] gap-x-2 gap-y-0.5 border-t border-border/40 pt-2"
          >
            <dt class="col-span-2 pb-0.5 font-medium text-surface-foreground">
              Batch incidents
            </dt>
            <template
              v-for="row in logCorpusLineView.incidentRows"
              :key="row.label"
            >
              <dt class="text-muted">
                {{ row.label }}
              </dt>
              <dd class="break-words font-mono text-[8px] text-surface-foreground">
                {{ row.value }}
              </dd>
            </template>
          </dl>
          <div
            v-if="logCorpusLineView.feedsTable"
            class="border-t border-border/40 pt-2"
          >
            <p class="mb-1 font-medium text-surface-foreground">
              Feeds
            </p>
            <div class="max-h-36 overflow-auto rounded border border-border">
              <table class="w-full border-collapse text-left text-[8px]">
                <thead class="sticky top-0 bg-overlay">
                  <tr>
                    <th
                      v-for="h in logCorpusLineView.feedsTable.headers"
                      :key="h"
                      class="border-b border-border px-1 py-0.5 font-medium text-muted"
                    >
                      {{ h }}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(r, ri) in logCorpusLineView.feedsTable.rows"
                    :key="ri"
                    class="border-b border-border/40 last:border-0"
                  >
                    <td
                      v-for="(c, ci) in r"
                      :key="ci"
                      class="px-1 py-0.5 align-top font-mono text-surface-foreground"
                    >
                      {{ c }}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <p
          v-if="!loading && !metricsHasContent && !tailErr"
          class="text-[8px] text-muted"
        >
          No structured metrics lines in this log tail yet.
        </p>

        <p v-if="tailTruncated" class="text-[8px] text-warning">
          Log tail was truncated — open the full log from Command & paths.
        </p>
        <div
          v-if="tailErr"
          class="rounded border border-danger/40 bg-danger/5 p-2 text-[8px] leading-snug text-danger"
          data-testid="pipeline-job-explore-metrics-tail-error"
        >
          <p class="font-medium text-danger">
            Log tail failed — Metrics needs the job log.
          </p>
          <p class="mt-1 text-surface-foreground">
            {{ tailErr }}
          </p>
          <p v-if="job.log_relpath" class="mt-1.5">
            <a
              class="font-medium text-primary underline decoration-dotted underline-offset-2 hover:decoration-solid"
              :href="pipelineJobLogUrl(corpusPath, job.job_id)"
              target="_blank"
              rel="noopener noreferrer"
              data-testid="pipeline-job-explore-metrics-full-log-fallback"
            >Open full log (new tab)</a>
          </p>
        </div>

        <details
          v-if="tailText"
          class="rounded border border-border/60 bg-overlay/30"
        >
          <summary class="cursor-pointer px-2 py-1 text-[8px] text-muted hover:text-surface-foreground">
            Raw log tail (text)
          </summary>
          <pre
            class="max-h-48 w-full overflow-auto whitespace-pre-wrap break-all p-2 font-mono text-[8px] text-canvas-foreground"
          >{{ tailText }}</pre>
        </details>
      </template>
    </div>

    <!-- Command & paths -->
    <div
      v-else
      class="min-h-[6rem] space-y-2 text-[9px]"
      role="tabpanel"
    >
      <p v-if="job.log_relpath">
        <a
          class="break-all font-mono text-surface-foreground underline decoration-dotted underline-offset-2 hover:decoration-solid"
          :href="pipelineJobLogUrl(corpusPath, job.job_id)"
          target="_blank"
          rel="noopener noreferrer"
          data-testid="pipeline-job-explore-full-log-link"
        >Open full log (new tab)</a>
      </p>
      <pre
        class="w-full max-w-full overflow-x-auto whitespace-pre-wrap break-all rounded border border-border bg-canvas p-2 text-left font-mono text-[8px] leading-snug text-canvas-foreground"
        role="region"
        :aria-label="`Job ${job.job_id} command and paths`"
      >{{ pipelineJobRunDetailsText(job) }}</pre>
    </div>
  </div>
</template>

<style scoped>
/* Summary: several label/value pairs per row to shorten the panel. */
.kv-compact-table {
  table-layout: fixed;
}
.kv-compact-th {
  padding: 1px 6px 1px 0;
  font-weight: 400;
  vertical-align: top;
  line-height: 1.25;
  word-break: break-word;
}
.kv-compact-td {
  min-width: 0;
  padding: 1px 6px 1px 0;
  vertical-align: top;
  line-height: 1.25;
  word-break: break-word;
}
/* Corpus meta: wider labels, values still flexible (paths, timestamps). */
.kv-compact-meta .kv-compact-th {
  width: 20%;
}
.kv-compact-meta .kv-compact-td {
  width: 13.33%;
}
/* Batch incidents grid: mostly small counts — give labels room, tighten value columns. */
.kv-compact-incidents .kv-compact-th {
  width: 27%;
}
.kv-compact-incidents .kv-compact-td {
  width: 6%;
  max-width: 3.5rem;
}
/* Long prose / paths: one row each, full width for the value. */
.kv-incident-long-table {
  table-layout: fixed;
}
.kv-incident-long-th {
  width: 22%;
  padding: 2px 8px 4px 0;
  font-weight: 400;
  vertical-align: top;
  line-height: 1.35;
  word-break: break-word;
}
.kv-incident-long-td {
  width: 78%;
  min-width: 0;
  padding: 2px 0 4px;
  vertical-align: top;
  line-height: 1.35;
  word-break: break-word;
}
</style>
