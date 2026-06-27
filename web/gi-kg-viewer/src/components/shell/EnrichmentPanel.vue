<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  getCorpusEnrichmentsCatalogue,
  getEnrichmentHealth,
  getEnrichmentMetrics,
  getEnrichmentRunSummary,
  getEnrichmentStatus,
  reEnableEnricher,
  submitEnrichmentJob,
  type CorpusEnrichmentsCatalogueItem,
  type EnricherHealthRecord,
  type EnrichmentJobAccepted,
  type EnrichmentMetricsResponse,
  type EnrichmentRunSummary,
  type EnrichmentStatusResponse,
} from '../../api/enrichmentApi'

interface Props {
  corpusPath: string
}
const props = defineProps<Props>()

/** Joined per-enricher view: health + last metrics + envelope availability. */
interface EnricherRow {
  enricher_id: string
  health: EnricherHealthRecord
  metrics?: EnrichmentMetricsResponse['per_enricher'][string]
  envelope?: CorpusEnrichmentsCatalogueItem
}

const status = ref<EnrichmentStatusResponse | null>(null)
const runSummary = ref<EnrichmentRunSummary | null>(null)
const rows = ref<EnricherRow[]>([])

const loading = ref(false)
const error = ref<string | null>(null)
const submitting = ref(false)
const submitNotice = ref<string | null>(null)
const reEnablingId = ref<string | null>(null)

const totalRuns = computed(() =>
  rows.value.reduce((acc, r) => acc + (r.metrics?.runs_total ?? 0), 0),
)
const autoDisabledCount = computed(
  () => rows.value.filter((r) => r.health.auto_disabled).length,
)

async function refresh(): Promise<void> {
  if (!props.corpusPath?.trim()) {
    error.value = 'No corpus path set.'
    return
  }
  loading.value = true
  error.value = null
  try {
    const [healthRes, metricsRes, statusRes, summaryRes, catRes] = await Promise.all([
      getEnrichmentHealth(props.corpusPath),
      getEnrichmentMetrics(props.corpusPath, '24h'),
      getEnrichmentStatus(props.corpusPath),
      getEnrichmentRunSummary(props.corpusPath),
      getCorpusEnrichmentsCatalogue(props.corpusPath),
    ])
    status.value = statusRes
    runSummary.value = summaryRes
    const catalogueById = new Map(catRes.enrichments.map((e) => [e.enricher_id, e]))
    const ids = new Set<string>([
      ...Object.keys(healthRes.enrichers ?? {}),
      ...Object.keys(metricsRes.per_enricher ?? {}),
      ...catalogueById.keys(),
    ])
    rows.value = [...ids]
      .map((id) => ({
        enricher_id: id,
        health: healthRes.enrichers[id] ?? {
          consecutive_failures: 0,
          auto_disabled: false,
          circuit_state: 'closed',
        },
        metrics: metricsRes.per_enricher[id],
        envelope: catalogueById.get(id),
      }))
      .sort((a, b) => a.enricher_id.localeCompare(b.enricher_id))
  } catch (exc) {
    error.value = exc instanceof Error ? exc.message : String(exc)
  } finally {
    loading.value = false
  }
}

async function runEnrichmentNow(): Promise<void> {
  submitting.value = true
  submitNotice.value = null
  error.value = null
  try {
    const accepted: EnrichmentJobAccepted = await submitEnrichmentJob(props.corpusPath, {})
    submitNotice.value = `Enrichment job ${accepted.job_id.slice(0, 12)}… ${accepted.status}.`
    await refresh()
  } catch (exc) {
    error.value = exc instanceof Error ? exc.message : String(exc)
  } finally {
    submitting.value = false
  }
}

async function reEnable(enricherId: string): Promise<void> {
  reEnablingId.value = enricherId
  error.value = null
  try {
    await reEnableEnricher(props.corpusPath, enricherId)
    await refresh()
  } catch (exc) {
    error.value = exc instanceof Error ? exc.message : String(exc)
  } finally {
    reEnablingId.value = null
  }
}

function statusBadgeClass(status: string | null | undefined): string {
  if (!status) return 'bg-overlay text-muted'
  if (status === 'ok') return 'bg-emerald-700/30 text-emerald-300'
  if (status === 'quarantined') return 'bg-amber-700/30 text-amber-300'
  if (status === 'failed' || status === 'timeout') return 'bg-rose-700/30 text-rose-300'
  if (status === 'cancelled' || status === 'skipped') return 'bg-slate-700/30 text-slate-300'
  return 'bg-overlay text-muted'
}

onMounted(refresh)
</script>

<template>
  <section data-testid="enrichment-panel" class="flex flex-col gap-3 text-[11px]">
    <header class="flex items-center justify-between gap-2">
      <div>
        <h2 class="text-sm font-semibold">Enrichment layer</h2>
        <p class="text-muted text-[10px]">
          RFC-088: per-enricher health + last-run status, drift signals, and re-enable.
        </p>
      </div>
      <div class="flex items-center gap-2">
        <button
          type="button"
          class="rounded border border-default bg-overlay px-2 py-1 hover:bg-overlay-2 disabled:opacity-50"
          data-testid="enrichment-refresh-btn"
          :disabled="loading"
          @click="void refresh()"
        >
          {{ loading ? 'Refreshing…' : 'Refresh' }}
        </button>
        <button
          type="button"
          class="rounded border border-emerald-700 bg-emerald-700/30 px-2 py-1 hover:bg-emerald-700/40 disabled:opacity-50"
          data-testid="enrichment-run-btn"
          :disabled="submitting || loading"
          @click="void runEnrichmentNow()"
        >
          {{ submitting ? 'Submitting…' : 'Run enrichment now' }}
        </button>
      </div>
    </header>

    <div v-if="error" class="rounded border border-rose-700 bg-rose-900/30 p-2 text-rose-200">
      {{ error }}
    </div>
    <div
      v-if="submitNotice"
      class="rounded border border-emerald-700 bg-emerald-900/30 p-2 text-emerald-200"
      data-testid="enrichment-submit-notice"
    >
      {{ submitNotice }}
    </div>

    <!-- Top-line glance: last run + counters -->
    <div class="grid grid-cols-3 gap-2 rounded border border-default bg-overlay p-2">
      <div>
        <div class="text-muted text-[10px]">Enrichers tracked</div>
        <div class="text-base font-semibold" data-testid="enrichment-row-count">{{ rows.length }}</div>
      </div>
      <div>
        <div class="text-muted text-[10px]">Total runs (last 24h)</div>
        <div class="text-base font-semibold" data-testid="enrichment-total-runs">{{ totalRuns }}</div>
      </div>
      <div>
        <div class="text-muted text-[10px]">Auto-disabled</div>
        <div
          class="text-base font-semibold"
          :class="autoDisabledCount > 0 ? 'text-amber-300' : ''"
          data-testid="enrichment-autodisabled-count"
        >
          {{ autoDisabledCount }}
        </div>
      </div>
    </div>

    <!-- Last run summary -->
    <div
      v-if="runSummary?.available !== false && runSummary?.run_id"
      class="rounded border border-default bg-overlay p-2"
      data-testid="enrichment-last-run"
    >
      <div class="flex items-center justify-between">
        <div>
          <span class="text-muted">Last run</span>
          <span class="ml-2 font-mono text-[10px]">{{ runSummary?.run_id?.slice(0, 12) }}…</span>
        </div>
        <span
          class="rounded px-2 py-0.5 text-[10px]"
          :class="statusBadgeClass(runSummary?.status ?? null)"
        >
          {{ runSummary?.status ?? '—' }}
        </span>
      </div>
      <div class="text-muted mt-1 text-[10px]">
        {{ runSummary?.finished_at ?? runSummary?.started_at ?? '—' }} ·
        {{ runSummary?.duration_ms ? (runSummary.duration_ms / 1000).toFixed(2) + 's' : '—' }}
        <span v-if="runSummary?.profile"> · profile {{ runSummary.profile }}</span>
      </div>
    </div>

    <!-- Per-enricher table -->
    <div class="overflow-x-auto rounded border border-default">
      <table class="w-full table-fixed text-[10px]" data-testid="enrichment-table">
        <thead class="bg-overlay text-muted">
          <tr>
            <th class="px-2 py-1 text-left">Enricher</th>
            <th class="px-2 py-1 text-left">Last status</th>
            <th class="px-2 py-1 text-right">Runs</th>
            <th class="px-2 py-1 text-right">OK</th>
            <th class="px-2 py-1 text-right">Failed</th>
            <th class="px-2 py-1 text-right">Consec. fails</th>
            <th class="px-2 py-1 text-left">Circuit</th>
            <th class="px-2 py-1 text-left">Auto-disabled</th>
            <th class="px-2 py-1 text-left">Envelope</th>
            <th class="px-2 py-1"></th>
          </tr>
        </thead>
        <tbody>
          <tr v-if="rows.length === 0" data-testid="enrichment-empty-row">
            <td colspan="10" class="text-muted px-2 py-2 italic">
              No enrichers seen yet. Click <em>Run enrichment now</em>.
            </td>
          </tr>
          <tr
            v-for="row in rows"
            :key="row.enricher_id"
            class="border-t border-default hover:bg-overlay/50"
            :data-testid="`enrichment-row-${row.enricher_id}`"
          >
            <td class="px-2 py-1 font-mono">{{ row.enricher_id }}</td>
            <td class="px-2 py-1">
              <span
                class="rounded px-2 py-0.5"
                :class="statusBadgeClass(row.health.last_status)"
              >
                {{ row.health.last_status ?? '—' }}
              </span>
            </td>
            <td class="px-2 py-1 text-right">{{ row.metrics?.runs_total ?? 0 }}</td>
            <td class="px-2 py-1 text-right text-emerald-300">{{ row.metrics?.runs_ok ?? 0 }}</td>
            <td class="px-2 py-1 text-right text-rose-300">{{ row.metrics?.runs_failed ?? 0 }}</td>
            <td class="px-2 py-1 text-right">{{ row.health.consecutive_failures }}</td>
            <td class="px-2 py-1">{{ row.health.circuit_state }}</td>
            <td class="px-2 py-1">
              <span v-if="row.health.auto_disabled" class="text-amber-300">yes</span>
              <span v-else class="text-muted">no</span>
            </td>
            <td class="px-2 py-1">
              <span v-if="row.envelope" class="text-emerald-300" :title="row.envelope.file">
                {{ row.envelope.enricher_version ?? 'present' }}
              </span>
              <span v-else class="text-muted">—</span>
            </td>
            <td class="px-2 py-1 text-right">
              <button
                v-if="row.health.auto_disabled"
                type="button"
                class="rounded border border-amber-700 bg-amber-700/30 px-2 py-0.5 text-[10px] hover:bg-amber-700/40 disabled:opacity-50"
                :data-testid="`enrichment-re-enable-${row.enricher_id}`"
                :disabled="reEnablingId === row.enricher_id"
                @click="void reEnable(row.enricher_id)"
              >
                {{ reEnablingId === row.enricher_id ? '…' : 'Re-enable' }}
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <p class="text-muted text-[10px]">
      Source endpoints: <code>/api/enrichment/{health,metrics,status,run-summary}</code> +
      <code>/api/corpus/enrichments</code>. Submit
      <code>POST /api/jobs/enrichment</code>. Re-enable
      <code>POST /api/enrichment/health/&lt;id&gt;/re-enable</code>.
    </p>
  </section>
</template>
