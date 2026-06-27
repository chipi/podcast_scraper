/**
 * RFC-088 enrichment-layer API client (viewer side).
 *
 * Wraps the server-side surfaces shipped in chunks 1 + 6a:
 *  - GET    /api/enrichment/status / health / metrics / events / run-summary
 *  - POST   /api/enrichment/health/{id}/re-enable
 *  - POST   /api/jobs/enrichment
 *  - GET    /api/corpus/enrichments        (catalogue)
 *  - GET    /api/corpus/enrichments/{id}   (single corpus envelope)
 */

import { fetchWithTimeout } from './httpClient'
import { readApiErrorMessage } from './readApiErrorMessage'

export interface EnricherHealthRecord {
  enricher_id?: string
  consecutive_failures: number
  auto_disabled: boolean
  auto_disabled_at?: string | null
  auto_disabled_reason?: string | null
  last_run_id?: string | null
  last_run_at?: string | null
  last_status?: string | null
  circuit_state: string
  circuit_opened_at?: string | null
  cooldown_until?: string | null
}

export interface EnrichmentHealthResponse {
  enrichers: Record<string, EnricherHealthRecord>
}

export interface EnrichmentStatusResponse {
  available?: boolean
  reason?: string
  run_id?: string
  profile?: string | null
  current_enricher?: string | null
  queue?: string[]
  completed?: Array<{ enricher_id: string; status: string; duration_ms?: number }>
  idle?: boolean
  started_at?: string
}

export interface EnrichmentMetricsResponse {
  window: string
  per_enricher: Record<
    string,
    {
      runs_total?: number
      runs_ok?: number
      runs_failed?: number
      runs_timeout?: number
      runs_quarantined?: number
      runs_cancelled?: number
      runs_skipped?: number
      retries_total?: number
      avg_duration_s?: number
      total_cost_usd?: number
    }
  >
}

export interface EnrichmentRunSummary {
  available?: boolean
  status?: string
  run_id?: string
  profile?: string | null
  started_at?: string
  finished_at?: string
  duration_ms?: number
  per_enricher?: Record<string, unknown>
}

export interface EnrichmentEventsResponse {
  events: Array<Record<string, unknown>>
  count: number
}

export interface CorpusEnrichmentsCatalogueItem {
  enricher_id: string
  enricher_version?: string | null
  schema_version?: string | null
  file: string
  size_bytes: number
}

export interface CorpusEnrichmentsCatalogue {
  enrichments: CorpusEnrichmentsCatalogueItem[]
}

export interface EnrichmentJobSubmitRequest {
  only?: string[]
  skip?: string[]
  corpus_only?: boolean
}

export interface EnrichmentJobAccepted {
  job_id: string
  status: 'running' | 'queued' | string
  corpus_path: string
}

function q(corpusPath: string, extra?: Record<string, string>): string {
  const u = new URLSearchParams({ path: corpusPath.trim() })
  if (extra) {
    for (const [k, v] of Object.entries(extra)) {
      u.set(k, v)
    }
  }
  return u.toString()
}

async function getJson<T>(url: string, detail: string): Promise<T> {
  const res = await fetchWithTimeout(url, undefined, { timeoutDetail: detail })
  if (!res.ok) {
    throw new Error(await readApiErrorMessage(res))
  }
  return (await res.json()) as T
}

export async function getEnrichmentStatus(
  corpusPath: string,
): Promise<EnrichmentStatusResponse> {
  return getJson<EnrichmentStatusResponse>(
    `/api/enrichment/status?${q(corpusPath)}`,
    'enrichment.status',
  )
}

export async function getEnrichmentHealth(
  corpusPath: string,
): Promise<EnrichmentHealthResponse> {
  return getJson<EnrichmentHealthResponse>(
    `/api/enrichment/health?${q(corpusPath)}`,
    'enrichment.health',
  )
}

export async function getEnrichmentMetrics(
  corpusPath: string,
  window: string = '24h',
): Promise<EnrichmentMetricsResponse> {
  return getJson<EnrichmentMetricsResponse>(
    `/api/enrichment/metrics?${q(corpusPath, { window })}`,
    'enrichment.metrics',
  )
}

export async function getEnrichmentRunSummary(
  corpusPath: string,
): Promise<EnrichmentRunSummary> {
  return getJson<EnrichmentRunSummary>(
    `/api/enrichment/run-summary?${q(corpusPath)}`,
    'enrichment.run_summary',
  )
}

export async function getEnrichmentEvents(
  corpusPath: string,
  opts: { enricher_id?: string; event_type?: string; limit?: number } = {},
): Promise<EnrichmentEventsResponse> {
  const params: Record<string, string> = {}
  if (opts.enricher_id) params.enricher_id = opts.enricher_id
  if (opts.event_type) params.event_type = opts.event_type
  if (opts.limit) params.limit = String(opts.limit)
  return getJson<EnrichmentEventsResponse>(
    `/api/enrichment/events?${q(corpusPath, params)}`,
    'enrichment.events',
  )
}

export async function getCorpusEnrichmentsCatalogue(
  corpusPath: string,
): Promise<CorpusEnrichmentsCatalogue> {
  return getJson<CorpusEnrichmentsCatalogue>(
    `/api/corpus/enrichments?${q(corpusPath)}`,
    'corpus.enrichments.list',
  )
}

export async function reEnableEnricher(
  corpusPath: string,
  enricherId: string,
  reason: string = 'operator re-enable from viewer',
): Promise<EnricherHealthRecord> {
  const res = await fetchWithTimeout(
    `/api/enrichment/health/${encodeURIComponent(enricherId)}/re-enable?${q(corpusPath)}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason }),
    },
    { timeoutDetail: 'enrichment.re_enable' },
  )
  if (!res.ok) {
    throw new Error(await readApiErrorMessage(res))
  }
  return (await res.json()) as EnricherHealthRecord
}

export async function submitEnrichmentJob(
  corpusPath: string,
  body: EnrichmentJobSubmitRequest = {},
): Promise<EnrichmentJobAccepted> {
  const res = await fetchWithTimeout(
    `/api/jobs/enrichment?${q(corpusPath)}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
    { timeoutDetail: 'jobs.enrichment.submit' },
  )
  if (!res.ok) {
    throw new Error(await readApiErrorMessage(res))
  }
  return (await res.json()) as EnrichmentJobAccepted
}
