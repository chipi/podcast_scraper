import { dedupeInFlight } from './inFlightDedupe'
import { fetchWithTimeout } from './httpClient'

export interface CorpusStatsResponse {
  path: string
  publish_month_histogram: Record<string, number>
  catalog_episode_count: number
  catalog_feed_count: number
  digest_topics_configured: number
}

export interface CorpusRunSummaryItem {
  relative_path: string
  run_id: string
  created_at: string | null
  run_duration_seconds: number | null
  episodes_scraped_total: number | null
  errors_total: number | null
  gi_artifacts_generated: number | null
  kg_artifacts_generated: number | null
  time_scraping_seconds: number | null
  time_parsing_seconds: number | null
  time_normalizing_seconds: number | null
  time_io_and_waiting_seconds: number | null
  episode_outcomes: Record<string, number>
  // #656 Stage B: #652 Part B filter counters. ``null`` when the run
  // predates #652 (dashboard renders these as "—").
  ads_filtered_count: number | null
  dialogue_insights_dropped_count: number | null
  topics_normalized_count: number | null
  entity_kinds_repaired_count: number | null
  // #656 Stage D: #663 pre-extraction ad-region excision counters.
  ad_chars_excised_preroll: number | null
  ad_chars_excised_postroll: number | null
  ad_episodes_with_excision_count: number | null
}

export interface CorpusRunsSummaryResponse {
  path: string
  runs: CorpusRunSummaryItem[]
}

export interface CorpusManifestFeed {
  feed_url?: string
  stable_feed_dir?: string
  episodes_processed?: number
  last_run_finished_at?: string
  ok?: boolean
}

export interface CorpusManifestDocument {
  schema_version?: string
  feeds?: CorpusManifestFeed[]
}

async function readJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || `HTTP ${res.status}`)
  }
  return (await res.json()) as T
}

export async function fetchCorpusStats(corpusPath: string): Promise<CorpusStatsResponse> {
  const key = corpusPath.trim()
  const q = new URLSearchParams({ path: key })
  const qs = q.toString()
  return dedupeInFlight(`GET|/api/corpus/stats?${qs}`, async () => {
    const res = await fetchWithTimeout(`/api/corpus/stats?${qs}`)
    return readJson<CorpusStatsResponse>(res)
  })
}

export async function fetchCorpusRunsSummary(
  corpusPath: string,
): Promise<CorpusRunsSummaryResponse> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const qs = q.toString()
  return dedupeInFlight(`GET|/api/corpus/runs/summary?${qs}`, async () => {
    const res = await fetchWithTimeout(`/api/corpus/runs/summary?${qs}`)
    return readJson<CorpusRunsSummaryResponse>(res)
  })
}

export async function fetchCorpusManifest(
  corpusPath: string,
): Promise<CorpusManifestDocument> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const qs = q.toString()
  return dedupeInFlight(`GET|/api/corpus/documents/manifest?${qs}`, async () => {
    const res = await fetchWithTimeout(`/api/corpus/documents/manifest?${qs}`)
    return readJson<CorpusManifestDocument>(res)
  })
}

/** ``corpus_run_summary.json`` at corpus root (multi-feed batch rollup). */
export type CorpusRunSummaryDocument = Record<string, unknown>

export async function fetchCorpusRunSummaryDocument(
  corpusPath: string,
): Promise<CorpusRunSummaryDocument> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const qs = q.toString()
  return dedupeInFlight(`GET|/api/corpus/documents/run-summary?${qs}`, async () => {
    const res = await fetchWithTimeout(`/api/corpus/documents/run-summary?${qs}`)
    return readJson<CorpusRunSummaryDocument>(res)
  })
}
